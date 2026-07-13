#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>
#include <set>
#include <sstream>
#include "ASTree.h"
#include "FastStack.h"
#include "pyc_numeric.h"
#include "bytecode.h"
#include "ASTree_priv.h"

// Triple-quote delimiter for interpolated f-string literals (see ASTree.cpp).
#define F_STRING_QUOTE "'''"

/* ==========================================================================
 * ASTRender.cpp -- PHASE 3 of the decompiler: AST -> source text.
 *
 * A recursive printer (print_src / print_block) plus the layout engine that
 * reproduces the original source positions: cmp_prec (minimal parenthesization),
 * padToCol, breakBeforeElem (line-faithful multi-line expressions), and
 * compound-statement joining. INVARIANT: the layout engine only adds/removes
 * whitespace and redundant parentheses, so it never changes the recompiled
 * byte-code -- only the reproduced co_positions. Shared state with the builder
 * lives in ASTree_priv.h.
 * ========================================================================== */

static int bin_arith_prec(int op)
{
    switch (op) {
    case ASTBinary::BIN_POWER:        return 7;   // right-associative (handled below)
    case ASTBinary::BIN_MULTIPLY:
    case ASTBinary::BIN_DIVIDE:
    case ASTBinary::BIN_FLOOR_DIVIDE:
    case ASTBinary::BIN_MODULO:
    case ASTBinary::BIN_MAT_MULTIPLY: return 6;
    case ASTBinary::BIN_ADD:
    case ASTBinary::BIN_SUBTRACT:     return 5;
    case ASTBinary::BIN_LSHIFT:
    case ASTBinary::BIN_RSHIFT:       return 4;
    case ASTBinary::BIN_AND:          return 3;
    case ASTBinary::BIN_XOR:          return 2;
    case ASTBinary::BIN_OR:           return 1;
    default:                          return -1;  // booleans / unknown
    }
}

static int cmp_prec(PycRef<ASTNode> parent, PycRef<ASTNode> child)
{
    /* Determine whether the parent has higher precedence than therefore
       child, so we don't flood the source code with extraneous parens.
       Else we'd have expressions like (((a + b) + c) + d) when therefore
       equivalent, a + b + c + d would suffice. */

    if (parent.type() == ASTNode::NODE_UNARY && parent.cast<ASTUnary>()->op() == ASTUnary::UN_NOT) {
        /* `not` binds looser than comparison/arithmetic/bitwise/attr/subscript
           (all tighter -> no parens) but tighter than boolean and/or. Only
           parenthesize the operand when it is itself an `and`/`or` expression;
           anything else the original source wrote without parens (`not x`,
           `not a.b`, `not a == b`). Ternary operands are still parenthesized by
           print_ordered. Parens here are co_code-inert. */
        if (child.type() == ASTNode::NODE_BINARY) {
            int cop = child.cast<ASTBinary>()->op();
            return (cop == ASTBinary::BIN_LOG_AND || cop == ASTBinary::BIN_LOG_OR) ? 1 : -1;
        }
        return -1;
    }
    if (child.type() == ASTNode::NODE_BINARY) {
        PycRef<ASTBinary> binChild = child.cast<ASTBinary>();
        if (parent.type() == ASTNode::NODE_BINARY) {
            PycRef<ASTBinary> binParent = parent.cast<ASTBinary>();
            int pp = bin_arith_prec(binParent->op());
            int cp = bin_arith_prec(binChild->op());
            /* Proper precedence for arithmetic/bitwise ops (the op-enum order is
               NOT a valid precedence, e.g. MUL/DIV/FLOOR_DIV/MOD share a level). */
            if (pp > 0 && cp > 0) {
                if (cp != pp)
                    return cp < pp ? 1 : -1;     // looser child -> parens
                /* Equal precedence. A same-level binary appearing as the RIGHT
                   operand only ever comes from explicit source parens (CPython
                   left-nests same-level ops), so it must be parenthesized to
                   reproduce the right-nested bytecode -- a - (b - c), a // (b * c),
                   a << (b << c), a + (b + c). '**' is right-associative, so the
                   rule is mirrored for it. Left operand at equal precedence is the
                   natural left-nesting and needs no parens. */
                bool isPow = binParent->op() == ASTBinary::BIN_POWER;
                bool onRight = (binParent->right() == child);
                return (onRight != isPow) ? 1 : -1;
            }
            if (binParent->right() == child) {
                if (binParent->op() == ASTBinary::BIN_SUBTRACT &&
                    binChild->op() == ASTBinary::BIN_ADD)
                    return 1;
                else if (binParent->op() == ASTBinary::BIN_DIVIDE &&
                         binChild->op() == ASTBinary::BIN_MULTIPLY)
                    return 1;
            }
            return binChild->op() - binParent->op();
        }
        else if (parent.type() == ASTNode::NODE_COMPARE)
            return (binChild->op() == ASTBinary::BIN_LOG_AND ||
                    binChild->op() == ASTBinary::BIN_LOG_OR) ? 1 : -1;
        else if (parent.type() == ASTNode::NODE_UNARY)
            /* `**` and attribute access bind tighter than a unary -/~/+, so
               `-a ** b` and `-a.b` need no parens; everything looser does. */
            return (binChild->op() == ASTBinary::BIN_POWER ||
                    binChild->op() == ASTBinary::BIN_ATTR) ? -1 : 1;
    } else if (child.type() == ASTNode::NODE_UNARY) {
        PycRef<ASTUnary> unChild = child.cast<ASTUnary>();
        if (parent.type() == ASTNode::NODE_BINARY) {
            PycRef<ASTBinary> binParent = parent.cast<ASTBinary>();
            if (binParent->op() == ASTBinary::BIN_LOG_AND ||
                binParent->op() == ASTBinary::BIN_LOG_OR)
                return -1;
            else if (unChild->op() == ASTUnary::UN_NOT)
                return 1;
            else if (binParent->op() == ASTBinary::BIN_POWER)
                /* The exponent (right operand of **) is a u_expr in the grammar,
                   so a unary exponent needs no parens: `a ** -b`. Only a unary
                   LEFT operand came from explicit source parens: `(-a) ** b`. */
                return (binParent->right() == child) ? -1 : 1;
            else
                return -1;
        } else if (parent.type() == ASTNode::NODE_COMPARE) {
            return (unChild->op() == ASTUnary::UN_NOT) ? 1 : -1;
        } else if (parent.type() == ASTNode::NODE_UNARY) {
            return unChild->op() - parent.cast<ASTUnary>()->op();
        }
    } else if (child.type() == ASTNode::NODE_COMPARE) {
        PycRef<ASTCompare> cmpChild = child.cast<ASTCompare>();
        if (parent.type() == ASTNode::NODE_BINARY)
            return (parent.cast<ASTBinary>()->op() == ASTBinary::BIN_LOG_AND ||
                    parent.cast<ASTBinary>()->op() == ASTBinary::BIN_LOG_OR) ? -1 : 1;
        else if (parent.type() == ASTNode::NODE_COMPARE)
            return 1;
        else if (parent.type() == ASTNode::NODE_UNARY)
            return (parent.cast<ASTUnary>()->op() == ASTUnary::UN_NOT) ? -1 : 1;
    }

    /* For normal nodes, don't parenthesize anything */
    return -1;
}

static void print_ordered(PycRef<ASTNode> parent, PycRef<ASTNode> child,
                          PycModule* mod, std::ostream& pyc_output)
{
    if (child.type() == ASTNode::NODE_BINARY ||
        child.type() == ASTNode::NODE_COMPARE) {
        if (cmp_prec(parent, child) > 0) {
            pyc_output << "(";
            print_src(child, mod, pyc_output);
            pyc_output << ")";
        } else {
            print_src(child, mod, pyc_output);
        }
    } else if (child.type() == ASTNode::NODE_UNARY) {
        if (cmp_prec(parent, child) > 0) {
            pyc_output << "(";
            print_src(child, mod, pyc_output);
            pyc_output << ")";
        } else {
            print_src(child, mod, pyc_output);
        }
    } else if (child.type() == ASTNode::NODE_TERNARY) {
        pyc_output << "(";
        print_src(child, mod, pyc_output);
        pyc_output << ")";
    } else {
        print_src(child, mod, pyc_output);
    }
}

void start_line(int indent, std::ostream& pyc_output)
{
    if (inLambda)
        return;
    for (int i=0; i<indent; i++)
        pyc_output << "    ";
}

void end_line(std::ostream& pyc_output)
{
    if (inLambda)
        return;
    pyc_output << "\n";
}

/* Column-layout engine: pad SPACES (forward only) so the next token lands at
   its original source start column (node->srcCol(), decoded from co_positions).
   Pure intra-line whitespace: co_code is unaffected. Guards:
   - never inside a lambda (blank/space injection there is unsafe) or while
     rendering a signature (g_noCompact) where spacing must stay canonical;
   - only pad forward (c > g_curCol); never remove;
   - a sanity cap prevents a stray huge column (e.g. a mis-decoded span) from
     blowing the line out. A leading space before any leaf token is always
     syntactically inert in Python. */
/* Layout engine: when > 0, collection displays do NOT compact to one line, and
   column padding is suppressed (signature spacing must stay canonical). */
int g_noCompact = 0;
/* Line-faithful expression rendering: a comma-separated element list (call args,
   tuple/list/set/dict) breaks before any element whose source line exceeds the
   previous element's, padding to its source column, reproducing the original
   multi-line layout. The newline inside the brackets/parens is a pure
   continuation, so co_code is byte-identical -- PROVIDED the wrapped expression
   contains NO nested code object: shifting a lambda/comprehension/genexpr onto a
   different source line changes its line-anchor NOP (the old Fix B wall). This
   master switch is on during a clean-build render; each wrap site additionally
   checks subtreeHasNestedCode on the specific expression (per-expression gate),
   so a flat literal is wrapped even when a lambda sits elsewhere in the same
   function. */
bool g_lineFaithful = false;
/* >0 while rendering inside a try/except/finally-protected body. Compound
   statement joining is disabled there: an inlined finally copy / exception
   cleanup emits line-marker NOPs sensitive to the exact line layout, so
   removing a line by joining `a; b` can drop a NOP and change co_code
   (asyncio/locks.wait: `await fut; return True` inside try/finally). */
static int g_protectedDepth = 0;
/* True if the subtree contains a nested code object (lambda/comprehension/def/
   class or a raw code const), whose source line would shift if the enclosing
   expression is re-wrapped. Conservative: an unrecognised node type returns true
   (assume unsafe) so wrapping is only ever suppressed, never wrongly applied. */
static bool subtreeHasNestedCode(const PycRef<ASTNode>& n, int depth = 0)
{
    if (n == nullptr)
        return false;
    if (depth > 60)
        return true;
    switch (n->type()) {
    case ASTNode::NODE_FUNCTION:
    case ASTNode::NODE_CLASS:
    case ASTNode::NODE_COMPREHENSION:
        return true;
    case ASTNode::NODE_OBJECT:
        return n.cast<ASTObject>()->object() != nullptr
                && n.cast<ASTObject>()->object().type() == PycObject::TYPE_CODE;
    case ASTNode::NODE_NAME:
    case ASTNode::NODE_KEYWORD:
    case ASTNode::NODE_IMPORT:
        return false;
    case ASTNode::NODE_BINARY:
    case ASTNode::NODE_COMPARE:
    case ASTNode::NODE_SLICE: {
        PycRef<ASTBinary> b = n.cast<ASTBinary>();
        return subtreeHasNestedCode(b->left(), depth + 1)
                || subtreeHasNestedCode(b->right(), depth + 1);
    }
    case ASTNode::NODE_UNARY:
        return subtreeHasNestedCode(n.cast<ASTUnary>()->operand(), depth + 1);
    case ASTNode::NODE_SUBSCR:
        return subtreeHasNestedCode(n.cast<ASTSubscr>()->name(), depth + 1)
                || subtreeHasNestedCode(n.cast<ASTSubscr>()->key(), depth + 1);
    case ASTNode::NODE_TERNARY: {
        PycRef<ASTTernary> t = n.cast<ASTTernary>();
        return subtreeHasNestedCode(t->if_block(), depth + 1)
                || subtreeHasNestedCode(t->if_expr(), depth + 1)
                || subtreeHasNestedCode(t->else_expr(), depth + 1);
    }
    case ASTNode::NODE_TUPLE:
        for (const auto& v : n.cast<ASTTuple>()->values())
            if (subtreeHasNestedCode(v, depth + 1))
                return true;
        return false;
    case ASTNode::NODE_LIST:
        for (const auto& v : n.cast<ASTList>()->values())
            if (subtreeHasNestedCode(v, depth + 1))
                return true;
        return false;
    case ASTNode::NODE_SET:
        for (const auto& v : n.cast<ASTSet>()->values())
            if (subtreeHasNestedCode(v, depth + 1))
                return true;
        return false;
    case ASTNode::NODE_MAP:
        for (const auto& e : n.cast<ASTMap>()->values())
            if (subtreeHasNestedCode(e.first, depth + 1)
                    || subtreeHasNestedCode(e.second, depth + 1))
                return true;
        return false;
    case ASTNode::NODE_CALL: {
        PycRef<ASTCall> c = n.cast<ASTCall>();
        if (subtreeHasNestedCode(c->func(), depth + 1))
            return true;
        for (const auto& p : c->pparams())
            if (subtreeHasNestedCode(p, depth + 1))
                return true;
        for (const auto& p : c->kwparams())
            if (subtreeHasNestedCode(p.second, depth + 1))
                return true;
        if (c->hasVar() && subtreeHasNestedCode(c->var(), depth + 1))
            return true;
        if (c->hasKW() && subtreeHasNestedCode(c->kw(), depth + 1))
            return true;
        return false;
    }
    default:
        return true;   // unrecognised: assume unsafe
    }
}
/* True when the code object being rendered contains a nested code const; only
   then does a wrap site pay for the per-expression subtreeHasNestedCode scan
   (a fully flat function needs no per-expression check). */
bool g_funcHasNestedCode = false;
/* Effective line-faithful decision for wrapping expression `node`. */
bool lfSafe(const PycRef<ASTNode>& node)
{
    if (!g_lineFaithful)
        return false;
    return !g_funcHasNestedCode || !subtreeHasNestedCode(node);
}
extern int cur_indent;   // defined below; current output indentation level
/* Break before a comma-separated element at its original line/column when the
   element starts a new source line (line-faithful mode). Returns true if it
   emitted a break (so the caller skips the plain ", "). */
static bool breakBeforeElem(int& prevLine, const PycRef<ASTNode>& elem,
                            std::ostream& out, bool first = false)
{
    if (!g_lineFaithful || inLambda || g_noCompact || elem == nullptr)
        return false;
    int cl = elem->srcLine();
    int cc = elem->srcCol();
    if (cl <= 0 || prevLine <= 0 || cl <= prevLine)
        return false;
    if (cl - prevLine > 400)   // guard against bad line data
        return false;
    /* The first element has no preceding one, so break with a bare newline
       (a leading comma `[, a]` would be a syntax error). */
    out << (first ? "\n" : ",\n");
    if (cc > 0 && cc <= 200)
        for (int i = 0; i < cc; i++)
            out << ' ';
    prevLine = cl;
    return true;
}
static inline void padToCol(const PycRef<ASTNode>& node, std::ostream& out)
{
    if (inLambda || g_noCompact)
        return;
    if (node == nullptr)
        return;
    int c = node->srcCol();
    if (c < 0 || g_curCol < 0)
        return;
    if (c <= g_curCol)
        return;
    /* Never re-indent a statement: the leading token sits at the indentation
       column (cur_indent*4). Padding it would push the whole statement right and
       can change block nesting (→ co_code). Only pad tokens already past the
       indent, i.e. interior-of-line whitespace, which is co_code-inert. */
    if (g_curCol <= (cur_indent < 0 ? 0 : cur_indent * 4))
        return;
    if (c - g_curCol > 64)
        return;
    for (int i = g_curCol; i < c; i++)
        out << ' ';
}

/* A const-string kwargs map key that is a valid Python identifier can render as a
   keyword argument (`k=v`) rather than a dict entry (`'k': v`). */
static bool isKwargIdentifier(const std::string& s)
{
    if (s.empty())
        return false;
    if (!(isalpha((unsigned char)s[0]) || s[0] == '_'))
        return false;
    for (char c : s)
        if (!(isalnum((unsigned char)c) || c == '_'))
            return false;
    return true;
}

int cur_indent = -1;
/* A def/class assignment self-manages its own start_line (and previously a
   leading blank line). Under absolute placement the enclosing renderer pads to
   its source line but must NOT also emit start_line, or the leading indent would
   land a blank line before it (over-run) / double-indent it. */
/* For a decorated def/class statement, the placement line is the FIRST
   decorator's line, not the store's line. CPython records that as the nested
   code object's co_firstlineno (its first instruction position), so return it
   when the store is a def/class carrying at least one decorator; otherwise -1
   (use the store's own srcLine). This makes the `@decorator` lines land where
   the original had them; the body re-aligns itself via absolute placement. */
static int decoratedDefFirstLine(const PycRef<ASTNode>& n)
{
    if (n == nullptr || n.type() != ASTNode::NODE_STORE)
        return -1;
    PycRef<ASTNode> src = n.cast<ASTStore>()->src();
    if (src == nullptr)
        return -1;
    PycRef<ASTNode> codeNode;
    if (src.type() == ASTNode::NODE_FUNCTION) {
        if (src.cast<ASTFunction>()->decorators().empty())
            return -1;
        codeNode = src.cast<ASTFunction>()->code();
    } else if (src.type() == ASTNode::NODE_CLASS) {
        if (src.cast<ASTClass>()->decorators().empty())
            return -1;
        codeNode = src.cast<ASTClass>()->code();
    } else {
        return -1;
    }
    if (codeNode == nullptr || codeNode.type() != ASTNode::NODE_OBJECT)
        return -1;
    PycRef<PycObject> obj = codeNode.cast<ASTObject>()->object();
    if (obj == nullptr || obj.type() != PycObject::TYPE_CODE)
        return -1;
    int fl = obj.cast<PycCode>()->firstLine();
    return fl > 0 ? fl : -1;
}

static bool selfPositions(const PycRef<ASTNode>& n)
{
    if (n == nullptr)
        return false;
    /* A try/except/finally wrapper (BLK_CONTAINER) renders its `try:` header via
       print_block of its child blocks, not inline, so like a def/class it
       positions itself and must not receive an outer start_line (which would
       become an indented blank line before `try:`). */
    if (n.type() == ASTNode::NODE_BLOCK)
        return n.cast<ASTBlock>()->blktype() == ASTBlock::BLK_CONTAINER;
    if (n.type() != ASTNode::NODE_STORE)
        return false;
    int st = n.cast<ASTStore>()->src().type();
    return st == ASTNode::NODE_FUNCTION || st == ASTNode::NODE_CLASS;
}
void print_block(PycRef<ASTBlock> blk, PycModule* mod,
                        std::ostream& pyc_output)
{
    ASTBlock::list_t lines = blk->nodes();

    if (lines.size() == 0) {
        PycRef<ASTNode> pass = new ASTKeyword(ASTKeyword::KW_PASS);
        start_line(cur_indent, pyc_output);
        print_src(pass, mod, pyc_output);
    }

    /* Layout engine: ABSOLUTE line placement. Each statement carries its source
       line (srcLine(), from the .pyc line table); g_curLine is the current
       output line (maintained by the newline-counting stream buffer). Before a
       statement, pad blank lines until the output line reaches its source line,
       so the rendered construct lands on its original line and a recompile
       reproduces the original co_positions. Purely blank lines -> co_code is
       unchanged. Padding is forward-only: if a preceding construct already
       over-ran past the target line, we cannot go back, so we just render where
       we are (that misalignment is what later over-run-reduction phases close).
       The gap is capped so bad/missing line data degrades gracefully. */
    /* Count non-suppressed statements so we can drop an all-suppressed body to a
       `pass` and emit separators correctly around suppressed nodes. */
    int nVisible = 0;
    for (const auto& n : lines)
        if (n == nullptr || !n->suppressed())
            nVisible++;
    if (nVisible == 0 && !lines.empty()) {
        PycRef<ASTNode> pass = new ASTKeyword(ASTKeyword::KW_PASS);
        start_line(cur_indent, pyc_output);
        print_src(pass, mod, pyc_output);
        return;
    }
    bool anyEmitted = false;
    /* Compound-statement joining: consecutive SIMPLE statements the compiler
       attributed to the SAME source line were written `a; b` on one physical
       line; render them `; `-joined instead of one-per-line so the line layout
       (and every later def's position) matches. co_code is byte-identical
       (verified). Only simple statements qualify -- a block header (if/for/
       while/try/with/def/class/match/case) always needs its own line, and a
       match/case body is co_code-load-bearing. */
    int prevJoinLine = -1;
    for (auto ln = lines.cbegin(); ln != lines.cend(); ++ln) {
        if (*ln != nullptr && (*ln)->suppressed())
            continue;   // layout-suppressed: no text, no line consumed
        bool isNodeList = (*ln).cast<ASTNode>().type() == ASTNode::NODE_NODELIST;
        bool joinable = !inLambda && !isNodeList && *ln != nullptr
                && g_protectedDepth == 0
                && (*ln).type() != ASTNode::NODE_BLOCK
                && !selfPositions(*ln);
        int myLine = (*ln != nullptr) ? (*ln)->srcLine() : -1;
        bool doJoin = anyEmitted && joinable && prevJoinLine > 0
                && myLine == prevJoinLine;
        if (doJoin) {
            pyc_output << "; ";
        } else {
            if (anyEmitted)
                end_line(pyc_output);
            if (!isNodeList) {
                int decoLine = decoratedDefFirstLine(*ln);
                int curLine = decoLine > 0 ? decoLine : (*ln)->srcLine();
                if (!inLambda && curLine > 0 && g_curLine > 0 && curLine > g_curLine
                        && curLine - g_curLine <= 2000) {
                    while (g_curLine < curLine)
                        pyc_output << "\n";
                }
                if (!selfPositions(*ln))
                    start_line(cur_indent, pyc_output);
            }
        }
        print_src(*ln, mod, pyc_output);
        prevJoinLine = joinable ? myLine : -1;
        anyEmitted = true;
    }
}

static int f_string_depth = 0;
static const char* f_string_quote()
{
    return (f_string_depth & 1) ? "\"\"\"" : "'''";
}

// A string literal that appears inside an f-string replacement field (the
// expression between '{' and '}') may not contain a backslash, because the
// surrounding f-string is a single token whose expression part is parsed
// without escape processing.  The default string printer escapes embedded
// quotes, newlines, tabs and so on with backslashes, which is rejected inside
// a replacement field.  When such a literal contains only characters that can
// be written verbatim, an equivalent backslash-free spelling exists by picking
// a quote delimiter that does not occur in the value (and, for embedded
// newlines, a triple-quoted form so the raw newline can be written literally).
// Returns true and emits the literal when a backslash-free form is available;
// returns false (printing nothing) when the value cannot be represented this
// way, so the caller can fall back to the normal escaped rendering.
static bool print_f_string_expr_literal(std::ostream& pyc_output,
                                        const PycRef<PycString>& str)
{
    switch (str->type()) {
    case PycObject::TYPE_UNICODE:
    case PycObject::TYPE_INTERNED:
    case PycObject::TYPE_ASCII:
    case PycObject::TYPE_ASCII_INTERNED:
    case PycObject::TYPE_SHORT_ASCII:
    case PycObject::TYPE_SHORT_ASCII_INTERNED:
        break;
    default:
        // Bytes literals and anything else keep their normal rendering.
        return false;
    }

    const std::string& v = str->strValue();
    if (v.empty())
        return false;   // The default "''" rendering is already backslash-free.
    bool hasNewline = false;
    const bool isUnicode = (str->type() == PycObject::TYPE_UNICODE);
    for (unsigned char ch : v) {
        if (ch == '\n' || ch == '\r') {
            // A raw newline can only appear inside a triple-quoted literal.
            hasNewline = true;
        } else if (ch == '\t') {
            // A tab can be written verbatim in any literal.
        } else if (ch < 0x20 || ch == 0x7F) {
            // Any other control character would require a backslash escape.
            return false;
        } else if (ch >= 0x80 && !isUnicode) {
            // A high byte in a non-unicode literal needs a \x escape.
            return false;
        }
    }

    // The delimiter was emitted before f_string_depth was incremented for this
    // expression, so the enclosing quote corresponds to the previous depth.
    const char outerChar = ((f_string_depth - 1) & 1) ? '"' : '\'';

    // The f-string scanner closes on the outer triple-quote run greedily, even
    // when that run lies inside a nested string literal, so a value containing
    // it cannot be embedded at all.
    if (v.find(std::string(3, outerChar)) != std::string::npos)
        return false;

    auto runOf = [&](char q) {
        return v.find(std::string(3, q)) != std::string::npos;
    };

    // Prefer the most compact delimiter that needs no escaping.  A raw newline
    // can only live inside a triple-quoted literal.
    const char* delim = nullptr;
    if (!hasNewline) {
        if (v.find('"') == std::string::npos && outerChar != '"')
            delim = "\"";
        else if (v.find('\'') == std::string::npos && outerChar != '\'')
            delim = "'";
    }
    if (!delim) {
        // Triple-quoted candidates: avoid a matching run, the outer delimiter,
        // and a boundary collision (a value starting or ending with the quote
        // char, which would merge with the opening/closing triple run).
        if (outerChar != '"' && !runOf('"') && v.front() != '"' && v.back() != '"')
            delim = "\"\"\"";
        else if (outerChar != '\'' && !runOf('\'') && v.front() != '\'' && v.back() != '\'')
            delim = "'''";
    }
    if (!delim)
        return false;

    pyc_output << delim << v << delim;
    return true;
}

void print_formatted_value(PycRef<ASTFormattedValue> formatted_value, PycModule* mod,
                           std::ostream& pyc_output)
{
    pyc_output << "{";
    print_src(formatted_value->val(), mod, pyc_output);

    switch (formatted_value->conversion() & ASTFormattedValue::CONVERSION_MASK) {
    case ASTFormattedValue::NONE:
        break;
    case ASTFormattedValue::STR:
        pyc_output << "!s";
        break;
    case ASTFormattedValue::REPR:
        pyc_output << "!r";
        break;
    case ASTFormattedValue::ASCII:
        pyc_output << "!a";
        break;
    }
    if (formatted_value->conversion() & ASTFormattedValue::HAVE_FMT_SPEC) {
        pyc_output << ":";
        PycRef<ASTNode> spec = formatted_value->format_spec();
        if (spec == nullptr) {
        } else if (spec.type() == ASTNode::NODE_OBJECT) {
            PycRef<PycString> s = spec.cast<ASTObject>()->object().try_cast<PycString>();
            if (s != nullptr)
                pyc_output << s->value();
            else
                print_const(pyc_output, spec.cast<ASTObject>()->object(), mod, F_STRING_QUOTE);
        } else if (spec.type() == ASTNode::NODE_JOINEDSTR) {
            for (const auto& val : spec.cast<ASTJoinedStr>()->values()) {
                if (val.type() == ASTNode::NODE_FORMATTEDVALUE)
                    print_formatted_value(val.cast<ASTFormattedValue>(), mod, pyc_output);
                else if (val.type() == ASTNode::NODE_OBJECT)
                    print_const(pyc_output, val.cast<ASTObject>()->object(), mod, F_STRING_QUOTE);
            }
        } else if (spec.type() == ASTNode::NODE_FORMATTEDVALUE) {
            print_formatted_value(spec.cast<ASTFormattedValue>(), mod, pyc_output);
        } else {
            print_src(spec, mod, pyc_output);
        }
    }
    pyc_output << "}";
}

static std::unordered_set<ASTNode *> node_seen;

static bool chainHasBreak(const PycRef<ASTBinary>& bin)
{
    if (bin == nullptr)
        return false;
    if (bin->breakBefore())
        return true;
    for (const PycRef<ASTNode>& side : { bin->left(), bin->right() }) {
        if (side != nullptr && side.type() == ASTNode::NODE_BINARY) {
            int op = side.cast<ASTBinary>()->op();
            if ((op == ASTBinary::BIN_LOG_AND || op == ASTBinary::BIN_LOG_OR)
                    && chainHasBreak(side.cast<ASTBinary>()))
                return true;
        }
    }
    return false;
}

static bool inBreakingBool = false;

/* Collect the distinct original short-circuit-jump lines of every LOG_AND/LOG_OR
   node in this subtree. A boolean sub-group whose operators all share one
   original line was written by the developer on a single line; it can be hoisted
   into a hanging paren rendered INLINE. A sub-group spanning multiple original
   lines must keep its own per-operator line breaks instead (an inline hoist
   would collapse them onto one line and change the recompiled threading). */
static bool boolAllSameLine(const PycRef<ASTNode>& n, int& line)
{
    if (n == nullptr || n->type() != ASTNode::NODE_BINARY)
        return true;
    PycRef<ASTBinary> b = n.cast<ASTBinary>();
    int op = b->op();
    if (op != ASTBinary::BIN_LOG_AND && op != ASTBinary::BIN_LOG_OR)
        return true;
    int l = b->scLine();
    if (l < 0)
        return false;
    if (line < 0)
        line = l;
    else if (line != l)
        return false;
    return boolAllSameLine(b->left(), line)
            && boolAllSameLine(b->right(), line);
}

/* When true, a hoisted boolean operand is rendered inline (one line) — set while
   printing the contents of a hanging-paren block (see printBoolOperand below). */
static bool boolInline = false;

static bool starUnpackNeedsParens(const PycRef<ASTNode>& n)
{
    if (n == nullptr)
        return false;
    if (n.type() == ASTNode::NODE_TERNARY)
        return true;
    if (n.type() == ASTNode::NODE_BINARY) {
        int op = n.cast<ASTBinary>()->op();
        return op == ASTBinary::BIN_LOG_AND || op == ASTBinary::BIN_LOG_OR;
    }
    return false;
}

/* Layout engine: true when every element of a collection shares one source line
   (or carries no line info), so a list/set/dict display can render compactly on
   a single output line instead of one element per line -- eliminating the
   over-run that a single-line source collection would otherwise incur. */
template <typename Container>
static bool seqOneSrcLine(const Container& c)
{
    int line = -1;
    for (const auto& n : c) {
        if (n == nullptr)
            continue;
        int l = n->srcLine();
        if (l <= 0)
            continue;
        if (line < 0)
            line = l;
        else if (l != line)
            return false;
    }
    return true;
}

/* Render an rvalue in a "bare tuple" statement context (return/yield value,
   assignment RHS). A multi-element tuple there is canonically written without
   surrounding parentheses (`return a, b`, not `return (a, b)`); the parens only
   widen the line and push the elements right of their source columns. Suppress
   them on the OUTERMOST tuple while rendering, restoring the flag afterwards so
   nested tuples keep their own parens. Single-element tuples keep the parens
   (the trailing-comma-only `(x,)` form would otherwise be ambiguous). Bytecode
   is identical -- BUILD_TUPLE is emitted either way. */
static void print_bare_tuple_value(PycRef<ASTNode> node, PycModule* mod,
                                   std::ostream& pyc_output)
{
    PycRef<ASTTuple> tup =
            (node != nullptr && node.type() == ASTNode::NODE_TUPLE)
            ? node.cast<ASTTuple>() : nullptr;
    if (tup != nullptr && tup->values().size() > 1 && tup->requireParens()) {
        tup->setRequireParens(false);
        print_src(node, mod, pyc_output);
        tup->setRequireParens(true);
    } else {
        print_src(node, mod, pyc_output);
    }
}

void print_src(PycRef<ASTNode> node, PycModule* mod, std::ostream& pyc_output)
{
    if (node == NULL) {
        pyc_output << "None";
        cleanBuild = true;
        return;
    }

    if (node_seen.find((ASTNode *)node) != node_seen.end()) {
        fputs("WARNING: Circular reference detected\n", stderr);
        return;
    }
    node_seen.insert((ASTNode *)node);

    switch (node->type()) {
    case ASTNode::NODE_BINARY:
    case ASTNode::NODE_COMPARE:
        {
            PycRef<ASTBinary> bin = node.cast<ASTBinary>();
            bool attrIntObj = false;
            if (node.type() == ASTNode::NODE_BINARY
                    && bin->op() == ASTBinary::BIN_ATTR
                    && bin->left().type() == ASTNode::NODE_OBJECT) {
                int t = bin->left().cast<ASTObject>()->object().type();
                attrIntObj = (t == PycObject::TYPE_INT || t == PycObject::TYPE_INT64
                              || t == PycObject::TYPE_LONG);
            }
            bool isLog = (node.type() == ASTNode::NODE_BINARY
                    && (bin->op() == ASTBinary::BIN_LOG_AND
                        || bin->op() == ASTBinary::BIN_LOG_OR));
            bool wrapBreak = false;
            if (isLog && !inLambda && !inBreakingBool && !boolInline
                    && chainHasBreak(bin)) {
                wrapBreak = true;
                inBreakingBool = true;
                pyc_output << "(";
            }
            /* Reproduce the original's per-operator short-circuit form (OR vs
               threaded CB). CPython 3.11's jump_thread folds a JUMP_IF_x_OR_POP
               into a POP_JUMP only when the jump and its target share a source
               line. A boolop sub-group written by the developer on its own
               line(s) (a hanging paren) keeps its short-circuit jump on a
               different line than its target, so it stays un-threaded (OR). When
               a nested boolean operand was un-threaded in the original
               (breakBefore), hoist it into a hanging-paren block on its own line
               so recompilation reproduces the same un-threaded form; the paren
               makes the newline a valid continuation even inside a lambda. */
            auto printBoolOperand = [&](const PycRef<ASTNode>& operand) {
                bool hang = false, inlineHang = false;
                if (isLog && !boolInline && operand != nullptr
                        && operand.type() == ASTNode::NODE_BINARY) {
                    PycRef<ASTBinary> ob = operand.cast<ASTBinary>();
                    int oop = ob->op();
                    if ((oop == ASTBinary::BIN_LOG_AND
                                || oop == ASTBinary::BIN_LOG_OR)
                            && ob->breakBefore()) {
                        int gl = -1;
                        hang = true;
                        /* A sub-group whose operators all share one original line
                           was written on a single source line and is hoisted
                           inline; one spanning several original lines keeps its own
                           per-operator line breaks (an inline hoist would collapse
                           it onto one line and change the recompiled threading). */
                        inlineHang = boolAllSameLine(operand, gl);
                    }
                }
                if (!hang) {
                    print_ordered(node, operand, mod, pyc_output);
                    return;
                }
                pyc_output << "(\n";
                for (int i = 0; i < cur_indent + 3; i++)
                    pyc_output << "    ";
                bool s1 = inBreakingBool, s2 = boolInline;
                inBreakingBool = true;
                boolInline = inlineHang;   // inline only for a single-line group
                print_src(operand, mod, pyc_output);
                inBreakingBool = s1;
                boolInline = s2;
                pyc_output << "\n";
                for (int i = 0; i < cur_indent + 2; i++)
                    pyc_output << "    ";
                pyc_output << ")";
            };
            if (attrIntObj)
                pyc_output << "(";
            printBoolOperand(bin->left());
            if (attrIntObj)
                pyc_output << ")";
            if (isLog && bin->breakBefore() && !inLambda && !boolInline) {
                pyc_output << "\n";
                for (int i = 0; i < cur_indent + 2; i++)
                    pyc_output << "    ";
                pyc_output << (bin->op_str() + 1);
            } else {
                pyc_output << bin->op_str();
            }
            /* A non-empty set literal on the right of `in`/`not in` is folded to a
               frozenset constant by the peephole optimizer. A frozenset has no
               literal syntax, so render the constant back as a set literal `{…}`
               rather than `frozenset({…})` (which recompiles to BUILD_SET + a call
               and diverges). Empty frozensets keep `frozenset()` (`{}` is a dict). */
            bool froze = false;
            if ((bin->op() == ASTCompare::CMP_IN || bin->op() == ASTCompare::CMP_NOT_IN)
                    && bin->right().type() == ASTNode::NODE_OBJECT) {
                PycRef<PycObject> ro = bin->right().cast<ASTObject>()->object();
                if (ro->type() == PycObject::TYPE_FROZENSET
                        && !ro.cast<PycSet>()->values().empty()) {
                    froze = true;
                    pyc_output << "{";
                    const PycSet::value_t& vs = ro.cast<PycSet>()->values();
                    auto it = vs.cbegin();
                    print_const(pyc_output, *it, mod);
                    while (++it != vs.cend()) {
                        pyc_output << ", ";
                        print_const(pyc_output, *it, mod);
                    }
                    pyc_output << "}";
                }
            }
            if (!froze)
                printBoolOperand(bin->right());
            if (wrapBreak) {
                pyc_output << ")";
                inBreakingBool = false;
            }
        }
        break;
    case ASTNode::NODE_CHAINCOMPARE:
        {
            static const char* s_cmp_strings[] = {
                " < ", " <= ", " == ", " != ", " > ", " >= ",
                " in ", " not in ", " is ", " is not ", "<EXCEPTION MATCH>", "<BAD>"
            };
            PycRef<ASTChainCompare> cc = node.cast<ASTChainCompare>();
            const auto& ops = cc->ops();
            const auto& operands = cc->operands();
            /* An operand that is ITSELF a comparison (`a in B != (c in D)`) must be
               parenthesized — otherwise it fuses into the chain as extra links and the
               re-parse diverges (`a in B != c in D` would chain a,B,c,D). */
            auto printOperand = [&](const PycRef<ASTNode>& o) {
                bool paren = (o != nullptr
                        && (o->type() == ASTNode::NODE_COMPARE
                            || o->type() == ASTNode::NODE_CHAINCOMPARE));
                if (paren) pyc_output << "(";
                print_src(o, mod, pyc_output);
                if (paren) pyc_output << ")";
            };
            printOperand(operands.front());
            for (size_t i = 0; i < ops.size(); ++i) {
                int op = ops[i];
                pyc_output << ((op >= 0 && op < (int)(sizeof(s_cmp_strings)/sizeof(*s_cmp_strings)))
                              ? s_cmp_strings[op] : " ? ");
                printOperand(operands[i + 1]);
            }
        }
        break;
    case ASTNode::NODE_AWAITABLE:
        {
            if (!node.cast<ASTAwaitable>()->implicit())
                pyc_output << "await ";
            print_src(node.cast<ASTAwaitable>()->expression(), mod, pyc_output);
        }
        break;
    case ASTNode::NODE_UNARY:
        {
            PycRef<ASTUnary> un = node.cast<ASTUnary>();
            pyc_output << un->op_str();
            if (un->op() == ASTUnary::UN_STAR
                    && starUnpackNeedsParens(un->operand())) {
                pyc_output << "(";
                print_src(un->operand(), mod, pyc_output);
                pyc_output << ")";
            } else {
                print_ordered(node, un->operand(), mod, pyc_output);
            }
        }
        break;
    case ASTNode::NODE_CALL:
        {
            PycRef<ASTCall> call = node.cast<ASTCall>();
            PycRef<ASTNode> fn = call->func();
            bool fnParen = fn != NULL && fn.type() == ASTNode::NODE_TERNARY;
            if (fnParen) pyc_output << "(";
            print_src(fn, mod, pyc_output);
            if (fnParen) pyc_output << ")";
            pyc_output << "(";
            bool first = true;
            /* A generator expression that is the SOLE argument of the call
               (one positional param, no keyword/`*`/`**` args) may drop its own
               parentheses: `sum(x for x in y)` rather than `sum((x for x in y))`.
               The genexpr bytecode (MAKE_FUNCTION + CALL) is identical. */
            bool soleGenArg = call->pparams().size() == 1
                    && call->kwparams().empty()
                    && !call->hasVar() && !call->hasKW()
                    && call->pparams().front() != nullptr
                    && call->pparams().front().type() == ASTNode::NODE_COMPREHENSION
                    && call->pparams().front().cast<ASTComprehension>()->comptype()
                           == ASTComprehension::COMP_GENERATOR;
            int elemLine = -1;   // src line of last-rendered arg (line-faithful breaks)
            bool callLF = lfSafe(node);   // per-expression gate
            for (const auto& param : call->pparams()) {
                if (!first) {
                    if (!(callLF && breakBeforeElem(elemLine, param, pyc_output)))
                        pyc_output << ", ";
                } else if (param != nullptr && param->srcLine() > 0) {
                    elemLine = param->srcLine();
                }
                if (soleGenArg)
                    suppressGenExprParens = true;
                /* A lambda argument's own parens are redundant (its body is a
                   `test`, so the next arg-list comma terminates it). */
                if (param != nullptr && param.type() == ASTNode::NODE_FUNCTION)
                    suppressLambdaParens = true;
                print_src(param, mod, pyc_output);
                suppressGenExprParens = false;
                suppressLambdaParens = false;
                first = false;
            }
            for (const auto& param : call->kwparams()) {
                if (!first) {
                    if (!(callLF && breakBeforeElem(elemLine, param.second, pyc_output)))
                        pyc_output << ", ";
                } else if (param.second != nullptr && param.second->srcLine() > 0) {
                    elemLine = param.second->srcLine();
                }
                if (param.first.type() == ASTNode::NODE_NAME) {
                    pyc_output << param.first.cast<ASTName>()->name()->value() << "=";
                } else {
                    PycRef<PycString> str_name = param.first.cast<ASTObject>()->object().cast<PycString>();
                    pyc_output << str_name->value() << "=";
                }
                if (param.second != nullptr
                        && param.second.type() == ASTNode::NODE_FUNCTION)
                    suppressLambdaParens = true;
                print_src(param.second, mod, pyc_output);
                suppressLambdaParens = false;
                first = false;
            }
            /* CALL_FUNCTION_EX with only keyword args still carries an empty
               positional tuple; `f(*(), **kw)` is equivalent to (and recompiles
               identically to) the canonical `f(**kw)`, so drop the empty `*()`
               when a `**` spread follows. */
            bool emptyVarWithKW = false;
            if (call->hasVar() && call->hasKW()
                    && call->var() != nullptr
                    && call->var().type() == ASTNode::NODE_TUPLE
                    && call->var().cast<ASTTuple>()->values().empty())
                emptyVarWithKW = true;
            if (call->hasVar() && !emptyVarWithKW) {
                /* CALL_FUNCTION_EX packs the positional side into one tuple.
                   When that tuple ALREADY contains a `*` spread element it was
                   built with a LIST/TUPLE spread, so the individual elements
                   recompile to the SAME bytecode whether written as
                   `f(*(a, *b))` or spread inline as `f(a, *b)`; the inline form
                   is canonical. This equivalence holds ONLY when an inner spread
                   is present -- a pure literal tuple (`f(*(a, b))`) compiles
                   differently from `f(a, b)`, so it keeps the `*(...)` form. */
                PycRef<ASTTuple> varTup =
                        (call->var() != nullptr
                         && call->var().type() == ASTNode::NODE_TUPLE)
                        ? call->var().cast<ASTTuple>() : nullptr;
                bool spreadInline = false;
                if (varTup != nullptr && varTup->values().size() > 1) {
                    for (const auto& el : varTup->values()) {
                        if (el != nullptr && el.type() == ASTNode::NODE_UNARY
                                && el.cast<ASTUnary>()->op() == ASTUnary::UN_STAR) {
                            spreadInline = true;
                            break;
                        }
                    }
                }
                if (spreadInline) {
                    for (const auto& el : varTup->values()) {
                        if (!first)
                            pyc_output << ", ";
                        print_src(el, mod, pyc_output);
                        first = false;
                    }
                } else {
                    if (!first)
                        pyc_output << ", ";
                    pyc_output << "*";
                    print_src(call->var(), mod, pyc_output);
                    first = false;
                }
            }
            if (call->hasKW()) {
                /* `f(*a, k=v, **rest)` bundles the explicit keyword args AND the
                   `**rest` spread into ONE kwargs map (BUILD_CONST_KEY_MAP {k:v} +
                   DICT_MERGE rest) for CALL_FUNCTION_EX. Rendering that map as a dict
                   display `**{k: v, **rest}` recompiles with a spurious
                   BUILD_MAP+DICT_UPDATE. When every entry is a const-IDENTIFIER key
                   (a real keyword arg) or the null-key `**` spread, unroll to
                   `k=v, **rest`. (A single `**X` spread unrolls to `**X`.) */
                PycRef<ASTNode> kwnode = call->kw();
                bool unrolled = false;
                if (kwnode->type() == ASTNode::NODE_MAP) {
                    const ASTMap::map_t& mv = kwnode.cast<ASTMap>()->values();
                    bool ok = !mv.empty();
                    for (const auto& e : mv) {
                        if (e.first == nullptr)
                            continue;               // `**` spread (null key)
                        PycRef<PycString> ks = e.first->type() == ASTNode::NODE_OBJECT
                                ? e.first.cast<ASTObject>()->object().try_cast<PycString>()
                                : nullptr;
                        if (ks == nullptr || !isKwargIdentifier(ks->value())) { ok = false; break; }
                    }
                    if (ok) {
                        for (const auto& e : mv) {
                            if (!first)
                                pyc_output << ", ";
                            if (e.first == nullptr) {
                                pyc_output << "**";
                                bool par = starUnpackNeedsParens(e.second);
                                if (par) pyc_output << "(";
                                print_src(e.second, mod, pyc_output);
                                if (par) pyc_output << ")";
                            } else {
                                pyc_output << e.first.cast<ASTObject>()->object().cast<PycString>()->value()
                                           << "=";
                                print_src(e.second, mod, pyc_output);
                            }
                            first = false;
                        }
                        unrolled = true;
                    }
                }
                /* `f(*a, k=v)` with NO `**rest` builds the kwargs as a const-key map
                   (BUILD_CONST_KEY_MAP); rendering it as a dict display `**{k: v}`
                   recompiles with a spurious BUILD_MAP+DICT_MERGE. Unroll to `k=v`
                   when every key is a const identifier. Values are stored in reverse
                   (the const-map render pairs keys[i] with values[n-1-i]). */
                else if (kwnode->type() == ASTNode::NODE_CONST_MAP) {
                    PycRef<ASTConstMap> cm = kwnode.cast<ASTConstMap>();
                    PycRef<ASTNode> keysNode = cm->keys();
                    if (keysNode != nullptr && keysNode.type() == ASTNode::NODE_OBJECT) {
                        PycRef<PycObject> ko = keysNode.cast<ASTObject>()->object();
                        if (ko != nullptr && (ko.type() == PycObject::TYPE_TUPLE
                                || ko.type() == PycObject::TYPE_SMALL_TUPLE)) {
                            PycTuple::value_t keys = ko.cast<PycTuple>()->values();
                            ASTConstMap::values_t vals = cm->values();
                            bool ok = !keys.empty() && keys.size() == vals.size();
                            for (const auto& k : keys) {
                                PycRef<PycString> ks = k.try_cast<PycString>();
                                if (ks == nullptr || !isKwargIdentifier(ks->value())) {
                                    ok = false;
                                    break;
                                }
                            }
                            if (ok) {
                                for (size_t i = 0; i < keys.size(); i++) {
                                    if (!first)
                                        pyc_output << ", ";
                                    pyc_output << keys[i].cast<PycString>()->value() << "=";
                                    print_src(vals[vals.size() - 1 - i], mod, pyc_output);
                                    first = false;
                                }
                                unrolled = true;
                            }
                        }
                    }
                }
                if (!unrolled) {
                    if (!first)
                        pyc_output << ", ";
                    pyc_output << "**";
                    bool par = starUnpackNeedsParens(kwnode);
                    if (par) pyc_output << "(";
                    print_src(kwnode, mod, pyc_output);
                    if (par) pyc_output << ")";
                    first = false;
                }
            }
            pyc_output << ")";
        }
        break;
    case ASTNode::NODE_DELETE:
        {
            pyc_output << "del ";
            print_src(node.cast<ASTDelete>()->value(), mod, pyc_output);
        }
        break;
    case ASTNode::NODE_EXEC:
        {
            PycRef<ASTExec> exec = node.cast<ASTExec>();
            pyc_output << "exec ";
            print_src(exec->statement(), mod, pyc_output);

            if (exec->globals() != NULL) {
                pyc_output << " in ";
                print_src(exec->globals(), mod, pyc_output);

                if (exec->locals() != NULL
                        && exec->globals() != exec->locals()) {
                    pyc_output << ", ";
                    print_src(exec->locals(), mod, pyc_output);
                }
            }
        }
        break;
    case ASTNode::NODE_FORMATTEDVALUE:
        {
            const char* fq = f_string_quote();
            pyc_output << "f" << fq;
            ++f_string_depth;
            print_formatted_value(node.cast<ASTFormattedValue>(), mod, pyc_output);
            --f_string_depth;
            pyc_output << fq;
        }
        break;
    case ASTNode::NODE_JOINEDSTR:
        {
            const char* fq = f_string_quote();
            pyc_output << "f" << fq;
            ++f_string_depth;
            for (const auto& val : node.cast<ASTJoinedStr>()->values()) {
                switch (val.type()) {
                case ASTNode::NODE_FORMATTEDVALUE:
                    print_formatted_value(val.cast<ASTFormattedValue>(), mod, pyc_output);
                    break;
                case ASTNode::NODE_OBJECT:
                    print_const(pyc_output, val.cast<ASTObject>()->object(), mod, fq);
                    break;
                default:
                    fprintf(stderr, "Unsupported node type %d in NODE_JOINEDSTR\n", val.type());
                }
            }
            --f_string_depth;
            pyc_output << fq;
        }
        break;
    case ASTNode::NODE_KEYWORD:
        pyc_output << node.cast<ASTKeyword>()->word_str();
        break;
    case ASTNode::NODE_LIST:
        {
            const ASTList::value_t& vals = node.cast<ASTList>()->values();
            pyc_output << "[";
            if (g_noCompact == 0 && seqOneSrcLine(vals)) {
                bool first = true;
                for (const auto& val : vals) {
                    if (!first)
                        pyc_output << ", ";
                    print_src(val, mod, pyc_output);
                    first = false;
                }
            } else if (lfSafe(node)) {
                bool first = true;
                int elemLine = g_curLine;   // output line of '[' ~ its source line
                for (const auto& val : vals) {
                    bool broke = breakBeforeElem(elemLine, val, pyc_output, first);
                    if (!broke && !first)
                        pyc_output << ", ";
                    print_src(val, mod, pyc_output);
                    first = false;
                }
            } else {
                bool first = true;
                cur_indent++;
                for (const auto& val : vals) {
                    pyc_output << (first ? "\n" : ",\n");
                    start_line(cur_indent, pyc_output);
                    print_src(val, mod, pyc_output);
                    first = false;
                }
                cur_indent--;
            }
            pyc_output << "]";
        }
        break;
    case ASTNode::NODE_SET:
        {
            const ASTSet::value_t& vals = node.cast<ASTSet>()->values();
            pyc_output << "{";
            if (g_noCompact == 0 && seqOneSrcLine(vals)) {
                bool first = true;
                for (const auto& val : vals) {
                    if (!first)
                        pyc_output << ", ";
                    print_src(val, mod, pyc_output);
                    first = false;
                }
            } else if (lfSafe(node)) {
                bool first = true;
                int elemLine = g_curLine;
                for (const auto& val : vals) {
                    bool broke = breakBeforeElem(elemLine, val, pyc_output, first);
                    if (!broke && !first)
                        pyc_output << ", ";
                    print_src(val, mod, pyc_output);
                    first = false;
                }
            } else {
                bool first = true;
                cur_indent++;
                for (const auto& val : vals) {
                    pyc_output << (first ? "\n" : ",\n");
                    start_line(cur_indent, pyc_output);
                    print_src(val, mod, pyc_output);
                    first = false;
                }
                cur_indent--;
            }
            pyc_output << "}";
        }
        break;
    case ASTNode::NODE_COMPREHENSION:
        {
            PycRef<ASTComprehension> comp = node.cast<ASTComprehension>();

            bool is_dict = comp->comptype() == ASTComprehension::COMP_DICT;
            bool is_set = comp->comptype() == ASTComprehension::COMP_SET;
            bool is_gen = comp->comptype() == ASTComprehension::COMP_GENERATOR;
            bool genNoParen = is_gen && suppressGenExprParens;
            /* The flag only applies to THIS outermost genexpr; a nested
               comprehension (in the result/iter/condition) keeps its parens. */
            suppressGenExprParens = false;
            if (!genNoParen)
                pyc_output << (is_gen ? "(" : (is_dict || is_set) ? "{" : "[");
            if (is_dict) {
                print_src(comp->key(), mod, pyc_output);
                pyc_output << ": ";
            }
            print_src(comp->result(), mod, pyc_output);

            for (const auto& gen : comp->generators()) {
                pyc_output << " for ";
                print_src(gen->index(), mod, pyc_output);
                pyc_output << " in ";
                if (gen->iter() != nullptr
                        && gen->iter().type() == ASTNode::NODE_TERNARY) {
                    pyc_output << "(";
                    print_src(gen->iter(), mod, pyc_output);
                    pyc_output << ")";
                } else {
                    print_src(gen->iter(), mod, pyc_output);
                }
                if (gen->condition()) {
                    pyc_output << " if ";
                    bool parenFilter = (gen->condition()->type() == ASTNode::NODE_TERNARY);
                    if (parenFilter) pyc_output << "(";
                    print_src(gen->condition(), mod, pyc_output);
                    if (parenFilter) pyc_output << ")";
                }
            }
            if (!genNoParen)
                pyc_output << (is_gen ? ")" : (is_dict || is_set) ? "}" : "]");
        }
        break;
    case ASTNode::NODE_MAP:
        {
            const ASTMap::map_t& entries = node.cast<ASTMap>()->values();
            /* One output line iff every key and value share one source line. */
            bool oneLine = (g_noCompact == 0);
            if (oneLine) {
                int line = -1;
                for (const auto& e : entries) {
                    for (const PycRef<ASTNode>& n : {e.first, e.second}) {
                        if (n == nullptr)
                            continue;
                        int l = n->srcLine();
                        if (l <= 0)
                            continue;
                        if (line < 0)
                            line = l;
                        else if (l != line) {
                            oneLine = false;
                            break;
                        }
                    }
                    if (!oneLine)
                        break;
                }
            }
            pyc_output << "{";
            bool first = true;
            bool lf = lfSafe(node) && !oneLine;
            int elemLine = g_curLine;
            if (!oneLine && !lf)
                cur_indent++;
            for (const auto& val : entries) {
                /* Anchor the break on whichever of key/value carries a source
                   line: a const dict key (BUILD_CONST_KEY_MAP) has none, so
                   fall back to the value's line. */
                PycRef<ASTNode> anchor =
                        (val.first != nullptr && val.first->srcLine() > 0)
                        ? val.first : val.second;
                if (oneLine) {
                    if (!first)
                        pyc_output << ", ";
                } else if (lf) {
                    bool broke = breakBeforeElem(elemLine, anchor, pyc_output, first);
                    if (!broke && !first)
                        pyc_output << ", ";
                } else {
                    pyc_output << (first ? "\n" : ",\n");
                    start_line(cur_indent, pyc_output);
                }
                if (val.first == nullptr) {
                    pyc_output << "**";
                    bool par = starUnpackNeedsParens(val.second);
                    if (par) pyc_output << "(";
                    print_src(val.second, mod, pyc_output);
                    if (par) pyc_output << ")";
                } else {
                    print_src(val.first, mod, pyc_output);
                    pyc_output << ": ";
                    print_src(val.second, mod, pyc_output);
                }
                first = false;
            }
            if (!oneLine && !lf)
                cur_indent--;
            pyc_output << (oneLine || lf ? "}" : " }");
        }
        break;
    case ASTNode::NODE_CONST_MAP:
        {
            PycRef<ASTConstMap> const_map = node.cast<ASTConstMap>();
            PycTuple::value_t keys = const_map->keys().cast<ASTObject>()->object().cast<PycTuple>()->values();
            ASTConstMap::values_t values = const_map->values();

            auto map = new ASTMap;
            for (const auto& key : keys) {
                // Values are pushed onto the stack in reverse order.
                PycRef<ASTNode> value = values.back();
                values.pop_back();

                map->add(new ASTObject(key), value);
            }

            print_src(map, mod, pyc_output);
        }
        break;
    case ASTNode::NODE_NAME:
        padToCol(node, pyc_output);
        pyc_output << demangleName(node.cast<ASTName>()->name()->strValue());
        break;
    case ASTNode::NODE_NODELIST:
        {
            cur_indent++;
            const ASTNodeList::list_t& nl = node.cast<ASTNodeList>()->nodes();
            /* Compound-statement joining (see print_block): consecutive simple
               statements on one source line render `; `-joined. */
            int prevJoinLine = -1;
            bool anyEmitted = false;
            for (auto ln = nl.cbegin(); ln != nl.cend(); ++ln) {
                bool isNL = (*ln).cast<ASTNode>().type() == ASTNode::NODE_NODELIST;
                bool joinable = !inLambda && !isNL && *ln != nullptr
                        && g_protectedDepth == 0
                        && (*ln).type() != ASTNode::NODE_BLOCK
                        && !selfPositions(*ln);
                int myLine = (*ln != nullptr) ? (*ln)->srcLine() : -1;
                bool doJoin = anyEmitted && joinable && prevJoinLine > 0
                        && myLine == prevJoinLine;
                if (doJoin) {
                    pyc_output << "; ";
                } else {
                    if (anyEmitted)
                        end_line(pyc_output);
                    if (!isNL) {
                        int decoLine = decoratedDefFirstLine(*ln);
                        int curLine = decoLine > 0 ? decoLine : (*ln)->srcLine();
                        if (!inLambda && curLine > 0 && g_curLine > 0 && curLine > g_curLine
                                && curLine - g_curLine <= 2000) {
                            while (g_curLine < curLine)
                                pyc_output << "\n";
                        }
                        if (!selfPositions(*ln))
                            start_line(cur_indent, pyc_output);
                    }
                }
                print_src(*ln, mod, pyc_output);
                prevJoinLine = joinable ? myLine : -1;
                anyEmitted = true;
            }
            cur_indent--;
        }
        break;
    case ASTNode::NODE_BLOCK:
        {
            PycRef<ASTBlock> blk = node.cast<ASTBlock>();
            if (blk->blktype() == ASTBlock::BLK_ELSE && blk->size() == 0)
                break;

            if (blk->blktype() == ASTBlock::BLK_CONTAINER) {
                bool hasElse = false, hasOther = false;
                for (const auto& n : blk->nodes()) {
                    if (n->type() != ASTNode::NODE_BLOCK) { hasOther = true; continue; }
                    if (n.cast<ASTBlock>()->blktype() == ASTBlock::BLK_ELSE) hasElse = true;
                }
                if (hasElse && !hasOther) {
                    ASTBlock::list_t b0, b1, b2, b3;
                    for (const auto& n : blk->nodes()) {
                        switch (n.cast<ASTBlock>()->blktype()) {
                        case ASTBlock::BLK_TRY:     b0.push_back(n); break;
                        case ASTBlock::BLK_EXCEPT:  b1.push_back(n); break;
                        case ASTBlock::BLK_ELSE:    b2.push_back(n); break;
                        case ASTBlock::BLK_FINALLY: b3.push_back(n); break;
                        default:                    b1.push_back(n); break;
                        }
                    }
                    while (blk->size()) blk->removeLast();
                    for (auto& n : b0) blk->append(n);
                    for (auto& n : b1) blk->append(n);
                    for (auto& n : b2) blk->append(n);
                    for (auto& n : b3) blk->append(n);
                }
                /* No leading end_line: the container is self-positioning (the
                   enclosing renderer padded to its source line and skipped
                   start_line), so print_block places `try:` directly. */
                print_block(blk, mod, pyc_output);
                end_line(pyc_output);
                break;
            }

            if (blk->blktype() == ASTBlock::BLK_WITH
                    && blk.cast<ASTWithBlock>()->isAsync())
                pyc_output << "async with";
            else
                pyc_output << blk->type_str();
            if (blk->blktype() == ASTBlock::BLK_IF
                    || blk->blktype() == ASTBlock::BLK_ELIF
                    || blk->blktype() == ASTBlock::BLK_WHILE) {
                if (blk.cast<ASTCondBlock>()->negative()) {
                    pyc_output << " not ";
                    PycRef<ASTNode> c = blk.cast<ASTCondBlock>()->cond();
                    bool paren = (c != NULL && c.type() == ASTNode::NODE_BINARY
                            && (c.cast<ASTBinary>()->op() == ASTBinary::BIN_LOG_AND
                                || c.cast<ASTBinary>()->op() == ASTBinary::BIN_LOG_OR));
                    if (paren) pyc_output << "(";
                    print_src(c, mod, pyc_output);
                    if (paren) pyc_output << ")";
                } else if (blk->blktype() == ASTBlock::BLK_WHILE
                        && blk.cast<ASTCondBlock>()->condRenderAsOne()) {
                    pyc_output << " 1";
                } else {
                    pyc_output << " ";
                    print_src(blk.cast<ASTCondBlock>()->cond(), mod, pyc_output);
                }
            } else if (blk->blktype() == ASTBlock::BLK_FOR || blk->blktype() == ASTBlock::BLK_ASYNCFOR) {
                pyc_output << " ";
                print_src(blk.cast<ASTIterBlock>()->index(), mod, pyc_output);
                pyc_output << " in ";
                print_src(blk.cast<ASTIterBlock>()->iter(), mod, pyc_output);
            } else if (blk->blktype() == ASTBlock::BLK_MATCH) {
                pyc_output << " ";
                print_src(blk.cast<ASTMatchBlock>()->subject(), mod, pyc_output);
            } else if (blk->blktype() == ASTBlock::BLK_CASE) {
                pyc_output << " ";
                print_src(blk.cast<ASTCaseBlock>()->pattern(), mod, pyc_output);
            } else if (blk->blktype() == ASTBlock::BLK_EXCEPT &&
                    blk.cast<ASTCondBlock>()->cond() != NULL) {
                pyc_output << " ";
                print_src(blk.cast<ASTCondBlock>()->cond(), mod, pyc_output);
                PycRef<ASTNode> ev = blk.cast<ASTCondBlock>()->exceptVar();
                if (ev != NULL) {
                    pyc_output << " as ";
                    print_src(ev, mod, pyc_output);
                }
            } else if (blk->blktype() == ASTBlock::BLK_WITH) {
                pyc_output << " ";
                print_src(blk.cast<ASTWithBlock>()->expr(), mod, pyc_output);
                PycRef<ASTNode> var = blk.try_cast<ASTWithBlock>()->var();
                if (var != NULL) {
                    pyc_output << " as ";
                    print_src(var, mod, pyc_output);
                }
            }
            /* Inline single-statement body: `if c: stmt` / `for x in y: stmt`
               when the sole body statement shares the header's source line (the
               original wrote it as a one-line suite). co_code is byte-identical.
               Only conditional/loop/with headers; never try/except/finally/match/
               case (line-sensitive / load-bearing) or def/class. */
            ASTBlock::BlkType bt = blk->blktype();
            /* Conditionals only. A loop (for/while) body carries a loop-back
               line-marker NOP that inlining can drop (co_code); `with`/`try`
               likewise. `if`/`elif`/`else` single-statement suites inline
               cleanly. */
            bool inlineable = bt == ASTBlock::BLK_IF || bt == ASTBlock::BLK_ELIF
                    || bt == ASTBlock::BLK_ELSE;
            PycRef<ASTNode> soleBody;
            int nvis = 0;
            for (const auto& bn : blk->nodes()) {
                if (bn == nullptr || bn->suppressed())
                    continue;
                nvis++;
                soleBody = bn;
            }
            bool inlineBody = inlineable && !inLambda && g_protectedDepth == 0
                    && nvis == 1 && soleBody != nullptr
                    && soleBody.type() != ASTNode::NODE_BLOCK
                    && soleBody.type() != ASTNode::NODE_NODELIST
                    && !selfPositions(soleBody)
                    && soleBody->srcLine() > 0 && blk->srcLine() > 0
                    && soleBody->srcLine() == blk->srcLine();
            if (inlineBody) {
                pyc_output << ": ";
                print_src(soleBody, mod, pyc_output);
            } else {
                pyc_output << ":\n";
                cur_indent++;
                bool prot = bt == ASTBlock::BLK_TRY
                        || bt == ASTBlock::BLK_EXCEPT
                        || bt == ASTBlock::BLK_FINALLY;
                if (prot) g_protectedDepth++;
                print_block(blk, mod, pyc_output);
                if (prot) g_protectedDepth--;
                cur_indent--;
            }
        }
        break;
    case ASTNode::NODE_OBJECT:
        {
            PycRef<PycObject> obj = node.cast<ASTObject>()->object();
            if (obj.type() != PycObject::TYPE_CODE)
                padToCol(node, pyc_output);
            /* Width-matched stripped-statement placeholder (floor b1): render an
               empty parenthesised group `(   )` (an empty-tuple expression) of
               exactly the stripped text's width instead of `...`, so the
               co_positions column span matches. Under -OO a bare constant
               expression is discarded to the SAME NOP as `...` (verified
               byte-identical, and unlike a bare string it is NOT treated as a
               docstring, so it is safe even in first-statement position). Only
               when a valid width (>=2) was recorded and not inside a
               lambda/f-string. */
            /* Multi-line stripped-statement placeholder: the original (a wrapped
               docstring) spanned layoutEndLine/EndCol; render the same open
               `(...` then blank continuation lines, closing `)` at the original
               end column, so the NOP's full (line, end-line, col, end-col) span
               is reproduced. Newlines inside the parentheses are pure whitespace
               to the compiler, so co_code is unchanged. */
            if (obj == Pyc_Ellipsis && node->layoutEndLine() > node->srcLine()
                    && node->srcLine() >= 0 && node->layoutEndCol() >= 1
                    && !inLambda && f_string_depth == 0) {
                pyc_output << "(...";
                for (int l = node->srcLine(); l < node->layoutEndLine(); l++)
                    pyc_output << '\n';
                for (int i = 0; i < node->layoutEndCol() - 1; i++)
                    pyc_output << ' ';
                pyc_output << ')';
                break;
            }
            if (obj == Pyc_Ellipsis && node->layoutWidth() >= 2
                    && !inLambda && f_string_depth == 0) {
                int w = node->layoutWidth();
                /* Self-documenting: `(...      )` keeps the `...` marker so the
                   stripped content is legible, while the trailing pad makes the
                   token span the original column width. `(...)` is an Ellipsis
                   constant expression, discarded to the SAME NOP as `...` under
                   -OO, so co_code is unchanged. Widths too narrow for `(...)`
                   fall back to an empty parenthesised pad. */
                if (w >= 5) {
                    pyc_output << "(...";
                    for (int i = 0; i < w - 5; i++)
                        pyc_output << ' ';
                    pyc_output << ')';
                } else {
                    pyc_output << '(';
                    for (int i = 0; i < w - 2; i++)
                        pyc_output << ' ';
                    pyc_output << ')';
                }
                break;
            }
            if (obj.type() == PycObject::TYPE_CODE) {
                PycRef<PycCode> code = obj.cast<PycCode>();
                decompyle(code, mod, pyc_output);
            } else {
                PycRef<PycString> str = (f_string_depth > 0)
                        ? obj.try_cast<PycString>() : nullptr;
                if (str == nullptr || !print_f_string_expr_literal(pyc_output, str))
                    print_const(pyc_output, obj, mod);
            }
        }
        break;
    case ASTNode::NODE_PRINT:
        {
            pyc_output << "print ";
            bool first = true;
            if (node.cast<ASTPrint>()->stream() != nullptr) {
                pyc_output << ">>";
                print_src(node.cast<ASTPrint>()->stream(), mod, pyc_output);
                first = false;
            }

            for (const auto& val : node.cast<ASTPrint>()->values()) {
                if (!first)
                    pyc_output << ", ";
                print_src(val, mod, pyc_output);
                first = false;
            }
            if (!node.cast<ASTPrint>()->eol())
                pyc_output << ",";
        }
        break;
    case ASTNode::NODE_RAISE:
        {
            PycRef<ASTRaise> raise = node.cast<ASTRaise>();
            const auto& params = raise->params();
            pyc_output << "raise ";
            if (mod->verCompare(3, 0) >= 0 && params.size() == 2) {
                print_src(params.front(), mod, pyc_output);
                pyc_output << " from ";
                print_src(params.back(), mod, pyc_output);
            } else {
                bool first = true;
                for (const auto& param : params) {
                    if (!first)
                        pyc_output << ", ";
                    print_src(param, mod, pyc_output);
                    first = false;
                }
            }
        }
        break;
    case ASTNode::NODE_RETURN:
        {
            PycRef<ASTReturn> ret = node.cast<ASTReturn>();
            PycRef<ASTNode> value = ret->value();
            bool asyncGenBareReturn = inAsyncGen
                    && ret->rettype() == ASTReturn::RETURN
                    && (value == nullptr
                        || (value.type() == ASTNode::NODE_OBJECT
                            && value.cast<ASTObject>()->object() == Pyc_None));
            if (ret->rettype() == ASTReturn::YIELD_EXPR) {
                pyc_output << "(yield ";
                print_src(value, mod, pyc_output);
                pyc_output << ")";
                break;
            }
            if (ret->rettype() == ASTReturn::YIELD_FROM_EXPR) {
                if (value.type() == ASTNode::NODE_AWAITABLE) {
                    pyc_output << "(await ";
                    print_src(value.cast<ASTAwaitable>()->expression(), mod, pyc_output);
                } else {
                    pyc_output << "(yield from ";
                    print_src(value, mod, pyc_output);
                }
                pyc_output << ")";
                break;
            }
            if (!inLambda) {
                switch (ret->rettype()) {
                case ASTReturn::RETURN:
                    pyc_output << (asyncGenBareReturn ? "return" : "return ");
                    break;
                case ASTReturn::YIELD:
                    pyc_output << "yield ";
                    break;
                case ASTReturn::YIELD_FROM:
                    if (value.type() == ASTNode::NODE_AWAITABLE) {
                        pyc_output << "await ";
                        value = value.cast<ASTAwaitable>()->expression();
                    } else {
                        pyc_output << "yield from ";
                    }
                    break;
                case ASTReturn::YIELD_EXPR:
                case ASTReturn::YIELD_FROM_EXPR:
                    break;
                }
            }
            if (!asyncGenBareReturn) {
                /* `return a, b` / `yield a, b` render the tuple without parens
                   (statement-level value context). Other rettypes (yield from,
                   the parenthesised yield-expr forms) are left untouched. */
                bool bareTupleCtx = !inLambda
                        && (ret->rettype() == ASTReturn::RETURN
                            || ret->rettype() == ASTReturn::YIELD);
                if (bareTupleCtx)
                    print_bare_tuple_value(value, mod, pyc_output);
                else
                    print_src(value, mod, pyc_output);
            }
        }
        break;
    case ASTNode::NODE_SLICE:
        {
            PycRef<ASTSlice> slice = node.cast<ASTSlice>();

            if (slice->op() & ASTSlice::SLICE1) {
                print_src(slice->left(), mod, pyc_output);
            }
            pyc_output << ":";
            if (slice->op() & ASTSlice::SLICE2) {
                print_src(slice->right(), mod, pyc_output);
            }
        }
        break;
    case ASTNode::NODE_IMPORT:
        {
            PycRef<ASTImport> import = node.cast<ASTImport>();
            PycRef<ASTNode> fl = import->fromlist();
            bool noFromlist = (fl == nullptr)
                    || (fl.type() == ASTNode::NODE_OBJECT
                        && fl.cast<ASTObject>()->object() == Pyc_None);
            if (import->stores().size() && noFromlist) {
                pyc_output << "import ";
                print_src(import->name(), mod, pyc_output);
                PycRef<ASTNode> dest = import->stores().back()->dest();
                const std::string full = import->name().cast<ASTName>()->name()->strValue();
                const std::string top = full.substr(0, full.find('.'));
                if (dest.cast<ASTName>()->name()->strValue() != top) {
                    pyc_output << " as ";
                    print_src(dest, mod, pyc_output);
                }
            } else if (import->stores().size()) {
                ASTImport::list_t stores = import->stores();

                pyc_output << "from ";
                for (int i = 0; i < import->level(); i++)
                    pyc_output << ".";
                if (import->name().type() == ASTNode::NODE_IMPORT)
                    print_src(import->name().cast<ASTImport>()->name(), mod, pyc_output);
                else
                    print_src(import->name(), mod, pyc_output);
                pyc_output << " import ";

                if (stores.size() == 1) {
                    auto src = stores.front()->src();
                    auto dest = stores.front()->dest();
                    print_src(src, mod, pyc_output);

                    if (src.cast<ASTName>()->name()->value() != dest.cast<ASTName>()->name()->value()) {
                        pyc_output << " as ";
                        print_src(dest, mod, pyc_output);
                    }
                } else {
                    bool first = true;
                    for (const auto& st : stores) {
                        if (!first)
                            pyc_output << ", ";
                        print_src(st->src(), mod, pyc_output);
                        first = false;

                        if (st->src().cast<ASTName>()->name()->value() != st->dest().cast<ASTName>()->name()->value()) {
                            pyc_output << " as ";
                            print_src(st->dest(), mod, pyc_output);
                        }
                    }
                }
            } else {
                pyc_output << "import ";
                print_src(import->name(), mod, pyc_output);
            }
        }
        break;
    case ASTNode::NODE_FUNCTION:
        {
            /* Actual named functions are NODE_STORE with a name */
            PycRef<ASTNode> code = node.cast<ASTFunction>()->code();
            PycRef<PycCode> code_src = code.cast<ASTObject>()->object().cast<PycCode>();
            /* A no-argument lambda is canonically `lambda:` (no space before the
               colon); only emit the space after `lambda` when a parameter list
               follows. */
            const bool lambdaHasParams =
                    code_src->argCount() != 0 || code_src->kwOnlyArgCount() != 0
                    || (code_src->flags() & (PycCode::CO_VARARGS | PycCode::CO_VARKEYWORDS)) != 0;
            bool lambdaNoParen = suppressLambdaParens;
            suppressLambdaParens = false;
            if (!lambdaNoParen)
                pyc_output << "(";
            pyc_output << (lambdaHasParams ? "lambda " : "lambda");
            ASTFunction::defarg_t defargs = node.cast<ASTFunction>()->defargs();
            ASTFunction::defarg_t kwdefargs = node.cast<ASTFunction>()->kwdefargs();
            auto da = defargs.cbegin();
            int narg = 0;
            for (int i=0; i<code_src->argCount(); i++) {
                if (narg)
                    pyc_output << ", ";
                pyc_output << code_src->getLocal(narg++)->value();
                if ((code_src->argCount() - i) <= (int)defargs.size()) {
                    pyc_output << "=";   // lambda params are never annotated
                    print_src(*da++, mod, pyc_output);
                }
            }
            /* `*args` / keyword-only params / `**kwargs` -- mirrors the def-signature
               path below (lambda params carry no annotations). Their names live in
               co_varnames after the positional and keyword-only params. Without this
               a `lambda *args:` / `lambda **kwargs:` lost the star params, dropping the
               varnames and shifting every LOAD_DEREF index in the body. */
            da = kwdefargs.cbegin();
            const int lamKwOnly = code_src->kwOnlyArgCount();
            const bool lamVararg = (code_src->flags() & PycCode::CO_VARARGS) != 0;
            if (lamVararg) {
                if (narg)
                    pyc_output << ", ";
                pyc_output << "*" << code_src->getLocal(code_src->argCount() + lamKwOnly)->value();
            } else if (lamKwOnly != 0) {
                pyc_output << (narg == 0 ? "*" : ", *");
            }
            for (int i = 0; i < lamKwOnly; ++i) {
                pyc_output << ", ";
                pyc_output << code_src->getLocal(code_src->argCount() + i)->value();
                if ((lamKwOnly - i) <= (int)kwdefargs.size()) {
                    pyc_output << "=";
                    print_src(*da++, mod, pyc_output);
                }
            }
            if (code_src->flags() & PycCode::CO_VARKEYWORDS) {
                if (narg || lamVararg || lamKwOnly)
                    pyc_output << ", ";
                pyc_output << "**" << code_src->getLocal(
                        code_src->argCount() + lamKwOnly + (lamVararg ? 1 : 0))->value();
            }
            pyc_output << ": ";

            inLambda = true;
            print_src(code, mod, pyc_output);
            inLambda = false;

            if (!lambdaNoParen)
                pyc_output << ")";
        }
        break;
    case ASTNode::NODE_STORE:
        {
            PycRef<ASTNode> src = node.cast<ASTStore>()->src();
            PycRef<ASTNode> dest = node.cast<ASTStore>()->dest();
            if (src.type() == ASTNode::NODE_FUNCTION) {
                PycRef<ASTNode> code = src.cast<ASTFunction>()->code();
                PycRef<PycCode> code_src = code.cast<ASTObject>()->object().cast<PycCode>();
                bool isLambda = false;

                if (strcmp(code_src->name()->value(), "<lambda>") == 0) {
                    start_line(cur_indent, pyc_output);
                    print_src(dest, mod, pyc_output);
                    /* `x = lambda: ...` (no space before the colon) when the
                       lambda takes no parameters; keep the space otherwise. */
                    const bool lambdaHasParams =
                            code_src->argCount() != 0 || code_src->kwOnlyArgCount() != 0
                            || (code_src->flags()
                                & (PycCode::CO_VARARGS | PycCode::CO_VARKEYWORDS)) != 0;
                    pyc_output << (lambdaHasParams ? " = lambda " : " = lambda");
                    isLambda = true;
                } else {
                    for (const auto& d : src.cast<ASTFunction>()->decorators()) {
                        start_line(cur_indent, pyc_output);
                        pyc_output << "@";
                        print_src(d, mod, pyc_output);
                        pyc_output << "\n";
                    }
                    start_line(cur_indent, pyc_output);
                    if (code_src->flags() & (PycCode::CO_COROUTINE | PycCode::CO_ASYNC_GENERATOR))
                        pyc_output << "async ";
                    pyc_output << "def ";
                    print_src(dest, mod, pyc_output);
                    pyc_output << "(";
                }

                ASTFunction::defarg_t defargs = src.cast<ASTFunction>()->defargs();
                ASTFunction::defarg_t kwdefargs = src.cast<ASTFunction>()->kwdefargs();
                const ASTFunction::annot_t& annots = src.cast<ASTFunction>()->annotations();
                /* A signature is LINE-FAITHFUL only when every annotation and
                   default expression sits on the same source line as the `def`
                   keyword: then the whole signature was single-line in the
                   source and can render compactly (collections follow the
                   ordinary seqOneSrcLine rule). When any annotation/default
                   lives on a later line, the source signature spanned multiple
                   lines and left one continuation-anchor NOP per extra line;
                   keeping collection compaction disabled (g_noCompact) preserves
                   the rendering that regenerates those anchors, so co_code stays
                   identical. */
                int sigDefLine = node->srcLine();
                bool sigSourceMultiLine = false;
                {
                    auto spans = [&](const PycRef<ASTNode>& n) {
                        if (n == nullptr)
                            return;
                        int l = n->srcLine();
                        if (l > 0 && sigDefLine > 0 && l != sigDefLine)
                            sigSourceMultiLine = true;
                    };
                    for (const auto& a : annots)
                        spans(a.second);
                    for (const auto& d : defargs)
                        spans(d);
                    for (const auto& d : kwdefargs)
                        spans(d);
                }
                /* When every constant default sat on its own source line, the
                   default lines were folded into a single tuple constant so each
                   default reports the `def` line and sigSourceMultiLine above
                   cannot see the wrap. sigNopAnchors() carries the anchor-NOP
                   count (== number of defaults) recovered at MAKE_FUNCTION; render
                   one parameter per line to regenerate those NOPs. */
                bool sigOnePerLine = src.cast<ASTFunction>()->sigNopAnchors() >= 0;
                /* Disable collection compaction for a genuinely multi-line
                   signature (see above); a single-line one renders compactly. */
                if (sigSourceMultiLine || sigOnePerLine)
                    g_noCompact++;
                /* A multi-line source signature emits one line-anchor NOP per
                   parameter continuation line (the compiler bumps the line before
                   building that parameter's annotation/default). Reproduce those
                   NOPs byte-for-byte by rendering each parameter on its own source
                   line -- pad blank continuation lines until g_curLine reaches the
                   parameter's line (taken from its annotation, else its default).
                   Returns true when it broke the line so the caller omits the inline
                   space after the comma. Newlines inside the signature parens are
                   pure whitespace, so co_code is unaffected EXCEPT for the intended
                   regenerated anchor NOPs. */
                auto padParamLine = [&](PycRef<ASTNode> annot,
                                        PycRef<ASTNode> defv) -> bool {
                    if (!sigSourceMultiLine || inLambda)
                        return false;
                    int tl = -1;
                    if (annot != nullptr && annot->srcLine() > 0)
                        tl = annot->srcLine();
                    else if (defv != nullptr && defv->srcLine() > 0)
                        tl = defv->srcLine();
                    if (tl <= 0 || tl <= g_curLine || tl - g_curLine > 2000)
                        return false;
                    while (g_curLine < tl)
                        pyc_output << "\n";
                    for (int s = 0; s < (cur_indent + 2) * 4; s++)
                        pyc_output << " ";
                    return true;
                };
                auto annotFor = [&](const char* nm) -> PycRef<ASTNode> {
                    for (const auto& a : annots)
                        if (a.first == nm) return a.second;
                    return nullptr;
                };
                bool futureAnnot = (code_src->flags() & PycCode::CO_FUTURE_ANNOTATIONS) != 0;
                auto printAnnot = [&](PycRef<ASTNode> at) {
                    if (futureAnnot && at != nullptr && at.type() == ASTNode::NODE_OBJECT) {
                        PycRef<PycObject> o = at.cast<ASTObject>()->object();
                        if (o->type() == PycObject::TYPE_STRING
                                || o->type() == PycObject::TYPE_UNICODE
                                || o->type() == PycObject::TYPE_INTERNED
                                || o->type() == PycObject::TYPE_ASCII
                                || o->type() == PycObject::TYPE_ASCII_INTERNED
                                || o->type() == PycObject::TYPE_SHORT_ASCII
                                || o->type() == PycObject::TYPE_SHORT_ASCII_INTERNED) {
                            pyc_output << o.cast<PycString>()->value();
                            return;
                        }
                    }
                    print_src(at, mod, pyc_output);
                };
                auto da = defargs.cbegin();
                int narg = 0;
                for (int i = 0; i < code_src->argCount(); ++i) {
                    const char* pname = code_src->getLocal(narg)->value();
                    PycRef<ASTNode> at = annotFor(pname);
                    bool hasDef = (code_src->argCount() - i) <= (int)defargs.size();
                    if (narg)
                        pyc_output << ",";
                    bool broke;
                    if (sigOnePerLine) {
                        /* Reproduce K anchor NOPs (one per source line carrying a
                           constant default) by laying the defaults out over K
                           continuation lines. Break before the FIRST default (so no
                           default sits on the `def` line -- a default there emits an
                           anchor only in a multi-line signature, which this now is)
                           and before enough later defaults to form K default lines:
                           the first continuation gets (Ndef-K+1) defaults, each
                           remaining line one. Blank/continuation whitespace is inert,
                           so exactly the K intended anchors are regenerated. */
                        int Kanch = src.cast<ASTFunction>()->sigNopAnchors();
                        int Ndef = (int)defargs.size();
                        int j = hasDef ? (i - (code_src->argCount() - Ndef)) : -1;
                        broke = (j == 0) || (j >= Ndef - Kanch + 1);
                        if (broke) {
                            pyc_output << "\n";
                            for (int s = 0; s < (cur_indent + 2) * 4; s++)
                                pyc_output << " ";
                        }
                    } else {
                        broke = padParamLine(at, hasDef ? *da : PycRef<ASTNode>());
                    }
                    if (narg && !broke)
                        pyc_output << " ";
                    ++narg;
                    pyc_output << pname;
                    bool annotated = false;
                    if (at != nullptr) {
                        pyc_output << ": ";
                        printAnnot(at);
                        annotated = true;
                    }
                    if (hasDef) {
                        /* Canonical spacing: `x=1` when unannotated, `x: T = 1`
                           when annotated. */
                        pyc_output << (annotated ? " = " : "=");
                        print_src(*da++, mod, pyc_output);
                    }
                }
                const int kwOnly = code_src->kwOnlyArgCount();
                const bool hasVararg = (code_src->flags() & PycCode::CO_VARARGS) != 0;
                if (hasVararg) {
                    if (narg)
                        pyc_output << ", ";
                    const char* vname = code_src->getLocal(code_src->argCount() + kwOnly)->value();
                    pyc_output << "*" << vname;
                    if (PycRef<ASTNode> at = annotFor(vname)) {
                        pyc_output << ": ";
                        printAnnot(at);
                    }
                } else if (kwOnly != 0) {
                    pyc_output << (narg == 0 ? "*" : ", *");
                }
                da = kwdefargs.cbegin();
                for (int i = 0; i < kwOnly; ++i) {
                    const char* kname = code_src->getLocal(code_src->argCount() + i)->value();
                    PycRef<ASTNode> at = annotFor(kname);
                    bool hasDef = (kwOnly - i) <= (int)kwdefargs.size();
                    pyc_output << ",";
                    bool broke = padParamLine(at, hasDef ? *da : PycRef<ASTNode>());
                    if (!broke)
                        pyc_output << " ";
                    pyc_output << kname;
                    bool kannotated = false;
                    if (at != nullptr) {
                        pyc_output << ": ";
                        printAnnot(at);
                        kannotated = true;
                    }
                    if (hasDef) {
                        pyc_output << (kannotated ? " = " : "=");
                        print_src(*da++, mod, pyc_output);
                    }
                }
                if (code_src->flags() & PycCode::CO_VARKEYWORDS) {
                    if (narg || hasVararg || kwOnly)
                        pyc_output << ", ";
                    const char* kwname = code_src->getLocal(
                            code_src->argCount() + kwOnly + (hasVararg ? 1 : 0))->value();
                    pyc_output << "**" << kwname;
                    if (PycRef<ASTNode> at = annotFor(kwname)) {
                        pyc_output << ": ";
                        printAnnot(at);
                    }
                }

                if (isLambda) {
                    pyc_output << ": ";
                } else {
                    pyc_output << ")";
                    if (PycRef<ASTNode> rt = annotFor("return")) {
                        pyc_output << " -> ";
                        printAnnot(rt);
                    }
                    pyc_output << ":\n";
                    printDocstringAndGlobals = true;
                }
                if (sigSourceMultiLine || sigOnePerLine)
                    g_noCompact--;   // signature done; body compacts normally

                bool preLambda = inLambda;
                bool preAsyncGen = inAsyncGen;
                inLambda |= isLambda;
                inAsyncGen = (code_src->flags() & PycCode::CO_ASYNC_GENERATOR) != 0;

                print_src(code, mod, pyc_output);

                inLambda = preLambda;
                inAsyncGen = preAsyncGen;
            } else if (src.type() == ASTNode::NODE_CLASS) {
                for (const auto& d : src.cast<ASTClass>()->decorators()) {
                    start_line(cur_indent, pyc_output);
                    pyc_output << "@";
                    print_src(d, mod, pyc_output);
                    pyc_output << "\n";
                }
                start_line(cur_indent, pyc_output);
                pyc_output << "class ";
                print_src(dest, mod, pyc_output);
                PycRef<ASTTuple> bases = src.cast<ASTClass>()->bases().cast<ASTTuple>();
                PycRef<ASTNode> ckw = src.cast<ASTClass>()->kwargs();
                bool hasKw = ckw != nullptr && ckw.type() == ASTNode::NODE_KW_NAMES_MAP
                             && !ckw.cast<ASTKwNamesMap>()->values().empty();
                if (bases->values().size() > 0 || hasKw) {
                    pyc_output << "(";
                    bool first = true;
                    for (const auto& val : bases->values()) {
                        if (!first)
                            pyc_output << ", ";
                        print_src(val, mod, pyc_output);
                        first = false;
                    }
                    if (hasKw) {
                        for (const auto& kv : ckw.cast<ASTKwNamesMap>()->values()) {
                            if (!first)
                                pyc_output << ", ";
                            first = false;
                            PycRef<ASTNode> k = kv.first;
                            PycRef<PycString> ks = (k != nullptr
                                    && k.type() == ASTNode::NODE_OBJECT)
                                ? k.cast<ASTObject>()->object().try_cast<PycString>()
                                : nullptr;
                            if (ks != nullptr)
                                pyc_output << ks->value() << "=";
                            print_src(kv.second, mod, pyc_output);
                        }
                    }
                    pyc_output << "):\n";
                } else {
                    // Don't put parens if there are no base classes
                    pyc_output << ":\n";
                }
                printClassDocstring = true;
                PycRef<ASTNode> code = src.cast<ASTClass>()->code().cast<ASTCall>()
                                       ->func().cast<ASTFunction>()->code();
                PycRef<PycCode> classCode =
                        code.cast<ASTObject>()->object().try_cast<PycCode>();
                if (classCode != nullptr && classCode->name() != nullptr)
                    g_classNameStack.push_back(classCode->name()->strValue());
                print_src(code, mod, pyc_output);
                if (classCode != nullptr && classCode->name() != nullptr)
                    g_classNameStack.pop_back();
            } else if (src.type() == ASTNode::NODE_IMPORT) {
                PycRef<ASTImport> import = src.cast<ASTImport>();
                if (import->fromlist() != NULL) {
                    PycRef<PycObject> fromlist = import->fromlist().cast<ASTObject>()->object();
                    if (fromlist != Pyc_None) {
                        pyc_output << "from ";
                        for (int i = 0; i < import->level(); i++)
                            pyc_output << ".";
                        if (import->name().type() == ASTNode::NODE_IMPORT)
                            print_src(import->name().cast<ASTImport>()->name(), mod, pyc_output);
                        else
                            print_src(import->name(), mod, pyc_output);
                        pyc_output << " import ";
                        if (fromlist.type() == PycObject::TYPE_TUPLE ||
                                fromlist.type() == PycObject::TYPE_SMALL_TUPLE) {
                            bool first = true;
                            for (const auto& val : fromlist.cast<PycTuple>()->values()) {
                                if (!first)
                                    pyc_output << ", ";
                                pyc_output << val.cast<PycString>()->value();
                                first = false;
                            }
                        } else {
                            pyc_output << fromlist.cast<PycString>()->value();
                        }
                    } else {
                        pyc_output << "import ";
                        print_src(import->name(), mod, pyc_output);
                    }
                } else {
                    pyc_output << "import ";
                    PycRef<ASTNode> import_name = import->name();
                    print_src(import_name, mod, pyc_output);
                    const std::string full = import_name.cast<ASTName>()->name()->strValue();
                    const std::string top = full.substr(0, full.find('.'));
                    if (dest.cast<ASTName>()->name()->strValue() != top) {
                        pyc_output << " as ";
                        print_src(dest, mod, pyc_output);
                    }
                }
            } else if (src.type() == ASTNode::NODE_BINARY
                    && src.cast<ASTBinary>()->is_inplace()) {
                print_src(src, mod, pyc_output);
            } else if (node.cast<ASTStore>()->isWalrus()) {
                pyc_output << "(";
                print_src(dest, mod, pyc_output);
                pyc_output << " := ";
                print_src(src, mod, pyc_output);
                pyc_output << ")";
            } else {
                /* A tuple assignment target is canonically written without
                   surrounding parentheses (`a, b = c`, not `(a, b) = c`); the
                   parens only widen the line. Suppress them on the OUTERMOST
                   target tuple while rendering (inner nested targets keep their
                   own parens). Bytecode is unaffected. */
                PycRef<ASTTuple> destTup =
                        (dest != nullptr && dest.type() == ASTNode::NODE_TUPLE)
                        ? dest.cast<ASTTuple>() : nullptr;
                bool savedParens = false;
                if (destTup != nullptr && destTup->values().size() > 1) {
                    savedParens = destTup->requireParens();
                    destTup->setRequireParens(false);
                }
                print_src(dest, mod, pyc_output);
                if (destTup != nullptr && destTup->values().size() > 1)
                    destTup->setRequireParens(savedParens);
                pyc_output << " = ";
                print_bare_tuple_value(src, mod, pyc_output);
            }
        }
        break;
    case ASTNode::NODE_CHAINSTORE:
        {
            for (auto& dest : node.cast<ASTChainStore>()->nodes()) {
                print_src(dest, mod, pyc_output);
                pyc_output << " = ";
            }
            print_bare_tuple_value(node.cast<ASTChainStore>()->src(), mod, pyc_output);
        }
        break;
    case ASTNode::NODE_SUBSCR:
        {
            print_src(node.cast<ASTSubscr>()->name(), mod, pyc_output);
            pyc_output << "[";
            print_src(node.cast<ASTSubscr>()->key(), mod, pyc_output);
            pyc_output << "]";
        }
        break;
    case ASTNode::NODE_CONVERT:
        {
            pyc_output << "`";
            print_src(node.cast<ASTConvert>()->name(), mod, pyc_output);
            pyc_output << "`";
        }
        break;
    case ASTNode::NODE_TUPLE:
        {
            PycRef<ASTTuple> tuple = node.cast<ASTTuple>();
            ASTTuple::value_t values = tuple->values();
            if (tuple->requireParens())
                pyc_output << "(";
            bool first = true;
            /* Line-faithful breaks only inside a parenthesised tuple: a newline
               is a valid continuation only within the parens. A bare tuple
               (`return a, b`) has no brackets, so it must stay on one line. */
            bool tupBreak = lfSafe(node) && tuple->requireParens();
            int elemLine = g_curLine;
            for (const auto& val : values) {
                if (tupBreak) {
                    bool broke = breakBeforeElem(elemLine, val, pyc_output, first);
                    if (!broke && !first)
                        pyc_output << ", ";
                } else if (!first) {
                    pyc_output << ", ";
                }
                print_src(val, mod, pyc_output);
                first = false;
            }
            if (values.size() == 1)
                pyc_output << ',';
            if (tuple->requireParens())
                pyc_output << ')';
        }
        break;
    case ASTNode::NODE_ANNOTATED_VAR:
        {
            PycRef<ASTAnnotatedVar> annotated_var = node.cast<ASTAnnotatedVar>();
            PycRef<ASTObject> name = annotated_var->name().cast<ASTObject>();
            PycRef<ASTNode> annotation = annotated_var->annotation();

            pyc_output << name->object().cast<PycString>()->value();
            pyc_output << ": ";
            /* Under `from __future__ import annotations` (PEP 563) a class/module
               annotation is stored as the STRING of its source expression
               (`x: bool` -> the const 'bool'); render that raw so it does not
               come out double-quoted as `x: 'bool'` and recompile to the wrong
               (quoted) string const. Mirrors the function-annotation path. The
               future flag is module-wide, so read it from the module code. */
            bool futureAnnotVar = (mod->code()->flags() & PycCode::CO_FUTURE_ANNOTATIONS) != 0;
            if (futureAnnotVar && annotation != nullptr
                    && annotation.type() == ASTNode::NODE_OBJECT) {
                PycRef<PycObject> o = annotation.cast<ASTObject>()->object();
                if (o->type() == PycObject::TYPE_STRING
                        || o->type() == PycObject::TYPE_UNICODE
                        || o->type() == PycObject::TYPE_INTERNED
                        || o->type() == PycObject::TYPE_ASCII
                        || o->type() == PycObject::TYPE_ASCII_INTERNED
                        || o->type() == PycObject::TYPE_SHORT_ASCII
                        || o->type() == PycObject::TYPE_SHORT_ASCII_INTERNED) {
                    pyc_output << o.cast<PycString>()->value();
                    break;
                }
            }
            print_src(annotation, mod, pyc_output);
        }
        break;
    case ASTNode::NODE_TERNARY:
        {
            /* parenthesis might be needed
             * 
             * when if-expr is part of numerical expression, ternary has the LOWEST precedence
             *     print(a + b if False else c)
             * output is c, not a+c (a+b is calculated first)
             * 
             * but, let's not add parenthesis - to keep the source as close to original as possible in most cases
             */
            PycRef<ASTTernary> ternary = node.cast<ASTTernary>();
            //pyc_output << "(";
            print_src(ternary->if_expr(), mod, pyc_output);
            const auto if_block = ternary->if_block().cast<ASTCondBlock>();
            pyc_output << " if ";
            if (if_block->negative())
                pyc_output << "not ";
            print_src(if_block->cond(), mod, pyc_output);
            pyc_output << " else ";
            print_src(ternary->else_expr(), mod, pyc_output);
            //pyc_output << ")";
        }
        break;
    default:
        pyc_output << "<NODE:" << node->type() << ">";
        fprintf(stderr, "Unsupported Node type: %d\n", node->type());
        cleanBuild = false;
        node_seen.erase((ASTNode *)node);
        return;
    }

    cleanBuild = true;
    node_seen.erase((ASTNode *)node);
}

bool print_docstring(PycRef<PycObject> obj, int indent, PycModule* mod,
                     std::ostream& pyc_output)
{
    // docstrings are translated from the bytecode __doc__ = 'string' to simply '''string'''
    auto doc = obj.try_cast<PycString>();
    if (doc != nullptr) {
        start_line(indent, pyc_output);
        doc->print(pyc_output, mod, true);
        pyc_output << "\n";
        return true;
    }
    return false;
}
