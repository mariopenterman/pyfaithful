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

/* ==========================================================================
 * ASTree.cpp -- the decompiler core: Python byte-code -> readable source.
 *
 * Decompilation happens in three phases (look for the "PHASE" banners; the
 * renderer physically sits between the builder and the faithfulness passes):
 *
 *   PHASE 1  BuildFromCode()      byte-code  ->  AST
 *       A stack-machine walk over the instruction stream that rebuilds
 *       expressions (an operand stack) and statements/control flow (a stack
 *       of open blocks: if/for/while/try/with/...). This is the large, hard
 *       part -- 3.11's jump-threaded control flow is reconstructed here.
 *
 *   PHASE 2  the faithfulness passes + decompyle()
 *       Post-build AST transforms that make the reconstructed source
 *       recompile to the *same* byte-code and source positions as the
 *       original: recover statements stripped under -OO (placeholders),
 *       fold if-return chains, suppress the implicit return-None epilogue,
 *       etc. decompyle() orchestrates build -> transform -> render.
 *
 *   PHASE 3  print_src() / print_block()   AST  ->  source text
 *       A recursive AST printer, plus the layout engine that reproduces the
 *       original line and column positions (multi-line expressions, compound
 *       statements, padding). INVARIANT: layout only ever adds whitespace, so
 *       it never changes the recompiled byte-code.
 *
 * Shared state: a number of file-scope flags/maps (e.g. inLambda, cur_indent,
 * g_lineFaithful) carry context between the phases; each is documented at its
 * definition below.
 * ========================================================================== */

// This must be a triple quote (''' or """), to handle interpolated string literals containing the opposite quote style.
// E.g. f'''{"interpolated "123' literal"}'''    -> valid.
// E.g. f"""{"interpolated "123' literal"}"""    -> valid.
// E.g. f'{"interpolated "123' literal"}'        -> invalid, unescaped quotes in literal.
// E.g. f'{"interpolated \"123\' literal"}'      -> invalid, f-string expression does not allow backslash.
// NOTE: Nested f-strings not supported.
#define F_STRING_QUOTE "'''"

static void append_to_chain_store(const PycRef<ASTNode>& chainStore,
        PycRef<ASTNode> item, FastStack& stack, const PycRef<ASTBlock>& curblock);

namespace {
struct ScanIns { int off; int op; int arg; };

static int parseStoreGroup(const std::vector<ScanIns>& ins, int i, bool& ok)
{
    int n = (int)ins.size();
    int d = 0;
    while (i < n) {
        switch (ins[i].op) {
        case Pyc::CACHE: case Pyc::PRECALL_A: case Pyc::RESUME_A: case Pyc::NOP:
            i++; continue;
        case Pyc::STORE_FAST_A: case Pyc::STORE_NAME_A:
        case Pyc::STORE_GLOBAL_A: case Pyc::STORE_DEREF_A:
            if (d != 0) { ok = false; return i; }
            ok = true; return i + 1;
        case Pyc::STORE_ATTR_A:
            if (d != 1) { ok = false; return i; }
            ok = true; return i + 1;
        case Pyc::STORE_SUBSCR:
            if (d != 2) { ok = false; return i; }
            ok = true; return i + 1;
        case Pyc::UNPACK_SEQUENCE_A: {
            if (d != 0) { ok = false; return i; }
            int cnt = ins[i].arg; i++;
            for (int k = 0; k < cnt; k++) {
                bool sub = false; i = parseStoreGroup(ins, i, sub);
                if (!sub) { ok = false; return i; }
            }
            ok = true; return i;
        }
        case Pyc::UNPACK_EX_A: {
            if (d != 0) { ok = false; return i; }
            int cnt = (ins[i].arg & 0xFF) + (ins[i].arg >> 8) + 1; i++;
            for (int k = 0; k < cnt; k++) {
                bool sub = false; i = parseStoreGroup(ins, i, sub);
                if (!sub) { ok = false; return i; }
            }
            ok = true; return i;
        }
        case Pyc::LOAD_FAST_A: case Pyc::LOAD_NAME_A: case Pyc::LOAD_DEREF_A:
        case Pyc::LOAD_CONST_A: case Pyc::LOAD_CLASSDEREF_A:
            d += 1; i++; continue;
        case Pyc::LOAD_GLOBAL_A:
            d += (ins[i].arg & 1) ? 2 : 1; i++; continue;
        case Pyc::LOAD_ATTR_A:
            i++; continue;
        case Pyc::BINARY_SUBSCR:
            d -= 1; i++; continue;
        default:
            ok = false; return i;
        }
    }
    ok = false; return i;
}

static std::unordered_set<int> scanChainAssignCopies(PycRef<PycCode> code, PycModule* mod)
{
    std::unordered_set<int> result;
    if (mod->verCompare(3, 11) < 0)
        return result;
    std::vector<ScanIns> ins;
    PycBuffer src(code->code()->value(), code->code()->length());
    int op, arg, pos = 0;
    while (!src.atEof()) {
        int off = pos;
        bc_next(src, mod, op, arg, pos);
        if (pos <= off) break;
        ins.push_back({off, op, arg});
    }
    int n = (int)ins.size();
    auto isNop = [](int op) {
        return op == Pyc::CACHE || op == Pyc::PRECALL_A
            || op == Pyc::RESUME_A || op == Pyc::NOP;
    };
    int i = 0;
    while (i < n) {
        if (ins[i].op == Pyc::COPY_A && ins[i].arg == 1) {
            std::vector<int> copies;
            int j = i, k = 0; bool good = true;
            for (;;) {
                copies.push_back(j);
                bool ok = false; j = parseStoreGroup(ins, j + 1, ok);
                if (!ok) { good = false; break; }
                k++;
                while (j < n && isNop(ins[j].op)) j++;
                if (j < n && ins[j].op == Pyc::COPY_A && ins[j].arg == 1)
                    continue;
                break;
            }
            if (good && k >= 1) {
                bool ok = false; parseStoreGroup(ins, j, ok);
                if (ok) {
                    for (int ci : copies) result.insert(ins[ci].off);
                    i = copies.back() + 1;
                    continue;
                }
            }
        }
        i++;
    }
    return result;
}
} // anonymous namespace

bool cleanBuild;

/* Set by BuildFromCode's pre-scan when the function body's final
   `LOAD_CONST None; RETURN_VALUE` (or `RETURN_CONST None`) block is the target of
   >=2 jumps — a SHARED merge point. CPython emits a single shared trailing-None
   block ONLY for an explicit `return None`; an implicit epilogue reached by
   several paths is duplicated per path. So when set, the trailing `return None`
   is kept (not stripped as the implicit epilogue), since dropping it recompiles
   to separate per-branch return-None blocks — a byte divergence. */
static bool keepFinalRetNone = false;

bool inLambda = false;
bool inAsyncGen = false;

/* When a generator expression is the SOLE positional argument of a call
   (`sum(x for x in y)`), its own delimiting parentheses coincide with the
   call's argument parentheses, so Python allows -- and canonically uses --
   the un-parenthesised form. Set while rendering that argument so the
   comprehension omits its own `(` / `)`. */
bool suppressGenExprParens = false;

/* When a lambda is a call argument its enclosing parentheses are redundant
   (`f(lambda x: x)` rather than `f((lambda x: x))`): the lambda body is a
   `test`, so an argument-list comma terminates it and cannot be swallowed.
   Set while rendering such an argument so the lambda omits its own `(` / `)`. */
bool suppressLambdaParens = false;

bool printDocstringAndGlobals = false;

bool printClassDocstring = true;

/* Stack of the class names lexically enclosing the node currently being
   rendered. Used to reverse CPython's private-name mangling: a name of the
   form `_<ClassName>__rest` (where <ClassName> is an enclosing class and the
   original spelling `__rest` does not end in `__`) was written `__rest` in the
   source. Un-mangling is co_code-safe because recompiling `__rest` inside the
   same class re-mangles to the identical `_<ClassName>__rest`. */
std::vector<std::string> g_classNameStack;

/* If `name` is a mangled private identifier for one of the enclosing classes,
   return its original (leading `__`) spelling; otherwise return it unchanged. */
std::string demangleName(const std::string& name)
{
    if (g_classNameStack.empty())
        return name;
    if (name.size() < 3 || name[0] != '_' || name[1] == '_')
        return name;  // mangled names begin with a single leading underscore
    for (auto it = g_classNameStack.rbegin(); it != g_classNameStack.rend(); ++it) {
        /* CPython strips leading underscores from the class name before
           mangling, and skips classes whose name is all underscores. */
        std::string cls = *it;
        size_t s = cls.find_first_not_of('_');
        if (s == std::string::npos)
            continue;
        cls = cls.substr(s);
        const std::string prefix = std::string("_") + cls + std::string("__");
        if (name.compare(0, prefix.size(), prefix) == 0) {
            std::string rest = name.substr(prefix.size() - 2);  // keep the `__`
            /* Names ending in two underscores are not mangled. */
            if (rest.size() >= 2 && rest.compare(rest.size() - 2, 2, "__") == 0)
                return name;
            return rest;
        }
    }
    return name;
}

static PycRef<ASTNode> StackPopTop(FastStack& stack)
{
    const auto node(stack.top());
    stack.pop();
    return node;
}

static PycRef<ASTNode> NegateCond(PycRef<ASTNode> node)
{
    if (node == NULL)
        return node;
    if (node.type() == ASTNode::NODE_UNARY
            && node.cast<ASTUnary>()->op() == ASTUnary::UN_NOT)
        return node.cast<ASTUnary>()->operand();
    if (node.type() == ASTNode::NODE_COMPARE) {
        PycRef<ASTCompare> c = node.cast<ASTCompare>();
        static const ASTCompare::CompareOp inv[] = {
            ASTCompare::CMP_GREATER_EQUAL,
            ASTCompare::CMP_GREATER,
            ASTCompare::CMP_NOT_EQUAL,
            ASTCompare::CMP_EQUAL,
            ASTCompare::CMP_LESS_EQUAL,
            ASTCompare::CMP_LESS,
            ASTCompare::CMP_NOT_IN,
            ASTCompare::CMP_IN,
            ASTCompare::CMP_IS_NOT,
            ASTCompare::CMP_IS,
        };
        int op = c->op();
        if (op >= ASTCompare::CMP_LESS && op <= ASTCompare::CMP_IS_NOT)
            return new ASTCompare(c->left(), c->right(), inv[op]);
    }
    return new ASTUnary(node, ASTUnary::UN_NOT);
}

static void CheckIfExpr(FastStack& stack, PycRef<ASTBlock> curblock)
{
    if (stack.empty())
        return;
    if (curblock->nodes().size() < 2)
        return;
    auto rit = curblock->nodes().crbegin();
    if ((*rit)->type() != ASTNode::NODE_BLOCK ||
        (*rit).cast<ASTBlock>()->blktype() != ASTBlock::BLK_ELSE)
        return;
    PycRef<ASTBlock> elseBlk = (*rit).cast<ASTBlock>();
    ++rit;
    if ((*rit)->type() != ASTNode::NODE_BLOCK ||
        (*rit).cast<ASTBlock>()->blktype() != ASTBlock::BLK_IF)
        return;
    PycRef<ASTBlock> ifBlk = (*rit).cast<ASTBlock>();
    if (ifBlk->size() != 0 || elseBlk->size() != 0)
        return;
    auto else_expr = StackPopTop(stack);
    curblock->removeLast();
    auto if_block = curblock->nodes().back();
    auto if_expr = StackPopTop(stack);
    curblock->removeLast();
    stack.push(new ASTTernary(std::move(if_block), std::move(if_expr), std::move(else_expr)));
}

/* Anchor-NOP offsets of the current code object's one-per-line const-default
   signatures. Populated by BuildFromCode, consumed by decompyle immediately
   after (before any child code object is decompiled). */
std::set<int> g_sigAnchorNopOffs;

static PycRef<ASTNode> InlineComprehension(PycRef<PycCode> code, PycModule* mod,
                                           PycRef<ASTNode> iter)
{
    bool savedClean = cleanBuild;
    /* BuildFromCode clears g_sigAnchorNopOffs on entry; save/restore it so this
       inline recursion does not wipe the enclosing code object's accumulated
       signature anchors (consumed by decompyle after the outer build). */
    std::set<int> savedSigAnchors = g_sigAnchorNopOffs;
    PycRef<ASTNode> body = BuildFromCode(code, mod);
    cleanBuild = savedClean;
    g_sigAnchorNopOffs = savedSigAnchors;
    PycRef<ASTComprehension> comp;
    if (body != nullptr && body.type() == ASTNode::NODE_NODELIST) {
        for (const auto& n : body.cast<ASTNodeList>()->nodes()) {
            PycRef<ASTNode> v = n;
            if (v != nullptr && v.type() == ASTNode::NODE_RETURN)
                v = v.cast<ASTReturn>()->value();
            if (v != nullptr && v.type() == ASTNode::NODE_COMPREHENSION) {
                comp = v.cast<ASTComprehension>();
                break;
            }
        }
    }
    if (comp == nullptr)
        return nullptr;
    if (!comp->generators().empty())
        comp->generators().front()->setIter(iter);
    return comp.cast<ASTNode>();
}

static bool nameStrEq(const PycRef<ASTNode>& a, const PycRef<ASTNode>& b)
{
    if (a == nullptr || b == nullptr) return false;
    if (a.type() != ASTNode::NODE_NAME || b.type() != ASTNode::NODE_NAME) return false;
    PycRef<PycString> sa = a.cast<ASTName>()->name();
    PycRef<PycString> sb = b.cast<ASTName>()->name();
    if (sa == nullptr || sb == nullptr) return false;
    return strcmp(sa->value(), sb->value()) == 0;
}

static bool tryAttachDecorators(PycRef<ASTBlock> curblock,
                                PycRef<ASTNode> value, PycRef<ASTNode> name)
{
    if (value == nullptr || value.type() != ASTNode::NODE_CALL)
        return false;
    if (name == nullptr || name.type() != ASTNode::NODE_NAME)
        return false;
    if (curblock == nullptr || curblock->nodes().empty())
        return false;
    PycRef<ASTNode> prev = curblock->nodes().back();
    if (prev.type() != ASTNode::NODE_STORE)
        return false;
    PycRef<ASTNode> defSrc = prev.cast<ASTStore>()->src();
    PycRef<ASTNode> defDest = prev.cast<ASTStore>()->dest();
    bool isFunc = defSrc.type() == ASTNode::NODE_FUNCTION;
    bool isClass = defSrc.type() == ASTNode::NODE_CLASS;
    if (!isFunc && !isClass)
        return false;
    if (!nameStrEq(defDest, name))
        return false;
    if (isFunc) {
        PycRef<PycCode> fc = defSrc.cast<ASTFunction>()->code()
                .cast<ASTObject>()->object().cast<PycCode>();
        if (fc->name()->isEqual("<lambda>"))
            return false;
    }
    std::vector<PycRef<ASTNode> > decos;
    PycRef<ASTNode> cur = value;
    while (cur.type() == ASTNode::NODE_CALL) {
        PycRef<ASTCall> call = cur.cast<ASTCall>();
        if (call->pparams().size() != 1 || !call->kwparams().empty())
            return false;
        decos.push_back(call->func());
        cur = call->pparams().front();
    }
    if (!nameStrEq(cur, name))
        return false;
    for (auto& d : decos) {
        if (isFunc) defSrc.cast<ASTFunction>()->addDecorator(d);
        else        defSrc.cast<ASTClass>()->addDecorator(d);
    }
    return true;
}

/* Recover a boolean OR whose constant-False left operand was folded from
   `<false-const> and <X>` — the flowgraph optimizer collapses the `and` to the
   left False but leaves <X> orphaned in co_consts and, having already run the
   or-fold pass, leaves the outer OR unfolded (the `__debug__ and '.pyc' or '.pyo'`
   idiom under -OO, where __debug__ is a compile-time False). A literal `False or Y`
   would itself fold away, so a surviving unfolded `LOAD_CONST False;
   JUMP_IF_TRUE_OR_POP` proves the False came from an inner `and`; re-attach the
   sole orphaned const as `False and <X>` so the source round-trips at every
   optimize level (semantically `False and X` is just `False`). */
static PycRef<ASTNode> recoverFoldedAndOperand(PycRef<ASTNode> left, bool isOr,
                                               PycRef<PycCode> code, PycModule* mod)
{
    if (!isOr || left == nullptr || left.type() != ASTNode::NODE_OBJECT)
        return left;
    PycRef<PycObject> lv = left.cast<ASTObject>()->object();
    if (lv == nullptr || lv.type() != PycObject::TYPE_FALSE || code->consts() == nullptr)
        return left;
    std::set<int> loaded;
    {
        PycBuffer s(code->code()->value(), code->code()->length());
        int o, a, p = 0;
        while (!s.atEof()) {
            bc_next(s, mod, o, a, p);
            if (o == Pyc::LOAD_CONST_A)
                loaded.insert(a);
        }
    }
    PycRef<PycObject> orphan;
    int norphan = 0;
    const int n = code->consts()->size();
    for (int i = 0; i < n; i++) {
        PycRef<PycObject> c = code->getConst(i);
        if (c != nullptr && c.type() != PycObject::TYPE_CODE
                && loaded.find(i) == loaded.end()) {
            orphan = c;
            norphan++;
        }
    }
    if (norphan != 1)
        return left;
    return PycRef<ASTNode>(new ASTBinary(left, new ASTObject(orphan),
                                         ASTBinary::BIN_LOG_AND));
}

/* ==========================================================================
 * PHASE 1 -- BuildFromCode: byte-code -> AST
 *
 * A single stack-machine pass over the instruction stream that rebuilds both
 * expressions and control flow. It is large (one function, ~250 opcode cases)
 * because CPython 3.11's jump-threaded control flow has to be reconstructed
 * here; the bulk of the code is not the opcodes themselves but the block
 * matching/coalescing that turns jumps back into if/elif/else, loops, and
 * try/except/finally.
 *
 * How to read it:
 *   - Two working stacks drive everything:
 *       `stack`   -- the operand stack: partial EXPRESSIONS being assembled
 *                    (an ADD pops two operands and pushes an ASTBinary, etc.).
 *       `blocks`  -- the block stack: currently-open STATEMENTS/suites, with
 *                    `curblock` the innermost. Statements are appended to
 *                    `curblock`; a jump target that matches a block's end pops
 *                    that block.
 *   - The prologue (right below) declares that state plus a large set of small
 *     bookkeeping maps/sets keyed by byte offset. Each map exists to remember a
 *     control-flow decision made at one offset so a later offset can act on it
 *     (e.g. "a finally copy starts here", "this jump is a loop continue, not a
 *     break"). They are grouped and commented at their declarations.
 *   - The main loop reads one instruction at a time (bc_next) and dispatches on
 *     `opcode` in a big switch; each case updates the two stacks and/or the
 *     bookkeeping maps.
 *   - The epilogue closes any still-open blocks and returns the assembled tree.
 *
 * NOTE: because the whole reconstruction shares this state, the function is not
 * split into per-opcode helpers -- doing so safely would mean promoting all of
 * the locals (and the ~50 `[&]` lambdas) to a shared object without changing a
 * single emitted byte. See ASTree_priv.h for the split between this builder,
 * the renderer (ASTRender.cpp), and the faithfulness passes (ASTFaithful.cpp).
 * ========================================================================== */
/* ---------------------------------------------------------------------------
 * CodeBuilder owns the Phase-1 reconstruction of a SINGLE code object. Today it
 * holds only the code object and its module and runs the entire pass in build();
 * the ~150 pieces of pass state (the two stacks, curblock, and the offset-keyed
 * bookkeeping maps) currently live as locals inside build(). They will migrate
 * onto this class member-by-member as cohesive opcode groups are lifted out of
 * the switch into documented handler methods -- because unqualified name lookup
 * and `[&]` capture inside a member function both resolve to members, promoting
 * a local to a member is a no-op at its use-sites and inside the lambdas.
 * BuildFromCode() stays the public entry point (one CodeBuilder per code object,
 * so recursion into nested code objects just constructs another builder).
 * ------------------------------------------------------------------------- */
class CodeBuilder {
public:
    CodeBuilder(PycRef<PycCode> code, PycModule* mod)
        : code(code), mod(mod),
          stack((mod->majorVer() == 1) ? 20 : code->stackSize()),
          source(code->code()->value(), code->code()->length()) {}
    PycRef<ASTNode> build();

private:
    /* A loop's [start, end) byte range and its exit offset (a continue records
       exit = the offset just past the loop). */
    struct LoopRange { int start; int end; int exit; };

    /* Per-opcode-group handlers lifted out of build()'s dispatch switch. */
    void handleUnaryOp(int opcode);
    void handleBinaryOp(int opcode, int operand);
    void handleIsContainsOp(int opcode, int operand);
    void handleBuildCollection(int opcode, int operand);
    void handleLoad(int opcode, int operand);
    void handleDelete(int opcode, int operand);
    void handleStoreSlice(int opcode);
    void handleSubscript(int opcode);
    void handleStackManip(int opcode, int operand);
    void handleCollectionUpdate(int opcode);
    void handlePrint(int opcode);
    void handleImport(int opcode, int operand);
    void handleExprWrap(int opcode, int operand);
    void handleRaiseVarargs(int operand);

    /* Simultaneous tuple-assignment helpers (see the tupleAssignStart prescan). */
    bool tupleAssignSafe(int K);
    void tupleStoreStep(PycRef<ASTNode> tname);
    /* Returns false to signal a bail-out (the caller returns the partial tree). */
    bool handleStore(int opcode, int operand);
    void handleUnpack(int opcode, int operand);
    void handleYield(int opcode);
    void handlePopBlock();
    /* Returns false to signal a bail-out (the caller returns the partial tree). */
    bool handlePopTop();
    void handleReraise();
    void handleSetupBlock(int opcode, int operand);
    void handleForIter(int opcode, int operand);
    bool handleSend();

    /* --- pass state (migrating from build() locals onto the class) --- */
    PycRef<PycCode> code;
    PycModule* mod;
    /* The operand stack: partial EXPRESSIONS being assembled. Promoted from a
       build() local so extracted handlers reach it as a member (unqualified
       name lookup and `[&]` capture inside build() resolve to it unchanged). */
    FastStack stack;
    /* The instruction stream being decoded, and the byte offset (pos) of the
       NEXT instruction to read; bc_next advances both. curpos (above) is the
       offset of the instruction currently being dispatched. Control-flow
       reconstruction rewinds/advances `source`+`pos` to re-read or skip ahead. */
    PycBuffer source;
    int pos = 0;
    /* The opcode and operand of the instruction currently being dispatched.
       Members (not build() locals) because some control-flow handlers re-read
       ahead with bc_next into these, and the loop epilogue reads opcode after
       the handler returns (to track lastSubstantialOp). */
    int opcode, operand;
    /* Saved operand-stack snapshots: a copy of `stack` is pushed when a block
       that may consume a fresh stack opens, and restored when a branch/handler
       unwinds back out of it. */
    stackhist_t stack_hist;
    /* Set to 1 by an in-place binary op so the STORE that follows renders as an
       augmented assignment (`x += y`) rather than `x = x + y`. */
    int inplaceStore = 0;
    /* Armed just before a comparison whose left operand continues a chained
       comparison (`a < b < c`), so the compare builds/extends an ASTChainCompare
       instead of a plain ASTCompare. Cleared after each comparison consumes it. */
    int chainCmp = 0;
    /* The block stack: currently-open statements/suites. curblock is the
       innermost (statements are appended to it); defblock is the outermost
       module/function body returned when the stream ends. Promoted from build()
       locals; default-constructed here and initialised at the top of build(). */
    std::stack<PycRef<ASTBlock> > blocks;
    PycRef<ASTBlock> defblock;
    PycRef<ASTBlock> curblock;
    /* Sequence unpacking (`a, b, *c = ...`): unpack counts the remaining targets
       of the active UNPACK_SEQUENCE/EX; unpackNest holds the outer counts while a
       nested tuple target is filled; unpackStar is the target index of a starred
       element, or -1 when there is none. */
    int unpack = 0;
    std::vector<int> unpackNest;
    int unpackStar = -1;
    /* Simultaneous tuple assignment `t1, ..., tN = v1, ..., vN` (equal arity,
       N >= 2). CPython drops the BUILD_TUPLE/UNPACK_SEQUENCE and instead pushes
       v1..vN then stores tN..t1 — a run of N consecutive simple STOREs. Without
       this, each store renders as its own statement, reordering the bytecode.
       tupleStore counts the remaining stores of the active run; the
       target/value nodes accumulate in reverse (bytecode/pop) order. */
    int tupleStore = 0;
    std::vector<PycRef<ASTNode> > tupleStoreTargets;
    std::vector<PycRef<ASTNode> > tupleStoreValues;
    /* Byte offset of the instruction currently being dispatched. */
    int curpos = 0;
    /* Walrus (`:=`) bookkeeping from the prescan: offsets whose STORE is the
       named target of an assignment expression (walrusStores), and the paired
       COPY offsets that duplicate the value (walrusCopies). */
    std::unordered_set<int> walrusCopies, walrusStores;
    /* Prescan map keyed by the STORE offset that begins a foldable simultaneous
       tuple assignment; the value is the number of parallel targets in the run. */
    std::unordered_map<int, int> tupleAssignStart;
    /* True once SETUP_ANNOTATIONS has been seen, so a store into __annotations__
       is recognised as a variable annotation (`name: type`) rather than a dict
       item assignment. */
    bool variable_annotations = false;
    /* The code object's exception table (3.11+ zero-cost exceptions) and a
       cursor into it: next_exception_entry is the index of the first entry not
       yet passed, advanced as `pos` moves forward. */
    std::vector<PycExceptionTableEntry> exception_entries;
    size_t next_exception_entry = 0;
    /* --- control-flow prescan maps (offset-keyed decisions the main loop acts
       on later); promoted from build() locals as their handlers are extracted. */
    /* At a RERAISE offset, the resume target of the enclosing try/except so the
       reconstructed handler closes and control continues past it. */
    std::unordered_map<int, int> finExcReraise;
    /* Set by SETUP_FINALLY and cleared by SETUP_EXCEPT: whether the container
       block just opened is a bare try/finally (no except clause). */
    bool need_try = false;
    /* Prescan-collected [start,end,exit) ranges of every loop, and a map from a
       FOR_ITER offset to an early loop-close offset when the loop body ends
       before the recorded block end. */
    std::vector<LoopRange> loopRanges;
    std::unordered_map<int, int> forEarlyClose;
    /* True while a comprehension's trailing filter (`if cond`) is still being
       threaded forward onto the comprehension's condition. */
    bool compFilterFwd = false;
};

PycRef<ASTNode> BuildFromCode(PycRef<PycCode> code, PycModule* mod)
{
    return CodeBuilder(code, mod).build();
}

PycRef<ASTNode> CodeBuilder::build()
{

    /* True when the byte range [from, to) contains nothing but NOP/CACHE
       padding — i.e. `to` is the same logical landing as `from` once the
       no-op fillers the compiler inserted are skipped. Used so a forward
       jump can still be recognised as targeting a loop exit even when a
       stray NOP sits between the recorded exit and the jump target. */
    auto onlyPaddingBetween = [&](int from, int to) -> bool {
        if (from > to) return false;
        PycBuffer s(code->code()->value(), code->code()->length());
        s.setPos(from);
        int o, a, p = from;
        while (p < to && !s.atEof()) {
            bc_next(s, mod, o, a, p);
            if (o != Pyc::NOP && o != Pyc::CACHE)
                return false;
        }
        return p == to;
    };

    defblock = new ASTBlock(ASTBlock::BLK_MAIN);
    defblock->init();
    curblock = defblock;
    blocks.push(defblock);

    struct BoolShortCircuit { PycRef<ASTNode> left; bool isOr; int target; int off; };
    std::vector<BoolShortCircuit> boolPending;
    bool else_pop = false;
    int lastSubstantialOp = Pyc::PYC_INVALID_OPCODE;
    std::unordered_map<int,int> dupHandlerEnd;
    std::unordered_set<int> dupActiveSkip;
    std::unordered_map<int, int> finallyReturnExit;
    /* try/FINALLY inside a loop whose NORMAL-exit continuation is a `return X`
       placed by the compiler as a shared tail block PAST the enclosing loop's
       exit statement (character `an observed function`: the finally's
       normal-copy `JUMP_FORWARD` skips the loop-exit `raise` and lands on a
       `LOAD_x; RETURN_VALUE`). This `return X` is the LAST STATEMENT OF THE INNER
       try body: the outer finally wraps an inner try/except, and on the try body's
       normal exit the compiler inlines the finally normal-copy then loads+returns X.
       Left alone, pycdc renders the inner try/except + outer finally correctly but
       DROPS the return — it resumes the finally at the return offset, closing the
       loop first (return > loop exit) and stranding `return X` as dead code AFTER
       the loop-exit `raise`. The finally normal-copy is the finallyCopySkip region
       whose start (the inner-except protected range's end) is this map's KEY: when
       that copy is about to be skipped and the inner BLK_TRY is still curblock, the
       synthesized `return X` is appended INTO the inner try body (so it renders as
       the try's last statement, BEFORE the except handler — matching the original's
       `…LOOP CB CB EXC MATCH…` layout where the inlined finally-copy precedes the
       inner EXC). `after` is still redirected to the loop exit so the loop closes
       and its exit `raise` renders post-loop; the tail-level emit at the return
       offset is suppressed via teoElseAbsorbSuppress. */
    std::unordered_map<int, PycRef<ASTNode> > finTryReturnRet;
    std::unordered_set<int> finallyReturnSkip;
    std::unordered_set<int> dupRetNoneSkip;
    std::unordered_set<int> raiseFinallyClose;
    std::unordered_set<int> loopTailFinClose;
    std::unordered_set<int> finallyEpilogueCut;
    /* Offsets of the run of NOPs seen immediately before the current instruction
       (see the LOAD_CONST tuple recording below): the compiler emits one such
       line-anchor NOP per source line carrying a constant default in a multi-line
       signature. sigTupleNopOffs maps a const-tuple LOAD_CONST's offset to those
       NOP offsets so MAKE_FUNCTION can recover a signature's anchor NOPs. */
    std::vector<int> pendingNopOffs;
    std::unordered_map<int, std::vector<int> > sigTupleNopOffs;
    /* Byte offsets of the anchor NOPs consumed by one-per-line const-default
       signatures (see g_sigAnchorNopOffs): excluded from the stripped-statement
       placeholder passes so the signature render (not a placeholder) reproduces
       them, with no double-count. Cleared per code object; consumed by decompyle
       right after this returns, before any child code object is decompiled. */
    g_sigAnchorNopOffs.clear();

    if (mod->verCompare(3, 11) >= 0) {
        exception_entries = code->exceptionTableEntries();
        const auto* ebuf = code->code()->value();
        int elen = (int)code->code()->length();
        auto gapIsReturnOnly = [&](int from, int to) -> bool {
            if (from >= to) return false;
            PycBuffer s(ebuf, elen); s.setPos(from);
            int o, a, p = from; bool sawReturn = false;
            while (p < to && !s.atEof()) {
                bc_next(s, mod, o, a, p);
                if (o == Pyc::CACHE || o == Pyc::NOP) continue;
                if (o == Pyc::RETURN_VALUE || o == Pyc::RETURN_CONST_A
                        || o == Pyc::INSTRUMENTED_RETURN_VALUE_A) {
                    sawReturn = true; continue;
                }
                if (o == Pyc::LOAD_CONST_A) continue;
                return false;
            }
            return sawReturn;
        };
        bool merged = true;
        while (merged) {
            merged = false;
            for (size_t i = 0; i + 1 < exception_entries.size(); ++i) {
                auto& a = exception_entries[i];
                auto& b = exception_entries[i + 1];
                if (a.push_lasti || b.push_lasti) continue;
                if (a.target != b.target) continue;
                if (a.stack_depth != b.stack_depth) continue;
                if (b.start_offset <= a.end_offset) continue;
                if (!gapIsReturnOnly(a.end_offset, b.start_offset)) continue;
                a.end_offset = std::max(a.end_offset, b.end_offset);
                exception_entries.erase(exception_entries.begin() + (i + 1));
                merged = true;
                break;
            }
        }

        /* `try: pass except ...: ...` with an EMPTY try body. CPython omits the
           body-protecting exception-table entry when the protected body cannot
           raise (it is just a NOP), leaving an orphan handler: the only entry is
           the handler's own lasti cleanup (start = the PUSH_EXC_INFO, target =
           the COPY/RERAISE stub). Without a body entry the standard try/except
           machinery never opens the block, so the orphan PUSH_EXC_INFO derails
           the build. Synthesize the missing body entry (NOP -> handler) so the
           normal machinery renders it as a `try: pass` with its handler. */
        {
            const char* sbuf = code->code()->value();
            int slen = (int)code->code()->length();
            const int SW = (int)sizeof(uint16_t);
            auto opAtS = [&](int off, int& arg) -> int {
                if (off < 0 || off >= slen) { arg = 0; return -1; }
                PycBuffer s(sbuf, slen); s.setPos(off);
                int o, a, p = off; bc_next(s, mod, o, a, p); arg = a; return o;
            };
            std::unordered_set<int> bodyTargets;
            for (const auto& e : exception_entries)
                if (!e.push_lasti) bodyTargets.insert(e.target);
            std::vector<PycExceptionTableEntry> synth;
            for (const auto& e : exception_entries) {
                if (!e.push_lasti) continue;
                int H = e.start_offset, ig;
                if (opAtS(H, ig) != Pyc::PUSH_EXC_INFO) continue;
                if (bodyTargets.count(H)) continue;          // not an orphan
                if (opAtS(e.target, ig) != Pyc::COPY_A) continue;
                int jfArg, nopArg;
                if (opAtS(H - 2 * SW, nopArg) != Pyc::NOP) continue;
                if (opAtS(H - SW, jfArg) != Pyc::JUMP_FORWARD_A) continue;
                int M = (H - SW + SW) + jfArg * SW;          // JF target (merge)
                // Handler must be a bare empty except: PUSH_EXC_INFO; POP_TOP;
                // POP_EXCEPT; JUMP_FORWARD -> M.
                int pa; PycBuffer s(sbuf, slen); s.setPos(H);
                int o, a, p = H;
                bc_next(s, mod, o, a, p);                     // PUSH_EXC_INFO
                bc_next(s, mod, o, a, p); if (o != Pyc::POP_TOP) continue;
                bc_next(s, mod, o, a, p); if (o != Pyc::POP_EXCEPT) continue;
                pa = p; bc_next(s, mod, o, a, p);
                if (o != Pyc::JUMP_FORWARD_A) continue;
                if (pa + SW + a * SW != M) continue;
                int depth = e.stack_depth > 0 ? e.stack_depth - 1 : 0;
                synth.emplace_back(H - 2 * SW, H - SW, H, depth, false);
            }
            for (auto& s : synth) {
                size_t pos = 0;
                while (pos < exception_entries.size()
                       && (exception_entries[pos].start_offset < s.start_offset
                           || (exception_entries[pos].start_offset == s.start_offset
                               && exception_entries[pos].end_offset <= s.end_offset)))
                    ++pos;
                exception_entries.insert(exception_entries.begin() + pos, s);
            }
        }
    }

    /* Detect a SHARED explicit trailing `return None` (see keepFinalRetNone). Find
       the last `LOAD_CONST None; RETURN_VALUE` / `RETURN_CONST None` block and count
       the jumps targeting it; >=2 means several branches converge on one trailing
       None-return, which the compiler emits only for an EXPLICIT `return None`. */
    keepFinalRetNone = false;
    if (mod->verCompare(3, 11) >= 0 && (code->flags() & PycCode::CO_OPTIMIZED)) {
        const char* ebuf = code->code()->value();
        int elen = (int)code->code()->length();
        const int W = (int)sizeof(uint16_t);
        int rnOff = -1, rnCount = 0;
        {
            PycBuffer s(ebuf, elen);
            int o, a, p = 0, prevOff = -1, prevOp = -1, prevArg = -1;
            while (!s.atEof()) {
                int ip = p;
                bc_next(s, mod, o, a, p);
                if ((o == Pyc::RETURN_VALUE || o == Pyc::INSTRUMENTED_RETURN_VALUE_A)
                        && prevOp == Pyc::LOAD_CONST_A
                        && code->getConst(prevArg).type() == PycObject::TYPE_NONE) {
                    rnOff = prevOff; ++rnCount;
                } else if (o == Pyc::RETURN_CONST_A
                        && code->getConst(a).type() == PycObject::TYPE_NONE) {
                    rnOff = ip; ++rnCount;
                }
                if (p <= ip) break;
                prevOff = ip; prevOp = o; prevArg = a;
            }
        }
        /* Require EXACTLY ONE None-return block: a shared block reached by several
           jumps comes only from an explicit `return None` (the implicit epilogue is
           emitted as a SEPARATE block per converging branch — e.g. an implicit
           compound-`if` false produces TWO None-blocks, one per short-circuit exit).
           Multiple None-blocks means the trailing one is a genuine per-branch
           implicit epilogue and must still be stripped. */
        if (rnOff >= 0 && rnCount == 1) {
            int hits = 0;
            PycBuffer s(ebuf, elen);
            int o, a, p = 0;
            while (!s.atEof()) {
                int ip = p;
                bc_next(s, mod, o, a, p);
                int tgt = -1;
                switch (o) {
                case Pyc::POP_JUMP_FORWARD_IF_FALSE_A:
                case Pyc::POP_JUMP_FORWARD_IF_TRUE_A:
                case Pyc::POP_JUMP_FORWARD_IF_NONE_A:
                case Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A:
                case Pyc::JUMP_FORWARD_A:
                    tgt = p + a * W; break;
                case Pyc::POP_JUMP_BACKWARD_IF_FALSE_A:
                case Pyc::POP_JUMP_BACKWARD_IF_TRUE_A:
                case Pyc::POP_JUMP_BACKWARD_IF_NONE_A:
                case Pyc::POP_JUMP_BACKWARD_IF_NOT_NONE_A:
                case Pyc::JUMP_BACKWARD_A:
                    tgt = p - a * W; break;
                default: break;
                }
                if (tgt == rnOff) ++hits;
                if (p <= ip) break;
            }
            if (hits >= 2) keepFinalRetNone = true;
        }
    }

    if (mod->verCompare(3, 11) >= 0 && !exception_entries.empty()) {
        const char* ebuf = code->code()->value();
        int elen = (int)code->code()->length();
        const int W = (int)sizeof(uint16_t);
        auto opAt = [&](int off) -> int {
            if (off < 0 || off >= elen) return -1;
            PycBuffer s(ebuf, elen); s.setPos(off);
            int o, a, p = off; bc_next(s, mod, o, a, p); return o;
        };
        auto seqMatch = [&](int A, int B, int K) -> bool {
            PycBuffer sa(ebuf, elen), sb(ebuf, elen); sa.setPos(A); sb.setPos(B);
            int oa, aa, pa = A, ob, ab, pb = B, n = 0;
            while (n < K) {
                do { if (pa >= elen) return false; bc_next(sa, mod, oa, aa, pa); }
                    while (oa == Pyc::CACHE || oa == Pyc::NOP);
                do { if (pb >= elen) return false; bc_next(sb, mod, ob, ab, pb); }
                    while (ob == Pyc::CACHE || ob == Pyc::NOP);
                if (oa != ob || aa != ab) return false;
                n++;
            }
            return true;
        };
        for (const auto& e : exception_entries) {
            if (e.push_lasti) continue;
            int T = e.target;
            if (dupHandlerEnd.count(T)) continue;
            if (opAt(T) != Pyc::PUSH_EXC_INFO) continue;
            int lastOp = -1, D = -1;
            { PycBuffer s(ebuf, elen); s.setPos(e.start_offset);
              int o, a, p = e.start_offset;
              while (p < e.end_offset && p < elen) {
                  bc_next(s, mod, o, a, p);
                  if (o == Pyc::CACHE || o == Pyc::NOP) continue;
                  lastOp = o; D = (o == Pyc::JUMP_FORWARD_A) ? (p + a * W) : -1;
              } }
            if (lastOp != Pyc::JUMP_FORWARD_A || D < 0) continue;
            int afterPush = T;
            { PycBuffer s(ebuf, elen); s.setPos(T); int o, a, p = T;
              bc_next(s, mod, o, a, p); afterPush = p; }
            if (!seqMatch(D, afterPush, 3)) continue;
            int tcleanup = -1;
            for (const auto& g : exception_entries)
                if (g.start_offset == T) { tcleanup = g.target; break; }
            if (tcleanup < 0 || opAt(tcleanup) != Pyc::COPY_A) continue;
            int dupEnd = tcleanup + 3 * W;
            if (dupEnd <= T || dupEnd > elen) continue;
            dupHandlerEnd[T] = dupEnd;
        }
    }

    std::unordered_set<int> chainCopyOffsets = scanChainAssignCopies(code, mod);

    {
        PycBuffer ws(code->code()->value(), code->code()->length());
        int wo, wa, wp = 0, prevOp = -1, prevArg = -1, prevOff = -1;
        int pp2Op = -1, pp2Arg = -1;
        while (!ws.atEof()) {
            int off = wp;
            bc_next(ws, mod, wo, wa, wp);
            if (wo == Pyc::CACHE) continue;
            bool selfStore =
                ((wo == Pyc::STORE_FAST_A && pp2Op == Pyc::LOAD_FAST_A)
                 || (wo == Pyc::STORE_NAME_A && pp2Op == Pyc::LOAD_NAME_A)
                 || (wo == Pyc::STORE_GLOBAL_A && pp2Op == Pyc::LOAD_GLOBAL_A)
                 || (wo == Pyc::STORE_DEREF_A && pp2Op == Pyc::LOAD_DEREF_A))
                && pp2Arg == wa;
            if ((wo == Pyc::STORE_FAST_A || wo == Pyc::STORE_NAME_A
                    || wo == Pyc::STORE_GLOBAL_A || wo == Pyc::STORE_DEREF_A)
                    && prevOp == Pyc::COPY_A && prevArg == 1
                    && !chainCopyOffsets.count(prevOff)
                    && !selfStore) {
                PycRef<PycString> nm =
                    (wo == Pyc::STORE_DEREF_A) ? code->getCellVar(mod, wa)
                    : (wo == Pyc::STORE_FAST_A) ? code->getLocal(wa)
                    : code->getName(wa);
                bool isClassCell = (nm != nullptr
                        && strcmp(nm->value(), "__classcell__") == 0);
                if (!isClassCell) {
                    walrusCopies.insert(prevOff);
                    walrusStores.insert(off);
                }
            }
            pp2Op = prevOp; pp2Arg = prevArg;
            prevOp = wo; prevArg = wa; prevOff = off;
        }
    }

    /* Prescan for simultaneous tuple assignment `t1, ..., tN = v1, ..., vN`
       (equal arity, N >= 2), which CPython compiles to N pushes followed by a
       run of N consecutive simple STOREs (tN..t1) with no BUILD_TUPLE/UNPACK.
       tupleAssignStart maps the run's first STORE offset -> N. The chained
       assign `a = b = <expr>` also ends in consecutive stores, but each is
       preceded by a COPY, so a run whose first store follows a COPY is
       excluded. */
    {
        auto isSimpleStore = [](int op) {
            return op == Pyc::STORE_FAST_A || op == Pyc::STORE_NAME_A
                || op == Pyc::STORE_GLOBAL_A || op == Pyc::STORE_DEREF_A;
        };
        std::vector<std::pair<int, int> > ins; // (op, off), CACHE skipped
        PycBuffer ts(code->code()->value(), code->code()->length());
        int to, ta, tp = 0;
        while (!ts.atEof()) {
            int off = tp;
            bc_next(ts, mod, to, ta, tp);
            if (to == Pyc::CACHE)
                continue;
            ins.emplace_back(to, off);
        }
        for (size_t i = 0; i < ins.size(); ) {
            if (isSimpleStore(ins[i].first)) {
                size_t j = i;
                while (j < ins.size() && isSimpleStore(ins[j].first))
                    ++j;
                int K = (int)(j - i);
                int prevOp = (i > 0) ? ins[i - 1].first : -1;
                /* A COPY before the run marks a chained assign `a = b = ...`;
                   an UNPACK_SEQUENCE/UNPACK_EX marks a real sequence unpack
                   `a, b = <iterable>` (handled by the unpack machinery). Only a
                   run fed by N independent value pushes is a literal tuple
                   assignment. */
                if (K >= 2 && prevOp != Pyc::COPY_A
                        && prevOp != Pyc::UNPACK_SEQUENCE_A
                        && prevOp != Pyc::UNPACK_EX_A)
                    tupleAssignStart[ins[i].second] = K;
                i = j;
            } else {
                ++i;
            }
        }
    }

    /* Prescan: terminal `if/else` at module- or class-body scope. CPython threads
       the shared exit of a terminal `if C: A else: B` (nothing follows the if/else
       at the same level) by ending the if-true arm A with an inline
       `LOAD_CONST None; RETURN_VALUE`, then begins the else arm B at the false-jump
       target. At module or class scope a `return` statement is illegal, so an inline
       return-None sitting right before a forward-jump target CANNOT be source code --
       it is compiler-generated exit threading, which proves the target region is a
       genuine `else:` and not unconditional post-`if` code. Without this the else is
       dropped: B renders as always-executed (a semantic change) and the recompiled
       co_code diverges (the shared exit re-threads differently). moduleElseAt maps
       the false-jump target offset -> the else-block end (the epilogue return-None).
       Restricted to module/class scope; at function scope the inline return-None is
       indistinguishable from a real trailing `return`. */
    std::unordered_map<int, int> moduleElseAt;
    {
        bool topScope = code->name() != nullptr && code->name()->isEqual("<module>");
        if (!topScope) {
            // class body: prologue is __name__ -> __module__, then __qualname__.
            PycBuffer cs(code->code()->value(), code->code()->length());
            int co, ca, cp = 0, real = 0;
            while (!cs.atEof() && real < 4) {
                bc_next(cs, mod, co, ca, cp);
                if (co == Pyc::CACHE || co == Pyc::RESUME_A || co == Pyc::NOP
                        || co == Pyc::EXTENDED_ARG_A)
                    continue;
                if (co == Pyc::STORE_NAME_A) {
                    PycRef<PycString> nm = code->getName(ca);
                    if (nm != nullptr && strcmp(nm->value(), "__module__") == 0) {
                        topScope = true;
                        break;
                    }
                }
                ++real;
            }
        }
        if (topScope) {
            const int W = (int)sizeof(uint16_t);
            struct MI { int op; int arg; int off; int end; };
            std::vector<MI> ins;
            PycBuffer ms(code->code()->value(), code->code()->length());
            int mo, ma, mp = 0;
            while (!ms.atEof()) {
                int off = mp;
                bc_next(ms, mod, mo, ma, mp);
                if (mo == Pyc::CACHE)
                    continue;
                ins.push_back({mo, ma, off, mp});
            }
            int n = (int)ins.size();
            auto isReal = [&](int i) {
                return ins[i].op != Pyc::NOP && ins[i].op != Pyc::EXTENDED_ARG_A;
            };
            // Trailing epilogue must be `LOAD_CONST None; RETURN_VALUE` (the module or
            // class body's implicit exit); its LOAD_CONST offset bounds every else.
            int lastReal = n - 1;
            while (lastReal >= 0 && !isReal(lastReal)) --lastReal;
            int epilogueOff = -1;
            if (lastReal >= 1 && ins[lastReal].op == Pyc::RETURN_VALUE) {
                int p = lastReal - 1;
                while (p >= 0 && !isReal(p)) --p;
                if (p >= 0 && ins[p].op == Pyc::LOAD_CONST_A
                        && code->getConst(ins[p].arg).type() == PycObject::TYPE_NONE)
                    epilogueOff = ins[p].off;
            }
            std::unordered_map<int, int> offToIdx;
            for (int i = 0; i < n; ++i)
                offToIdx[ins[i].off] = i;
            if (epilogueOff >= 0) {
                for (int i = 0; i < n; ++i) {
                    int op = ins[i].op;
                    if (op != Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                            && op != Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                            && op != Pyc::POP_JUMP_FORWARD_IF_NONE_A
                            && op != Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A)
                        continue;
                    int tgt = ins[i].end + ins[i].arg * W;
                    /* tgt must leave a non-empty else body before the epilogue:
                       tgt == epilogueOff is a plain `if C: A` (no else) whose false
                       path falls straight to the shared exit. */
                    if (tgt <= ins[i].off || tgt >= epilogueOff)
                        continue;
                    auto it = offToIdx.find(tgt);
                    if (it == offToIdx.end())
                        continue;
                    int j = it->second;
                    // The instruction physically preceding the target must be an
                    // inline `LOAD_CONST None; RETURN_VALUE` (the if-true arm's
                    // threaded exit), so nothing falls through into the target.
                    int r = j - 1;
                    while (r >= 0 && !isReal(r)) --r;
                    int l = r - 1;
                    while (l >= 0 && !isReal(l)) --l;
                    if (r >= 1 && ins[r].op == Pyc::RETURN_VALUE
                            && l >= 0 && ins[l].op == Pyc::LOAD_CONST_A
                            && code->getConst(ins[l].arg).type() == PycObject::TYPE_NONE)
                        moduleElseAt[tgt] = epilogueOff;
                }
            }
        }
    }

    struct FinallyCoalesce { int target; int finBodyStart; int finBodyEnd; int after; int tryEnd; };
    std::unordered_map<int, std::vector<FinallyCoalesce> > finallyOpenAt;
    auto finHasPlan = [&](int s, int t) -> bool {
        auto it = finallyOpenAt.find(s);
        if (it == finallyOpenAt.end()) return false;
        for (const auto& p : it->second) if (p.target == t) return true;
        return false;
    };
    std::unordered_map<int, FinallyCoalesce> finallyPlanByTarget;
    std::unordered_set<int> finallyP1Wrap;
    std::unordered_map<int, int> finallyCopySkip;
    std::unordered_set<int> openFinallyTargets;
    /* Targets of an EMPTY finally (`try: B finally: pass`): the handler is just
       PUSH_EXC_INFO; RERAISE 0 (no body). The try-close emits an empty BLK_FINALLY
       and resumes directly at `after` (past the dead handler) WITHOUT jumping to a
       finally body — there is none, and the handler region must not render. */
    std::unordered_set<int> emptyFinallyTargets;
    /* NESTED empty finally (`try: (try/except) finally: pass`): the wrapped
       try/except's inner BLK_EXCEPT (+ its container) dangle on top of the
       empty-finally BLK_TRY at the shared MERGE offset M. Drain them at M so the
       finally-try (end==M) is exposed and the empty-finally close fires. */
    std::unordered_set<int> emptyFinNestedMerge;                // merge offset M
    /* `try: B finally: raise` — the finally body is a single bare `raise` (a
       RAISE_VARARGS 0). Modelled exactly like an empty finally (no inlined finally
       copy on the normal path, so the generic copy machinery drops the whole
       try/finally), but the close emits `finally:` containing a `raise` instead of
       `pass`. */
    std::unordered_set<int> raiseFinallyTargets;
    std::unordered_map<int, int> returnInFinallyAfter;
    std::unordered_map<int, int> finallyExceptReturnExit;
    /* break-escape finally: the finBodyEnd offset of a coalesced finally whose body
       ends in a terminal `else: break` whose recorded end OVERSHOOTS finBodyEnd (it
       is the loop bottom-test).  At finBodyEnd we drain that terminal branch into the
       BLK_FINALLY so the finally closes + resumes at `after` (the bottom test). */
    std::unordered_set<int> breakEscFinBodyEnd;
    /* break-escape finally: an inlined finally copy at a loop-body `break` (all of
       its exits jump to the loop exit L).  Since the finally is rendered ONCE as
       the coalesced `finally:` block, the copy is skipped and replaced by a single
       `break` (the break runs the finally then exits).  Maps copy-start -> skipTo. */
    std::unordered_map<int, int> breakEscCopySkip;
    std::set<int> loopFinBreakExit;
    std::unordered_map<int, std::pair<int,int>> forElseFinBreak;
    std::unordered_map<int, int> exceptOpenAt;
    /* When two try/except blocks share the same body start offset (an outer try
       whose body begins with another try), exceptOpenAt — keyed by start — can
       only record one of them. The handler with the SMALLER target is the inner
       one and wins the slot; the outer (larger target) is recorded here so it can
       be opened first (outermost) at that offset. */
    std::unordered_map<int, std::vector<int>> exceptOpenAtExtra;
    std::unordered_set<int> openExceptTargets;
    std::unordered_map<int, int> ship215ExceptEnd;
    std::unordered_map<int, int> loopTailElseAt;
    std::unordered_map<int, int> loopBodyElseAt;
    /* SHIP-263: nested loop-bearing try/finally (asyncio Condition.wait). An OUTER
       try/finally whose finally body BEARS A LOOP (the `while True: try: await
       acquire() except CancelledError: …` re-acquire) and which WRAPS an inner
       try/finally, with the inner try's deferred `return` placed AFTER both finally
       normal copies. The global handler classifier reads the inner `except`'s
       CHECK_EXC_MATCH inside the loop-bearing finally and mis-classes the OUTER
       handler as an `except` (relaxing that globally regresses
       multiprocessing/resource_tracker), so this exact shape is detected up front
       and rendered from the NORMAL finally copies via two coalesce plans. This map
       carries a CONST deferred return (`return True`) to synthesize into the inner
       try at its close — the value is not on the stack there, it is loaded after
       both finally copies; a STACK-value deferred return (`return g()`) reuses
       finallyReturnExit instead. */
    std::unordered_map<int, int> nestedFinReturnConst;        // inner try end -> const index
    struct OrReconStep { bool isFinal; bool andOfOrs; bool neg; std::vector<int> groupSizes; };
    std::unordered_map<int, OrReconStep> orReconStep;
    std::vector<PycRef<ASTNode> > orReconAcc;
    /* Final-operand offsets of an OR-of-AND-groups condition whose TRAILING AND
       group ends in a NEGATED conjunct (`… or (C and not D)`): the compiler emits
       that last conjunct as POP_JUMP_FORWARD_IF_TRUE -> exit, so the final operand's
       opcode is IF_TRUE. The orReconStep assembler already builds the fully-correct
       POSITIVE condNode (the negated conjunct is wrapped via st.neg), so the
       fall-through BLK_IF open must NOT re-negate it (its `neg` is normally derived
       from the IF_TRUE opcode). Force neg=false at exactly these offsets. */
    std::unordered_set<int> orReconFinalNoNeg;
    std::unordered_set<int> orReconLoopBody;
    std::unordered_map<int, int> finElseAt;
    std::unordered_map<int, int> exceptElseAt;
    std::set<ASTBlock*> exceptElseNoCollapse;
    std::unordered_set<int> elseStartOffsets;
    /* else-starts admitted by the two-typed-clause terminal-else path (asyncio
       sslproto._do_shutdown): only for THESE may the else-boundary be recorded through
       an open inner if-chain (see the walk-down at the try-close). Single-clause
       terminal-else starts keep the original BLK_TRY-on-top boundary rule so an
       ordinary try-body tail ending in an `if` is not reshaped into a spurious else. */
    std::unordered_set<int> elseStartMultiClause;
    std::unordered_map<int, int> elseBoundaryByTryEnd;
    std::unordered_map<int, int> nestedElseHandler;
    std::unordered_set<int> nestedElseHandlerVals;
    /* TERMINAL-else / terminal try/except (clauses all return/raise): handler offset
       -> post-handler merge M, and whether the handler is a multi-clause chain. The
       clauses have no jump-to-merge, so the LAST clause has no inferable end — bound
       it at M so the code after the try/except[/else] is not absorbed. */
    std::unordered_map<int, int> terminalExceptMerge;
    std::unordered_map<int, bool> terminalExceptMulti;
    std::unordered_map<int, int> teoExceptMerge;
    /* A try/except/else clause whose terminating JUMP_FORWARD targets a `LOAD_x;
       RETURN_VALUE` that sits PAST the enclosing loop's exit and is reached by NO
       other predecessor: the return is logically the else clause's own continuation
       (`else: …; return X`), but the compiler placed it as a shared tail block and
       pycdc would otherwise render it as dead code after the loop-exit statement. Map
       else-start -> the synthesized `return X` node (built in the else-detection
       pre-scan from the LOAD op at the JF target); teoSplitElse appends it into the
       BLK_ELSE, and teoElseAbsorbSuppress marks the JF-target offset so it is not
       re-emitted at function level. */
    std::unordered_map<int, PycRef<ASTNode> > teoElseAbsorbRet;
    std::unordered_set<int> teoElseAbsorbSuppress;
    auto teoSplitElse = [&](PycRef<ASTBlock> tryb) -> PycRef<ASTBlock> {
        if (tryb == nullptr || tryb->blktype() != ASTBlock::BLK_TRY)
            return PycRef<ASTBlock>(nullptr);
        auto it = elseBoundaryByTryEnd.find(tryb->end());
        if (it == elseBoundaryByTryEnd.end())
            return PycRef<ASTBlock>(nullptr);
        int bc = it->second;
        if ((int)tryb->size() <= bc)
            return PycRef<ASTBlock>(nullptr);
        PycRef<ASTBlock> elseb = new ASTBlock(ASTBlock::BLK_ELSE, 0, 1);
        std::list<PycRef<ASTNode> > tail;
        while ((int)tryb->size() > bc) {
            tail.push_front(tryb->nodes().back());
            tryb->removeLast();
        }
        for (auto& n : tail)
            elseb->append(n);
        /* Absorb the else's shared-tail `return X` (see teoElseAbsorbRet). The
           else nodes ended at the JUMP_FORWARD that targets the return; append
           the synthesized return so the else reads `…; return X`. */
        auto ar = teoElseAbsorbRet.find(tryb->end());
        if (ar != teoElseAbsorbRet.end())
            elseb->append(ar->second);
        return elseb;
    };
    std::vector<std::pair<int,int> > nestedFinRegions;
    std::vector<std::pair<int,int> > excHandlerSpan;
    if (mod->verCompare(3, 11) >= 0 && !exception_entries.empty()) {
        const PycString* codeBuf = code->code();
        auto opAt = [&](int off, int& op, int& arg) -> int {
            PycBuffer b(codeBuf->value(), codeBuf->length());
            b.setPos(off);
            int o = Pyc::PYC_INVALID_OPCODE, a = 0, np = off;
            if (!b.atEof()) bc_next(b, mod, o, a, np);
            op = o; arg = a; return np;
        };
        std::unordered_map<int, std::vector<const PycExceptionTableEntry*>> prot;
        std::unordered_map<int, int> lastiCleanup;
        for (const auto& e : exception_entries) {
            if (e.push_lasti)
                lastiCleanup[e.start_offset] = e.target;
            else
                prot[e.target].push_back(&e);
        }
        for (const auto& kv : prot) {
            int T = kv.first;
            int hop = 0, harg = 0;
            opAt(T, hop, harg);
            if (hop != Pyc::PUSH_EXC_INFO)
                continue;
            auto lit = lastiCleanup.find(T);
            if (lit == lastiCleanup.end())
                continue;
            int cleanupStart = lit->second;
            int A = -1;
            PycBuffer xs(codeBuf->value(), codeBuf->length());
            xs.setPos(cleanupStart);
            int xo, xa, xp = cleanupStart;
            while (!xs.atEof()) {
                int xip = xp;
                bc_next(xs, mod, xo, xa, xp);
                if (xo == Pyc::RERAISE || xo == Pyc::RERAISE_A) { A = xp; break; }
                if (xo == Pyc::PUSH_EXC_INFO) break;
                if (xp <= xip) break;
            }
            if (A > T)
                excHandlerSpan.push_back(std::make_pair(T, A));
        }
        std::set<int> forIterExits200;
        std::unordered_map<int,int> forIterExit2Start200;
        std::unordered_map<int,int> whileExit200;
        {
            PycBuffer scan(codeBuf->value(), codeBuf->length());
            int sop, sarg, spos = 0;
            while (!scan.atEof()) {
                int ioff = spos;
                bc_next(scan, mod, sop, sarg, spos);
                if (sop == Pyc::FOR_ITER_A || sop == Pyc::INSTRUMENTED_FOR_ITER_A) {
                    int ex = sarg;
                    if (mod->verCompare(3, 10) >= 0) ex *= sizeof(uint16_t);
                    forIterExits200.insert(spos + ex);
                    forIterExit2Start200[spos + ex] = ioff;
                } else if (sop == Pyc::JUMP_BACKWARD_A) {
                    int tgt = spos - sarg * (int)sizeof(uint16_t);
                    if (tgt < ioff) whileExit200[spos] = tgt;
                }
            }
        }
        for (auto& kv : prot) {
            int T = kv.first;
            std::vector<const PycExceptionTableEntry*>& P = kv.second;
            if (P.empty())
                continue;
            int op0 = 0, arg0 = 0;
            int afterPush = opAt(T, op0, arg0);
            if (op0 != Pyc::PUSH_EXC_INFO)
                continue;
            auto lit = lastiCleanup.find(T);
            if (lit == lastiCleanup.end())
                continue;
            int X = lit->second;
            if (X <= afterPush)
                continue;
            int rerOff = -1;
            bool isExcept = false;
            bool skipNested = true;
            /* Does THIS handler body contain a nested try/EXCEPT (a non-lasti
               protection entry, i.e. a typed/bare except with CHECK_EXC_MATCH)?
               Only then does the nested-finally-normal-copy skip below apply — that
               is the poplib shape (`try: A finally: … try: B except: … finally: C`),
               whose exceptional finally copy contains an except that would make the
               classifier read the whole handler as an except and drop the outer
               try/finally. A PURE nested try/finally (MemoryHandler.close) has only
               lasti entries and must be left to the existing classification. */
            bool bodyHasNestedExcept = false;
            /* For a nested try/EXCEPT inline in this finally body, map its try-body
               start to the offset just past its except handler's terminal RERAISE
               cleanup — the point at which THIS handler's own flow resumes. Used only
               by the nested-except skip below (multiprocessing/connection.wait). */
            std::unordered_map<int,int> nestedExcSpanEnd;
            for (const auto& e : exception_entries) {
                if (!e.push_lasti
                        && e.start_offset >= afterPush && e.start_offset < X
                        && e.target > afterPush && e.target < X) {
                    /* Distinguish a nested try/EXCEPT from a nested try/FINALLY: a
                       nested try/finally ALSO emits a non-lasti body-protection entry
                       pointing at its exceptional finally copy (PUSH_EXC_INFO; body;
                       RERAISE, no CHECK_EXC_MATCH). Only a genuine except handler
                       (CHECK_EXC_MATCH, or a bare-except POP_TOP that is not a finally)
                       makes the classifier read this whole handler as an except.
                       MemoryHandler.close has only nested finallies -> must stay
                       false; poplib POP3.close has an OSError except -> true. */
                    int ho, ha, hp = e.target;
                    int hend = X; { auto hl = lastiCleanup.find(e.target); if (hl != lastiCleanup.end()) hend = hl->second; }
                    bool isExc = false;
                    while (hp < hend) {
                        int hn = opAt(hp, ho, ha);
                        if (ho == Pyc::CHECK_EXC_MATCH) { isExc = true; break; }
                        if (ho == Pyc::RERAISE || ho == Pyc::RERAISE_A) break;
                        if (hn <= hp) break;
                        hp = hn;
                    }
                    if (isExc) {
                        bodyHasNestedExcept = true;
                        /* Record the nested try/except's resume point: walk its except
                           handler's lasti cleanup (COPY; POP_EXCEPT; RERAISE) to just
                           past the terminal RERAISE within [., X). */
                        int spanEnd = -1;
                        auto hl2 = lastiCleanup.find(e.target);
                        if (hl2 != lastiCleanup.end()) {
                            int co, ca, cp = hl2->second;
                            while (cp < X) {
                                int cn = opAt(cp, co, ca);
                                if (co == Pyc::RERAISE || co == Pyc::RERAISE_A) { spanEnd = cn; break; }
                                if (cn <= cp) break;
                                cp = cn;
                            }
                        }
                        if (spanEnd > e.start_offset)
                            nestedExcSpanEnd[e.start_offset] = spanEnd;
                    }
                }
            }
            /* Is THIS handler the OUTER finally of a nested loop-bearing try/finally
               owned by the dedicated SHIP-263 coalesce pre-scan (asyncio
               Condition.wait)? That shape is: this handler body is loop-bearing AND
               somewhere there is an INNER try/finally (a non-lasti entry whose target
               is a non-loop PUSH_EXC_INFO handler reaching its own RERAISE, no
               CHECK_EXC_MATCH) whose lasti cleanup chains back to T. When present the
               nested-except skip below must NOT fire: SHIP-263 owns this handler and
               already emits its own coalesce plan; also skipping would let the standard
               classifier read it as a finally and register a SECOND plan -> a cyclic
               block structure that hangs the renderer (asyncio/locks.pyc). Mirrors
               SHIP-263 steps (a)+(c). */
            bool bodyWrapsInnerFinally = false;
            {
                bool oLoop = false;
                { int o, a, p = afterPush;
                  while (p < X) {
                      int np = opAt(p, o, a);
                      if (o == Pyc::FOR_ITER_A || o == Pyc::JUMP_BACKWARD_A) { oLoop = true; break; }
                      if (np <= p) break;
                      p = np;
                  } }
                if (oLoop) {
                    for (const auto& e : exception_entries) {
                        if (e.push_lasti) continue;
                        int cand = e.target;
                        if (cand == T) continue;
                        int io, ia; opAt(cand, io, ia);
                        if (io != Pyc::PUSH_EXC_INFO) continue;
                        auto li = lastiCleanup.find(cand);
                        if (li == lastiCleanup.end()) continue;
                        int Xi = li->second;
                        if (Xi <= cand) continue;
                        bool iIsExcept = false, iLoop = false; int iRer = -1;
                        { int o, a, p = cand;
                          while (p < Xi) {
                              int np = opAt(p, o, a);
                              if (o == Pyc::FOR_ITER_A || o == Pyc::JUMP_BACKWARD_A) { iLoop = true; break; }
                              if (o == Pyc::CHECK_EXC_MATCH) { iIsExcept = true; break; }
                              if (o == Pyc::RERAISE || o == Pyc::RERAISE_A) { iRer = p; break; }
                              if (np <= p) break;
                              p = np;
                          } }
                        if (iIsExcept || iLoop || iRer < 0) continue;
                        for (const auto& e2 : exception_entries)
                            if (!e2.push_lasti && e2.start_offset == Xi && e2.target == T) {
                                bodyWrapsInnerFinally = true; break;
                            }
                        if (bodyWrapsInnerFinally) break;
                    }
                }
            }
            { int bo, ba; opAt(afterPush, bo, ba);
              if (bo == Pyc::POP_TOP) skipNested = false; }
            /* Does THIS handler body contain a nested try/except OR try/finally? Such a
               shape has a non-lasti body-protection entry strictly inside [afterPush, X).
               The with-exit classifyScan skip below is only safe when the handler body
               is FLAT (no nested try covering a sub-range) — a with directly in the
               finally body, as in concurrent.futures._base.as_completed. When the body
               instead wraps a nested try/finally (aiohttp.web.run_app: `finally: try: …
               with …: … finally: …`), the with belongs to that nested try and skipping
               it over-extends rerOff past the nested finally, mis-coalescing it. Leave
               such nested shapes to the existing classification. */
            bool bodyHasNestedProt = false;
            for (const auto& e : exception_entries) {
                if (!e.push_lasti
                        && e.start_offset >= afterPush && e.start_offset < X
                        && e.target > afterPush && e.target < X) {
                    bodyHasNestedProt = true; break;
                }
            }
            if (skipNested) {
                int o, a, p = afterPush; const int W = (int)sizeof(uint16_t);
                while (p < X && p < (int)codeBuf->length()) {
                    int np = opAt(p, o, a);
                    if (o == Pyc::FOR_ITER_A || o == Pyc::JUMP_BACKWARD_A) { skipNested = false; break; }
                    if (o == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A) {
                        int bt = np - a * W, bo2 = -1, ba2; opAt(bt, bo2, ba2);
                        if (bo2 != Pyc::SEND_A) { skipNested = false; break; }
                    }
                    if (np <= p) break;
                    p = np;
                }
            }
            auto classifyScan = [&](bool skip) {
                isExcept = false; rerOff = -1;
                int o, a, p = afterPush;
                while (p < X && p < (int)codeBuf->length()) {
                    int np = opAt(p, o, a);
                    /* NESTED try/EXCEPT inline in this finally handler body: skip the
                       whole nested try/except (from its try-body start to just past its
                       handler's terminal RERAISE) so the scan reaches THIS handler's own
                       terminal RERAISE instead of latching onto the nested except's
                       CHECK_EXC_MATCH -- which would mis-class the outer try/finally as a
                       swallowing bare `except:` (multiprocessing/connection.wait). Runs
                       REGARDLESS of `skip`/`skipNested`: a finally body may contain real
                       for-loops which force skipNested false. Tightly gated: only a
                       genuine nested try/EXCEPT span recorded above, and NOT when this
                       finally wraps an inner try/finally owned by the SHIP-263 pre-scan
                       (else a conflicting double coalesce plan hangs the renderer). */
                    if (bodyHasNestedExcept && !bodyWrapsInnerFinally) {
                        auto ne = nestedExcSpanEnd.find(p);
                        if (ne != nestedExcSpanEnd.end() && ne->second > p && ne->second < X) {
                            p = ne->second; continue;
                        }
                    }
                    /* NESTED WITH-EXIT exceptional copy inside this handler body: a
                       `with` inside the finally body compiles its cleanup as a lasti
                       handler `PUSH_EXC_INFO; WITH_EXCEPT_START; POP_JUMP …; RERAISE;
                       COPY; POP_EXCEPT; RERAISE`. Its inner RERAISE is NOT this
                       handler's terminal reraise; if the classifier latches onto it we
                       mis-size the handler and drop the outer `finally:` to a swallowing
                       bare `except:` (concurrent.futures._base.as_completed). Always skip
                       past a with-exit's cleanup so the scan reaches THIS handler's own
                       terminal RERAISE. Tightly gated on the WITH_EXCEPT_START signature
                       (a lasti-cleanup PUSH_EXC_INFO whose next op is WITH_EXCEPT_START),
                       so it never affects nested try/except or try/finally shapes. This
                       is independent of `skip` because a `with` inside the finally body
                       does not require the nested-copy elision that `skip` gates. */
                    if (o == Pyc::PUSH_EXC_INFO && !bodyHasNestedProt) {
                        int wo, wa; opAt(np, wo, wa);
                        auto wl = lastiCleanup.find(p);
                        if (wo == Pyc::WITH_EXCEPT_START && wl != lastiCleanup.end()) {
                            int co, ca, cp = wl->second, skipTo = -1;
                            while (cp < X) {
                                int cn = opAt(cp, co, ca);
                                if (co == Pyc::RERAISE || co == Pyc::RERAISE_A) { skipTo = cn; break; }
                                if (co == Pyc::PUSH_EXC_INFO || co == Pyc::CHECK_EXC_MATCH) break;
                                if (cn <= cp) break;
                                cp = cn;
                            }
                            if (skipTo > p) { p = skipTo; continue; }
                        }
                    }
                    if (o == Pyc::PUSH_EXC_INFO && skip) {
                        auto nl = lastiCleanup.find(p);
                        if (nl != lastiCleanup.end()) {
                            int co, ca, cp = nl->second, skipTo = -1;
                            while (cp < X) {
                                int cn = opAt(cp, co, ca);
                                if (co == Pyc::RERAISE || co == Pyc::RERAISE_A) { skipTo = cn; break; }
                                if (cn <= cp) break;
                                cp = cn;
                            }
                            if (skipTo > p) { p = skipTo; continue; }
                        }
                    }
                    /* NESTED FINALLY normal-copy inside this handler body: when THIS
                       handler is itself a finally whose body contains a nested
                       try/finally, the nested finally is inlined here as a NORMAL
                       copy ending in a RERAISE (offset p, a lasti-protected range not
                       starting with PUSH_EXC_INFO) IMMEDIATELY followed by its
                       EXCEPTIONAL copy (PUSH_EXC_INFO). The classifier would latch
                       onto the nested finally's RERAISE, mis-sizing THIS handler and
                       dropping the outer try/finally (poplib POP3.close). Skip the
                       normal copy so the following PUSH_EXC_INFO branch skips the
                       exceptional copy, reaching THIS handler's own terminal RERAISE.
                       Tightly gated: the normal copy's RERAISE must be directly
                       followed by a PUSH_EXC_INFO that is itself a lasti-cleanup
                       handler ending in RERAISE (the nested finally's own exceptional
                       copy) — this excludes ordinary finally-body lasti atoms
                       (returns / cleanup) that also carry lasti entries. */
                    if (skip && bodyHasNestedExcept && o != Pyc::PUSH_EXC_INFO && lastiCleanup.count(p)) {
                        int co, ca, cp = p, rer = -1;
                        while (cp < X) {
                            int cn = opAt(cp, co, ca);
                            if (co == Pyc::RERAISE || co == Pyc::RERAISE_A) { rer = cp; break; }
                            if (co == Pyc::CHECK_EXC_MATCH || co == Pyc::PUSH_EXC_INFO) break;
                            if (cn <= cp) break;
                            cp = cn;
                        }
                        if (rer > p) {
                            int no, na; int afterRer = opAt(rer, no, na);
                            int fo, fa; int afterPush2 = opAt(afterRer, fo, fa);
                            /* The following handler must be the nested finally's
                               EXCEPTIONAL copy: a PUSH_EXC_INFO body that is a plain
                               statement copy terminated by RERAISE, with NO
                               WITH_EXCEPT_START (a `with`) and NO CHECK_EXC_MATCH (a
                               typed except) — those are different constructs whose
                               classification the old scan already handles correctly. */
                            bool pureFinallyExc = false;
                            if (fo == Pyc::PUSH_EXC_INFO) {
                                auto el = lastiCleanup.find(afterRer);
                                if (el != lastiCleanup.end()) {
                                    int eo, ea, ep = afterPush2; bool ok = false;
                                    while (ep < el->second) {
                                        int en = opAt(ep, eo, ea);
                                        if (eo == Pyc::RERAISE || eo == Pyc::RERAISE_A) { ok = true; break; }
                                        if (eo == Pyc::WITH_EXCEPT_START
                                                || eo == Pyc::CHECK_EXC_MATCH
                                                || eo == Pyc::PUSH_EXC_INFO) { ok = false; break; }
                                        if (en <= ep) break;
                                        ep = en;
                                    }
                                    pureFinallyExc = ok;
                                }
                            }
                            if (pureFinallyExc) {
                                p = afterRer; continue;
                            }
                        }
                    }
                    if (o == Pyc::CHECK_EXC_MATCH) { isExcept = true; break; }
                    if (o == Pyc::RERAISE || o == Pyc::RERAISE_A) { rerOff = p; break; }
                    if (np <= p) break;
                    p = np;
                }
            };
            classifyScan(skipNested);
            if (skipNested && !isExcept && rerOff > 0) {
                int xo, xa; int xn = opAt(rerOff, xo, xa);
                if (xn != X) { skipNested = false; classifyScan(false); }
            }
            /* ===== EMPTY finally (`try: B finally: pass`) =====
               The handler is just `PUSH_EXC_INFO; RERAISE 0` — the finally body is
               EMPTY, so rerOff == afterPush. With no body there is no inlined finally
               copy on the normal path, so the copy-based machinery bails and the
               whole try/finally is dropped. Register a plan that opens the BLK_TRY
               and (via emptyFinallyTargets at the close) emits an empty `finally:`
               then resumes past the handler. tryEnd = the body's normal exit op
               (JUMP_FORWARD over the handler, or the loop back-edge JUMP_BACKWARD
               that precedes it) so the BLK_TRY contains the body's result-discard
               POP_TOP but NOT the exit jump. */
            /* The empty finally may wrap a plain body (P.size()==1) OR a nested
               try/except (multiple protected sub-ranges all pointing at THIS empty
               finally handler — `try: (try: B except E: H) finally: pass`, e.g. a
               `finally: pass` wrapping a per-item try/except cleanup). For the nested
               case S is the MIN protected start and E the MAX protected end; the inner
               try/except is opened by the ordinary except machinery INSIDE the BLK_TRY
               (its own handler target differs from T, so the finallyOpenAt consume loop
               leaves it). Require every P entry to be self-contained within [minS, maxE)
               so we don't straddle an unrelated construct. */
            bool nestedInP = P.size() > 1;
            if (nestedInP) {
                int mn = T, mx = 0;
                for (const PycExceptionTableEntry* e : P) {
                    if (e->start_offset < mn) mn = e->start_offset;
                    if (e->end_offset > mx) mx = e->end_offset;
                }
                for (const PycExceptionTableEntry* e : P)
                    if (e->target < mn || e->target > T) { nestedInP = false; break; }
                if (mx >= T) nestedInP = false;
            }
            if (!isExcept && rerOff == afterPush && rerOff > 0
                    && (P.size() == 1 || nestedInP)
                    && !finHasPlan(P.front()->start_offset, T)) {
                const int W = (int)sizeof(uint16_t);
                int S = P.front()->start_offset, E = P.front()->end_offset;
                if (nestedInP) {
                    S = T; E = 0;
                    for (const PycExceptionTableEntry* e : P) {
                        if (e->start_offset < S) S = e->start_offset;
                        if (e->end_offset > E) E = e->end_offset;
                    }
                    /* The finally's protected sub-ranges only cover the parts that can
                       raise while the finally is armed (the wrapped try/except's JF and
                       handler cleanup); the wrapped try's BODY is protected by the INNER
                       handler (a different, closer target) so it is NOT in P. Extend S
                       backward through any nested handler whose body abuts S or whose
                       target lands inside the finally body [S,T) — this reaches the true
                       body start (the wrapped `try:`). */
                    bool ext = true;
                    while (ext) {
                        ext = false;
                        for (const auto& e : exception_entries) {
                            if (e.start_offset < S
                                    && (e.end_offset == S
                                        || (e.target >= S && e.target < T))) {
                                S = e.start_offset; ext = true;
                            }
                        }
                    }
                }
                int tryEnd = -1, after = -1; bool isFwd = false;
                /* ===== NESTED empty finally (`try: (try/except) finally: pass`) =====
                   The wrapped try/except's normal-exit jumps converge at the MERGE
                   point M, which is exactly E (the max protected end — the except's
                   lasti-cleanup range ends there). The BLK_TRY closes at M; the inner
                   try/except is rendered inside it by the ordinary machinery. */
                if (nestedInP) {
                    int M = E;
                    int eo = -1, ea; opAt(M, eo, ea);
                    if (eo == Pyc::JUMP_BACKWARD_A) {
                        /* IN A LOOP: the merge IS the loop back-edge. Close the BLK_TRY
                           at the back-edge (M) so the inner try/except renders inside,
                           then emit `finally: pass` and RESUME PAST the dead handler's
                           lasti cleanup X (the FOR_ITER exit) — skipping BOTH the dead
                           handler AND the back-edge, exactly like the plain-body loop
                           empty finally. The enclosing for-loop is modelled from its
                           FOR_ITER, so the back-edge must NOT render as a spurious
                           explicit `continue`. */
                        int aftL = X;
                        { int o, a, p = X;
                          while (p < (int)codeBuf->length()) {
                              int np = opAt(p, o, a);
                              if (o == Pyc::RERAISE || o == Pyc::RERAISE_A) { aftL = np; break; }
                              if (np <= p) break;
                              p = np;
                          } }
                        if (aftL > M) {
                            FinallyCoalesce plan = { T, afterPush, afterPush, aftL, M };
                            finallyOpenAt[S].push_back(plan);
                            finallyPlanByTarget[T] = plan;
                            emptyFinallyTargets.insert(T);
                            emptyFinNestedMerge.insert(M);
                            continue;
                        }
                    }
                    /* NOT in a loop: close the BLK_TRY at the merge M and resume PAST
                       the dead handler's lasti cleanup X (skip the empty-finally
                       PUSH_EXC_INFO/RERAISE/COPY/POP_EXCEPT/RERAISE). */
                    int aft = X;
                    { int o, a, p = X;
                      while (p < (int)codeBuf->length()) {
                          int np = opAt(p, o, a);
                          if (o == Pyc::RERAISE || o == Pyc::RERAISE_A) { aft = np; break; }
                          if (np <= p) break;
                          p = np;
                      } }
                    if (M > S && aft > M) {
                        FinallyCoalesce plan = { T, afterPush, afterPush, aft, M };
                        finallyOpenAt[S].push_back(plan);
                        finallyPlanByTarget[T] = plan;
                        emptyFinallyTargets.insert(T);
                        emptyFinNestedMerge.insert(M);
                        continue;
                    }
                }
                { int o, a, p = E;
                  while (p < T) {
                      int np = opAt(p, o, a);
                      if (o == Pyc::JUMP_FORWARD_A) { tryEnd = p; after = np + a * W; isFwd = true; break; }
                      if (o == Pyc::JUMP_BACKWARD_A) { tryEnd = p; break; }
                      if (np <= p) break;
                      p = np;
                  } }
                if (!isFwd && tryEnd >= 0) {
                    /* loop back-edge: resume PAST the handler's lasti cleanup X
                       (skipping the dead handler AND the back-edge — the enclosing
                       loop is modelled from its FOR_ITER/while exit). */
                    int o, a, p = X;
                    while (p < (int)codeBuf->length()) {
                        int np = opAt(p, o, a);
                        if (o == Pyc::RERAISE || o == Pyc::RERAISE_A) { after = np; break; }
                        if (np <= p) break;
                        p = np;
                    }
                }
                if (tryEnd >= E && after > tryEnd) {
                    FinallyCoalesce plan = { T, afterPush, afterPush, after, tryEnd };
                    finallyOpenAt[S].push_back(plan);
                    finallyPlanByTarget[T] = plan;
                    emptyFinallyTargets.insert(T);
                    continue;
                }
            }
            /* ===== EMPTY finally wrapping a TERMINATING try/EXCEPT =====
               (`try: (try: B except E: … raise/return) finally: pass`, e.g.
               service/mgt/godma godma Godma.activate's inner finally). The inner
               empty finally (rerOff==afterPush) wraps a try/except whose except
               clauses TERMINATE (return/raise) — there is no jump-to-merge, so the
               except handler renders to its own no-match RERAISE and CLOSES there.
               The plain empty-finally path above bails: P has >1 entry (the except's
               normal-copy region + its cleanup stub) and the last stub ends AT T so
               nestedInP's `mx >= T` guard rejects it. Detect the exact shape here:
               among P there is a cleanup-stub entry (start == the except handler's
               lasti-cleanup chain) whose start is the LARGEST P start, and the
               except handler Hexc's body-protection entry gives the wrapped try's
               body start. Open a BLK_TRY over the whole try/except ending at the
               except's terminal RERAISE (just before that stub), emit `finally:
               pass`, and resume at the stub start (where the enclosing construct's
               dead-cleanup absorption already resumes). Tightly gated so nothing
               else can match. */
            if (!isExcept && rerOff == afterPush && rerOff > 0 && P.size() >= 2
                    && !finHasPlan(P.front()->start_offset, T)
                    && !emptyFinallyTargets.count(T)) {
                /* every P entry must target T */
                bool allT = true;
                for (const PycExceptionTableEntry* e : P)
                    if (e->target != T) { allT = false; break; }
                /* the LARGEST P start = the except-cleanup stub that ends AT T. */
                int stubStart = -1;
                for (const PycExceptionTableEntry* e : P)
                    if (e->start_offset > stubStart) stubStart = e->start_offset;
                bool stubEndsAtT = false;
                for (const PycExceptionTableEntry* e : P)
                    if (e->start_offset == stubStart && e->end_offset == T)
                        stubEndsAtT = true;
                /* stubStart must be a lasti-cleanup target of some nested except
                   handler Hexc (the except's `COPY 3; POP_EXCEPT; RERAISE 1` stub
                   that the inner finally protects). Find Hexc: an except handler
                   (PUSH_EXC_INFO + CHECK_EXC_MATCH) whose lasti-cleanup chain reaches
                   stubStart, and whose body-protection entry gives the wrapped try's
                   body start Sbody. */
                int Hexc = -1, Sbody = -1;
                if (allT && stubEndsAtT) {
                    for (const auto& e : exception_entries) {
                        if (e.push_lasti) continue;
                        int cand = e.target;                 // an except handler?
                        int io, ia; opAt(cand, io, ia);
                        if (io != Pyc::PUSH_EXC_INFO) continue;
                        /* handler must contain a CHECK_EXC_MATCH (typed except) */
                        auto lc = lastiCleanup.find(cand);
                        if (lc == lastiCleanup.end()) continue;
                        bool isExc = false;
                        { int o, a, p = cand; int Xh = lc->second;
                          while (p < Xh) { int np = opAt(p, o, a);
                              if (o == Pyc::CHECK_EXC_MATCH) { isExc = true; break; }
                              if (o == Pyc::RERAISE || o == Pyc::RERAISE_A) break;
                              if (np <= p) break;
                              p = np; } }
                        if (!isExc) continue;
                        /* the stub range [stubStart, T) must be Hexc's own outer lasti
                           cleanup: some lasti entry at/after Hexc targets stubStart. */
                        bool reaches = false;
                        for (const auto& e2 : exception_entries)
                            if (e2.push_lasti && e2.target == stubStart
                                    && e2.start_offset >= cand) { reaches = true; break; }
                        if (!reaches) continue;
                        /* body start = the except's body-protection entry start */
                        for (const auto& e2 : exception_entries)
                            if (!e2.push_lasti && e2.target == cand) {
                                if (Sbody < 0 || e2.start_offset < Sbody)
                                    Sbody = e2.start_offset;
                            }
                        if (Sbody >= 0) { Hexc = cand; break; }
                    }
                }
                if (Hexc > 0 && Sbody > 0 && Sbody < stubStart) {
                    /* extend Sbody backward through any nested handler abutting it or
                       whose target lands inside [Sbody, T) (reaches the true try:). */
                    int S = Sbody;
                    bool ext = true;
                    while (ext) { ext = false;
                        for (const auto& e : exception_entries)
                            if (e.start_offset < S
                                    && (e.end_offset == S
                                        || (e.target >= S && e.target < T))) {
                                S = e.start_offset; ext = true;
                            }
                    }
                    /* tryEnd = the cleanup stub start (= the except handler's own
                       lasti cleanup): the wrapped try/except renders + closes just
                       before this, so the empty-finally BLK_TRY closes at stubStart
                       (== pos there) and emits `finally: pass`. */
                    int tryEnd = stubStart;
                    /* resume PAST the whole dead empty-finally region (its exceptional
                       PUSH_EXC_INFO;RERAISE copy + cleanup stub) — chain lasti cleanups
                       from stubStart to the FIRST offset that is not part of this dead
                       region (the enclosing construct picks up there). */
                    int after = X;
                    { int o, a, p = X;
                      while (p < (int)codeBuf->length()) {
                          int np = opAt(p, o, a);
                          if (o == Pyc::RERAISE || o == Pyc::RERAISE_A) { after = np; break; }
                          if (np <= p) break;
                          p = np;
                      } }
                    if (tryEnd > S && after > tryEnd && S < Hexc) {
                        FinallyCoalesce plan = { T, afterPush, afterPush, after, tryEnd };
                        finallyOpenAt[S].push_back(plan);
                        finallyPlanByTarget[T] = plan;
                        emptyFinallyTargets.insert(T);
                        continue;
                    }
                }
            }
            /* ===== `finally: raise` ===== handler is PUSH_EXC_INFO; RAISE_VARARGS 0;
               COPY; POP_EXCEPT; RERAISE — the whole finally body is one bare re-raise.
               Like the empty finally, there is no RERAISE-terminated copy for the
               classifier to latch onto (the body is RAISE_VARARGS, not RERAISE) and
               the normal-path finally copy is a lone RAISE_VARARGS sitting exactly at
               the protected body's end; the generic machinery sees neither an except
               nor a finally and drops the construct (and the dangling normal-copy raise
               renders as a duplicate bare `raise`). Open the BLK_TRY over the body and
               emit `finally: raise` at the close. */
            if (!isExcept && rerOff < 0 && P.size() == 1
                    && !finHasPlan(P[0]->start_offset, T)) {
                int S = P[0]->start_offset, E = P[0]->end_offset;
                int bo = -1, ba = 0; int bn = opAt(afterPush, bo, ba);
                bool bodyIsBareRaise = (bo == Pyc::RAISE_VARARGS_A && ba == 0
                        && bn == X);
                int eo = -1, ea; opAt(E, eo, ea);
                bool normIsRaise = (eo == Pyc::RAISE_VARARGS_A);
                if (bodyIsBareRaise && normIsRaise && E > S && E < T) {
                    int after = -1;
                    { int o, a, p = X;
                      while (p < (int)codeBuf->length()) {
                          int np = opAt(p, o, a);
                          if (o == Pyc::RERAISE || o == Pyc::RERAISE_A) { after = np; break; }
                          if (np <= p) break;
                          p = np;
                      } }
                    if (after > X) {
                        FinallyCoalesce plan = { T, afterPush, afterPush, after, E };
                        finallyOpenAt[S].push_back(plan);
                        finallyPlanByTarget[T] = plan;
                        raiseFinallyTargets.insert(T);
                        continue;
                    }
                }
            }
            if (!isExcept && rerOff < 0 && !finHasPlan(P.front()->start_offset, T)) {
                std::vector<std::pair<int,int> > hops;
                { int o, a, p = afterPush;
                  while (p < X) {
                      int np = opAt(p, o, a);
                      hops.push_back(std::make_pair(o, p));
                      if (np <= p) break;
                      p = np;
                  } }
                int n = (int)hops.size();
                bool formA = n >= 5
                        && hops[n-1].first == Pyc::RETURN_VALUE
                        && hops[n-2].first == Pyc::POP_EXCEPT
                        && hops[n-3].first == Pyc::SWAP_A
                        && hops[n-4].first == Pyc::POP_TOP
                        && hops[n-5].first == Pyc::SWAP_A;
                bool pureExpr = formA;
                if (formA) {
                    int swapOff0 = hops[n-5].second;
                    for (int i = 0; i < n - 5; ++i) {
                        int op = hops[i].first;
                        if (hops[i].second >= swapOff0) break;
                        if (op == Pyc::STORE_FAST_A || op == Pyc::STORE_GLOBAL_A
                                || op == Pyc::STORE_NAME_A || op == Pyc::STORE_DEREF_A
                                || op == Pyc::STORE_ATTR_A || op == Pyc::STORE_SUBSCR
                                || op == Pyc::DELETE_FAST_A || op == Pyc::DELETE_GLOBAL_A
                                || op == Pyc::DELETE_NAME_A || op == Pyc::DELETE_DEREF_A
                                || op == Pyc::DELETE_ATTR_A || op == Pyc::DELETE_SUBSCR
                                || op == Pyc::POP_TOP || op == Pyc::FOR_ITER_A
                                || op == Pyc::GET_ITER || op == Pyc::PUSH_EXC_INFO
                                || op == Pyc::JUMP_FORWARD_A || op == Pyc::JUMP_BACKWARD_A
                                || op == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A
                                || op == Pyc::SEND_A || op == Pyc::YIELD_VALUE
                                || op == Pyc::GET_AWAITABLE
                                || op == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                                || op == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                                || op == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                                || op == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A
                                || op == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A
                                || op == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A) {
                            pureExpr = false; break;
                        }
                    }
                }
                if (pureExpr) {
                    int swapOff = hops[n-5].second;
                    int computeLen = swapOff - afterPush;
                    int retOff = hops[n-1].second;
                    int retLen = X - retOff;
                    int E = T - computeLen - retLen;
                    int no = -1, na;
                    int nn = (E > 0 && computeLen > 0) ? opAt(E + computeLen, no, na) : -1;
                    if (E > 0 && computeLen > 0 && E < T
                            && memcmp(codeBuf->value() + afterPush,
                                      codeBuf->value() + E, computeLen) == 0
                            && no == Pyc::RETURN_VALUE && nn == T) {
                        int S = T;
                        for (const PycExceptionTableEntry* e : P)
                            S = std::min(S, e->start_offset);
                        bool ext = true;
                        while (ext) {
                            ext = false;
                            for (const auto& e : exception_entries) {
                                if (e.start_offset < S
                                        && (e.end_offset == S
                                            || (e.target >= S && e.target < T))) {
                                    S = e.start_offset; ext = true;
                                }
                            }
                        }
                        int after = X;
                        { int oo, aa; for (int i = 0; i < 3 && after < (int)codeBuf->length(); ++i) {
                              int np = opAt(after, oo, aa); if (np <= after) break; after = np; } }
                        if (S < E && after > X) {
                            FinallyCoalesce plan = { T, E, T, after, 0 };
                            finallyOpenAt[S].push_back(plan);
                            returnInFinallyAfter[T] = after;
                            continue;
                        }
                    }
                }
                /* ===== PLAIN try/finally with a returning (terminal) finally whose
                   body is NOT a bare `return <expr>` (formA above), but a COMPOUND
                   statement sequence — stores, a for-loop, conditionals — ending in a
                   `return`/`raise`.  `try: A finally: <stmts…; return X>`.  The compiler
                   inlines the finally body TWICE: a NORMAL copy [E, T) after the try body
                   ending in RETURN_VALUE (its next op == T, the handler start), and an
                   EXCEPTIONAL copy in the handler that discards the caught exception
                   (SWAP; POP_TOP; SWAP; POP_EXCEPT) then RETURN_VALUEs — so the finally
                   SWALLOWS.  Because the two copies are NOT byte-identical (the
                   exceptional one carries the exception-discard dance) the generic
                   copy-matcher bails and the whole construct renders as `try: A; B
                   except: B` — a bare swallowing `except:` with the finally body
                   duplicated.  formA above handles this only when the finally body is a
                   lone return expression (pureExpr); this handles the compound-body
                   variant.  Render the finally from the CLEAN NORMAL copy [E, T) via the
                   p1wrap-style plan (finallyPlanByTarget re-renders it as the BLK_FINALLY;
                   finallyCopySkip elides it from the try body) so it emits `try: A
                   finally: B`, resuming past the dead exceptional copy.  Tightly gated:
                   single self-contained try-body entry (P.size()==1, no nested typed
                   except), exceptional body ends in the exact discard-then-RETURN dance
                   (formA n>=5 tail), and the normal copy is a clean straight-line region
                   (loops allowed, but no PUSH_EXC_INFO / CHECK_EXC_MATCH / SETUP_FINALLY /
                   nested exception coverage) ending in RETURN_VALUE at T. */
                if (formA && P.size() == 1
                        && !finHasPlan(P.front()->start_offset, T)) {
                    int S = P.front()->start_offset;
                    int E = P.front()->end_offset;
                    { int o, a; int np = opAt(E, o, a); if (o == Pyc::NOP && np > E) E = np; }
                    /* the normal copy [E, T) must end in a RETURN_VALUE whose next op == T
                       (the finally body's normal-path terminal return). */
                    bool normReturns = false;
                    { int o, a, p = E;
                      while (p < T) {
                          int np = opAt(p, o, a);
                          if (np == T && (o == Pyc::RETURN_VALUE
                                  || o == Pyc::INSTRUMENTED_RETURN_VALUE_A)) {
                              normReturns = true;
                          }
                          if (np <= p) break;
                          p = np;
                      } }
                    /* the normal copy must be a clean finally body: no exception-table
                       coverage of any sub-range other than the try-body entry itself, and
                       no exception-management / SETUP_FINALLY opcodes inside it (a genuine
                       nested try would need its own handling). A for-loop (FOR_ITER /
                       JUMP_BACKWARD) is fine — the finally body may contain one. */
                    bool cleanNorm = normReturns && E > S && E < T;
                    if (cleanNorm) {
                        int o, a, p = E;
                        while (p < T) {
                            int np = opAt(p, o, a);
                            if (o == Pyc::PUSH_EXC_INFO || o == Pyc::CHECK_EXC_MATCH
                                    || o == Pyc::SETUP_FINALLY_A) { cleanNorm = false; break; }
                            if (np <= p) break;
                            p = np;
                        }
                    }
                    if (cleanNorm) {
                        for (const auto& e : exception_entries) {
                            /* any entry whose protected range lies inside the normal copy
                               (a nested try) disqualifies it. */
                            if (e.start_offset >= E && e.start_offset < T
                                    && !(e.start_offset == E && e.end_offset == E)) {
                                cleanNorm = false; break;
                            }
                        }
                    }
                    if (cleanNorm) {
                        /* extend S back through any nested handler abutting it (an inner
                           try inside the TRY body). */
                        bool ext = true;
                        while (ext) {
                            ext = false;
                            for (const auto& e : exception_entries) {
                                if (e.start_offset < S
                                        && (e.end_offset == S
                                            || (e.target >= S && e.target < T))) {
                                    S = e.start_offset; ext = true;
                                }
                            }
                        }
                        /* resume past the dead exceptional copy: chain from X to just
                           past its terminal RERAISE. */
                        int after = X;
                        { int o, a, p = X;
                          while (p < (int)codeBuf->length()) {
                              int np = opAt(p, o, a);
                              if (o == Pyc::RERAISE || o == Pyc::RERAISE_A) { after = np; break; }
                              if (np <= p) break;
                              p = np;
                          } }
                        if (S < E && after > X) {
                            FinallyCoalesce plan = { T, E, T, after, 0 };
                            finallyOpenAt[S].push_back(plan);
                            finallyPlanByTarget[T] = plan;
                            finallyCopySkip[E] = T - E;
                            continue;
                        }
                    }
                }
            }
            if (!isExcept && rerOff > afterPush
                    && !finHasPlan(P.front()->start_offset, T)) {
                const int W = (int)sizeof(uint16_t);
                int ro, ra; opAt(rerOff, ro, ra);
                bool rerIsReraise = (ro == Pyc::RERAISE || ro == Pyc::RERAISE_A);
                int popJ = -1, popJOp = -1;
                if (rerIsReraise) {
                    int o, a, p = afterPush;
                    while (p < rerOff) {
                        int np = opAt(p, o, a);
                        if ((o == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                                || o == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                                || o == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                                || o == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A)
                                && np + a * W == rerOff) { popJ = p; popJOp = o; break; }
                        if (np <= p) break;
                        p = np;
                    }
                }
                if (popJ > afterPush) {
                    int pjo, pja; int popJEnd = opAt(popJ, pjo, pja);
                    int o1, a1; int n1 = opAt(popJEnd, o1, a1);
                    int o2 = -1, a2; int n2 = (o1 == Pyc::POP_TOP) ? opAt(n1, o2, a2) : -1;
                    if (o1 == Pyc::POP_TOP && o2 == Pyc::POP_EXCEPT) {
                        int retComputeStart = n2;
                        int condLen = popJ - afterPush;
                        int popJLen = popJEnd - popJ;
                        int retLen2 = rerOff - retComputeStart;
                        int normEnd = -1, jfTgt = -1;
                        { int o, a, p = (T - 8 > afterPush) ? T - 8 : 0;
                          while (p < T) {
                              int np = opAt(p, o, a);
                              if (np == T && o == Pyc::JUMP_FORWARD_A) {
                                  normEnd = p; jfTgt = np + a * W;
                              }
                              if (np <= p) break;
                              p = np;
                          } }
                        int normalLen = condLen + popJLen + retLen2;
                        int E = (normEnd > 0) ? normEnd - normalLen : -1;
                        int co, ca, cn = (E > 0) ? opAt(E + condLen, co, ca) : -1;
                        if (E > 0 && jfTgt > T
                                && condLen > 0 && retLen2 > 0
                                && E + normalLen == normEnd
                                && memcmp(codeBuf->value() + afterPush,
                                          codeBuf->value() + E, condLen) == 0
                                && co == popJOp && cn + ca * W == normEnd
                                && memcmp(codeBuf->value() + retComputeStart,
                                          codeBuf->value() + E + condLen + popJLen,
                                          retLen2) == 0) {
                            int S = T;
                            for (const PycExceptionTableEntry* e : P)
                                S = std::min(S, e->start_offset);
                            bool ext = true;
                            while (ext) {
                                ext = false;
                                for (const auto& e : exception_entries) {
                                    if (e.start_offset < S
                                            && (e.end_offset == S
                                                || (e.target >= S && e.target < T))) {
                                        S = e.start_offset; ext = true;
                                    }
                                }
                            }
                            if (S < E) {
                                FinallyCoalesce plan = { T, E, T, jfTgt, normEnd };
                                finallyOpenAt[S].push_back(plan);
                                returnInFinallyAfter[T] = jfTgt;
                                continue;
                            }
                        }
                    }
                }
            }
            if (isExcept) {
                int S = T;
                for (const PycExceptionTableEntry* e : P) S = std::min(S, e->start_offset);
                int firstFrag = S;
                bool ext = true;
                while (ext) {
                    ext = false;
                    for (const auto& e : exception_entries) {
                        if (e.start_offset < S
                                && ((e.end_offset == S && e.target <= T)
                                    || (e.target >= S && e.target <= T))) {
                            S = e.start_offset; ext = true;
                        }
                    }
                }
                if (S < firstFrag || P.size() >= 2) {
                    if (!exceptOpenAt.count(S)) {
                        exceptOpenAt[S] = T;
                    } else if (exceptOpenAt[S] != T) {
                        /* Two try/except blocks begin at the same offset. Keep the
                           smaller-target (inner) one in the primary slot and stash
                           the other so both still open, nested correctly. */
                        int held = exceptOpenAt[S];
                        int inner = std::min(held, T), outer = std::max(held, T);
                        exceptOpenAt[S] = inner;
                        std::vector<int>& ex = exceptOpenAtExtra[S];
                        bool dup = false;
                        for (int v : ex) if (v == outer) { dup = true; break; }
                        if (!dup)
                            ex.push_back(outer);
                    }
                }
                continue;
            }
            {
                bool bareExcept = false;
                { int o, a; opAt(afterPush, o, a); if (o == Pyc::POP_TOP) bareExcept = true; }
                if (bareExcept && P.size() >= 2) {
                    int S = T;
                    for (const PycExceptionTableEntry* e : P) S = std::min(S, e->start_offset);
                    bool ext = true;
                    while (ext) {
                        ext = false;
                        for (const auto& e : exception_entries) {
                            if (e.start_offset < S
                                    && (e.end_offset == S
                                        || (e.target >= S && e.target <= T))) {
                                S = e.start_offset; ext = true;
                            }
                        }
                    }
                    if (!exceptOpenAt.count(S))
                        exceptOpenAt[S] = T;
                    continue;
                }
            }
            if (rerOff < 0)
                continue;
            if (P.size() == 1 && !finHasPlan(P[0]->start_offset, T)) {
                const int W = (int)sizeof(uint16_t);
                int S = P[0]->start_offset, E = P[0]->end_offset;
                int Flen = rerOff - afterPush;
                auto skipNops = [&](int p) -> int {
                    int o, a;
                    while (p < (int)codeBuf->length()) {
                        int np = opAt(p, o, a);
                        if (o != Pyc::NOP) break;
                        if (np <= p) break;
                        p = np;
                    }
                    return p;
                };
                auto isFcopy = [&](int p) -> bool {
                    if (Flen <= 0 || p + Flen > (int)codeBuf->length()) return false;
                    return memcmp(codeBuf->value() + afterPush,
                                  codeBuf->value() + p, Flen) == 0;
                };
                auto exitAfter = [&](int p, int& kind, int& jtgt) -> int {
                    int o, a, np = opAt(p, o, a);
                    if (o == Pyc::JUMP_FORWARD_A) { kind = 2; jtgt = np + a * W; return np; }
                    if (o == Pyc::RETURN_CONST_A) { kind = 1; jtgt = -1; return np; }
                    if (o == Pyc::LOAD_CONST_A || o == Pyc::LOAD_FAST_A
                            || o == Pyc::LOAD_GLOBAL_A || o == Pyc::LOAD_DEREF_A) {
                        int n2 = opAt(np, o, a);
                        if (o == Pyc::RETURN_VALUE || o == Pyc::INSTRUMENTED_RETURN_VALUE_A) {
                            kind = 1; jtgt = -1; return n2;
                        }
                    }
                    kind = 0; return -1;
                };
                int copyA = skipNops(E);
                int kA = 0, jA = -1, afterA = -1;
                if (isFcopy(copyA)) afterA = exitAfter(copyA + Flen, kA, jA);
                int elseT = -1;
                if (afterA > 0 && kA == 1) {
                    int o, a, p = S;
                    while (p < E) {
                        int np = opAt(p, o, a);
                        bool fwdCond = (o == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                                || o == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                                || o == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                                || o == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A);
                        if (fwdCond) {
                            int tg = np + a * W;
                            if (tg > E && tg < T) elseT = tg;
                        }
                        if (np <= p) break;
                        p = np;
                    }
                }
                int copyB = (elseT > 0) ? skipNops(elseT) : -1;
                int kB = 0, jB = -1, afterB = -1;
                if (copyB > 0 && isFcopy(copyB)) afterB = exitAfter(copyB + Flen, kB, jB);
                if (afterA > 0 && kA == 1 && afterB == T && kB != 0) {
                    int aft = -1;
                    if (kB == 2) aft = jB;
                    else {
                        int o, a, p = X;
                        while (p < (int)codeBuf->length()) {
                            int np = opAt(p, o, a);
                            if (o == Pyc::RERAISE || o == Pyc::RERAISE_A) { aft = np; break; }
                            if (np <= p) break;
                            p = np;
                        }
                    }
                    if (aft > T) {
                        FinallyCoalesce plan = { T, afterPush, rerOff, aft, elseT };
                        finallyOpenAt[S].push_back(plan);
                        finallyPlanByTarget[elseT] = plan;
                        finallyCopySkip[copyA] = Flen;
                        continue;
                    }
                }
            }
            {
                bool bareExcept = false;
                { int o, a; opAt(afterPush, o, a); if (o == Pyc::POP_TOP) bareExcept = true; }
                bool nested = false;
                if (!bareExcept) {
                    for (const auto& e : exception_entries) {
                        if (!e.push_lasti
                                && e.start_offset >= afterPush && e.start_offset < X
                                && e.target > afterPush && e.target < X) {
                            nested = true; break;
                        }
                    }
                }
                if (nested) {
                    int S = T, E = 0;
                    for (const PycExceptionTableEntry* e : P) {
                        S = std::min(S, e->start_offset);
                        E = std::max(E, e->end_offset);
                    }
                    bool ext = true;
                    while (ext) {
                        ext = false;
                        for (const auto& e : exception_entries) {
                            if (e.start_offset < S
                                    && (e.end_offset == S
                                        || (e.target >= S && e.target < T))) {
                                S = e.start_offset; ext = true;
                            }
                        }
                    }
                    int after = X;
                    { int o, a; for (int i = 0; i < 3 && after < (int)codeBuf->length(); ++i) {
                          int np = opAt(after, o, a); if (np <= after) break; after = np; } }
                    if (S < E && E <= T && after > X && !finHasPlan(S, T)) {
                        FinallyCoalesce plan = { T, afterPush, X, after, 0 };
                        finallyOpenAt[S].push_back(plan);
                        finallyPlanByTarget[T] = plan;
                        finallyCopySkip[E] = T - E;
                        nestedFinRegions.push_back(std::make_pair(afterPush, X));
                        { int o, a, p = afterPush, prevop = -1;
                          while (p < X) {
                              int np = opAt(p, o, a);
                              if ((o == Pyc::RERAISE || (o == Pyc::RERAISE_A && a == 0))
                                      && (p == rerOff || prevop == Pyc::POP_EXCEPT))
                                  finExcReraise[p] = X;
                              prevop = o;
                              if (np <= p) break;
                              p = np;
                          } }
                        { int o, a, p = E, lastOp = -1;
                          int mainEnd = T;
                          { int oo, aa, q = E;
                            while (q < T) {
                                int nq = opAt(q, oo, aa);
                                if (oo == Pyc::PUSH_EXC_INFO) { mainEnd = q; break; }
                                if (nq <= q) break;
                                q = nq;
                            } }
                          while (p < mainEnd) { int np = opAt(p, o, a); if (np <= p) break; lastOp = o; p = np; }
                          if (lastOp == Pyc::RETURN_VALUE || lastOp == Pyc::INSTRUMENTED_RETURN_VALUE_A)
                              finallyReturnExit[T] = T;
                        }
                        if (rerOff > afterPush) {
                            int fl = rerOff - afterPush;
                            for (const PycExceptionTableEntry* e : P) {
                                int cs = e->end_offset;
                                if (cs == E || cs + fl > (int)codeBuf->length())
                                    continue;
                                if (finallyCopySkip.count(cs))
                                    continue;
                                { int eo = 0, ea = 0;
                                  int en = opAt(e->start_offset, eo, ea);
                                  if (eo != Pyc::POP_EXCEPT || en != cs)
                                      continue; }
                                bool match = true;
                                int a1 = cs, a2 = afterPush;
                                while (a2 < rerOff) {
                                    int o1, r1, o2, r2;
                                    int n1 = opAt(a1, o1, r1), n2 = opAt(a2, o2, r2);
                                    if (n1 <= a1 || n2 <= a2 || o1 != o2 || r1 != r2) { match = false; break; }
                                    a1 = n1; a2 = n2;
                                }
                                if (!match) continue;
                                int tOff = cs + fl, to = 0, ta = 0;
                                int tn = opAt(tOff, to, ta);
                                int retOff = -1;
                                if (to == Pyc::RETURN_VALUE || to == Pyc::INSTRUMENTED_RETURN_VALUE_A
                                        || to == Pyc::RETURN_CONST_A || to == Pyc::INSTRUMENTED_RETURN_CONST_A) {
                                    retOff = tOff;
                                } else if (to == Pyc::LOAD_CONST_A) {
                                    int no = 0, na = 0; (void)opAt(tn, no, na);
                                    if (no == Pyc::RETURN_VALUE || no == Pyc::INSTRUMENTED_RETURN_VALUE_A)
                                        retOff = tn;
                                }
                                if (retOff < 0) continue;
                                int hs = -1;
                                { int o, a, p = S;
                                  while (p < cs) {
                                      int np = opAt(p, o, a);
                                      if (o == Pyc::PUSH_EXC_INFO) hs = p;
                                      if (np <= p) break;
                                      p = np;
                                  } }
                                int merge = T;
                                if (hs > S) {
                                    int o, a, p = S, prevStart = -1;
                                    while (p < hs) {
                                        int np = opAt(p, o, a);
                                        if (np == hs) { prevStart = p; break; }
                                        if (np <= p) break;
                                        p = np;
                                    }
                                    if (prevStart >= 0) {
                                        int po, pa; int pn = opAt(prevStart, po, pa);
                                        if (po == Pyc::JUMP_FORWARD_A) {
                                            int tgt = pn + pa * (int)sizeof(uint16_t);
                                            if (tgt > cs && tgt <= T) merge = tgt;
                                        }
                                    }
                                }
                                finallyCopySkip[cs] = fl;
                                finallyExceptReturnExit[retOff] = merge;
                            }
                        }
                    }
                    continue;
                }
            }
            if (rerOff > afterPush) {
                const int W = (int)sizeof(uint16_t);
                int elseT = -1, condBr = -1, condBrNext = -1, lastRer = -1;
                auto isBranch = [](int o) {
                    return o == Pyc::POP_JUMP_FORWARD_IF_FALSE_A || o == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                        || o == Pyc::POP_JUMP_FORWARD_IF_NONE_A || o == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A
                        || o == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A || o == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A
                        || o == Pyc::POP_JUMP_BACKWARD_IF_NONE_A || o == Pyc::POP_JUMP_BACKWARD_IF_NOT_NONE_A
                        || o == Pyc::JUMP_IF_TRUE_OR_POP_A || o == Pyc::JUMP_IF_FALSE_OR_POP_A
                        || o == Pyc::JUMP_FORWARD_A || o == Pyc::JUMP_BACKWARD_A
                        || o == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A || o == Pyc::FOR_ITER_A
                        || o == Pyc::PUSH_EXC_INFO || o == Pyc::CHECK_EXC_MATCH;
                };
                bool finBodyHasLoop = false;
                { int sLo = T; for (const PycExceptionTableEntry* e : P) sLo = std::min(sLo, e->start_offset);
                  int o, a, p = sLo;
                  while (p < X) {
                      int np = opAt(p, o, a);
                      if (o == Pyc::FOR_ITER_A || o == Pyc::JUMP_BACKWARD_A) { finBodyHasLoop = true; break; }
                      if (o == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A) {
                          int bt = np - a * W, bo = -1, ba; opAt(bt, bo, ba);
                          if (bo != Pyc::SEND_A) { finBodyHasLoop = true; break; }
                      }
                      if (np <= p) break;
                      p = np;
                  } }
                auto armHardOp = [](int o) {
                    return o == Pyc::PUSH_EXC_INFO || o == Pyc::CHECK_EXC_MATCH
                        || o == Pyc::FOR_ITER_A || o == Pyc::JUMP_BACKWARD_A
                        || o == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A || o == Pyc::SETUP_FINALLY_A
                        || o == Pyc::JUMP_FORWARD_A
                        || o == Pyc::JUMP_IF_TRUE_OR_POP_A || o == Pyc::JUMP_IF_FALSE_OR_POP_A
                        || o == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A || o == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A
                        || o == Pyc::POP_JUMP_BACKWARD_IF_NONE_A || o == Pyc::POP_JUMP_BACKWARD_IF_NOT_NONE_A;
                };
                { int o, a, p = afterPush;
                  while (p < rerOff) {
                      int np = opAt(p, o, a);
                      bool fwdBr = (o == Pyc::POP_JUMP_FORWARD_IF_FALSE_A || o == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                              || o == Pyc::POP_JUMP_FORWARD_IF_NONE_A || o == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A);
                      if (fwdBr) { condBr = p; condBrNext = np; elseT = np + a * W; break; }
                      if (o == Pyc::JUMP_FORWARD_A || o == Pyc::JUMP_BACKWARD_A
                              || o == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A || o == Pyc::FOR_ITER_A
                              || o == Pyc::PUSH_EXC_INFO) break;
                      if (np <= p) break;
                      p = np;
                  } }
                bool trueArmClean = (condBr > 0);
                if (trueArmClean) {
                    int o, a, p = condBrNext;
                    while (p < rerOff) {
                        int np = opAt(p, o, a);
                        if (finBodyHasLoop ? isBranch(o) : armHardOp(o)) { trueArmClean = false; break; }
                        if (np <= p) break;
                        p = np;
                    }
                }
                bool ifElse = false;
                if (trueArmClean && condBr > 0 && elseT > rerOff && elseT < X) {
                    int o, a, p = elseT; bool clean = true;
                    while (p < X) {
                        int np = opAt(p, o, a);
                        if (finBodyHasLoop ? isBranch(o) : armHardOp(o)) { clean = false; break; }
                        if (o == Pyc::RERAISE || o == Pyc::RERAISE_A) { lastRer = p; }
                        if (np <= p) break;
                        p = np;
                    }
                    if (clean && lastRer > elseT) {
                        int so, sa; int snp = opAt(lastRer, so, sa);
                        if (snp == X) ifElse = true;
                    }
                }
                if (ifElse && P.size() >= 2) {
                    int S = T, E = 0;
                    for (const PycExceptionTableEntry* e : P) {
                        S = std::min(S, e->start_offset);
                        E = std::max(E, e->end_offset);
                    }
                    bool ext = true;
                    while (ext) {
                        ext = false;
                        for (const auto& e : exception_entries) {
                            if (e.start_offset < S
                                    && (e.end_offset == S
                                        || (e.target >= S && e.target < T))) {
                                S = e.start_offset; ext = true;
                            }
                        }
                    }
                    int after = X;
                    { int o, a; for (int i = 0; i < 3 && after < (int)codeBuf->length(); ++i) {
                          int np = opAt(after, o, a); if (np <= after) break; after = np; } }
                    bool simpleNormal = (E < T);
                    for (const auto& e : exception_entries)
                        if (e.start_offset > E && e.start_offset < T) { simpleNormal = false; break; }
                    if (S < E && E <= T && after > X && simpleNormal && !finHasPlan(S, T)) {
                        FinallyCoalesce plan = { T, afterPush, X, after, 0 };
                        finallyOpenAt[S].push_back(plan);
                        finallyPlanByTarget[T] = plan;
                        finallyCopySkip[E] = T - E;
                        finElseAt[elseT] = lastRer;
                        continue;
                    }
                }
            }
            /* SHARED-MERGE if/ELSE finally (`try: B finally: if C: A else: D`,
               canonically a `@contextmanager` generator `try: yield finally: …`).
               The finally body is a single top-level if/else whose if-true arm
               JUMP_FORWARDs over the else to ONE trailing RERAISE both arms share
               (rerOff). The exceptional copy [afterPush, rerOff) is the clean if/else;
               the NORMAL copy [E, T) inlines the same body but each arm ends in the
               generator's `return None` epilogue instead of the reraise — so matchCopy
               can't byte-pair the divergent arm tails and the coalesce bails, dropping
               the `finally:` to a swallowing bare `except:`. Render the finally from the
               exceptional copy (the if/else reconstructs natively from the if-true arm's
               JUMP_FORWARD-to-merge) and skip the whole normal copy. Gated TIGHT: one
               protected fragment, both arms straight-line, no nested handler in [E, T),
               and the normal copy's prefix byte-matches the exceptional copy's. */
            if (P.size() == 1 && rerOff > afterPush && !finHasPlan(P[0]->start_offset, T)) {
                const int W = (int)sizeof(uint16_t);
                auto fwdCond = [](int o) {
                    return o == Pyc::POP_JUMP_FORWARD_IF_FALSE_A || o == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                        || o == Pyc::POP_JUMP_FORWARD_IF_NONE_A || o == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A;
                };
                auto ctrlOp = [](int o) {   // any jump/branch/loop/exception op
                    return o == Pyc::POP_JUMP_FORWARD_IF_FALSE_A || o == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                        || o == Pyc::POP_JUMP_FORWARD_IF_NONE_A || o == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A
                        || o == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A || o == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A
                        || o == Pyc::POP_JUMP_BACKWARD_IF_NONE_A || o == Pyc::POP_JUMP_BACKWARD_IF_NOT_NONE_A
                        || o == Pyc::JUMP_IF_TRUE_OR_POP_A || o == Pyc::JUMP_IF_FALSE_OR_POP_A
                        || o == Pyc::JUMP_FORWARD_A || o == Pyc::JUMP_BACKWARD_A
                        || o == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A || o == Pyc::FOR_ITER_A
                        || o == Pyc::PUSH_EXC_INFO || o == Pyc::CHECK_EXC_MATCH
                        || o == Pyc::SETUP_FINALLY_A;
                };
                /* the top-level forward cond in [afterPush, rerOff). */
                int condBr = -1, condNext = -1, elseT = -1;
                { int o, a, p = afterPush;
                  while (p < rerOff) {
                      int np = opAt(p, o, a);
                      if (fwdCond(o)) { condBr = p; condNext = np; elseT = np + a * W; break; }
                      if (ctrlOp(o)) break;
                      if (np <= p) break;
                      p = np;
                  } }
                bool shape = (condBr > 0 && elseT > condNext && elseT < rerOff);
                /* if-true arm [condNext, elseT): straight-line, ending in exactly one
                   JUMP_FORWARD -> rerOff as its terminal op (the jump over the else). */
                if (shape) {
                    int o, a, p = condNext, jf = -1;
                    while (p < elseT) {
                        int np = opAt(p, o, a);
                        if (o == Pyc::JUMP_FORWARD_A) { jf = p; if (np != elseT) { shape = false; break; } }
                        else if (ctrlOp(o)) { shape = false; break; }
                        if (np <= p) break;
                        p = np;
                    }
                    if (shape) {
                        int jo, ja; int jn = (jf > 0) ? opAt(jf, jo, ja) : -1;
                        shape = (jf > 0 && jn == elseT && jn + ja * W == rerOff);
                    }
                }
                /* else arm [elseT, rerOff): straight-line, falling to the shared RERAISE. */
                if (shape) {
                    int o, a, p = elseT;
                    while (p < rerOff) {
                        int np = opAt(p, o, a);
                        if (ctrlOp(o)) { shape = false; break; }
                        if (np <= p) break;
                        p = np;
                    }
                }
                /* confirm [E, T) is the inlined normal copy: contiguous (no nested
                   handler), and its body (past a leading yield POP_TOP) byte-matches the
                   exceptional copy's prefix [afterPush, condBr). */
                if (shape) {
                    int E = P[0]->end_offset;
                    bool simpleNormal = (E < T);
                    for (const auto& e : exception_entries)
                        if (e.start_offset > E && e.start_offset < T) { simpleNormal = false; break; }
                    int nb = E;
                    { int o, a;
                      while (nb < T) {
                          int np = opAt(nb, o, a);
                          if (o != Pyc::POP_TOP && o != Pyc::NOP) break;
                          if (np <= nb) break;
                          nb = np;
                      } }
                    int preLen = condBr - afterPush;
                    bool prefixMatch = simpleNormal && preLen > 0 && nb + preLen <= T
                        && memcmp(codeBuf->value() + afterPush, codeBuf->value() + nb, preLen) == 0;
                    if (prefixMatch) {
                        int S = T;
                        for (const PycExceptionTableEntry* e : P) S = std::min(S, e->start_offset);
                        bool ext = true;
                        while (ext) { ext = false;
                            for (const auto& e : exception_entries)
                                if (e.start_offset < S && (e.end_offset == S || (e.target >= S && e.target < T))) {
                                    S = e.start_offset; ext = true;
                                } }
                        int after = X;
                        { int o, a; for (int i = 0; i < 3 && after < (int)codeBuf->length(); ++i) {
                              int np = opAt(after, o, a); if (np <= after) break; after = np; } }
                        if (S < E && after >= X) {
                            FinallyCoalesce plan = { T, afterPush, rerOff, after, 0 };
                            finallyOpenAt[S].push_back(plan);
                            finallyPlanByTarget[T] = plan;
                            finallyCopySkip[E] = T - E;
                            continue;
                        }
                    }
                }
            }
            int finBodyStart = afterPush, finBodyEnd = rerOff, finLen = finBodyEnd - finBodyStart;
            if (finLen <= 0)
                continue;
            bool splitTableFin = (P.size() >= 2);
            { bool cf = false; int o, a, p = finBodyStart; const int W = (int)sizeof(uint16_t);
              while (p < finBodyEnd) {
                  int np = opAt(p, o, a);
                  if (o == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A) {
                      int bt = np - a * W, bo = -1, ba;
                      opAt(bt, bo, ba);
                      if (bo != Pyc::SEND_A) { cf = true; break; }
                  } else if (o == Pyc::PUSH_EXC_INFO || o == Pyc::SETUP_FINALLY_A) {
                      /* A with-exit's exceptional copy (PUSH_EXC_INFO; WITH_EXCEPT_START)
                         inside the finally body is NOT the finally's own control flow — it
                         is a nested `with` cleanup that matchCopy pairs byte-for-byte across
                         the normal/exceptional finally copies. Only bail on a PUSH_EXC_INFO
                         that is NOT a with-exit (a nested try/except or try/finally). */
                      bool withExit = false;
                      if (o == Pyc::PUSH_EXC_INFO && !bodyHasNestedProt) {
                          int wo, wa; opAt(np, wo, wa);
                          if (wo == Pyc::WITH_EXCEPT_START) withExit = true;
                      }
                      if (!withExit) { cf = true; break; }
                  }
                  else if (o == Pyc::FOR_ITER_A || o == Pyc::JUMP_BACKWARD_A) {
                      if (!splitTableFin) { cf = true; break; }
                  }
                  if (o == Pyc::POP_JUMP_FORWARD_IF_TRUE_A || o == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                          || o == Pyc::POP_JUMP_FORWARD_IF_NONE_A || o == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A) {
                      int bt = np + a * W;
                      if (bt >= finBodyEnd) {
                          int bo = -1, ba, bp = bt;
                          while (true) { int bn = opAt(bp, bo, ba); if (bo != Pyc::NOP || bn <= bp) break; bp = bn; }
                          if (bo != Pyc::RERAISE && bo != Pyc::RERAISE_A) { cf = true; break; }
                      }
                  }
                  if (np <= p) break;
                  p = np;
              }
              bool selfContainedFin = false;
              if (cf && P.size() >= 2 && X > afterPush) {
                  bool ok = true, hasCond = false; int lastReal = -1;
                  int o, a, p = afterPush;
                  while (p < X) {
                      int np = opAt(p, o, a);
                      if (np <= p) { ok = false; break; }
                      if (o == Pyc::PUSH_EXC_INFO || o == Pyc::CHECK_EXC_MATCH
                              || o == Pyc::SETUP_FINALLY_A || o == Pyc::SETUP_EXCEPT_A
                              || o == Pyc::FOR_ITER_A || o == Pyc::JUMP_BACKWARD_A
                              || o == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A || o == Pyc::SEND_A) { ok = false; break; }
                      bool fwdCond = (o == Pyc::POP_JUMP_FORWARD_IF_TRUE_A || o == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                              || o == Pyc::POP_JUMP_FORWARD_IF_NONE_A || o == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A);
                      bool fwdAny = fwdCond || o == Pyc::JUMP_FORWARD_A
                              || o == Pyc::JUMP_IF_TRUE_OR_POP_A || o == Pyc::JUMP_IF_FALSE_OR_POP_A;
                      if (fwdCond) hasCond = true;
                      if (fwdAny) {
                          int bt = np + a * W;
                          if (bt < afterPush || bt > X) { ok = false; break; }
                      }
                      if (o != Pyc::CACHE && o != Pyc::NOP) lastReal = o;
                      p = np;
                  }
                  if (ok && hasCond && (lastReal == Pyc::RERAISE || lastReal == Pyc::RERAISE_A))
                      selfContainedFin = true;
              }
              if (selfContainedFin) {
                  int S = T, E = 0;
                  for (const PycExceptionTableEntry* e : P) {
                      S = std::min(S, e->start_offset);
                      E = std::max(E, e->end_offset);
                  }
                  bool ext = true;
                  while (ext) {
                      ext = false;
                      for (const auto& e : exception_entries) {
                          if (e.start_offset < S
                                  && (e.end_offset == S
                                      || (e.target >= S && e.target < T))) {
                              S = e.start_offset; ext = true;
                          }
                      }
                  }
                  int after = X;
                  { int o, a; for (int i = 0; i < 3 && after < (int)codeBuf->length(); ++i) {
                        int np = opAt(after, o, a); if (np <= after) break; after = np; } }
                  bool simpleNormal = (E < T);
                  for (const auto& e : exception_entries)
                      if (e.start_offset > E && e.start_offset < T) { simpleNormal = false; break; }
                  if (S < E && E <= T && after > X && simpleNormal && !finHasPlan(S, T)) {
                      { int o, a, p = afterPush;
                        while (p < X) {
                            int np = opAt(p, o, a);
                            if (np <= p) break;
                            bool fwdCond = (o == Pyc::POP_JUMP_FORWARD_IF_TRUE_A || o == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                                    || o == Pyc::POP_JUMP_FORWARD_IF_NONE_A || o == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A);
                            if (fwdCond) {
                                int F = np + a * W;
                                if (F > p && F < X && !finElseAt.count(F)) {
                                    int bo = -1, ba, bp = afterPush, lastBefore = -1;
                                    while (bp < F) { int bn = opAt(bp, bo, ba); if (bn <= bp) break; lastBefore = bo; bp = bn; }
                                    if (lastBefore == Pyc::RERAISE || lastBefore == Pyc::RERAISE_A) {
                                        int eo = -1, ea, ep = F, eRer = -1;
                                        while (ep < X) { int en = opAt(ep, eo, ea); if (en <= ep) break;
                                            if (eo == Pyc::RERAISE || eo == Pyc::RERAISE_A) { eRer = ep; break; } ep = en; }
                                        if (eRer > F) finElseAt[F] = eRer;
                                    }
                                }
                            }
                            p = np;
                        } }
                      FinallyCoalesce plan = { T, afterPush, X, after, 0 };
                      finallyOpenAt[S].push_back(plan);
                      finallyPlanByTarget[T] = plan;
                      finallyCopySkip[E] = T - E;
                      continue;
                  }
              }
              if (cf) continue; }
            int S = T;
            for (const PycExceptionTableEntry* e : P) S = std::min(S, e->start_offset);
            bool extended = true;
            while (extended) {
                extended = false;
                for (const auto& e : exception_entries) {
                    if (e.start_offset < S
                            && (e.end_offset == S
                                || (e.target >= S && e.target < T))) {
                        S = e.start_offset; extended = true;
                    }
                }
            }
            int afterDefault = X;
            { int o, a; for (int i = 0; i < 3 && afterDefault < (int)codeBuf->length(); ++i) {
                  int np = opAt(afterDefault, o, a); if (np <= afterDefault) break; afterDefault = np; } }
            int after = afterDefault;
            bool ok = true, singleJumpf = false;
            bool loopBreakOverHandler = false;
            bool loopCtrlCopy = false;
            std::vector<std::pair<int,int>> copies;
            std::vector<std::pair<int,int>> retExits;
            auto matchCopy = [&](int cs, int& skipOut, int& afterOut,
                                 bool& isFwd, int& retOffOut) -> int {
                if (cs + finLen > (int)codeBuf->length())
                    return 0;
                bool branchInBody = false;
                int a1 = cs, a2 = finBodyStart;
                while (a2 < finBodyEnd) {
                    int o1, r1, o2, r2; int n1 = opAt(a1, o1, r1); int n2 = opAt(a2, o2, r2);
                    if (n1 <= a1 || n2 <= a2 || o1 != o2) return 0;
                    if (o2 == Pyc::POP_JUMP_FORWARD_IF_FALSE_A || o2 == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                            || o2 == Pyc::POP_JUMP_FORWARD_IF_NONE_A || o2 == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A)
                        branchInBody = true;
                    if (r1 != r2) {
                        const int W = (int)sizeof(uint16_t);
                        bool fwdBr = (o1 == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                                || o1 == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                                || o1 == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                                || o1 == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A);
                        int bt1 = n1 + r1 * W, bt2 = n2 + r2 * W, end1 = cs + finLen;
                        bool trailingIf = (fwdBr && bt1 > a1 && bt2 > a2
                                && bt1 >= end1 - 2*W && bt1 <= end1 + 4*W
                                && bt2 >= finBodyEnd - 2*W && bt2 <= finBodyEnd + 4*W);
                        bool internalBr = (fwdBr && splitTableFin
                                && bt1 > a1 && bt2 > a2
                                && (bt1 - cs) == (bt2 - finBodyStart)
                                && (bt1 - cs) > 0 && (bt1 - cs) <= finLen);
                        bool exitPadBr = false;
                        if (fwdBr && bt1 > a1 && bt2 >= finBodyEnd) {
                            int eo = -1, ea, ep = bt2;
                            while (true) { int en = opAt(ep, eo, ea); if (eo != Pyc::NOP || en <= ep) break; ep = en; }
                            bool excExit = (eo == Pyc::RERAISE || eo == Pyc::RERAISE_A);
                            int no = -1, na, np = bt1;
                            while (true) { int nn = opAt(np, no, na); if (no != Pyc::NOP || nn <= np) break; np = nn; }
                            bool normExit = (no == Pyc::RETURN_VALUE || no == Pyc::INSTRUMENTED_RETURN_VALUE_A
                                    || no == Pyc::RETURN_CONST_A || no == Pyc::INSTRUMENTED_RETURN_CONST_A
                                    || no == Pyc::LOAD_CONST_A || no == Pyc::JUMP_FORWARD_A);
                            exitPadBr = excExit && normExit;
                        }
                        /* A FOR_ITER whose loop-exhaustion exit leaves the copy: in the
                           exceptional finally copy that exit falls into the trailing
                           RERAISE (re-raise after finally), while in the normal copy it
                           reaches the post-finally continuation (a JUMP_FORWARD over the
                           handler, or the function epilogue). The exit arg therefore
                           differs by the one-instruction RERAISE/JUMP_FORWARD shift, but
                           the loop body is otherwise byte-identical, so pair it. */
                        bool forIterExitPad = false;
                        if (!exitPadBr && o1 == Pyc::FOR_ITER_A
                                && bt2 >= finBodyEnd && bt1 >= end1) {
                            int eo = -1, ea, ep = bt2;
                            while (true) { int en = opAt(ep, eo, ea); if (eo != Pyc::NOP || en <= ep) break; ep = en; }
                            bool excExit = (eo == Pyc::RERAISE || eo == Pyc::RERAISE_A);
                            int no = -1, na, np = bt1;
                            while (true) { int nn = opAt(np, no, na); if (no != Pyc::NOP || nn <= np) break; np = nn; }
                            bool normExit = (no == Pyc::RETURN_VALUE || no == Pyc::INSTRUMENTED_RETURN_VALUE_A
                                    || no == Pyc::RETURN_CONST_A || no == Pyc::INSTRUMENTED_RETURN_CONST_A
                                    || no == Pyc::LOAD_CONST_A || no == Pyc::JUMP_FORWARD_A);
                            forIterExitPad = excExit && normExit;
                        }
                        if (!trailingIf && !internalBr && !exitPadBr && !forIterExitPad)
                            return 0;
                    }
                    a1 = n1; a2 = n2;
                }
                int skip = finLen;
                int tOff = cs + finLen;
                int to = 0, ta = 0; int tn = opAt(tOff, to, ta);
                isFwd = false;
                retOffOut = -1;
                if (to == Pyc::JUMP_FORWARD_A) {
                    int jt = tn + ta * (int)sizeof(uint16_t);
                    if (jt > T && whileExit200.count(jt) && whileExit200[jt] <= S) {
                        skipOut = skip;
                        return 1;
                    }
                    if (jt >= T) { skip += (tn - tOff); afterOut = jt; isFwd = true; }
                    else return 2;
                } else if (to == Pyc::JUMP_BACKWARD_A || to == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A) {
                    skipOut = skip;
                    return 1;
                } else if (tOff > T) {
                    return 2;
                } else if (to == Pyc::RETURN_VALUE
                        || to == Pyc::INSTRUMENTED_RETURN_VALUE_A
                        || to == Pyc::RETURN_CONST_A
                        || to == Pyc::INSTRUMENTED_RETURN_CONST_A) {
                    retOffOut = tOff;
                    if (branchInBody) {
                        int so = 0, sa = 0; (void)opAt(tn, so, sa);
                        if (so == Pyc::RETURN_VALUE || so == Pyc::INSTRUMENTED_RETURN_VALUE_A
                                || so == Pyc::RETURN_CONST_A || so == Pyc::INSTRUMENTED_RETURN_CONST_A) {
                            skip += (tn - tOff);
                            retOffOut = tn;
                        }
                    }
                } else if (to == Pyc::LOAD_CONST_A) {
                    int nop = 0, narg = 0;
                    int np2 = opAt(tn, nop, narg);
                    if (nop == Pyc::RETURN_VALUE
                            || nop == Pyc::INSTRUMENTED_RETURN_VALUE_A) {
                        retOffOut = tn;
                        if (branchInBody) {
                            /* A finally body whose tail is a nest of `if` statements
                               duplicates the deferred `return <const>` into every branch
                               exit — a two-level `if a: if b: X` yields three such exits.
                               Walk the run of consecutive identical `LOAD_CONST c;
                               RETURN_VALUE` pairs starting at tOff and skip all but the
                               last, so a single `return` survives. A single trailing `if`
                               is the one-pair case. */
                            std::vector<int> starts;
                            int p = tOff, retEnd = tn;
                            while (true) {
                                int lo = 0, la = 0; int rn = opAt(p, lo, la);
                                if (lo == Pyc::LOAD_CONST_A && la == ta) {
                                    int ro = 0, ra = 0; int r2 = opAt(rn, ro, ra);
                                    if (ro == Pyc::RETURN_VALUE
                                            || ro == Pyc::INSTRUMENTED_RETURN_VALUE_A) {
                                        starts.push_back(p);
                                        retEnd = rn;
                                        p = r2;
                                        continue;
                                    }
                                }
                                break;
                            }
                            if (starts.size() >= 2) {
                                int keep = starts.back();   // keep the LAST pair
                                skip += (keep - tOff);
                                retOffOut = retEnd;
                            }
                        }
                    }
                    (void)np2;
                }
                skipOut = skip;
                return 1;
            };
            for (const PycExceptionTableEntry* e : P) {
                int cs = e->end_offset;
                int skip = 0, afterCand = after, retOff = -1;
                bool isFwd = false;
                int r = matchCopy(cs, skip, afterCand, isFwd, retOff);
                /* The try body's own normal-exit inlined finally copy can begin one
                   instruction after this fragment's end, because 3.11 emits a
                   line-boundary NOP between the last body op and the copy. matchCopy
                   at the NOP fails; retry once past a single leading NOP (as the
                   non-P copy scan already does). Guarded on the direct match having
                   failed, so a real copy at e->end_offset is unaffected. */
                if (r == 0) {
                    int o, a, np = opAt(cs, o, a);
                    if (o == Pyc::NOP && np > cs && np < T) {
                        int cs2 = np, skip2 = 0, afterCand2 = after, retOff2 = -1;
                        bool isFwd2 = false;
                        int r2 = matchCopy(cs2, skip2, afterCand2, isFwd2, retOff2);
                        if (r2 == 1) { cs = cs2; skip = skip2; afterCand = afterCand2;
                                       retOff = retOff2; isFwd = isFwd2; r = r2; }
                    }
                }
                if (r == 0) continue;
                if (r == 2) { ok = false; break; }
                if (isFwd) { after = afterCand; singleJumpf = true; }
                /* The normal finally copy may close a loop-body branch by falling
                   straight into the loop back-edge (a `continue`): its terminal op
                   is a JUMP_BACKWARD whose target lies at or before the protected
                   region's start. That is a loop-control copy just like the ones
                   the loop-tail scan below recognises, so flag it — otherwise a
                   single-handler try/finally inside a loop branch is dropped for
                   want of a p1wrap/with marker, taking the following sibling
                   branches with it. */
                if (!isFwd && retOff < 0) {
                    int to = 0, ta = 0; int jn = opAt(cs + skip, to, ta);
                    if (to == Pyc::JUMP_BACKWARD_A
                            || to == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A) {
                        int bt = jn - ta * (int)sizeof(uint16_t);
                        /* Only treat this as a continue-style loop-control copy when
                           the post-handler resume (`after`) lands BACK INSIDE the
                           enclosing loop body — i.e. more sibling statements of the
                           loop follow this branch. If `after` is itself the loop's
                           exit (the try/finally is the loop's last statement), the
                           ordinary non-plan reconstruction already renders it; forcing
                           a plan there would fold the post-loop code into the finally. */
                        if (bt <= S && !forIterExits200.count(after))
                            loopCtrlCopy = true;
                    }
                }
                copies.push_back({ cs, skip });
                if (retOff >= 0) retExits.push_back({ retOff, T });
            }
            if (!ok)
                continue;
            (void)singleJumpf;
            {
                std::set<int> seen;
                for (const auto& c : copies) seen.insert(c.first);
                for (const auto& e : exception_entries) {
                    if (e.target == T)
                        continue;
                    int cs = e.end_offset;
                    {
                        int o, a, p = cs;
                        while (p < T) {
                            int np = opAt(p, o, a);
                            if (o != Pyc::NOP)
                                break;
                            if (np <= p)
                                break;
                            p = np;
                        }
                        cs = p;
                    }
                    if (cs <= S || cs >= T)
                        continue;
                    if (seen.count(cs))
                        continue;
                    int skip = 0, afterCand = after, retOff = -1;
                    bool isFwd = false;
                    int r = matchCopy(cs, skip, afterCand, isFwd, retOff);
                    if (r != 1) continue;
                    seen.insert(cs);
                    copies.push_back({ cs, skip });
                    /* When THIS non-P copy is the try body's own normal-exit inlined
                       finally (a forward JUMP over the handler to the post-finally
                       continuation), propagate its merge target to `after`. Layer-2
                       anchors copies at exception-entry ends and previously discarded
                       the forward merge, leaving `after` at the past-handler reraise
                       continuation (afterDefault). When the merge lands PAST an
                       enclosing loop's exit on a bare `LOAD_x; RETURN_VALUE` reached
                       by ONLY this jump, that return is the finally's NORMAL-exit
                       continuation which the compiler placed as a shared tail block
                       behind the loop-exit statement (character
                       `an observed function`: the finally's normal exit is
                       `return X`, laid out after the loop-exit `raise`).
                       Resuming the finally at that offset closes the loop first and
                       strands the return as dead code after the `raise`. Instead
                       synthesize the `return X` and append it as the LAST STATEMENT
                       OF THE INNER TRY BODY (keyed by THIS finally-copy start `cs` =
                       the inner-except protected range's end, drained at the
                       finallyCopySkip point while the inner BLK_TRY is still open —
                       so the return renders inside the inner try, matching the
                       original's inlined-finally-copy-precedes-inner-EXC layout).
                       Resume the finally at the LOOP EXIT (so the loop closes and the
                       exit statement renders post-loop), and suppress the tail-level
                       emit at the return offset. This is the try/FINALLY analogue of
                       teoElseAbsorbRet. Gated to: `after`
                       not yet claimed, a single predecessor of the return offset, and
                       a loop whose exit lies in (T, afterCand) with a back-edge before
                       S. Absent that exact shape, `after` is LEFT UNTOUCHED (the
                       original Layer-2 behaviour) — a plain forward-merge propagation
                       here regressed a rotated `while` whose in-loop try/finally
                       normal-exits forward (godma `prime`). */
                    if (isFwd && after == afterDefault && afterCand > T) {
                        const int W = (int)sizeof(uint16_t);
                        /* the merge (afterCand) must be a bare `LOAD_x; RETURN_VALUE`. */
                        int lo = 0, la = 0; int ln = opAt(afterCand, lo, la);
                        bool loadRet = false; PycRef<ASTNode> rval(nullptr);
                        if (lo == Pyc::LOAD_FAST_A || lo == Pyc::LOAD_NAME_A
                                || lo == Pyc::LOAD_GLOBAL_A || lo == Pyc::LOAD_DEREF_A
                                || lo == Pyc::LOAD_CLASSDEREF_A || lo == Pyc::LOAD_CONST_A) {
                            int ro = 0, ra = 0; (void)opAt(ln, ro, ra);
                            if (ro == Pyc::RETURN_VALUE || ro == Pyc::INSTRUMENTED_RETURN_VALUE_A)
                                loadRet = true;
                        }
                        /* a loop whose exit E_exit lies strictly in (T, afterCand):
                           a forward guard (POP_JUMP_FORWARD_IF_*) at/before the loop
                           top (before S) targets E_exit, AND a JUMP_BACKWARD whose
                           top is before S closes the loop (confirming E_exit is a
                           genuine loop exit, not an ordinary forward branch). */
                        int loopExit = -1;
                        if (loadRet) {
                            int bo = 0, ba = 0, bp = 0;
                            while (bp < S) {
                                int bn = opAt(bp, bo, ba);
                                if (bn <= bp) break;
                                bool fwdCond = (bo == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                                        || bo == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                                        || bo == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                                        || bo == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A);
                                if (fwdCond) {
                                    int tg = bn + ba * W;
                                    if (tg > T && tg < afterCand) { loopExit = tg; break; }
                                }
                                bp = bn;
                            }
                        }
                        /* confirm a back-edge (loop top before S) closes to before the
                           exit — i.e. this is a real loop wrapping the try/finally. */
                        if (loopExit > 0) {
                            bool hasBackEdge = false;
                            int bo = 0, ba = 0, bp = 0;
                            while (bp < loopExit) {
                                int bn = opAt(bp, bo, ba);
                                if (bn <= bp) break;
                                if ((bo == Pyc::JUMP_BACKWARD_A
                                        || bo == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A)
                                        && (bn - ba * W) < S && bp > S) {
                                    hasBackEdge = true; break;
                                }
                                bp = bn;
                            }
                            if (!hasBackEdge) loopExit = -1;
                        }
                        /* single predecessor of the return offset. */
                        int preds = 0;
                        if (loopExit > 0) {
                            int po = 0, pa = 0, pp = 0;
                            while (pp < (int)codeBuf->length()) {
                                int pip = pp; int pn = opAt(pp, po, pa);
                                if (pn <= pip) break;
                                int pt = -1;
                                if (po == Pyc::JUMP_FORWARD_A
                                        || po == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                                        || po == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                                        || po == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                                        || po == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A)
                                    pt = pn + pa * W;
                                else if (po == Pyc::JUMP_BACKWARD_A
                                        || po == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A
                                        || po == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A
                                        || po == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A)
                                    pt = pn - pa * W;
                                if (pt == afterCand) preds++;
                                pp = pn;
                            }
                        }
                        if (loopExit > 0 && preds == 1) {
                            int lo2 = 0, la2 = 0; (void)opAt(afterCand, lo2, la2);
                            if (lo2 == Pyc::LOAD_FAST_A)
                                rval = new ASTName(code->getLocal(la2));
                            else if (lo2 == Pyc::LOAD_NAME_A)
                                rval = new ASTName(code->getName(la2));
                            else if (lo2 == Pyc::LOAD_DEREF_A || lo2 == Pyc::LOAD_CLASSDEREF_A)
                                rval = new ASTName(code->getCellVar(mod, la2));
                            else if (lo2 == Pyc::LOAD_GLOBAL_A)
                                rval = new ASTName(code->getName(la2 >> 1));
                            else if (lo2 == Pyc::LOAD_CONST_A)
                                rval = new ASTObject(code->getConst(la2));
                            if (rval != nullptr) {
                                finTryReturnRet[cs] = new ASTReturn(rval);
                                teoElseAbsorbSuppress.insert(afterCand);
                                after = loopExit;
                            }
                        }
                    }
                    if (retOff >= 0) retExits.push_back({ retOff, T });
                }
            }
            {
                bool finCoalesceable = true;
                { int o, a, p = finBodyStart; const int W = (int)sizeof(uint16_t);
                  while (p < finBodyEnd) {
                      int np = opAt(p, o, a);
                      bool fwdBr = (o == Pyc::POP_JUMP_FORWARD_IF_TRUE_A || o == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                              || o == Pyc::POP_JUMP_FORWARD_IF_NONE_A || o == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A);
                      if (fwdBr) {
                          if (np + a * W < finBodyEnd - 2 * W) { finCoalesceable = false; break; }
                      } else if (o == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A || o == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A
                              || o == Pyc::POP_JUMP_BACKWARD_IF_NONE_A || o == Pyc::POP_JUMP_BACKWARD_IF_NOT_NONE_A
                              || o == Pyc::POP_JUMP_IF_TRUE_A || o == Pyc::POP_JUMP_IF_FALSE_A
                              || o == Pyc::JUMP_IF_TRUE_OR_POP_A || o == Pyc::JUMP_IF_FALSE_OR_POP_A) {
                          finCoalesceable = false; break;
                      }
                      if (np <= p) break;
                      p = np;
                  } }
                if (finCoalesceable && finLen >= 6) {
                    std::set<int> seen3;
                    for (const auto& c : copies) seen3.insert(c.first);
                    int lo, la, lp = S;
                    while (lp < T) {
                        int lnp = opAt(lp, lo, la);
                        if (lnp <= lp) break;
                        if (!seen3.count(lp)) {
                            int skip = 0, afterCand = after, retOff = -1;
                            bool isFwd = false;
                            int r = matchCopy(lp, skip, afterCand, isFwd, retOff);
                            bool loopCtrl = false;
                            if (r == 1 && retOff < 0 && !isFwd) {
                                int to = 0, ta = 0; int tn = opAt(lp + skip, to, ta);
                                if (to == Pyc::JUMP_BACKWARD_A
                                        || to == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A)
                                    loopCtrl = true;
                                else if (to == Pyc::POP_TOP) {
                                    int o2 = 0, a2 = 0; int bn = opAt(tn, o2, a2);
                                    if (o2 == Pyc::JUMP_FORWARD_A) {
                                        int bt = bn + a2 * (int)sizeof(uint16_t);
                                        if (bt > T) {
                                            if (forIterExits200.count(bt)) {
                                                loopCtrl = true;
                                                loopFinBreakExit.insert(bt);
                                            } else if (whileExit200.count(bt)
                                                    && whileExit200[bt] <= S) {
                                                loopCtrl = true;
                                                loopFinBreakExit.insert(bt);
                                            } else {
                                                int bestE = -1, bestStart = -1;
                                                for (auto& fe : forIterExit2Start200) {
                                                    int E = fe.first, st = fe.second;
                                                    if (E > T && E < bt && st < S
                                                            && (bestE < 0 || E > bestE)) {
                                                        bestE = E; bestStart = st;
                                                    }
                                                }
                                                if (bestE > 0) {
                                                    loopCtrl = true;
                                                    forElseFinBreak[bestStart] = { bestE, bt };
                                                } else {
                                                    loopBreakOverHandler = true;
                                                }
                                            }
                                        } else loopCtrl = true;
                                    }
                                }
                            }
                            if (r == 1 && (retOff >= 0 || isFwd || loopCtrl)) {
                                seen3.insert(lp);
                                copies.push_back({ lp, skip });
                                if (isFwd) after = afterCand;
                                if (retOff >= 0) retExits.push_back({ retOff, T });
                                if (loopCtrl) loopCtrlCopy = true;
                                lp += skip;
                                continue;
                            }
                        }
                        lp = lnp;
                    }
                }
            }
            if (copies.empty() && splitTableFin) {
                int finNormStart = 0;
                for (const PycExceptionTableEntry* e : P)
                    finNormStart = std::max(finNormStart, e->end_offset);
                if (finNormStart > S && finNormStart < T) {
                    int cs = finNormStart;
                    { int o, a; int np = opAt(cs, o, a);
                      if (o == Pyc::NOP && np > cs) cs = np; }
                    int skip = 0, afterCand = after, retOff = -1;
                    bool isFwd = false;
                    int r = matchCopy(cs, skip, afterCand, isFwd, retOff);
                    if (r == 1 && (retOff >= 0 || isFwd)) {
                        copies.push_back({ cs, skip });
                        if (isFwd) after = afterCand;
                        if (retOff >= 0) retExits.push_back({ retOff, T });
                    }
                }
            }
            {
                std::set<int> seenLc;
                for (const auto& c : copies) seenLc.insert(c.first);
                int lo, la, lp = S;
                while (lp < T) {
                    int lnp = opAt(lp, lo, la);
                    if (lnp <= lp) break;
                    if (!seenLc.count(lp)) {
                        int skip = 0, afterCand = after, retOff = -1;
                        bool isFwd = false;
                        int r = matchCopy(lp, skip, afterCand, isFwd, retOff);
                        if (r == 1 && retOff < 0 && !isFwd) {
                            int to = 0, ta = 0; int tn = opAt(lp + skip, to, ta);
                            bool lc = false;
                            if (to == Pyc::JUMP_BACKWARD_A
                                    || to == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A) {
                                lc = true;
                            } else if (to == Pyc::JUMP_FORWARD_A) {
                                int bt = tn + ta * (int)sizeof(uint16_t);
                                if (bt > T && whileExit200.count(bt)
                                        && whileExit200[bt] <= S) {
                                    lc = true;
                                    loopFinBreakExit.insert(bt);
                                }
                            } else if (to == Pyc::POP_TOP) {
                                int o2 = 0, a2 = 0; int bn = opAt(tn, o2, a2);
                                if (o2 == Pyc::JUMP_FORWARD_A) {
                                    int bt = bn + a2 * (int)sizeof(uint16_t);
                                    if (bt > T && forIterExits200.count(bt)) {
                                        lc = true;
                                        loopFinBreakExit.insert(bt);
                                    } else if (bt > T && whileExit200.count(bt)
                                            && whileExit200[bt] <= S) {
                                        lc = true;
                                        loopFinBreakExit.insert(bt);
                                    } else if (bt > T) {
                                        int bestE = -1, bestStart = -1;
                                        for (auto& fe : forIterExit2Start200) {
                                            int E = fe.first, st = fe.second;
                                            if (E > T && E < bt && st < S
                                                    && (bestE < 0 || E > bestE)) {
                                                bestE = E; bestStart = st;
                                            }
                                        }
                                        if (bestE > 0) {
                                            lc = true;
                                            forElseFinBreak[bestStart] = { bestE, bt };
                                        } else {
                                            loopBreakOverHandler = true;
                                        }
                                    }
                                }
                            }
                            if (lc) {
                                seenLc.insert(lp);
                                copies.push_back({ lp, skip });
                                loopCtrlCopy = true;
                                lp += skip;
                                continue;
                            }
                        }
                    }
                    lp = lnp;
                }
            }
            if (copies.empty())
                continue;
            if (loopBreakOverHandler)
                continue;
            bool p1wrap = false;
            if (P.size() < 2) {
                for (const auto& e : exception_entries) {
                    if (e.push_lasti || e.target == T
                            || e.start_offset < S || e.end_offset > T
                            || e.start_offset >= e.end_offset)
                        continue;
                    bool innerExcept = false;
                    { int o, a, p = e.target;
                      int np0 = opAt(p, o, a);
                      if (o == Pyc::PUSH_EXC_INFO) {
                          int o2, a2; opAt(np0, o2, a2);
                          if (o2 == Pyc::POP_TOP) innerExcept = true;
                      }
                      while (!innerExcept && p < (int)codeBuf->length()) {
                          int np = opAt(p, o, a);
                          if (o == Pyc::CHECK_EXC_MATCH) { innerExcept = true; break; }
                          if (o == Pyc::RERAISE || o == Pyc::RERAISE_A) break;
                          if (np <= p) break;
                          p = np;
                      } }
                    if (innerExcept) { p1wrap = true; break; }
                }
                bool finBodyWith = false;
                { int o, a, p = finBodyStart;
                  while (p < finBodyEnd) {
                      int np = opAt(p, o, a);
                      if (o == Pyc::BEFORE_WITH || o == Pyc::SETUP_WITH_A) { finBodyWith = true; break; }
                      if (np <= p) break;
                      p = np;
                  } }
                if (!p1wrap && !loopCtrlCopy && !finBodyWith) continue;
            }
            FinallyCoalesce plan = { T, finBodyStart, finBodyEnd, after, 0 };
            if (!finHasPlan(S, T)) finallyOpenAt[S].push_back(plan);
            finallyPlanByTarget[T] = plan;
            if (p1wrap) finallyP1Wrap.insert(T);
            for (auto& c : copies) finallyCopySkip[c.first] = c.second;
            for (auto& r : retExits) finallyExceptReturnExit[r.first] = r.second;
        }

        /* ===== SHIP-263: nested loop-bearing try/finally coalesce =====
           Detect the EXACT asyncio Condition.wait shape — an outer try/finally
           whose finally body bears a loop and wraps an inner try/finally, the
           inner try's `return` deferred past BOTH finally normal copies — and
           emit NORMAL-copy coalesce plans for both levels (NO finallyPlanByTarget,
           so the finally renders from the inlined normal copy in place). Kept a
           SEPARATE tightly-gated pre-scan rather than relaxing the global handler
           classifier: a broad skip-nested-for-loops allowance regresses
           multiprocessing/resource_tracker. */
        for (auto& okv : prot) {
            int To = okv.first;                                  // candidate OUTER finally handler
            std::vector<const PycExceptionTableEntry*>& Po = okv.second;
            if (Po.empty()) continue;
            int oop = 0, oarg = 0;
            int afterPushO = opAt(To, oop, oarg);
            if (oop != Pyc::PUSH_EXC_INFO) continue;
            auto lo = lastiCleanup.find(To);
            if (lo == lastiCleanup.end()) continue;
            int Xo = lo->second;                                 // outer handler lasti cleanup
            if (Xo <= afterPushO) continue;

            /* (a) the outer handler body [afterPushO, Xo) is LOOP-BEARING. */
            bool loopBearing = false;
            { int o, a, p = afterPushO;
              while (p < Xo) {
                  int np = opAt(p, o, a);
                  if (o == Pyc::FOR_ITER_A || o == Pyc::JUMP_BACKWARD_A) { loopBearing = true; break; }
                  if (np <= p) break;
                  p = np;
              } }
            if (!loopBearing) continue;

            /* (b) skipping nested handler regions, the outer handler is a genuine
               FINALLY (reaches its own RERAISE before any un-nested CHECK_EXC_MATCH). */
            bool oIsExcept = false; int oRer = -1;
            { int o, a, p = afterPushO;
              while (p < Xo) {
                  int np = opAt(p, o, a);
                  if (o == Pyc::PUSH_EXC_INFO) {
                      auto nl = lastiCleanup.find(p);
                      if (nl != lastiCleanup.end()) {
                          int co, ca, cp = nl->second, skipTo = -1;
                          while (cp < Xo) {
                              int cn = opAt(cp, co, ca);
                              if (co == Pyc::RERAISE || co == Pyc::RERAISE_A) { skipTo = cn; break; }
                              if (cn <= cp) break;
                              cp = cn;
                          }
                          if (skipTo > p) { p = skipTo; continue; }
                      }
                  }
                  if (o == Pyc::CHECK_EXC_MATCH) { oIsExcept = true; break; }
                  if (o == Pyc::RERAISE || o == Pyc::RERAISE_A) { oRer = p; break; }
                  if (np <= p) break;
                  p = np;
              } }
            if (oIsExcept || oRer < 0) continue;

            /* (c) find the NESTED inner finally Ti: a (non-lasti) entry whose target
               is a non-loop finally handler whose lasti cleanup chains back into To
               (an entry starting at Xi targeting To). */
            int Ti = -1, innerBodyStart = -1, innerBodyEnd = -1;
            for (const auto& e : exception_entries) {
                if (e.push_lasti) continue;
                int cand = e.target;
                if (cand == To) continue;
                int io, ia; opAt(cand, io, ia);
                if (io != Pyc::PUSH_EXC_INFO) continue;
                auto li = lastiCleanup.find(cand);
                if (li == lastiCleanup.end()) continue;
                int Xi = li->second;
                if (Xi <= cand) continue;
                bool iIsExcept = false, iLoop = false; int iRer = -1;
                { int o, a, p = cand;
                  while (p < Xi) {
                      int np = opAt(p, o, a);
                      if (o == Pyc::FOR_ITER_A || o == Pyc::JUMP_BACKWARD_A) { iLoop = true; break; }
                      if (o == Pyc::CHECK_EXC_MATCH) { iIsExcept = true; break; }
                      if (o == Pyc::RERAISE || o == Pyc::RERAISE_A) { iRer = p; break; }
                      if (np <= p) break;
                      p = np;
                  } }
                if (iIsExcept || iLoop || iRer < 0) continue;
                bool chains = false;
                for (const auto& e2 : exception_entries)
                    if (!e2.push_lasti && e2.start_offset == Xi && e2.target == To) { chains = true; break; }
                if (!chains) continue;
                Ti = cand; innerBodyStart = e.start_offset; innerBodyEnd = e.end_offset;
                break;
            }
            if (Ti < 0) continue;

            /* (d) the inner finally normal copy = the prot[To] fragment that begins
               at/after the inner try body end; the prefix frags (if any) are the
               outer try body before the inner try. */
            int innerFinStart = -1, innerFinEnd = -1;
            int outerBodyStart = innerBodyStart;
            for (const PycExceptionTableEntry* e : Po) {
                outerBodyStart = std::min(outerBodyStart, e->start_offset);
                if (e->start_offset >= innerBodyEnd
                        && (innerFinStart < 0 || e->start_offset < innerFinStart)) {
                    innerFinStart = e->start_offset; innerFinEnd = e->end_offset;
                }
            }
            if (innerFinStart < 0) continue;
            int outerFinStart = innerFinEnd;                     // outer finally normal copy start

            /* (e) the deferred return sits in [outerFinStart, Ti): the RETURN closest
               to Ti. A bare RETURN_VALUE returns an inherited stack value (reuse
               finallyReturnExit at the inner try close); LOAD_CONST + RETURN_VALUE is
               a const return to synthesize into the inner try. */
            int returnStmt = -1, retConst = -1, retValOff = -1;
            { int o, a, p = outerFinStart, prevStart = -1, prevOp = -1, prevArg = -1;
              while (p < Ti) {
                  int np = opAt(p, o, a);
                  if (o == Pyc::RETURN_VALUE || o == Pyc::INSTRUMENTED_RETURN_VALUE_A) {
                      retValOff = p;
                      if (prevOp == Pyc::LOAD_CONST_A) { returnStmt = prevStart; retConst = prevArg; }
                      else { returnStmt = p; retConst = -1; }
                  }
                  prevStart = p; prevOp = o; prevArg = a;
                  if (np <= p) break;
                  p = np;
              } }
            if (retValOff < 0) continue;
            int outerFinEnd = returnStmt;                        // outer finally body ends before the return

            /* (f) after = past the outer handler's lasti cleanup RERAISE. */
            int afterO = -1;
            { int o, a, p = Xo;
              while (p < (int)codeBuf->length()) {
                  int np = opAt(p, o, a);
                  if (o == Pyc::RERAISE || o == Pyc::RERAISE_A) { afterO = np; break; }
                  if (np <= p) break;
                  p = np;
              } }
            if (afterO <= outerFinEnd) continue;

            /* ordered, contiguous layout sanity. */
            if (!(outerBodyStart <= innerBodyStart && innerBodyStart < innerBodyEnd
                    && innerBodyEnd <= innerFinStart && innerFinStart < innerFinEnd
                    && innerFinEnd == outerFinStart && outerFinStart < outerFinEnd
                    && outerFinEnd <= retValOff && retValOff < Ti && Ti < To))
                continue;
            if (finHasPlan(outerBodyStart, To))
                continue;

            /* The standard coalesce pre-scan above already registered a plan for the
               INNER finally — but it modelled the inner try body as [S, Ti), wrongly
               absorbing the OUTER finally normal copy + the deferred return into the
               inner try and rendering the inner finally from its exception copy. For
               the nested-loop shape that geometry is wrong; OVERRIDE it with a
               normal-copy plan whose inner try body ends at innerFinStart and whose
               finally is the inlined normal copy. Clear the stale registrations. */
            finallyPlanByTarget.erase(Ti);
            finallyCopySkip.erase(innerFinStart);
            {
                auto it = finallyOpenAt.find(innerBodyStart);
                if (it != finallyOpenAt.end()) {
                    std::vector<FinallyCoalesce> keep;
                    for (const auto& p : it->second)
                        if (p.target != Ti) keep.push_back(p);
                    it->second = keep;
                }
            }
            /* The outer loop-bearing finally was classified as an `except` by the
               handler classifier (it reads the inner `except`'s CHECK_EXC_MATCH), so
               the try/except body-hoist pre-scan registered a spurious exceptOpenAt
               for To. Drop it — the outer coalesce plan opens the try instead. */
            {
                std::vector<int> drop;
                for (const auto& kv : exceptOpenAt)
                    if (kv.second == To) drop.push_back(kv.first);
                for (int s : drop) exceptOpenAt.erase(s);
                for (auto& kv : exceptOpenAtExtra) {
                    std::vector<int> kept;
                    for (int v : kv.second) if (v != To) kept.push_back(v);
                    kv.second = kept;
                }
            }

            /* emit the two NORMAL-copy coalesce plans (no finallyPlanByTarget). */
            FinallyCoalesce outerPlan = { To, outerFinStart, outerFinEnd, afterO, outerFinStart };
            FinallyCoalesce innerPlan = { Ti, innerFinStart, innerFinEnd, outerFinStart, innerFinStart };
            finallyOpenAt[outerBodyStart].push_back(outerPlan);
            finallyOpenAt[innerBodyStart].push_back(innerPlan);
            /* hoist the deferred return into the inner try at its close. */
            if (retConst >= 0) nestedFinReturnConst[innerFinStart] = retConst;
            else finallyReturnExit[innerFinStart] = retValOff;
        }

        /* ===== try/except/FINALLY combo whose finally body BRANCH-RETURNS =====
           `try: B except: H finally: <stmts; if C: return X elif D: return Y>`.
           The inner finally wraps a try/except and its body has RETURNs reached via
           branches. On the EXCEPTIONAL path each return first discards the in-flight
           exception (SWAP; POP_TOP; SWAP; POP_EXCEPT; RETURN), so the exceptional
           finally copy is NOT byte-identical to the NORMAL copy (which returns
           directly) — matchCopy can never pair them and the generic Layer-3 coalescer
           bails (`copies.empty()`), DROPPING the `finally:` (the normal copy renders as
           plain post-try code, the exceptional copy is lost). This is the deferred
           try/except/FINALLY combo the classifier's skip-validation falls back on (a
           finally whose body only re-raises is the self-contained variant handled
           elsewhere; this is the BRANCH-RETURN variant).

           Render the finally from the CLEAN NORMAL copy [E, normEnd) — a straight-line
           region ending in a JUMP_FORWARD over the dead handler — via a finallyOpenAt
           plan whose BLK_TRY ends at E; the container's BLK_FINALLY then spans
           [E, normEnd) and resumes at `after`, skipping the exceptional copy. Gated as
           tightly as possible to this exact shape. */
        for (auto& fkv : prot) {
            int T = fkv.first;
            std::vector<const PycExceptionTableEntry*>& P = fkv.second;
            if (P.size() < 2) continue;                          // split table (wraps a nested handler)
            int fop = 0, farg = 0;
            int afterPush = opAt(T, fop, farg);
            if (fop != Pyc::PUSH_EXC_INFO) continue;
            auto lit = lastiCleanup.find(T);
            if (lit == lastiCleanup.end()) continue;
            int X = lit->second;
            if (X <= afterPush) continue;
            /* This handler must be a genuine FINALLY whose fall-through re-raises: no
               un-nested CHECK_EXC_MATCH (not an except), and the last op before X is a
               RERAISE (rerOff), reached by skipping nested handler regions. */
            bool isExc = false; int rerOff = -1;
            { int o, a, p = afterPush;
              while (p < X) {
                  int np = opAt(p, o, a);
                  if (o == Pyc::PUSH_EXC_INFO) {
                      auto nl = lastiCleanup.find(p);
                      if (nl != lastiCleanup.end()) {
                          int co, ca, cp = nl->second, skipTo = -1;
                          while (cp < X) {
                              int cn = opAt(cp, co, ca);
                              if (co == Pyc::RERAISE || co == Pyc::RERAISE_A) { skipTo = cn; break; }
                              if (cn <= cp) break;
                              cp = cn;
                          }
                          if (skipTo > p) { p = skipTo; continue; }
                      }
                  }
                  if (o == Pyc::CHECK_EXC_MATCH) { isExc = true; break; }
                  if (o == Pyc::RERAISE || o == Pyc::RERAISE_A) { rerOff = p; break; }
                  if (np <= p) break;
                  p = np;
              } }
            if (isExc || rerOff <= afterPush) continue;
            /* rerOff must be the finally's OWN fall-through re-raise: its next op == X
               (the lasti cleanup). A rerOff whose next op != X sits inside a nested
               construct — not this shape. */
            { int xo, xa; if (opAt(rerOff, xo, xa) != X) continue; }
            /* The exceptional finally body [afterPush, rerOff) must BRANCH-RETURN: at
               least one RETURN_VALUE preceded by the exception-discard dance
               SWAP;POP_TOP;SWAP;POP_EXCEPT (this is exactly why the copies diverge and
               the generic coalescer bailed). Also require a forward conditional (the
               branch). No loops / nested finally in the body. */
            bool hasBranchReturn = false, hasCond = false, bad = false;
            { int o, a, p = afterPush;
              int h4 = -1, h3 = -1, h2 = -1, h1 = -1;
              while (p < rerOff) {
                  int np = opAt(p, o, a);
                  if (o == Pyc::FOR_ITER_A || o == Pyc::JUMP_BACKWARD_A
                          || o == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A
                          || o == Pyc::SETUP_FINALLY_A) { bad = true; break; }
                  if (o == Pyc::POP_JUMP_FORWARD_IF_FALSE_A || o == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                          || o == Pyc::POP_JUMP_FORWARD_IF_NONE_A || o == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A)
                      hasCond = true;
                  if ((o == Pyc::RETURN_VALUE || o == Pyc::INSTRUMENTED_RETURN_VALUE_A)
                          && h1 == Pyc::POP_EXCEPT && h2 == Pyc::SWAP_A
                          && h3 == Pyc::POP_TOP && h4 == Pyc::SWAP_A)
                      hasBranchReturn = true;
                  h4 = h3; h3 = h2; h2 = h1; h1 = o;
                  if (np <= p) break;
                  p = np;
              } }
            if (bad || !hasBranchReturn || !hasCond) continue;
            /* Inner-handler wrap check: an entry inside [S, T) whose target is a
               nested handler (the inner try's except/finally) — the split table's
               reason for existing. S = min P-fragment start, extended back through
               abutting / inside-targeted entries (the inner try body). */
            int S = T;
            for (const PycExceptionTableEntry* e : P) S = std::min(S, e->start_offset);
            bool ext = true;
            while (ext) {
                ext = false;
                for (const auto& e : exception_entries) {
                    if (e.start_offset < S
                            && (e.end_offset == S
                                || (e.target >= S && e.target < T))) {
                        S = e.start_offset; ext = true;
                    }
                }
            }
            bool wrapsHandler = false;
            for (const auto& e : exception_entries) {
                if (e.push_lasti) continue;
                if (e.start_offset >= S && e.end_offset <= T
                        && e.target > S && e.target < T && e.target != T) {
                    wrapsHandler = true; break;
                }
            }
            if (!wrapsHandler) continue;
            /* Normal copy: E = largest P-fragment end (where the try-body's normal-exit
               lands), skipping a leading NOP. normEnd = a JUMP_FORWARD ending exactly at
               T whose target `after` is PAST T (the finally's normal-exit over the dead
               handler). The region [E, normEnd) must be loop/handler-free (a clean
               straight-line-with-branches finally body). */
            int E = 0;
            for (const PycExceptionTableEntry* e : P) E = std::max(E, e->end_offset);
            if (E <= S || E >= T) continue;
            { int o, a; int np = opAt(E, o, a); if (o == Pyc::NOP && np > E) E = np; }
            int normEnd = -1, after = -1;
            { int o, a, p = (T - 8 > E) ? T - 8 : E; const int W = (int)sizeof(uint16_t);
              while (p < T) {
                  int np = opAt(p, o, a);
                  if (np == T && o == Pyc::JUMP_FORWARD_A) { normEnd = p; after = np + a * W; }
                  if (np <= p) break;
                  p = np;
              } }
            if (normEnd <= E || after <= T) continue;
            bool cleanNorm = true;
            { int o, a, p = E; const int W = (int)sizeof(uint16_t);
              while (p < normEnd) {
                  int np = opAt(p, o, a);
                  if (o == Pyc::PUSH_EXC_INFO || o == Pyc::CHECK_EXC_MATCH
                          || o == Pyc::SETUP_FINALLY_A || o == Pyc::FOR_ITER_A
                          || o == Pyc::JUMP_BACKWARD_A
                          || o == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A) { cleanNorm = false; break; }
                  bool fwd = (o == Pyc::POP_JUMP_FORWARD_IF_FALSE_A || o == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                          || o == Pyc::POP_JUMP_FORWARD_IF_NONE_A || o == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A
                          || o == Pyc::JUMP_FORWARD_A);
                  if (fwd) { int bt = np + a * W; if (bt < E || bt > after) { cleanNorm = false; break; } }
                  if (np <= p) break;
                  p = np;
              } }
            if (!cleanNorm) continue;
            if (finHasPlan(S, T)) continue;
            /* p1wrap-style plan rendering the finally from the NORMAL copy [E, normEnd):
               the BLK_TRY spans [S, T) wrapping the inner try/except; the normal copy
               is SKIPPED in the try body (finallyCopySkip) and re-rendered as the
               BLK_FINALLY via finallyPlanByTarget (which jumps source to finBodyStart=E).
               finallyP1Wrap drains the dangling inner except at T so the finally-try is
               exposed. Unlike the generic p1wrap this needs no byte-identical
               exceptional copy — the finally renders straight from the clean normal
               copy, resuming at `after`. */
            FinallyCoalesce plan = { T, E, normEnd, after, 0 };
            finallyOpenAt[S].push_back(plan);
            finallyPlanByTarget[T] = plan;
            finallyP1Wrap.insert(T);
            finallyCopySkip[E] = T - E;
        }

        /* ===== try/except/FINALLY combo whose finally body BRANCH-BREAKS inside a
           rotated `while` =====  (the break-escaping analogue of the branch-return
           coalescer above.)  `while not C: try: … <inner breaks> except A: … except
           B: … finally: <stmts; if …: … elif not C: if …: … else: break>`.
           The keepalive `finally:` is inlined by the compiler at every loop-body
           break and at the normal fall-through + the exceptional PUSH_EXC_INFO copy.
           The finally BODY itself contains an `else: break` -> `POP_TOP; POP_EXCEPT;
           JUMP_FORWARD -> L` that ESCAPES the finally span to the loop exit L.  pycdc
           mis-processes the inlined copies, DRAINING the loop-guard BLK_IF so the
           rotated-while bottom test can't convert -> bail.

           This handler is a genuine FINALLY (rerOff fall-through re-raise, next==X),
           its EXCEPTIONAL body [afterPush, rerOff) contains a break-escape JUMP_FORWARD
           to a while-loop exit L that is OUTSIDE [afterPush, X) and whose loop wraps the
           whole try (whileExit <= S), and the NORMAL copy [E, normEnd) ends in a
           JUMP_FORWARD -> B (the loop bottom-test, B < L) while ALSO containing its own
           break-escape JUMP_FORWARD -> L.  Render the finally ONCE from the clean NORMAL
           copy resuming at B (NOT past the handler) so the guard BLK_IF survives on the
           block stack and the rotated-while `BLK_IF->BLK_WHILE` conversion fires.
           Gated as tightly as possible to this exact shape. */
        /* Rotated-while exits: a CONDITIONAL back-edge (`POP_JUMP_BACKWARD_IF_*`)
           re-tests the loop guard at the bottom; its exit is the next offset and its
           target is the loop top.  (The global loop-break-finally map only tracks
           UNCONDITIONAL JUMP_BACKWARD.)  Built locally so that map is untouched. */
        std::unordered_map<int,int> rotWhileExit;              // exit -> loop top
        {
            PycBuffer scan(codeBuf->value(), codeBuf->length());
            int sop, sarg, spos = 0;
            while (!scan.atEof()) {
                int ioff = spos;
                bc_next(scan, mod, sop, sarg, spos);
                if (sop == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A
                        || sop == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A) {
                    int tgt = spos - sarg * (int)sizeof(uint16_t);
                    if (tgt < ioff) rotWhileExit[spos] = tgt;
                }
            }
        }
        for (auto& fkv : prot) {
            int T = fkv.first;
            std::vector<const PycExceptionTableEntry*>& P = fkv.second;
            if (P.empty()) continue;
            int fop = 0, farg = 0;
            int afterPush = opAt(T, fop, farg);
            if (fop != Pyc::PUSH_EXC_INFO) continue;
            auto lit = lastiCleanup.find(T);
            if (lit == lastiCleanup.end()) continue;
            int X = lit->second;
            if (X <= afterPush) continue;
            if (finallyPlanByTarget.count(T)) continue;   // already handled above
            /* genuine FINALLY: no un-nested CHECK_EXC_MATCH; fall-through RERAISE
               (rerOff) whose next op == X. */
            bool isExc = false; int rerOff = -1;
            { int o, a, p = afterPush;
              while (p < X) {
                  int np = opAt(p, o, a);
                  if (o == Pyc::PUSH_EXC_INFO) {
                      auto nl = lastiCleanup.find(p);
                      if (nl != lastiCleanup.end()) {
                          int co, ca, cp = nl->second, skipTo = -1;
                          while (cp < X) {
                              int cn = opAt(cp, co, ca);
                              if (co == Pyc::RERAISE || co == Pyc::RERAISE_A) { skipTo = cn; break; }
                              if (cn <= cp) break;
                              cp = cn;
                          }
                          if (skipTo > p) { p = skipTo; continue; }
                      }
                  }
                  if (o == Pyc::CHECK_EXC_MATCH) { isExc = true; break; }
                  if (o == Pyc::RERAISE || o == Pyc::RERAISE_A) { rerOff = p; break; }
                  if (np <= p) break;
                  p = np;
              } }
            if (isExc || rerOff <= afterPush) continue;
            { int xo, xa; if (opAt(rerOff, xo, xa) != X) continue; }
            /* EXCEPTIONAL body [afterPush, rerOff) must BRANCH-BREAK: a `POP_TOP;
               POP_EXCEPT; JUMP_FORWARD -> L` where L is a while-loop EXIT past X and
               OUTSIDE [afterPush, X), whose loop wraps the whole try (top <= S below).
               Also require a forward conditional (the branch).  No loops / nested
               finally in the body. */
            const int W = (int)sizeof(uint16_t);
            int loopExitL = -1;
            bool hasBreakEsc = false, hasCondE = false, badE = false;
            { int o, a, p = afterPush;
              int h2 = -1, h1 = -1;
              while (p < rerOff) {
                  int np = opAt(p, o, a);
                  if (o == Pyc::FOR_ITER_A || o == Pyc::JUMP_BACKWARD_A
                          || o == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A
                          || o == Pyc::SETUP_FINALLY_A) { badE = true; break; }
                  if (o == Pyc::POP_JUMP_FORWARD_IF_FALSE_A || o == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                          || o == Pyc::POP_JUMP_FORWARD_IF_NONE_A || o == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A)
                      hasCondE = true;
                  if (o == Pyc::JUMP_FORWARD_A
                          && h1 == Pyc::POP_EXCEPT && h2 == Pyc::POP_TOP) {
                      int bt = np + a * W;
                      if ((bt < afterPush || bt >= X) && bt > X
                              && rotWhileExit.count(bt)) {
                          hasBreakEsc = true; loopExitL = bt;
                      }
                  }
                  h2 = h1; h1 = o;
                  if (np <= p) break;
                  p = np;
              } }
            if (badE || !hasBreakEsc || !hasCondE || loopExitL < 0) continue;
            /* Inner-handler wrap + try-body start S (extended back through the inner
               try body / handler fragments), exactly as the branch-return scan. */
            int S = T;
            for (const PycExceptionTableEntry* e : P) S = std::min(S, e->start_offset);
            bool ext = true;
            while (ext) {
                ext = false;
                for (const auto& e : exception_entries) {
                    if (e.start_offset < S
                            && (e.end_offset == S
                                || (e.target >= S && e.target < T))) {
                        S = e.start_offset; ext = true;
                    }
                }
            }
            /* the escaping loop must wrap the whole try (back-edge target <= S). */
            if (rotWhileExit[loopExitL] > S) continue;
            bool wrapsHandler = false;
            for (const auto& e : exception_entries) {
                if (e.push_lasti) continue;
                if (e.start_offset >= S && e.end_offset <= T
                        && e.target > S && e.target < T && e.target != T) {
                    wrapsHandler = true; break;
                }
            }
            if (!wrapsHandler) continue;
            /* Normal copy: E = largest P-fragment end, skip a leading NOP.  normEnd =
               a JUMP_FORWARD ending exactly at T whose target B is PAST T but BEFORE
               the loop exit L (B is the loop bottom-test; resuming there keeps the
               guard BLK_IF alive).  The region [E, normEnd) is loop/handler-free and
               all forward targets are internal OR the break-escape to L. */
            int E = 0;
            for (const PycExceptionTableEntry* e : P) E = std::max(E, e->end_offset);
            if (E <= S || E >= T) continue;
            { int o, a; int np = opAt(E, o, a); if (o == Pyc::NOP && np > E) E = np; }
            int normEnd = -1, B = -1;
            { int o, a, p = (T - 8 > E) ? T - 8 : E;
              while (p < T) {
                  int np = opAt(p, o, a);
                  if (np == T && o == Pyc::JUMP_FORWARD_A) { normEnd = p; B = np + a * W; }
                  if (np <= p) break;
                  p = np;
              } }
            if (normEnd <= E || B <= T || B >= loopExitL) continue;
            bool cleanNorm = true; bool normHasBreak = false;
            { int o, a, p = E;
              while (p < normEnd) {
                  int np = opAt(p, o, a);
                  if (o == Pyc::PUSH_EXC_INFO || o == Pyc::CHECK_EXC_MATCH
                          || o == Pyc::SETUP_FINALLY_A || o == Pyc::FOR_ITER_A
                          || o == Pyc::JUMP_BACKWARD_A
                          || o == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A) { cleanNorm = false; break; }
                  bool fwd = (o == Pyc::POP_JUMP_FORWARD_IF_FALSE_A || o == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                          || o == Pyc::POP_JUMP_FORWARD_IF_NONE_A || o == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A
                          || o == Pyc::JUMP_FORWARD_A);
                  if (fwd) {
                      int bt = np + a * W;
                      if (bt == loopExitL) normHasBreak = true;       // the else:break
                      else if (bt < E || bt > B) { cleanNorm = false; break; }
                  }
                  if (np <= p) break;
                  p = np;
              } }
            if (!cleanNorm || !normHasBreak) continue;
            if (finHasPlan(S, T)) continue;
            /* Register the p1wrap-style plan: BLK_TRY [S, T) wrapping the inner
               try/except(s); the normal copy [E, normEnd) is SKIPPED in the try body
               and re-rendered as the BLK_FINALLY (finallyPlanByTarget jumps source to
               finBodyStart=E); finallyP1Wrap drains the dangling inner except at T so
               the finally-try is exposed; the finally resumes at B (the bottom test). */
            /* Collect the inlined finally copies at loop-body BREAKS: a region in the
               try body [S, E) whose opcode SEQUENCE matches the normal finally copy
               [E, normEnd) (tolerating jump-arg differences — the per-copy exits point
               at L or an intermediate copy offset) AND ALL of whose forward exits land
               at the loop exit L (the copy is a `<finally>; break`).  Each is skipped +
               replaced by a single `break` (the coalesced `finally:` renders the body
               once; the break runs it then exits).  Scan for a match beginning at each
               offset where the opcode == the normal copy's first opcode. */
            int nfo = -1, nfa; opAt(E, nfo, nfa);      // normal copy's first opcode
            { int o, a, p = S;
              while (p < E) {
                  int np = opAt(p, o, a);
                  if (np <= p) break;
                  if (o == nfo && p != E) {
                      /* try to match the full copy sequence starting at p. */
                      int c = p, n = E; bool match = true;
                      while (n < normEnd) {
                          int co, ca, no, na;
                          int cn = opAt(c, co, ca); int nn = opAt(n, no, na);
                          if (cn <= c || nn <= n || co != no) { match = false; break; }
                          bool fwd = (co == Pyc::POP_JUMP_FORWARD_IF_FALSE_A || co == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                                  || co == Pyc::POP_JUMP_FORWARD_IF_NONE_A || co == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A
                                  || co == Pyc::JUMP_FORWARD_A);
                          if (fwd) {
                              int cbt = cn + ca * W;
                              /* a copy-internal forward branch stays inside the copy;
                                 an EXIT branch must go to L (break). */
                              if (cbt < p) { match = false; break; }
                              if (cbt >= p && cbt <= c + (normEnd - E) + 4 * W) {
                                  /* internal (relative to copy) — ok */
                              } else if (cbt == loopExitL) {
                                  /* break exit — ok */
                              } else { match = false; break; }
                          }
                          c = cn; n = nn;
                      }
                      if (match) {
                          /* the copy is byte-equal to the finally body and every one
                             of its exits goes to L — a `<finally>; break`.  Extend
                             copyEnd over the trailing run of JUMP_FORWARD -> L. */
                          int copyEnd = c;
                          { int to, ta, tp = copyEnd;
                            while (tp < E) {
                                int tn = opAt(tp, to, ta);
                                if (to == Pyc::JUMP_FORWARD_A && (tn + ta * W) == loopExitL) { tp = tn; continue; }
                                break;
                            }
                            copyEnd = tp; }
                          breakEscCopySkip[p] = copyEnd - p;
                          p = copyEnd; continue;
                      }
                  }
                  p = np;
              } }
            FinallyCoalesce plan = { T, E, normEnd, B, 0 };
            finallyOpenAt[S].push_back(plan);
            finallyPlanByTarget[T] = plan;
            finallyP1Wrap.insert(T);
            finallyCopySkip[E] = T - E;
            breakEscFinBodyEnd.insert(normEnd);
        }
    }

    std::unordered_map<int, int> forStartToExit;
    std::vector<std::pair<int,int>> fwdJumps;
    std::unordered_map<int, int> opEndingBefore;
    {
        PycBuffer scan(code->code()->value(), code->code()->length());
        int sop, sarg, spos = 0;
        while (!scan.atEof()) {
            int ioff = spos;
            bc_next(scan, mod, sop, sarg, spos);
            opEndingBefore[spos] = sop;
            if (sop == Pyc::JUMP_BACKWARD_A
                    || sop == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A
                    || sop == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A
                    || sop == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A
                    || sop == Pyc::POP_JUMP_BACKWARD_IF_NONE_A
                    || sop == Pyc::POP_JUMP_BACKWARD_IF_NOT_NONE_A) {
                int sa = sarg;
                if (mod->verCompare(3, 10) >= 0)
                    sa *= sizeof(uint16_t);
                loopRanges.push_back({ spos - sa, ioff, spos });
            } else if (sop == Pyc::FOR_ITER_A
                    || sop == Pyc::INSTRUMENTED_FOR_ITER_A) {
                int ex = sarg;
                if (mod->verCompare(3, 10) >= 0)
                    ex *= sizeof(uint16_t);
                forStartToExit[ioff] = spos + ex;
            } else if (sop == Pyc::JUMP_FORWARD_A
                    || sop == Pyc::INSTRUMENTED_JUMP_FORWARD_A) {
                int fa = sarg;
                if (mod->verCompare(3, 10) >= 0)
                    fa *= sizeof(uint16_t);
                fwdJumps.push_back({ ioff, spos + fa });
            }
        }
    }
    /* Exit offsets of a rotated-`while cond:`'s CONTINUE back-edge. A rotated
       `while cond:` whose body contains an explicit `continue` emits TWO backward
       edges to the SAME loop: the `continue` is an unconditional JUMP_BACKWARD to
       the loop HEADER (re-evaluating the top guard), and the natural fall-off-end
       is a CONDITIONAL bottom re-test (POP_JUMP_BACKWARD_IF_*) to the loop BODY
       start (the guard's fall-through — the guard was already checked). The
       loopRanges entry for the continue records exit = (offset right after the
       JUMP_BACKWARD), which is where the bottom re-test lives — NOT the loop exit.
       A forward jump landing there is the body's normal completion FALLING THROUGH
       into the bottom re-test, NOT a `break`. Collect those continue-exits so the
       JUMP_FORWARD break-detection excludes them (character
       `an observed function`: the outermost of three nested `while x
       is None:` — the inner loops' normal exits fall through a staircase of bottom
       re-tests, but the outermost also has a mid-body `continue`, so its bottom
       re-test's continue-range was mis-read as the loop exit and the fall-through
       into it rendered as a spurious `break`, dropping the bottom re-test). */
    std::unordered_set<int> rotWhileContinueExit;
    {
        const int W = (int)sizeof(uint16_t);
        PycBuffer scan(code->code()->value(), code->code()->length());
        int sop, sarg, spos = 0;
        std::vector<std::pair<int,int>> uncondBacks;   // (target header, exit)
        std::vector<int> condBackTargets;              // bottom re-test targets
        while (!scan.atEof()) {
            bc_next(scan, mod, sop, sarg, spos);
            int sa = sarg * W;
            if (sop == Pyc::JUMP_BACKWARD_A
                    || sop == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A)
                uncondBacks.push_back({ spos - sa, spos });
            else if (sop == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A
                    || sop == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A
                    || sop == Pyc::POP_JUMP_BACKWARD_IF_NONE_A
                    || sop == Pyc::POP_JUMP_BACKWARD_IF_NOT_NONE_A)
                condBackTargets.push_back(spos - sa);
        }
        for (const auto& ub : uncondBacks) {
            int H = ub.first;      /* the continue's target = loop header */
            /* Find the header guard: scan forward from H over value ops to the
               first forward conditional; its fall-through is the loop body start.
               Stop at any control-flow / store op (no guard = not a rotated while). */
            PycBuffer hs(code->code()->value(), code->code()->length());
            hs.setPos(H);
            int ho, ha, hp = H, bodyStart = -1;
            while (!hs.atEof() && hp < ub.second) {
                bc_next(hs, mod, ho, ha, hp);
                if (ho == Pyc::CACHE || ho == Pyc::NOP) continue;
                if (ho == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                        || ho == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                        || ho == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                        || ho == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A) {
                    bodyStart = hp; break;
                }
                if (ho == Pyc::STORE_FAST_A || ho == Pyc::STORE_NAME_A
                        || ho == Pyc::STORE_GLOBAL_A || ho == Pyc::STORE_DEREF_A
                        || ho == Pyc::STORE_SUBSCR || ho == Pyc::STORE_ATTR_A
                        || ho == Pyc::JUMP_FORWARD_A || ho == Pyc::JUMP_BACKWARD_A
                        || ho == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A
                        || ho == Pyc::FOR_ITER_A || ho == Pyc::GET_ITER
                        || ho == Pyc::RETURN_VALUE || ho == Pyc::RAISE_VARARGS_A
                        || ho == Pyc::PUSH_EXC_INFO || ho == Pyc::SEND_A
                        || ho == Pyc::YIELD_VALUE)
                    break;
            }
            /* A conditional bottom re-test targeting that body start, sitting AFTER
               the continue's exit, confirms this is a rotated `while cond:` whose
               unconditional back-edge is a `continue`: record its exit. */
            if (bodyStart > 0) {
                for (int t : condBackTargets) {
                    if (t == bodyStart) { rotWhileContinueExit.insert(ub.second); break; }
                }
            }
        }
    }
    /* `if C: <body> else: raise` where the if-body's jump-over-else is NESTED
       (its last statement is an inner if/else whose one arm jumps to the merge,
       so the body's PHYSICAL last op is terminal — return/raise — not a direct
       jump-over). The standard else-open (which keys off a trailing
       JUMP_FORWARD) then misses it, the BLK_IF closes at its false-target, and
       the `raise` renders UNCONDITIONALLY — which, inside a `while True:` whose
       only continue path runs through this else's fall-through, makes the loop
       back-edge unreachable so it is dropped on recompile. Record E (the if's
       false-target = a RAISE_VARARGS) -> M (the merge the body jumps to, past
       the raise) so the close machinery opens a real BLK_ELSE[E, M). */
    std::unordered_map<int, int> ifElseRaise;
    {
        PycBuffer scan(code->code()->value(), code->code()->length());
        int sop, sarg, spos = 0;
        std::vector<std::pair<int,int>> condJumps;
        while (!scan.atEof()) {
            int ioff = spos;
            bc_next(scan, mod, sop, sarg, spos);
            if (sop == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                    || sop == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                    || sop == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                    || sop == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A) {
                int t = spos + sarg * (int)sizeof(uint16_t);
                if (t > ioff) condJumps.push_back({ ioff, t });
            }
        }
        for (const auto& cj : condJumps) {
            int J = cj.first, E = cj.second;
            auto pe = opEndingBefore.find(E);
            if (pe == opEndingBefore.end()) continue;
            if (pe->second != Pyc::RETURN_VALUE && pe->second != Pyc::RETURN_CONST_A
                    && pe->second != Pyc::RAISE_VARARGS_A && pe->second != Pyc::RERAISE
                    && pe->second != Pyc::RERAISE_A)
                continue;
            PycBuffer eb(code->code()->value(), code->code()->length());
            eb.setPos(E);
            int eo, ea, ep = E;
            if (eb.atEof()) continue;
            bc_next(eb, mod, eo, ea, ep);
            if (eo != Pyc::RAISE_VARARGS_A) continue;
            int M = ep;
            bool jumpsToMerge = false;
            for (const auto& fj : fwdJumps)
                if (fj.first > J && fj.first < E && fj.second == M) { jumpsToMerge = true; break; }
            if (jumpsToMerge)
                ifElseRaise[E] = M;
        }
    }
    std::set<int> forExitFallThrough;
    for (const auto& kv : forStartToExit) {
        auto it = opEndingBefore.find(kv.second);
        if (it == opEndingBefore.end())
            continue;
        int pop = it->second;
        bool terminal = (pop == Pyc::JUMP_BACKWARD_A
                || pop == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A
                || pop == Pyc::JUMP_FORWARD_A
                || pop == Pyc::INSTRUMENTED_JUMP_FORWARD_A
                || pop == Pyc::JUMP_ABSOLUTE_A
                || pop == Pyc::RETURN_VALUE
                || pop == Pyc::RETURN_CONST_A
                || pop == Pyc::RERAISE
                || pop == Pyc::RERAISE_A
                || pop == Pyc::RAISE_VARARGS_A);
        if (!terminal)
            forExitFallThrough.insert(kv.first);
    }
    std::set<int> forHasBreak;
    for (const auto& kv : forStartToExit) {
        int s = kv.first, x = kv.second;
        for (const auto& j : fwdJumps) {
            if (j.first > s && j.first < x && j.second >= x) {
                forHasBreak.insert(s);
                break;
            }
        }
    }
    std::set<int> forBreakBeyondExit;
    for (const auto& kv : forStartToExit) {
        int s = kv.first, x = kv.second;
        for (const auto& j : fwdJumps) {
            if (j.first > s && j.first < x && j.second > x) {
                forBreakBeyondExit.insert(s);
                break;
            }
        }
    }
    /* A for-loop whose body has BOTH a forward jump-break (a JUMP_FORWARD->exit
       preceded by the iterator-pop POP_TOP) AND a fall-through break (the op ending
       before the exit is a POP_TOP), plus an inner continue back-edge. The two
       breaks' circular guards each disable the other, so the construct mis-renders;
       mark this exact shape so both breaks emit `break`. */
    std::set<int> doubleBreakContLoop;
    for (const auto& kv : forStartToExit) {
        int s = kv.first, E = kv.second;
        if (!forExitFallThrough.count(s) || !forHasBreak.count(s)
                || forBreakBeyondExit.count(s))
            continue;
        auto eit = opEndingBefore.find(E);
        if (eit == opEndingBefore.end() || eit->second != Pyc::POP_TOP)
            continue;
        bool breakA = false;
        for (const auto& j : fwdJumps) {
            if (j.first > s && j.first < E && j.second == E) {
                auto pit = opEndingBefore.find(j.first);
                if (pit != opEndingBefore.end() && pit->second == Pyc::POP_TOP) {
                    breakA = true; break;
                }
            }
        }
        if (breakA)
            doubleBreakContLoop.insert(s);
    }
    std::set<int> forExitOffsets;
    for (const auto& kv : forStartToExit)
        forExitOffsets.insert(kv.second);
    std::set<int> backwardJumpOffsets;
    {
        PycBuffer scan(code->code()->value(), code->code()->length());
        int sop, sarg, spos = 0;
        while (!scan.atEof()) {
            int ioff = spos;
            bc_next(scan, mod, sop, sarg, spos);
            if (sop == Pyc::JUMP_BACKWARD_A || sop == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A)
                backwardJumpOffsets.insert(ioff);
        }
    }
    bool compPureOr = false;
    struct CompAndOrOp { int sense; bool groupEnd; bool isLast; };
    std::unordered_map<int, CompAndOrOp> compAndOrOp;
    int compAndOrKeep = -1;
    bool compAndOrTopAnd = true;
    PycRef<ASTNode> compAndOrGroup, compAndOrAcc;
    {
        const char* cn = code->name() ? code->name()->value() : "";
        bool isComp = cn && (strcmp(cn, "<listcomp>") == 0
                || strcmp(cn, "<setcomp>") == 0 || strcmp(cn, "<dictcomp>") == 0
                || strcmp(cn, "<genexpr>") == 0);
        if (isComp) {
            struct In { int op, arg, off, next; };
            std::vector<In> v;
            PycBuffer scan(code->code()->value(), code->code()->length());
            int so, sa, sp = 0;
            while (!scan.atEof()) {
                int io = sp; bc_next(scan, mod, so, sa, sp);
                if (so == Pyc::CACHE) continue;
                v.push_back({ so, sa, io, sp });
            }
            const int W = (int)sizeof(uint16_t);
            int forCount = 0, forOff = -1, addOff = -1;
            for (const auto& i : v) {
                if (i.op == Pyc::FOR_ITER_A || i.op == Pyc::INSTRUMENTED_FOR_ITER_A) {
                    ++forCount; forOff = i.off;
                } else if (addOff < 0 && (i.op == Pyc::LIST_APPEND_A || i.op == Pyc::SET_ADD_A
                        || i.op == Pyc::MAP_ADD_A || i.op == Pyc::YIELD_VALUE
                        || i.op == Pyc::YIELD_VALUE_A))
                    addOff = i.off;
            }
            auto isBackOp = [](int op) {
                return op == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A || op == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A
                    || op == Pyc::POP_JUMP_BACKWARD_IF_NONE_A || op == Pyc::POP_JUMP_BACKWARD_IF_NOT_NONE_A
                    || op == Pyc::JUMP_BACKWARD_A || op == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A; };
            auto tgt = [&](const In& i) {
                return isBackOp(i.op) ? i.next - i.arg * W : i.next + i.arg * W; };
            auto isCond = [](int op) {
                return op == Pyc::POP_JUMP_FORWARD_IF_FALSE_A || op == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                    || op == Pyc::POP_JUMP_FORWARD_IF_NONE_A || op == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A
                    || op == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A || op == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A
                    || op == Pyc::POP_JUMP_BACKWARD_IF_NONE_A || op == Pyc::POP_JUMP_BACKWARD_IF_NOT_NONE_A; };
            if (forCount >= 1 && addOff > forOff) {
                std::vector<const In*> ops;
                bool clean = true;
                for (const auto& i : v) {
                    if (i.off <= forOff || i.off >= addOff) continue;
                    if (isCond(i.op)) ops.push_back(&i);
                    else if (i.op == Pyc::JUMP_FORWARD_A || i.op == Pyc::JUMP_ABSOLUTE_A)
                        clean = false;
                }
                auto isSkip = [&](int t) {
                    if (t == forOff) return true;
                    for (const auto& i : v)
                        if (i.off == t && (i.op == Pyc::JUMP_BACKWARD_A
                                || i.op == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A) && tgt(i) == forOff)
                            return true;
                    return false;
                };
                int keep = ops.empty() ? -1 : ops.back()->next;
                auto tryShape = [&](bool topAnd) -> bool {
                    if (!(clean && ops.size() >= 2 && keep > 0)) return false;
                    std::vector<std::pair<int,bool>> recs;
                    int merge = -1;
                    bool sawNonTrivialGroup = false;
                    for (size_t k = 0; k < ops.size(); ++k) {
                        const In* o = ops[k];
                        int t = tgt(*o);
                        bool none = (o->op == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                                || o->op == Pyc::POP_JUMP_BACKWARD_IF_NONE_A);
                        bool notnone = (o->op == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A
                                || o->op == Pyc::POP_JUMP_BACKWARD_IF_NOT_NONE_A);
                        bool iffalse = (o->op == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                                || o->op == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A);
                        int jumpSense = none ? 2 : notnone ? 3 : iffalse ? 1 : 0;
                        int negSense  = none ? 3 : notnone ? 2 : iffalse ? 0 : 1;
                        bool last = (k + 1 == ops.size());
                        bool tSkip = isSkip(t), tKeep = (t == keep);
                        bool tFwd  = (t > forOff && t < keep && !tSkip);
                        bool isFinal = topAnd ? tSkip : tKeep;
                        int otherSentinel = topAnd ? keep : -2;
                        if (isFinal) {
                            recs.push_back({ topAnd ? negSense : jumpSense, true });
                            int m = o->next;
                            if (merge < 0) merge = m; else if (merge != m) return false;
                            int wantLast = topAnd ? keep : -2;
                            if (last) { if (!(merge == keep)) return false; }
                            else if (!(merge > forOff && merge < keep)) return false;
                            (void)wantLast;
                            merge = -1;
                        } else if (tKeep || tSkip || tFwd) {
                            recs.push_back({ topAnd ? jumpSense : negSense, false });
                            int m = tFwd ? t : otherSentinel;
                            if (merge < 0) merge = m; else if (merge != m) return false;
                            if (last) {
                                if (o->next != keep) return false;
                                recs.back().second = true;
                            }
                        } else return false;
                    }
                    if (recs.empty() || !recs.back().second) return false;
                    { int run = 0; for (auto& r : recs) { ++run;
                        if (r.second) { if (run >= 2) sawNonTrivialGroup = true; run = 0; } } }
                    if (!sawNonTrivialGroup) return false;
                    compAndOrTopAnd = topAnd;
                    compAndOrKeep = keep;
                    for (size_t k = 0; k < ops.size(); ++k)
                        compAndOrOp[ops[k]->off] = { recs[k].first, recs[k].second,
                                                     k + 1 == ops.size() };
                    return true;
                };
                if (!tryShape(true)) { compAndOrOp.clear(); compAndOrKeep = -1; tryShape(false); }
            }
        }
    }
    {
        const char* cn = code->name() ? code->name()->value() : "";
        bool isComp = cn && (strcmp(cn, "<listcomp>") == 0
                || strcmp(cn, "<setcomp>") == 0 || strcmp(cn, "<dictcomp>") == 0
                || strcmp(cn, "<genexpr>") == 0);
        if (isComp) {
            struct Ins { int op; int arg; int off; int next; };
            std::vector<Ins> v;
            PycBuffer scan(code->code()->value(), code->code()->length());
            int sop, sarg, spos = 0;
            while (!scan.atEof()) {
                int ioff = spos;
                bc_next(scan, mod, sop, sarg, spos);
                if (sop == Pyc::CACHE) continue;
                v.push_back({ sop, sarg, ioff, spos });
            }
            int forCount = 0, forOff = -1, addOff = -1;
            for (size_t i = 0; i < v.size(); ++i) {
                if (v[i].op == Pyc::FOR_ITER_A || v[i].op == Pyc::INSTRUMENTED_FOR_ITER_A) {
                    ++forCount; forOff = v[i].off;
                } else if (addOff < 0 && (v[i].op == Pyc::LIST_APPEND_A
                        || v[i].op == Pyc::SET_ADD_A || v[i].op == Pyc::MAP_ADD_A
                        || v[i].op == Pyc::YIELD_VALUE || v[i].op == Pyc::YIELD_VALUE_A)) {
                    addOff = v[i].off;
                }
            }
            auto isFwd = [](int op) {
                return op == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                        || op == Pyc::POP_JUMP_FORWARD_IF_FALSE_A; };
            auto isBwd = [](int op) {
                return op == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A
                        || op == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A; };
            if (forCount == 1 && addOff > forOff) {
                std::vector<const Ins*> jmps;
                bool sawOtherCond = false;
                for (const auto& ins : v) {
                    if (ins.off <= forOff || ins.off >= addOff) continue;
                    if (isFwd(ins.op) || isBwd(ins.op)) jmps.push_back(&ins);
                    else if (ins.op == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                            || ins.op == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A
                            || ins.op == Pyc::POP_JUMP_BACKWARD_IF_NONE_A
                            || ins.op == Pyc::POP_JUMP_BACKWARD_IF_NOT_NONE_A
                            || ins.op == Pyc::JUMP_FORWARD_A)
                        sawOtherCond = true;
                }
                if (!sawOtherCond && jmps.size() >= 2) {
                    const Ins* last = jmps.back();
                    int keep = last->next;
                    int loopTop = forOff;
                    bool ok = isBwd(last->op);
                    {
                        int t = last->arg; if (mod->verCompare(3, 10) >= 0) t *= sizeof(uint16_t);
                        if (last->next - t != loopTop) ok = false;
                    }
                    for (size_t i = 0; ok && i + 1 < jmps.size(); ++i) {
                        const Ins* j = jmps[i];
                        if (!isFwd(j->op)) { ok = false; break; }
                        int t = j->arg; if (mod->verCompare(3, 10) >= 0) t *= sizeof(uint16_t);
                        if (j->next + t != keep) { ok = false; break; }
                    }
                    if (ok) compPureOr = true;
                }
            }
        }
    }
    std::set<int> forStuckExit;
    {
        PycBuffer scan(code->code()->value(), code->code()->length());
        int sop, sarg, spos = 0;
        while (!scan.atEof()) {
            bc_next(scan, mod, sop, sarg, spos);
            if (forExitOffsets.count(spos)
                    && (sop == Pyc::RAISE_VARARGS_A || sop == Pyc::RETURN_VALUE))
                forStuckExit.insert(spos);
        }
    }
    for (const auto& kv : forStartToExit) {
        int fstart = kv.first, E = kv.second;
        if (forExitFallThrough.count(fstart))
            continue;
        PycBuffer sc(code->code()->value(), code->code()->length());
        sc.setPos(E);
        int so, sa, sp = E, steps = 0;
        bool terminal = false, straight = true;
        while (!sc.atEof() && steps++ < 80) {
            bc_next(sc, mod, so, sa, sp);
            if (so == Pyc::RAISE_VARARGS_A
                    || so == Pyc::RETURN_VALUE || so == Pyc::RETURN_CONST_A) {
                terminal = true; break;
            }
            if (so == Pyc::POP_JUMP_FORWARD_IF_FALSE_A || so == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                    || so == Pyc::POP_JUMP_FORWARD_IF_NONE_A || so == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A
                    || so == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A || so == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A
                    || so == Pyc::JUMP_FORWARD_A || so == Pyc::JUMP_BACKWARD_A
                    || so == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A || so == Pyc::JUMP_ABSOLUTE_A
                    || so == Pyc::JUMP_IF_TRUE_OR_POP_A || so == Pyc::JUMP_IF_FALSE_OR_POP_A
                    || so == Pyc::FOR_ITER_A || so == Pyc::SEND_A
                    || so == Pyc::SETUP_FINALLY_A || so == Pyc::SETUP_EXCEPT_A
                    || so == Pyc::PUSH_EXC_INFO) {
                straight = false; break;
            }
        }
        if (terminal && straight)
            forStuckExit.insert(E);
    }
    std::unordered_set<int> classCellRet;
    int classCellLastRet = -1;
    int classCellMerge = -1;
    {
        PycBuffer scan(code->code()->value(), code->code()->length());
        int sop, sarg, spos = 0;
        int p1 = -1, p2 = -1, p3 = -1;
        int o1 = -1, o2 = -1, o3 = -1;
        int a1 = -1;
        while (!scan.atEof()) {
            int ioff = spos;
            bc_next(scan, mod, sop, sarg, spos);
            if (sop == Pyc::RETURN_VALUE
                    && p1 == Pyc::STORE_NAME_A
                    && p2 == Pyc::COPY_A
                    && p3 == Pyc::LOAD_CLOSURE_A
                    && a1 >= 0
                    && code->getName(a1)->isEqual("__classcell__")) {
                classCellRet.insert(ioff);
                classCellLastRet = ioff;
                classCellMerge = o3;
            }
            p3 = p2; p2 = p1; p1 = sop;
            o3 = o2; o2 = o1; o1 = ioff;
            a1 = sarg;
        }
    }
    std::unordered_map<int,int> forElseMerge;
    std::set<ASTBlock*> loopElseBlocks;
    for (const auto& kv : forStartToExit) {
        int s = kv.first, E = kv.second;
        int M = -1; bool ok = true;
        for (const auto& j : fwdJumps) {
            if (j.first > s && j.first < E && j.second > E) {
                if (M < 0) M = j.second;
                else if (M != j.second) { ok = false; break; }
            }
        }
        if (ok && M > E)
            forElseMerge[E] = M;
    }
    {
        struct Guard { int fallthrough; int target; };
        std::vector<Guard> guards;
        std::vector<std::pair<int,int>> condBack;
        PycBuffer sc(code->code()->value(), code->code()->length());
        int o, a, p = 0;
        while (!sc.atEof()) {
            bc_next(sc, mod, o, a, p);
            int aa = a;
            if (mod->verCompare(3, 10) >= 0) aa *= sizeof(uint16_t);
            if (o == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                    || o == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                    || o == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                    || o == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A)
                guards.push_back({ p, p + aa });
            else if (o == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A
                    || o == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A
                    || o == Pyc::POP_JUMP_BACKWARD_IF_NONE_A
                    || o == Pyc::POP_JUMP_BACKWARD_IF_NOT_NONE_A)
                condBack.push_back({ p, p - aa });
        }
        for (const auto& cb : condBack) {
            int E = cb.first, H = cb.second;
            if (forElseMerge.count(E))
                continue;
            bool guarded = false;
            for (const auto& g : guards)
                if (g.fallthrough == H && g.target == E) { guarded = true; break; }
            if (!guarded)
                continue;
            int M = -1; bool ok = true;
            for (const auto& j : fwdJumps) {
                if (j.first >= H && j.first < E && j.second > E) {
                    if (M < 0) M = j.second;
                    else if (M != j.second) { ok = false; break; }
                }
            }
            if (ok && M > E)
                forElseMerge[E] = M;
        }
    }
    std::unordered_map<int, int> backedgeCount;
    for (const auto& lr : loopRanges)
        backedgeCount[lr.start]++;
    std::unordered_map<int, int> chainedBareExcept;
    std::set<int> forUncondBreak;
    for (const auto& kv : forStartToExit) {
        int fstart = kv.first, E = kv.second;
        if (backedgeCount.count(fstart) && backedgeCount[fstart] != 0)
            continue;
        if (!forExitFallThrough.count(fstart))
            continue;
        if (forHasBreak.count(fstart) || forBreakBeyondExit.count(fstart))
            continue;
        auto it = opEndingBefore.find(E);
        if (it != opEndingBefore.end() && it->second == Pyc::POP_TOP)
            forUncondBreak.insert(E);
    }
    std::set<int> loopCloseAtExit;
    std::unordered_map<int,int> dupForBreakJF;
    std::unordered_map<int,int> dupForBreakIfEnd;
    {
        struct II { int op, arg, off, next; };
        std::vector<II> iv;
        std::unordered_map<int,size_t> byNext;
        PycBuffer sc(code->code()->value(), code->code()->length());
        int o, a, p = 0;
        while (!sc.atEof()) { int s = p; bc_next(sc, mod, o, a, p); byNext[p] = iv.size(); iv.push_back({ o, a, s, p }); }
        auto endingAt = [&](int nextOff) -> const II* {
            auto f = byNext.find(nextOff); return f == byNext.end() ? nullptr : &iv[f->second]; };
        auto isCondJump = [&](int op) {
            return op == Pyc::POP_JUMP_FORWARD_IF_FALSE_A || op == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                || op == Pyc::POP_JUMP_IF_FALSE_A || op == Pyc::POP_JUMP_IF_TRUE_A
                || op == Pyc::POP_JUMP_FORWARD_IF_NONE_A || op == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A; };
        for (const auto& kv : forStartToExit) {
            int s = kv.first, E = kv.second;
            const II* cPop = endingAt(E);
            if (!cPop || cPop->op != Pyc::POP_TOP) continue;
            const II* jf = endingAt(cPop->off);
            if (!jf || !(jf->op == Pyc::JUMP_FORWARD_A || jf->op == Pyc::INSTRUMENTED_JUMP_FORWARD_A)) continue;
            if (jf->next + jf->arg * (int)sizeof(uint16_t) != E) continue;
            const II* aPop = endingAt(jf->off);
            if (!aPop || aPop->op != Pyc::POP_TOP) continue;
            bool innerIf = false;
            for (const auto& x : iv) {
                if (x.off < s || x.off >= jf->off) continue;
                if (isCondJump(x.op)
                        && x.next + x.arg * (int)sizeof(uint16_t) == cPop->off) {
                    innerIf = true; break;
                }
            }
            if (!innerIf) continue;
            dupForBreakJF[jf->off] = E;
            dupForBreakIfEnd[jf->off] = cPop->off;
            loopCloseAtExit.insert(E);
        }
    }
    for (int E : loopFinBreakExit)
        loopCloseAtExit.insert(E);
    std::set<int> forElseBreakLoop;
    for (const auto& kv : forStartToExit) {
        int s = kv.first, E = kv.second;
        if (!(forElseMerge.count(E) && forBreakBeyondExit.count(s)))
            continue;
        bool nested = false;
        for (const auto& k2 : forStartToExit)
            if (k2.first > s && k2.first < E) { nested = true; break; }
        if (!nested)
            for (const auto& lr : loopRanges)
                if (lr.start > s && lr.start < E && lr.end < E) { nested = true; break; }
        if (nested)
            continue;
        loopCloseAtExit.insert(E);
        forElseBreakLoop.insert(s);
    }
    /* Nested for-else "search loop" idiom:
           for x:
               for y: …; if c: break       # inner for-else
               else: continue              # inner exhaust -> continue the outer
               break                       # outer break (only on inner-break path)
           else: E                         # outer exhaust -> else clause
       The OUTER for-else has a nested loop, so the flat arming above skips it. Arm
       it when the inner loop is itself a for-else+break (forElseMerge + a break past
       its exit) whose FOR_ITER exit is a JUMP_BACKWARD straight to the OUTER header
       (that back-edge IS the inner `else: continue` == the outer's only back-edge).
       The inner-else `continue` then emits via the forElseBreakLoop gate in the
       JUMP_BACKWARD else-handler, and the loop-exit open renders the outer `else:`. */
    {
        struct JI { int op, arg, off, next; };
        std::vector<JI> jiv;
        PycBuffer sc(code->code()->value(), code->code()->length());
        int o, a, p = 0;
        while (!sc.atEof()) { int st = p; bc_next(sc, mod, o, a, p); jiv.push_back({ o, a, st, p }); }
        auto opAt = [&](int off) -> const JI* {
            for (const auto& x : jiv) if (x.off == off) return &x;
            return nullptr; };
        for (const auto& kv : forStartToExit) {
            int s = kv.first, E = kv.second;
            if (forElseBreakLoop.count(s))
                continue;
            if (!(forElseMerge.count(E) && forBreakBeyondExit.count(s)))
                continue;
            bool idiom = false;
            for (const auto& k2 : forStartToExit) {
                int s2 = k2.first, E2 = k2.second;
                if (!(s2 > s && E2 < E))
                    continue;
                if (!(forElseMerge.count(E2) && forBreakBeyondExit.count(s2)))
                    continue;
                const JI* je = opAt(E2);
                if (!je || je->op != Pyc::JUMP_BACKWARD_A)
                    continue;
                int tgt = je->next - je->arg * (int)sizeof(uint16_t);
                if (tgt != s)
                    continue;
                idiom = true;
                break;
            }
            if (!idiom)
                continue;
            loopCloseAtExit.insert(E);
            forElseBreakLoop.insert(s);
        }
    }
    for (const auto& kv : forElseFinBreak) {
        int s = kv.first, E = kv.second.first;
        loopCloseAtExit.insert(E);
        forElseBreakLoop.insert(s);
    }
    bool recoverThisExcept = false;

    std::unordered_map<int, int> retPadCanon;
    {
        PycBuffer scan(code->code()->value(), code->code()->length());
        int sop, sarg, spos = 0, prevOp = -1, prevArg = -1, prevPos = 0;
        std::unordered_map<int, int> firstPadForConst;
        while (!scan.atEof()) {
            int ioff = spos;
            bc_next(scan, mod, sop, sarg, spos);
            if (sop == Pyc::RETURN_VALUE && prevOp == Pyc::LOAD_CONST_A) {
                auto it = firstPadForConst.find(prevArg);
                if (it == firstPadForConst.end())
                    firstPadForConst[prevArg] = prevPos;
                retPadCanon[prevPos] = firstPadForConst[prevArg];
            }
            prevOp = sop; prevArg = sarg; prevPos = ioff;
        }
    }
    {
        auto tailCat = [](int op) -> int {
            if (op == Pyc::RETURN_VALUE) return 1;
            if (op == Pyc::RERAISE || op == Pyc::RERAISE_A) return 2;
            return 0;
        };
        PycBuffer scan(code->code()->value(), code->code()->length());
        int sop, sarg, spos = 0, prevCat = 0, runStart = -1;
        while (!scan.atEof()) {
            int ioff = spos;
            bc_next(scan, mod, sop, sarg, spos);
            if (sop == Pyc::CACHE) continue;
            int cat = tailCat(sop);
            if (cat != 0 && cat == prevCat && runStart >= 0) {
                retPadCanon[runStart] = runStart;
                retPadCanon[ioff] = runStart;
            } else if (cat != 0) {
                runStart = ioff;
            }
            prevCat = cat;
            if (spos <= ioff) break;
        }
    }
    auto canonTarget = [&retPadCanon](int off) {
        auto it = retPadCanon.find(off);
        return it == retPadCanon.end() ? off : it->second;
    };
    std::unordered_map<int, int> popRetPadCanon;
    {
        PycBuffer scan(code->code()->value(), code->code()->length());
        int sop, sarg, spos = 0;
        int op1 = -1, arg1 = -1, off1 = -1;
        int op2 = -1, off2 = -1;
        std::unordered_map<int, int> firstPopRetForConst;
        while (!scan.atEof()) {
            int ioff = spos;
            bc_next(scan, mod, sop, sarg, spos);
            if (sop == Pyc::RETURN_VALUE && op1 == Pyc::LOAD_CONST_A
                    && op2 == Pyc::POP_TOP) {
                auto it = firstPopRetForConst.find(arg1);
                if (it == firstPopRetForConst.end())
                    firstPopRetForConst[arg1] = off2;
                popRetPadCanon[off2] = firstPopRetForConst[arg1];
            }
            op2 = op1; off2 = off1;
            op1 = sop; arg1 = sarg; off1 = ioff;
        }
    }
    /* Canonicalize the compiler-generated `except E as v:` name-cleanup blocks.
       CPython 3.11 emits, at EACH exit path of an `except … as v:` handler
       body, an implicit cleanup copy:
           POP_EXCEPT; LOAD_CONST None; STORE_FAST v; DELETE_FAST v
       (same v). When the handler body holds an `if A and B:` chain, each
       comparison's FALSE branch jumps to its OWN separate copy of this cleanup
       block, at DIFFERENT offsets. The simple two-comparison boolean fold's
       isAnd test — `canonTarget(firstCompFalseTarget) == canonTarget(secondCompFalseTarget)`
       — then sees two DISTINCT targets and mis-reads the `A and B` as an OR,
       emitting the de Morgan residual `err.code < 400 or err.code < 500` instead
       of `err.code >= 400 and err.code < 500`
       (urllib.robotparser.RobotFileParser.read; asyncio.base_subprocess). Since
       every such copy is semantically the same handler-exit merge, map each copy
       start (keyed on the cleaned-up var v) to the first copy seen for that var,
       folding into BOTH popRetPadCanon (canonPop, used by the grouped boolean
       fold) and retPadCanon (canonTarget, used by the simple two-comparison
       isAnd test). Tightly gated to the exact 4-op byte sequence.

       NOTE: this fix is scoped to the boolean-polarity defect only. The spurious
       `return None` copies that also appear for a function-terminal `except … as
       v:` handler are NOT touched here: a REAL source `return`/`return None`
       inside such a handler compiles to a byte-IDENTICAL cleanup sequence, so the
       copies cannot be distinguished from implicit epilogues and suppressing them
       drops real returns (adversarial `if C: return 5 else: return`). */
    {
        PycBuffer scan(code->code()->value(), code->code()->length());
        const char* ebuf = code->code()->value();
        int elen = (int)code->code()->length();
        int sop, sarg, spos = 0;
        std::unordered_map<int, int> firstCleanupForVar;
        while (!scan.atEof()) {
            int ioff = spos;
            bc_next(scan, mod, sop, sarg, spos);
            if (sop != Pyc::POP_EXCEPT)
                continue;
            /* Walk POP_EXCEPT; LOAD_CONST None; STORE_FAST v; DELETE_FAST v. */
            PycBuffer ns(ebuf, elen);
            ns.setPos(ioff);
            int no, na, np = ioff;
            auto step = [&]() -> bool {
                int ip = np;
                if (ns.atEof()) return false;
                bc_next(ns, mod, no, na, np);
                return np > ip;
            };
            if (!(step() && no == Pyc::POP_EXCEPT))
                continue;
            if (!(step() && no == Pyc::LOAD_CONST_A
                    && code->getConst(na).type() == PycObject::TYPE_NONE))
                continue;
            if (!(step() && no == Pyc::STORE_FAST_A))
                continue;
            int cleanupVar = na;
            if (!(step() && no == Pyc::DELETE_FAST_A && na == cleanupVar))
                continue;
            auto it = firstCleanupForVar.find(cleanupVar);
            if (it == firstCleanupForVar.end())
                firstCleanupForVar[cleanupVar] = ioff;
            int canon = firstCleanupForVar[cleanupVar];
            popRetPadCanon[ioff] = canon;
            /* Also fold into retPadCanon (used by canonTarget) so the simple
               two-comparison `if A and B:` isAnd test — which compares each
               comparison's false-jump target via canonTarget — sees these
               distinct cleanup copies as one merge point. */
            retPadCanon[ioff] = canon;
        }
    }
    auto canonPop = [&popRetPadCanon](int off) {
        auto it = popRetPadCanon.find(off);
        return it == popRetPadCanon.end() ? off : it->second;
    };
    std::set<int> fwdJumpTargets;
    for (const auto& j : fwdJumps)
        fwdJumpTargets.insert(j.second);

    std::unordered_map<int, int> withOpen;
    std::unordered_map<int, int> withResume;
    std::unordered_map<int, int> withHandlerSkip;
    std::unordered_set<int> withInTryResume;
    std::unordered_map<int, int> withExitSkip;
    std::unordered_set<int> withHandlerTargets;
    std::unordered_set<int> asyncWithBody;
    struct AsyncForRec { int loopTop, storeAt, loopEnd; };
    std::unordered_map<int, AsyncForRec> asyncForInfo;
    if (mod->verCompare(3, 11) >= 0) {
        const auto* buf = code->code()->value();
        int len = (int)code->code()->length();
        {
            PycBuffer s(buf, len);
            int so, sa, sp = 0;
            bool pendingAW = false;
            while (!s.atEof()) {
                bc_next(s, mod, so, sa, sp);
                if (so == Pyc::BEFORE_ASYNC_WITH) { pendingAW = true; continue; }
                if (pendingAW && so == Pyc::SEND_A) {
                    int bt = sp + sa * (int)sizeof(uint16_t);
                    /* skip a leading NOP at the await landing — present when the
                       async-with is wrapped in a try/except, so the body / protected
                       region starts one op later (the STORE of `as var`). Keeps
                       asyncWithBody aligned with the withGroup body start (else the
                       BEFORE_ASYNC_WITH gate's withOpen lookup misses and bails). */
                    { PycBuffer ns(buf, len); ns.setPos(bt);
                      int no, na, np = bt;
                      if (!ns.atEof()) { bc_next(ns, mod, no, na, np);
                          if (no == Pyc::NOP) bt = np; } }
                    asyncWithBody.insert(bt);
                    pendingAW = false;
                }
            }
        }
        {
            auto opAtF = [&](int off) -> int {
                if (off < 0 || off >= len) return -1;
                PycBuffer b(buf, len); b.setPos(off);
                int o, a, p = off; bc_next(b, mod, o, a, p); return o;
            };
            PycBuffer s(buf, len);
            int so, sa, sp = 0;
            while (!s.atEof()) {
                int aioff = sp;
                bc_next(s, mod, so, sa, sp);
                if (so != Pyc::GET_AITER) continue;
                int loopTop = sp;
                PycBuffer t(buf, len); t.setPos(sp);
                int o, a, p = sp, sendOff = -1, storeAt = -1;
                for (int k = 0; k < 6 && !t.atEof(); ++k) {
                    int io = p; bc_next(t, mod, o, a, p);
                    if (o == Pyc::SEND_A) { sendOff = io; storeAt = p + a * (int)sizeof(uint16_t); break; }
                }
                if (sendOff < 0) continue;
                int loopEnd = -1;
                for (const auto& e : exception_entries)
                    if (e.start_offset <= sendOff && sendOff < e.end_offset
                            && opAtF(e.target) == Pyc::END_ASYNC_FOR) {
                        loopEnd = e.target; break;
                    }
                if (loopEnd < 0 || storeAt < 0) continue;
                asyncForInfo[aioff] = { loopTop, storeAt, loopEnd };
            }
            for (size_t i = 0; i < exception_entries.size(); ) {
                if (opAtF(exception_entries[i].target) == Pyc::END_ASYNC_FOR)
                    exception_entries.erase(exception_entries.begin() + i);
                else ++i;
            }
        }
        std::unordered_map<int, std::pair<int,int>> withGroup;
        for (const auto& e : exception_entries) {
            if (!e.push_lasti)
                continue;
            bool isWith = false;
            {
                PycBuffer hs(buf, len); hs.setPos(e.target);
                int ho, ha, hp;
                for (int i = 0; i < 4 && !hs.atEof(); ++i) {
                    bc_next(hs, mod, ho, ha, hp);
                    if (ho == Pyc::WITH_EXCEPT_START) { isWith = true; break; }
                    if (ho != Pyc::PUSH_EXC_INFO && ho != Pyc::NOP) break;
                }
            }
            if (!isWith)
                continue;
            withHandlerTargets.insert(e.target);
            auto it = withGroup.find(e.target);
            if (it == withGroup.end())
                withGroup[e.target] = { e.start_offset, e.end_offset };
            else {
                it->second.first  = std::min(it->second.first, e.start_offset);
                it->second.second = std::max(it->second.second, e.end_offset);
            }
        }
        if (getenv("PYCDB")) {
            for (auto& kv : finallyCopySkip) fprintf(stderr, "[FIN] copySkip[%d]=%d (->%d)\n", kv.first, kv.second, kv.first+kv.second);
            for (auto& kv : finallyReturnExit) fprintf(stderr, "[FIN] returnExit[%d]=%d\n", kv.first, kv.second);
            for (auto& kv : finallyExceptReturnExit) fprintf(stderr, "[FIN] exceptReturnExit[%d]=%d\n", kv.first, kv.second);
            for (auto& kv : withGroup) fprintf(stderr, "[FIN] withGroup target=%d body=[%d,%d)\n", kv.first, kv.second.first, kv.second.second);
        }
        for (const auto& g : withGroup) {
            int target = g.first;
            int bodyStart = g.second.first;
            int bodyEnd = g.second.second;
            bool thisAsync = asyncWithBody.count(bodyStart) > 0;
            int q = -1;
            if (thisAsync && bodyEnd != target) {
                PycBuffer s(buf, len); s.setPos(bodyEnd);
                int so, sa, sp = bodyEnd;
                bool sawCall = false;
                for (int i = 0; i < 30 && !s.atEof(); ++i) {
                    bc_next(s, mod, so, sa, sp);
                    if (so == Pyc::CACHE)
                        continue;
                    if (!sawCall) {
                        if (so == Pyc::CALL_A && sa == 2) sawCall = true;
                        continue;
                    }
                    if (so == Pyc::POP_TOP) { q = sp; break; }
                }
            } else {
                /* An async with whose body has no normal-exit teardown
                   (bodyEnd == target) exits only via in-body return/break/
                   continue (the inline awaited __aexit__ is already skipped).
                   Anchor the case-b resume on the first POP_TOP in the dead
                   handler, like the plain with path, with the wider scan the
                   awaited-__aexit__ wrapper needs. Skip it when the async with
                   is wrapped by an enclosing try/except whose handler follows
                   the with-handler — there the except would mis-nest in the
                   body, so bail honestly. */
                bool asyncNoExit = (thisAsync && bodyEnd == target);
                int enclHandler = -1;
                if (asyncNoExit) {
                    for (const auto& e : exception_entries) {
                        if (e.start_offset >= target || e.end_offset <= bodyStart)
                            continue;
                        if (e.target <= target || withHandlerTargets.count(e.target))
                            continue;
                        if (enclHandler < 0 || e.target < enclHandler)
                            enclHandler = e.target;
                    }
                    if (enclHandler >= 0) asyncNoExit = false;
                }
                if (thisAsync && bodyEnd == target && enclHandler >= 0) {
                    /* The async no-normal-exit with is wrapped by an enclosing
                       try/except. Open the with, resume at the enclosing except
                       handler (past the dead with-handler), and flag it in-try so
                       the with-close re-runs the loop and the except opens at the
                       try's level instead of mis-nesting in the body. */
                    withOpen[bodyStart] = bodyEnd;
                    withResume[bodyEnd] = enclHandler;
                    withInTryResume.insert(bodyEnd);
                    continue;
                }
                int budget = asyncNoExit ? 30 : 12;
                PycBuffer s(buf, len); s.setPos(bodyEnd);
                int so, sa, sp = bodyEnd;
                for (int i = 0; i < budget && !s.atEof(); ++i) {
                    bc_next(s, mod, so, sa, sp);
                    if (so == Pyc::POP_TOP) { q = sp; break; }
                }
            }
            if (q < 0)
                continue;
            if (!thisAsync) {
                PycBuffer cs2(buf, len); cs2.setPos(q);
                int o2 = 0, a2 = 0, p2 = q;
                bool gotSwap = false, gotPop = false; int popEnd = -1;
                if (!cs2.atEof()) {
                    do { bc_next(cs2, mod, o2, a2, p2); } while (o2 == Pyc::CACHE && !cs2.atEof());
                    if (o2 == Pyc::SWAP_A) {
                        gotSwap = true;
                        int o3 = 0, a3 = 0, p3 = p2;
                        do { bc_next(cs2, mod, o3, a3, p3); } while (o3 == Pyc::CACHE && !cs2.atEof());
                        if (o3 == Pyc::POP_TOP) { gotPop = true; popEnd = p3; }
                        if (gotPop) {
                            int o4 = 0, a4 = 0, p4 = p3;
                            do { bc_next(cs2, mod, o4, a4, p4); } while (o4 == Pyc::CACHE && !cs2.atEof());
                            if (o4 == Pyc::RETURN_VALUE || o4 == Pyc::INSTRUMENTED_RETURN_VALUE_A)
                                q = popEnd;
                        }
                    }
                }
                (void)gotSwap;
                {
                    int probe = q;
                    while (true) {
                        PycBuffer ts(buf, len); ts.setPos(probe);
                        int to, ta, tp = probe; int ends[7]; bool ok = true;
                        for (int k = 0; k < 7; ++k) {
                            do { if (ts.atEof()) { ok = false; break; } bc_next(ts, mod, to, ta, tp); }
                            while (to == Pyc::CACHE);
                            if (!ok) break;
                            ends[k] = tp;
                            if (k == 0 && !(to == Pyc::SWAP_A && ta == 2)) { ok = false; break; }
                            if ((k == 1 || k == 2 || k == 3)
                                    && !(to == Pyc::LOAD_CONST_A
                                         && code->getConst(ta).type() == PycObject::TYPE_NONE)) { ok = false; break; }
                            if (k == 4 && !(to == Pyc::PRECALL_A && ta == 2)) { ok = false; break; }
                            if (k == 5 && !(to == Pyc::CALL_A && ta == 2)) { ok = false; break; }
                            if (k == 6 && to != Pyc::POP_TOP) { ok = false; break; }
                        }
                        if (!ok) break;
                        probe = ends[6];
                    }
                    if (probe != q) {
                        PycBuffer rs(buf, len); int ro, ra, rp = probe;
                        rs.setPos(probe);
                        if (!rs.atEof()) {
                            do { bc_next(rs, mod, ro, ra, rp); } while (ro == Pyc::CACHE && !rs.atEof());
                            if (ro == Pyc::RETURN_VALUE || ro == Pyc::INSTRUMENTED_RETURN_VALUE_A)
                                q = probe;
                            else {
                                int skipUntil = probe, retOff = -1, budget = 0;
                                PycBuffer fs(buf, len); fs.setPos(probe);
                                int fo, fa, fp = probe;
                                while (!fs.atEof() && budget++ < 80) {
                                    int opStart = fp;
                                    bc_next(fs, mod, fo, fa, fp);
                                    if (fo == Pyc::CACHE) continue;
                                    if (opStart < skipUntil) continue;
                                    for (const auto& xe : exception_entries)
                                        if (xe.start_offset == opStart && xe.end_offset > skipUntil)
                                            skipUntil = xe.end_offset;
                                    if (opStart < skipUntil) continue;
                                    if (fo == Pyc::RETURN_VALUE
                                            || fo == Pyc::INSTRUMENTED_RETURN_VALUE_A) {
                                        retOff = opStart; break;
                                    }
                                    if (fo == Pyc::BEFORE_WITH || fo == Pyc::FOR_ITER_A
                                            || fo == Pyc::RETURN_CONST_A
                                            || fo == Pyc::JUMP_FORWARD_A
                                            || fo == Pyc::JUMP_BACKWARD_A
                                            || fo == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A)
                                        break;
                                }
                                if (retOff > probe) {
                                    finallyCopySkip[probe] = retOff - probe;
                                    q = retOff;
                                }
                            }
                        }
                    }
                }
            }
            if (!thisAsync) {
                struct RI { int op, arg, off, next; };
                std::vector<RI> rv;
                PycBuffer rs(buf, len); rs.setPos(bodyEnd);
                int ro, ra, rp = bodyEnd;
                while (rp < target && !rs.atEof()) {
                    int o0 = rp;
                    bc_next(rs, mod, ro, ra, rp);
                    if (ro == Pyc::CACHE) continue;
                    rv.push_back({ ro, ra, o0, rp });
                }
                int normStart = -1, normJumpTgt = -1;
                bool sawCondRet = false;
                std::vector<std::pair<int,int>> breakSkips;
                for (size_t i = 0; i + 5 < rv.size(); ++i) {
                    if (rv[i].op == Pyc::LOAD_CONST_A
                            && code->getConst(rv[i].arg).type() == PycObject::TYPE_NONE
                            && rv[i+1].op == Pyc::LOAD_CONST_A
                            && rv[i+2].op == Pyc::LOAD_CONST_A
                            && rv[i+3].op == Pyc::PRECALL_A && rv[i+3].arg == 2
                            && rv[i+4].op == Pyc::CALL_A && rv[i+4].arg == 2
                            && rv[i+5].op == Pyc::POP_TOP) {
                        bool hasJF = (i + 6 < rv.size()
                                && rv[i+6].op == Pyc::JUMP_FORWARD_A);
                        if (hasJF && rv[i+6].next == target) {
                            normStart = rv[i].off;
                            normJumpTgt = rv[i+6].next
                                    + rv[i+6].arg * (int)sizeof(uint16_t);
                        } else if (hasJF) {
                            breakSkips.push_back({ rv[i].off, rv[i+6].off });
                            sawCondRet = true;
                        } else {
                            sawCondRet = true;
                        }
                    }
                }
                if (normStart > 0 && normJumpTgt > 0 && sawCondRet) {
                    int ns = normStart;
                    for (const auto& r : rv)
                        if (r.next == ns && r.op == Pyc::NOP) { ns = r.off; break; }
                    withOpen[bodyStart] = ns;
                    withResume[ns] = normJumpTgt;
                    for (const auto& bs : breakSkips)
                        withExitSkip[bs.first] = bs.second;
                    continue;
                }
            }
            withOpen[bodyStart] = bodyEnd;
            PycBuffer s2(buf, len); s2.setPos(q);
            int qo, qa, qp = q;
            if (!s2.atEof()) bc_next(s2, mod, qo, qa, qp);
            if (qo == Pyc::JUMP_FORWARD_A) {
                withResume[bodyEnd] = qp + qa * (int)sizeof(uint16_t);
            } else {
                int suppressT = -1;
                {
                    PycBuffer hs(buf, len); hs.setPos(target);
                    int ho, ha, hp = target;
                    int hlim = thisAsync ? 16 : 8;
                    for (int i = 0; i < hlim && !hs.atEof(); ++i) {
                        bc_next(hs, mod, ho, ha, hp);
                        if (ho == Pyc::POP_JUMP_FORWARD_IF_TRUE_A) {
                            suppressT = hp + ha * (int)sizeof(uint16_t); break;
                        }
                    }
                }
                int postCode = -1;
                int hEnd = -1;
                if (suppressT > 0) {
                    PycBuffer hs(buf, len); hs.setPos(suppressT);
                    int ho, ha, hp = suppressT;
                    for (int i = 0; i < 24 && !hs.atEof(); ++i) {
                        int opStart = hp;
                        bc_next(hs, mod, ho, ha, hp);
                        if (ho == Pyc::POP_TOP || ho == Pyc::POP_EXCEPT)
                            continue;
                        if (ho == Pyc::LOAD_CONST_A
                                && code->getConst(ha).type() == PycObject::TYPE_NONE) {
                            PycBuffer ps(buf, len); ps.setPos(opStart);
                            int po, pa, pp = opStart; bool isExit = true;
                            int seq[6];
                            for (int k = 0; k < 6 && !ps.atEof(); ++k) {
                                bc_next(ps, mod, po, pa, pp); seq[k] = po;
                            }
                            if (seq[0]==Pyc::LOAD_CONST_A && seq[1]==Pyc::LOAD_CONST_A
                                    && seq[2]==Pyc::LOAD_CONST_A
                                    && seq[3]==Pyc::PRECALL_A && seq[4]==Pyc::CALL_A
                                    && seq[5]==Pyc::POP_TOP) {
                                hs.setPos(pp); hp = pp; continue;
                            }
                            isExit = false;
                            { PycBuffer rs(buf, len); rs.setPos(hp);
                              int ro, ra, rp = hp;
                              if (!rs.atEof()) bc_next(rs, mod, ro, ra, rp);
                              if (ro == Pyc::RETURN_VALUE) { hEnd = rp; break; } }
                            if (!isExit) { postCode = opStart; break; }
                        }
                        else if (ho == Pyc::RERAISE || ho == Pyc::RERAISE_A
                                || ho == Pyc::JUMP_FORWARD_A) { hEnd = hp; break; }
                        else if (ho == Pyc::RETURN_CONST_A
                                && code->getConst(ha).type() == PycObject::TYPE_NONE) {
                            hEnd = hp; break;
                        }
                        else { postCode = opStart; break; }
                    }
                }
                bool hasRetSwap = false;
                {
                    struct WI { int op, arg; };
                    std::vector<WI> wv;
                    PycBuffer ss(buf, len); ss.setPos(bodyEnd);
                    int sso, ssa, ssp = bodyEnd;
                    while (ssp < q && !ss.atEof()) {
                        bc_next(ss, mod, sso, ssa, ssp);
                        if (sso == Pyc::CACHE) continue;
                        wv.push_back({ sso, ssa });
                    }
                    auto isN = [&](size_t k){
                        return k < wv.size() && wv[k].op == Pyc::LOAD_CONST_A
                            && code->getConst(wv[k].arg).type() == PycObject::TYPE_NONE; };
                    for (size_t k = 0; k + 5 < wv.size(); ++k) {
                        if (wv[k].op == Pyc::SWAP_A && wv[k].arg == 2
                                && isN(k+1) && isN(k+2) && isN(k+3)
                                && wv[k+4].op == Pyc::PRECALL_A && wv[k+4].arg == 2
                                && wv[k+5].op == Pyc::CALL_A && wv[k+5].arg == 2) {
                            hasRetSwap = true; break;
                        }
                    }
                }
                bool withInTry = false;
                if (postCode <= 0 && hEnd > 0 && hasRetSwap) {
                    for (const auto& xe : exception_entries) {
                        if (xe.start_offset >= qp || xe.end_offset <= bodyEnd)
                            continue;
                        bool handlerIsWith = false;
                        PycBuffer hs(buf, len); hs.setPos(xe.target);
                        int ho, ha, hp;
                        for (int i = 0; i < 4 && !hs.atEof(); ++i) {
                            bc_next(hs, mod, ho, ha, hp);
                            if (ho == Pyc::WITH_EXCEPT_START) { handlerIsWith = true; break; }
                            if (ho != Pyc::PUSH_EXC_INFO && ho != Pyc::NOP) break;
                        }
                        if (!handlerIsWith) { withInTry = true; break; }
                    }
                }
                if (qo == Pyc::RETURN_VALUE && (postCode > 0 || (hEnd > 0 && hasRetSwap && !withInTry))) {
                    withOpen[bodyStart] = qp;
                    withResume[qp] = (postCode > 0) ? postCode : hEnd;
                    /* Only skip the dead handler separately when it starts AFTER the
                       with-body end. If it coincides with qp (handler right after the
                       return — an if/else arm whose other arm follows the handler),
                       the early handler-skip would fire before the with-close and pos
                       would skip past qp without closing the BLK_WITH, swallowing the
                       following arm; the withResume[qp]=hEnd jump already skips it. */
                    if (postCode <= 0 && hEnd > 0 && target > qp)
                        withHandlerSkip[target] = hEnd;
                } else if (qo == Pyc::RETURN_VALUE && hEnd > 0 && hasRetSwap && withInTry) {
                    withOpen[bodyStart] = qp;
                    withResume[qp] = hEnd;
                    withInTryResume.insert(qp);
                } else {
                    withResume[bodyEnd] = q;
                    if (hEnd > 0)
                        withHandlerSkip[target] = hEnd;
                    else if (postCode > 0 && !withInTry) {
                        bool constRet = false; int pastRet = -1;
                        { PycBuffer rs(buf, len); rs.setPos(q);
                          int ro, ra, rp = q;
                          if (!rs.atEof()) {
                              bc_next(rs, mod, ro, ra, rp);
                              if (ro == Pyc::RETURN_CONST_A) { constRet = true; pastRet = rp; }
                              else if (ro == Pyc::LOAD_CONST_A && !rs.atEof()) {
                                  int ro2, ra2, rp2 = rp;
                                  bc_next(rs, mod, ro2, ra2, rp2);
                                  if (ro2 == Pyc::RETURN_VALUE
                                          || ro2 == Pyc::INSTRUMENTED_RETURN_VALUE_A) {
                                      constRet = true; pastRet = rp2;
                                  }
                              }
                          } }
                        if (constRet && pastRet > 0) {
                            withOpen[bodyStart] = pastRet;
                            withResume[pastRet] = postCode;
                            withResume.erase(bodyEnd);
                        }
                    }
                }
            }
        }
    }

    if (mod->verCompare(3, 11) >= 0) {
        struct Ins { int op; int arg; int off; int next; };
        std::vector<Ins> v;
        PycBuffer scan(code->code()->value(), code->code()->length());
        int sop, sarg, spos = 0;
        while (!scan.atEof()) {
            int ioff = spos;
            bc_next(scan, mod, sop, sarg, spos);
            if (sop == Pyc::CACHE)
                continue;
            v.push_back({ sop, sarg, ioff, spos });
        }
        auto isNone = [&](int idx) {
            return v[idx].op == Pyc::LOAD_CONST_A
                    && code->getConst(v[idx].arg).type() == PycObject::TYPE_NONE;
        };
        for (size_t i = 0; i + 5 < v.size(); ++i) {
            if (!(isNone(i) && isNone(i + 1) && isNone(i + 2)
                    && v[i+3].op == Pyc::PRECALL_A && v[i+3].arg == 2
                    && v[i+4].op == Pyc::CALL_A && v[i+4].arg == 2
                    && v[i+5].op == Pyc::POP_TOP))
                continue;
            if (i + 6 < v.size() && v[i+6].op == Pyc::JUMP_FORWARD_A) {
                int jfTgt = v[i+6].next + v[i+6].arg * (int)sizeof(uint16_t);
                if (withHandlerTargets.count(jfTgt))
                    continue;
                withExitSkip[v[i].off] = v[i+6].off;
                continue;
            }
            if (i >= 1 && v[i-1].op == Pyc::SWAP_A && v[i-1].arg == 2) {
                size_t k = i + 6;
                int groups = 1;
                while (k + 6 < v.size() && v[k].op == Pyc::SWAP_A && v[k].arg == 2
                        && isNone(k+1) && isNone(k+2) && isNone(k+3)
                        && v[k+4].op == Pyc::PRECALL_A && v[k+4].arg == 2
                        && v[k+5].op == Pyc::CALL_A && v[k+5].arg == 2
                        && v[k+6].op == Pyc::POP_TOP) {
                    groups++;
                    k += 7;
                }
                if (groups >= 2 && k < v.size()
                        && (v[k].op == Pyc::RETURN_VALUE
                            || v[k].op == Pyc::INSTRUMENTED_RETURN_VALUE_A)) {
                    withExitSkip[v[i-1].off] = v[k].off;
                    continue;
                }
                if (groups >= 2 && k < v.size()
                        && finallyCopySkip.count(v[k].off)) {
                    withExitSkip[v[i-1].off] = v[k].off;
                    continue;
                }
            }
            int skipStart = v[i].off;
            int skipEnd = v[i+5].next;
            bool iterCleanup = (i + 8 < v.size()
                    && v[i+6].op == Pyc::SWAP_A && v[i+6].arg == 2
                    && v[i+7].op == Pyc::POP_TOP
                    && (v[i+8].op == Pyc::RETURN_VALUE
                        || v[i+8].op == Pyc::INSTRUMENTED_RETURN_VALUE_A));
            if (iterCleanup)
                skipEnd = v[i+8].off;
            if (i >= 1 && v[i-1].op == Pyc::SWAP_A && v[i-1].arg == 2) {
                if (iterCleanup) {
                    skipStart = v[i-1].off;
                } else {
                    for (size_t k = i + 6; k < v.size() && k - i <= 16; ++k) {
                        int o = v[k].op;
                        if (o == Pyc::RETURN_VALUE || o == Pyc::INSTRUMENTED_RETURN_VALUE_A) {
                            skipStart = v[i-1].off; break;
                        }
                        if (!(o == Pyc::LOAD_CONST_A || o == Pyc::LOAD_FAST_A
                                || o == Pyc::LOAD_GLOBAL_A || o == Pyc::LOAD_DEREF_A
                                || o == Pyc::LOAD_NAME_A || o == Pyc::LOAD_ATTR_A
                                || o == Pyc::STORE_FAST_A || o == Pyc::STORE_NAME_A
                                || o == Pyc::STORE_GLOBAL_A || o == Pyc::STORE_DEREF_A
                                || o == Pyc::STORE_ATTR_A || o == Pyc::NOP))
                            break;
                    }
                }
            }
            withExitSkip[skipStart] = skipEnd;
        }
        for (size_t i = 0; i + 5 < v.size(); ++i) {
            if (!(isNone(i) && isNone(i + 1) && isNone(i + 2)
                    && v[i+3].op == Pyc::PRECALL_A && v[i+3].arg == 2
                    && v[i+4].op == Pyc::CALL_A && v[i+4].arg == 2
                    && (v[i+5].op == Pyc::GET_AWAITABLE
                        || v[i+5].op == Pyc::GET_AWAITABLE_A)))
                continue;
            size_t j = i + 6;
            bool okShape = true;
            while (j < v.size() && v[j].op != Pyc::POP_TOP) {
                int o = v[j].op;
                if (o != Pyc::LOAD_CONST_A && o != Pyc::SEND_A
                        && o != Pyc::YIELD_VALUE && o != Pyc::YIELD_VALUE_A
                        && o != Pyc::RESUME_A
                        && o != Pyc::JUMP_BACKWARD_NO_INTERRUPT_A
                        && o != Pyc::CACHE) {
                    okShape = false; break;
                }
                if (j - i > 12) { okShape = false; break; }
                j++;
            }
            if (!okShape || j >= v.size() || v[j].op != Pyc::POP_TOP)
                continue;
            if (j + 1 < v.size() && v[j+1].op == Pyc::JUMP_FORWARD_A)
                continue;
            int skipStart = v[i].off;
            if (i >= 1 && v[i-1].op == Pyc::SWAP_A && v[i-1].arg == 2
                    && j + 1 < v.size() && v[j+1].op == Pyc::RETURN_VALUE)
                skipStart = v[i-1].off;
            withExitSkip[skipStart] = v[j].next;
        }
    }

    std::unordered_map<int, PycRef<PycString>> exceptAsBind;
    std::unordered_set<int> exceptAsSkip;
    if (mod->verCompare(3, 11) >= 0) {
        struct Ins { int op; int arg; int off; };
        std::vector<Ins> v;
        PycBuffer scan(code->code()->value(), code->code()->length());
        int sop, sarg, spos = 0;
        while (!scan.atEof()) {
            int ioff = spos;
            bc_next(scan, mod, sop, sarg, spos);
            if (sop == Pyc::CACHE)
                continue;
            v.push_back({ sop, sarg, ioff });
        }
        auto isStore = [](int op) {
            return op == Pyc::STORE_FAST_A || op == Pyc::STORE_NAME_A
                    || op == Pyc::STORE_GLOBAL_A || op == Pyc::STORE_DEREF_A;
        };
        auto storeName = [&](const Ins& in) -> PycRef<PycString> {
            switch (in.op) {
            case Pyc::STORE_FAST_A:   return code->getLocal(in.arg);
            case Pyc::STORE_NAME_A:
            case Pyc::STORE_GLOBAL_A: return code->getName(in.arg);
            case Pyc::STORE_DEREF_A:  return code->getCellVar(mod, in.arg);
            default:                  return nullptr;
            }
        };
        auto delName = [&](const Ins& in) -> PycRef<PycString> {
            switch (in.op) {
            case Pyc::DELETE_FAST_A:   return code->getLocal(in.arg);
            case Pyc::DELETE_NAME_A:
            case Pyc::DELETE_GLOBAL_A: return code->getName(in.arg);
            case Pyc::DELETE_DEREF_A:  return code->getCellVar(mod, in.arg);
            default:                   return nullptr;
            }
        };
        auto isNone = [&](const Ins& in) {
            return in.op == Pyc::LOAD_CONST_A
                    && code->getConst(in.arg).type() == PycObject::TYPE_NONE;
        };
        std::unordered_set<std::string> bindNames;
        for (size_t i = 0; i + 2 < v.size(); ++i) {
            if (v[i].op == Pyc::CHECK_EXC_MATCH
                    && (v[i+1].op == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                        || v[i+1].op == Pyc::POP_JUMP_IF_FALSE_A)
                    && isStore(v[i+2].op)) {
                PycRef<PycString> nm = storeName(v[i+2]);
                if (nm != nullptr) {
                    exceptAsBind[v[i+2].off] = nm;
                    bindNames.insert(nm->strValue());
                }
            }
        }
        if (!bindNames.empty()) {
            for (size_t i = 0; i + 2 < v.size(); ++i) {
                if (!isNone(v[i]) || !isStore(v[i+1].op))
                    continue;
                PycRef<PycString> sn = storeName(v[i+1]);
                PycRef<PycString> dn = delName(v[i+2]);
                if (sn != nullptr && dn != nullptr
                        && sn->strValue() == dn->strValue()
                        && bindNames.count(sn->strValue())) {
                    exceptAsSkip.insert(v[i].off);
                    exceptAsSkip.insert(v[i+1].off);
                    exceptAsSkip.insert(v[i+2].off);
                }
            }
        }
    }

    std::unordered_set<int> chainIfSwap;
    std::unordered_set<int> chainIfGlue;
    /* Offsets of the dup-cleanup POP_TOP a chained comparison emits at each link's
       false-landing (`SWAP;COPY;COMPARE;IF_FALSE->Lc; …; Lc: POP_TOP`). It discards
       the duplicated shared operand and is INTERNAL to the ONE chain operand, so it
       must not be treated as a statement boundary by cleanOperand's POP_TOP reject. */
    std::unordered_set<int> chainCleanupPop;
    std::unordered_map<int, int> chainIfTrue;
    std::unordered_set<int> chainIfNoneFinal;
    std::unordered_map<int, int> chainWhileBottomExit;
    std::unordered_set<int> chainAndLeadGuard;
    std::unordered_set<int> chainLeadSkip;
    std::unordered_set<int> chainWhileBwdEnd;
    if (mod->verCompare(3, 11) >= 0) {
        struct Ins { int op; int arg; int off; int next; };
        std::vector<Ins> v;
        PycBuffer scan(code->code()->value(), code->code()->length());
        int sop, sarg, spos = 0;
        while (!scan.atEof()) {
            int ioff = spos;
            bc_next(scan, mod, sop, sarg, spos);
            if (sop == Pyc::CACHE)
                continue;
            v.push_back({ sop, sarg, ioff, spos });
        }
        std::unordered_map<int, int> offToIdx;
        for (size_t k = 0; k < v.size(); ++k)
            offToIdx[v[k].off] = (int)k;
        auto isPJF = [](int op) {
            return op == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                    || op == Pyc::POP_JUMP_FORWARD_IF_TRUE_A;
        };
        auto isPJB = [](int op) {
            return op == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A
                    || op == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A;
        };
        auto isPJNone = [](int op) {
            return op == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                    || op == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A;
        };
        auto fwdTarget = [&](const Ins& in) { return in.next + in.arg * (int)sizeof(uint16_t); };
        auto bwdTarget = [&](const Ins& in) { return in.next - in.arg * (int)sizeof(uint16_t); };
        auto isCmpLink = [](int op) {
            return op == Pyc::COMPARE_OP_A || op == Pyc::IS_OP_A || op == Pyc::CONTAINS_OP_A;
        };
        for (size_t i = 0; i + 4 < v.size(); ++i) {
            if (!(v[i].op == Pyc::SWAP_A && v[i].arg == 2
                    && v[i+1].op == Pyc::COPY_A && v[i+1].arg == 2
                    && isCmpLink(v[i+2].op)
                    && isPJF(v[i+3].op)))
                continue;
            int Lc = fwdTarget(v[i+3]);
            auto lcIt = offToIdx.find(Lc);
            if (lcIt == offToIdx.end() || v[lcIt->second].op != Pyc::POP_TOP)
                continue;
            std::vector<int> swaps = { (int)i };
            std::vector<int> glues = { (int)(i + 3) };
            size_t j = i + 4;
            bool bail = false, done = false;
            int finalJ = -1;
            int finalBwd = -1;
            bool finalExitBwd = false;
            int noneFinalOff = -1;
            while (j < v.size() && !done) {
                int swapIdx = -1;
                size_t k = j;
                while (k + 1 < v.size()) {
                    /* The chain LINK's comparison is a cmp op IMMEDIATELY followed by
                       its short-circuit POP_JUMP (or a bare `is None` POP_JUMP). */
                    if (isCmpLink(v[k].op) && (isPJF(v[k+1].op) || isPJB(v[k+1].op))) break;
                    if (isPJNone(v[k].op)) break;
                    if (v[k].op == Pyc::SWAP_A && v[k].arg == 2
                            && v[k+1].op == Pyc::COPY_A && v[k+1].arg == 2) {
                        swapIdx = (int)k; k += 2; continue;
                    }
                    /* A comparison op NOT followed by a POP_JUMP is a NESTED operand
                       sub-expression (`… != (a in b)` — the final link's right side is
                       itself a containment/comparison); skip it and keep scanning for
                       the real link comparison. */
                    if (isCmpLink(v[k].op)) { ++k; continue; }
                    if (v[k].op == Pyc::SWAP_A || v[k].op == Pyc::COPY_A
                            || isPJF(v[k].op)) { bail = true; break; }
                    ++k;
                }
                if (bail) break;
                if (k < v.size() && isPJNone(v[k].op)) {
                    if (swapIdx >= 0) { bail = true; break; }
                    if (k + 1 >= v.size() || v[k+1].op != Pyc::JUMP_FORWARD_A) { bail = true; break; }
                    finalJ = (int)(k + 1);
                    noneFinalOff = v[k].off;
                    done = true;
                    break;
                }
                if (k + 1 >= v.size() || !isCmpLink(v[k].op)
                        || !(isPJF(v[k+1].op) || isPJB(v[k+1].op))) { bail = true; break; }
                bool bwdFinal = isPJB(v[k+1].op);
                int tgt = bwdFinal ? bwdTarget(v[k+1]) : fwdTarget(v[k+1]);
                if (!bwdFinal && tgt == Lc) {
                    if (swapIdx < 0) { bail = true; break; }
                    swaps.push_back(swapIdx);
                    glues.push_back((int)(k + 1));
                    j = k + 2;
                } else {
                    if (swapIdx >= 0) { bail = true; break; }
                    if (k + 2 >= v.size()) { bail = true; break; }
                    bool jf = v[k+2].op == Pyc::JUMP_FORWARD_A;
                    bool jb = bwdFinal && v[k+2].op == Pyc::JUMP_BACKWARD_A;
                    if (!jf && !jb) { bail = true; break; }
                    finalJ = (int)(k + 2);
                    if (bwdFinal)
                        finalBwd = v[k+1].off;
                    if (jb)
                        finalExitBwd = true;
                    done = true;
                }
            }
            if (bail || !done)
                continue;
            for (int s : swaps) chainIfSwap.insert(v[s].off);
            for (int g : glues) {
                chainIfGlue.insert(v[g].off);
                /* the glue link's false-target lands on the dup-cleanup POP_TOP */
                int lc = fwdTarget(v[g]);
                int co = Pyc::PYC_INVALID_OPCODE, ca = 0, cp = lc;
                PycBuffer cb2(code->code()->value(), code->code()->length());
                cb2.setPos(lc);
                if (!cb2.atEof()) {
                    bc_next(cb2, mod, co, ca, cp);
                    if (co == Pyc::POP_TOP)
                        chainCleanupPop.insert(lc);
                }
            }
            if (noneFinalOff >= 0)
                chainIfNoneFinal.insert(noneFinalOff);
            if (finalBwd >= 0 && finalExitBwd) {
                chainWhileBottomExit[finalBwd] = 0;
                chainWhileBwdEnd.insert(finalBwd);
            } else if (finalBwd >= 0) {
                int E = fwdTarget(v[finalJ]);
                chainWhileBottomExit[finalBwd] = E;
                for (int z = (int)i - 1; z >= 0; --z) {
                    int zo = v[z].op;
                    if (isPJF(zo) || isPJB(zo) || isPJNone(zo)) {
                        if (isPJF(zo) && fwdTarget(v[z]) == E) {
                            chainAndLeadGuard.insert(finalBwd);
                            chainLeadSkip.insert(v[z].off);
                        }
                        break;
                    }
                    if (zo == Pyc::JUMP_FORWARD_A || zo == Pyc::JUMP_BACKWARD_A
                            || zo == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A)
                        break;
                }
            } else
                chainIfTrue[v[finalJ].off] = fwdTarget(v[finalJ]);
        }
    }

    if (mod->verCompare(3, 11) >= 0 && !chainWhileBottomExit.empty()) {
        const PycString* cb = code->code();
        const int W = (int)sizeof(uint16_t);
        struct BJ { int off, nextoff, target; };
        std::vector<BJ> bj;
        PycBuffer scan(cb->value(), cb->length());
        int o, a, p = 0;
        while (!scan.atEof()) {
            int ioff = p;
            bc_next(scan, mod, o, a, p);
            if (o == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A
                    || o == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A
                    || o == Pyc::POP_JUMP_BACKWARD_IF_NONE_A
                    || o == Pyc::POP_JUMP_BACKWARD_IF_NOT_NONE_A)
                bj.push_back({ ioff, p, p - a * W });
        }
        std::unordered_map<int, std::vector<int>> byTarget;
        for (size_t k = 0; k < bj.size(); ++k)
            byTarget[bj[k].target].push_back((int)k);
        for (auto& kv : byTarget) {
            if (kv.second.size() < 2) continue;
            int firstBwd = bj[kv.second.front()].off;
            int lastNext = bj[kv.second.front()].nextoff;
            for (int idx : kv.second) {
                if (bj[idx].off < firstBwd) firstBwd = bj[idx].off;
                if (bj[idx].nextoff > lastNext) lastNext = bj[idx].nextoff;
            }
            auto it = chainWhileBottomExit.find(firstBwd);
            if (it == chainWhileBottomExit.end()) continue;
            it->second = lastNext;
        }
    }

    {
        const PycString* cb = code->code();
        const int W = (int)sizeof(uint16_t);
        auto opAt3 = [&](int off, int& op, int& arg) -> int {
            PycBuffer b(cb->value(), cb->length());
            b.setPos(off);
            int o = Pyc::PYC_INVALID_OPCODE, a = 0, np = off;
            if (!b.atEof()) bc_next(b, mod, o, a, np);
            op = o; arg = a; return np;
        };
        std::unordered_set<int> ternaryInternal;
        {
            int o, a, p = 0;
            while (p < (int)cb->length()) {
                int np = opAt3(p, o, a);
                if (np <= p) break;
                if (o == Pyc::POP_JUMP_FORWARD_IF_FALSE_A) {
                    int E = np + a * W;
                    if (E > np && E < (int)cb->length()) {
                        auto armBad = [](int op) {
                            return op == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                                || op == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                                || op == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                                || op == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A
                                || op == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A
                                || op == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A
                                || op == Pyc::JUMP_FORWARD_A || op == Pyc::JUMP_BACKWARD_A
                                || op == Pyc::PUSH_EXC_INFO || op == Pyc::FOR_ITER_A
                                || op == Pyc::POP_TOP || op == Pyc::RETURN_VALUE
                                || op == Pyc::RETURN_CONST_A || op == Pyc::RAISE_VARARGS_A
                                || op == Pyc::STORE_FAST_A || op == Pyc::STORE_NAME_A
                                || op == Pyc::STORE_GLOBAL_A || op == Pyc::STORE_DEREF_A
                                || op == Pyc::STORE_ATTR_A || op == Pyc::STORE_SUBSCR;
                        };
                        int jo = 0, ja = 0, jp = np, joff = -1;
                        bool armClean = true;
                        while (jp < E) {
                            joff = jp;
                            int n = opAt3(jp, jo, ja);
                            if (n <= jp) { armClean = false; break; }
                            if (n != E && armBad(jo))
                                armClean = false;
                            jp = n;
                        }
                        if (armClean && jp == E && jo == Pyc::JUMP_FORWARD_A) {
                            int M = E + ja * W;
                            if (M > E) {
                                bool elseClean = true;
                                int eo, ea, ep = E;
                                while (ep < M) {
                                    int n = opAt3(ep, eo, ea);
                                    if (n <= ep) { elseClean = false; break; }
                                    if (armBad(eo)) { elseClean = false; break; }
                                    ep = n;
                                }
                                if (elseClean) {
                                    ternaryInternal.insert(p);
                                    ternaryInternal.insert(joff);
                                }
                            }
                        }
                    }
                }
                p = np;
            }
        }
        struct CJ { int off, op, nxt, tgt; };
        std::vector<CJ> cjs;
        {
            int o, a, p = 0;
            while (p < (int)cb->length()) {
                int ip = p;
                int np = opAt3(p, o, a);
                if (np <= p) break;
                if ((o == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                        || o == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                        || o == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                        || o == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A)
                        && !chainIfGlue.count(ip)
                        && !ternaryInternal.count(ip))
                    cjs.push_back({ ip, o, np, np + a * W });
                p = np;
            }
        }
        auto cleanOperand = [&](int start, int end) -> bool {
            if (start >= end) return false;
            int o, a, p = start;
            while (p < end) {
                int np = opAt3(p, o, a);
                if (np <= p) return false;
                if (chainIfGlue.count(p) || chainIfTrue.count(p)) { p = np; continue; }
                /* a chained-comparison dup-cleanup POP_TOP is INTERNAL to the one
                   chain operand, not a statement boundary — don't reject it below. */
                if (chainCleanupPop.count(p)) { p = np; continue; }
                if (ternaryInternal.count(p)) { p = np; continue; }
                switch (o) {
                case Pyc::POP_JUMP_FORWARD_IF_FALSE_A:
                case Pyc::POP_JUMP_FORWARD_IF_TRUE_A:
                case Pyc::POP_JUMP_BACKWARD_IF_FALSE_A:
                case Pyc::POP_JUMP_BACKWARD_IF_TRUE_A:
                case Pyc::POP_JUMP_FORWARD_IF_NONE_A:
                case Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A:
                case Pyc::POP_JUMP_BACKWARD_IF_NONE_A:
                case Pyc::POP_JUMP_BACKWARD_IF_NOT_NONE_A:
                case Pyc::POP_JUMP_IF_FALSE_A:
                case Pyc::POP_JUMP_IF_TRUE_A:
                case Pyc::JUMP_FORWARD_A:
                case Pyc::JUMP_BACKWARD_A:
                case Pyc::JUMP_BACKWARD_NO_INTERRUPT_A:
                case Pyc::JUMP_ABSOLUTE_A:
                case Pyc::JUMP_IF_FALSE_OR_POP_A:
                case Pyc::JUMP_IF_TRUE_OR_POP_A:
                case Pyc::FOR_ITER_A:
                case Pyc::RETURN_VALUE:
                case Pyc::RETURN_CONST_A:
                case Pyc::RAISE_VARARGS_A:
                case Pyc::RERAISE:
                case Pyc::RERAISE_A:
                case Pyc::PUSH_EXC_INFO:
                case Pyc::CHECK_EXC_MATCH:
                case Pyc::POP_EXCEPT:
                case Pyc::BEFORE_WITH:
                case Pyc::BEFORE_ASYNC_WITH:
                case Pyc::SEND_A:
                case Pyc::YIELD_VALUE:
                case Pyc::GET_AITER:
                case Pyc::GET_ANEXT:
                case Pyc::STORE_FAST_A:
                case Pyc::STORE_NAME_A:
                case Pyc::STORE_GLOBAL_A:
                case Pyc::STORE_DEREF_A:
                case Pyc::STORE_ATTR_A:
                case Pyc::STORE_SUBSCR:
                /* POP_TOP discards a value — a boolean OPERAND's value is always kept
                   (it may be the short-circuit result), so a bare POP_TOP means the
                   gap is a STATEMENT, not one operand. Without this, a condition's
                   body that begins right after the final operand (a bare expression
                   statement like a method call: `self.write_line_break()`) is wrongly
                   absorbed as a "clean operand", extending the boolean chain across
                   the `if` body into the NEXT statement's jump — yaml/emitter
                   write_indent. */
                case Pyc::POP_TOP:
                    return false;
                default: break;
                }
                p = np;
            }
            return true;
        };
        size_t i0 = 0;
        while (i0 < cjs.size()) {
            size_t e = i0;
            while (e + 1 < cjs.size()
                    && cleanOperand(cjs[e].nxt, cjs[e + 1].off))
                ++e;
            auto sFalse = [](int op) {
                return op == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                    || op == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                    || op == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A;
            };
            auto sTrue = [](int op) {
                return op == Pyc::POP_JUMP_FORWARD_IF_TRUE_A;
            };
            int bodyStart = cjs[e].nxt;
            int term2 = cjs[e].tgt;
            std::vector<int> groupSizes;
            std::vector<bool> negv(cjs.size(), false);
            bool andOfOrs = false;
            bool ok = false;
            /* OR-of-ANDs is normally recognized by cjs[e].op==IF_FALSE (the final
               conjunct is positive, FALSE->exit). But when the TRAILING AND group's
               last conjunct is NEGATED (`… or (C and not D)`), the compiler emits it
               as IF_TRUE->exit, so cjs[e].op==IF_TRUE. Distinguish that from a genuine
               AND-of-ORs (which also has cjs[e].op==IF_TRUE): in OR-of-ANDs at least
               one EARLIER operand jumps to bodyStart (an OR group-end), whereas
               AND-of-ORs operands only jump to bodyStart on FALSE (AND short-circuit)
               with IF_TRUE jumps going to term2. We ONLY take the IF_TRUE branch of
               OR-of-ANDs when: (a) the final jump is IF_TRUE->exit (term2, past body),
               and (b) some earlier operand is a positive OR group-end (IF_TRUE->body).
               This is the yaml/emitter write_indent shape. The full group validation
               below still gates ok=false on any deviation, so this only widens the
               pre-filter for this exact tail-negated shape. */
            bool orFinalNegExit = false;
            if (cjs[e].op == Pyc::POP_JUMP_FORWARD_IF_TRUE_A && term2 != bodyStart
                    && term2 > bodyStart && e > i0) {
                for (size_t i = i0; i < e; ++i) {
                    if (cjs[i].op == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                            && cjs[i].tgt == bodyStart) { orFinalNegExit = true; break; }
                }
            }
            if ((cjs[e].op == Pyc::POP_JUMP_FORWARD_IF_FALSE_A || orFinalNegExit)
                    && term2 != bodyStart) {
                ok = true;
                size_t g = i0;
                bool finalMode = false;
                for (size_t i = i0; i <= e && ok; ++i) {
                    int tgt = cjs[i].tgt;
                    bool grpEndPos = sTrue(cjs[i].op) && tgt == bodyStart;
                    bool grpEndNeg = cjs[i].op == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                                     && tgt == bodyStart;
                    bool grpEndNone = (cjs[i].op == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                                       || cjs[i].op == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A)
                                      && tgt == bodyStart;
                    if (grpEndPos || grpEndNeg || grpEndNone) {
                        if (finalMode) { ok = false; break; }
                        int ngs = cjs[i].nxt;
                        for (size_t j = g; j < i; ++j) {
                            if (sFalse(cjs[j].op) && cjs[j].tgt == ngs)
                                negv[j] = false;
                            else if (cjs[j].op == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                                     && cjs[j].tgt == ngs)
                                negv[j] = true;
                            else { ok = false; break; }
                        }
                        if (!ok) break;
                        negv[i] = grpEndNeg || grpEndNone;
                        groupSizes.push_back((int)(i - g + 1));
                        g = i + 1;
                    } else if (sFalse(cjs[i].op) && tgt == term2) {
                        finalMode = true;
                    } else if (cjs[i].op == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                               && tgt == term2) {
                        finalMode = true;
                    } else if (!finalMode && tgt > cjs[i].off && tgt < bodyStart
                               && (sFalse(cjs[i].op)
                                   || cjs[i].op == Pyc::POP_JUMP_FORWARD_IF_TRUE_A)) {
                    } else { ok = false; break; }
                }
                if (ok && finalMode && g <= e) {
                    for (size_t j = g; j <= e; ++j) {
                        if (sFalse(cjs[j].op) && cjs[j].tgt == term2)
                            negv[j] = false;
                        else if (cjs[j].op == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                                 && cjs[j].tgt == term2)
                            negv[j] = true;
                        else { ok = false; break; }
                    }
                    if (ok) groupSizes.push_back((int)(e - g + 1));
                } else ok = false;
            }
            /* If the tail-negated OR-of-ANDs attempt didn't validate, fall back to
               the AND-of-ORs interpretation (both share cjs[e].op==IF_TRUE). Reset
               the accumulated state so the fallback starts clean. */
            if (orFinalNegExit && !ok) {
                groupSizes.clear();
                std::fill(negv.begin(), negv.end(), false);
                andOfOrs = false;
            }
            if (!ok && cjs[e].op == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                       && term2 != bodyStart && term2 > bodyStart) {
                andOfOrs = true;
                ok = true;
                size_t g = i0;
                bool lastGroup = false;
                for (size_t i = i0; i <= e && ok; ++i) {
                    int tgt = cjs[i].tgt;
                    if (sFalse(cjs[i].op) && tgt == bodyStart) {
                        if (lastGroup) { ok = false; break; }
                        int ngs = cjs[i].nxt;
                        for (size_t j = g; j < i; ++j)
                            if (!(sTrue(cjs[j].op) && cjs[j].tgt == ngs)) { ok = false; break; }
                        if (!ok) break;
                        groupSizes.push_back((int)(i - g + 1));
                        g = i + 1;
                    } else if (sTrue(cjs[i].op) && tgt == term2) {
                        lastGroup = true;
                    } else if (sTrue(cjs[i].op) && !lastGroup
                               && tgt > cjs[i].off && tgt <= bodyStart) {
                    } else { ok = false; break; }
                }
                if (ok && lastGroup && g <= e) {
                    for (size_t j = g; j <= e; ++j)
                        if (!(sTrue(cjs[j].op) && cjs[j].tgt == term2)) { ok = false; break; }
                    if (ok) groupSizes.push_back((int)(e - g + 1));
                } else ok = false;
            }
            int maxg = 0;
            for (int gsz : groupSizes) if (gsz > maxg) maxg = gsz;
            bool hasChainOperand = false;
            if (ok && groupSizes.size() >= 2) {
                int lo = cjs[i0].off, hi = cjs[e].off;
                for (int g : chainIfGlue)
                    if (g >= lo && g <= hi) { hasChainOperand = true; break; }
            }
            if (ok && groupSizes.size() >= 2
                    && (maxg >= 2 || hasChainOperand
                        || (!andOfOrs && groupSizes.size() >= 3))) {
                for (size_t i = i0; i <= e; ++i) {
                    OrReconStep st;
                    st.isFinal = (i == e);
                    st.andOfOrs = andOfOrs;
                    st.neg = negv[i];
                    if (st.isFinal) st.groupSizes = groupSizes;
                    orReconStep[cjs[i].off] = st;
                }
                orReconLoopBody.insert(bodyStart);
                /* tail-negated OR-of-ANDs (`… or (C and not D)`, final op IF_TRUE):
                   condNode is already the correct positive condition, so the
                   fall-through BLK_IF open must not re-negate on the IF_TRUE opcode. */
                if (orFinalNegExit && !andOfOrs)
                    orReconFinalNoNeg.insert(cjs[e].off);
            }
            i0 = e + 1;
        }
    }

    std::unordered_set<int> whileBottomSkip;
    std::unordered_set<int> compoundWhileBody;
    std::unordered_set<int> compoundAndLeadGuard;
    std::unordered_set<int> rotWhileGuard;
    if (mod->verCompare(3, 11) >= 0) {
        struct CJump { int off; int nextoff; int target; bool backward; };
        std::vector<CJump> cj;
        std::unordered_set<int> forIters;
        {
            PycBuffer scan(code->code()->value(), code->code()->length());
            int sop, sarg, spos = 0;
            while (!scan.atEof()) {
                int ioff = spos;
                bc_next(scan, mod, sop, sarg, spos);
                if (sop == Pyc::FOR_ITER_A)
                    forIters.insert(ioff);
                bool fwd = sop == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                        || sop == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                        || sop == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                        || sop == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A;
                bool bwd = sop == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A
                        || sop == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A
                        || sop == Pyc::POP_JUMP_BACKWARD_IF_NONE_A
                        || sop == Pyc::POP_JUMP_BACKWARD_IF_NOT_NONE_A;
                if (fwd || bwd) {
                    int tgt = bwd ? spos - sarg * (int)sizeof(uint16_t)
                                  : spos + sarg * (int)sizeof(uint16_t);
                    cj.push_back({ ioff, spos, tgt, bwd });
                }
            }
        }
        auto isRetNone = [&](int off) -> bool {
            PycBuffer b(code->code()->value(), code->code()->length());
            b.setPos(off);
            int o = -1, a = 0, p = off;
            if (b.atEof()) return false;
            bc_next(b, mod, o, a, p);
            if (o != Pyc::LOAD_CONST_A
                    || code->getConst(a).type() != PycObject::TYPE_NONE)
                return false;
            if (b.atEof()) return false;
            int o2 = -1, a2 = 0, p2 = p;
            bc_next(b, mod, o2, a2, p2);
            return o2 == Pyc::RETURN_VALUE || o2 == Pyc::INSTRUMENTED_RETURN_VALUE_A;
        };
        const bool isGenCo = (code->flags()
            & (PycCode::CO_GENERATOR | PycCode::CO_COROUTINE
               | PycCode::CO_ASYNC_GENERATOR)) != 0;
        for (size_t i = 0; i < cj.size(); ++i) {
            if (!cj[i].backward)
                continue;
            int T = cj[i].target;
            int B = cj[i].off;
            int E = cj[i].nextoff;
            if (T >= B)
                continue;
            if (forIters.count(T))
                continue;
            size_t start = i;
            bool usedRetNoneRelax = false;
            std::unordered_set<int> botOpStart;
            botOpStart.insert(cj[i].nextoff);
            bool sweptForward = false, interleaved = false;
            while (start > 0) {
                const CJump& p = cj[start - 1];
                if (p.off <= T)
                    break;
                if (p.target == T || p.target == E) {
                    if (p.backward && sweptForward)
                        interleaved = true;
                    if (!p.backward)
                        sweptForward = true;
                    botOpStart.insert(p.nextoff);
                    start--;
                    continue;
                }
                if (isGenCo && isRetNone(p.target) && isRetNone(E)) {
                    usedRetNoneRelax = true;
                    sweptForward = true;
                    botOpStart.insert(p.nextoff);
                    start--;
                    continue;
                }
                if (!p.backward && retPadCanon.count(p.target)
                        && canonTarget(p.target) == canonTarget(E)) {
                    usedRetNoneRelax = true;
                    sweptForward = true;
                    botOpStart.insert(p.nextoff);
                    start--;
                    continue;
                }
                break;
            }
            if (start == i)
                continue;
            std::vector<int> orSkip;
            if (!interleaved || orReconLoopBody.count(T)) {
                size_t os = start;
                while (os > 0) {
                    const CJump& p = cj[os - 1];
                    if (p.off <= T || p.backward || !botOpStart.count(p.target))
                        break;
                    botOpStart.insert(p.nextoff);
                    orSkip.push_back(p.off);
                    os--;
                }
            }
            bool guard = false;
            int entryOps = 0;
            int guardExit = -1;
            for (size_t k = 0; k < cj.size(); ++k) {
                if (!cj[k].backward && cj[k].nextoff == T
                        && (cj[k].target == E
                            || (isGenCo && isRetNone(cj[k].target) && isRetNone(E))
                            || (retPadCanon.count(cj[k].target)
                                && canonTarget(cj[k].target) == canonTarget(E)))) {
                    guard = true;
                    guardExit = cj[k].target;
                    int exit = cj[k].target;
                    entryOps = 1;
                    size_t g = k;
                    while (g > 0) {
                        const CJump& pg = cj[g - 1];
                        if (pg.backward)
                            break;
                        if (!(pg.target == exit || pg.target == E
                              || (isGenCo && isRetNone(pg.target) && isRetNone(E))
                              || (retPadCanon.count(pg.target)
                                  && canonTarget(pg.target) == canonTarget(E))))
                            break;
                        entryOps++;
                        g--;
                    }
                    break;
                }
            }
            if (!guard)
                continue;
            int bottomOps = (int)(i - start) + 1;
            /* The generator distinct-return-None relax is safe whenever the entry
               guard re-tests the same operand count as the bottom: marking the body
               compoundWhileBody lets the entry-guard AND-combine fold every operand
               (including the body-adjacent final one, otherwise peeled into a nested
               `if`) into one BLK_IF that the final backward jump converts. */
            if (usedRetNoneRelax && !(entryOps == bottomOps && bottomOps >= 2))
                continue;
            for (size_t k = start; k < i; ++k)
                whileBottomSkip.insert(cj[k].off);
            for (int off : orSkip)
                whileBottomSkip.insert(off);
            compoundWhileBody.insert(T);
            {
                size_t j = start;
                while (j < i && !cj[j].backward && cj[j].target == E)
                    j++;
                if (j > start && j < i && cj[j].backward && cj[j].target == T)
                    compoundAndLeadGuard.insert(E);
                if (!orSkip.empty()) {
                    compoundAndLeadGuard.insert(E);
                    if (guardExit >= 0)
                        compoundAndLeadGuard.insert(guardExit);
                }
            }
        }
    }

    std::unordered_map<int, int> whileTrueHdr;
    std::unordered_set<int> whileContMerge;
    std::unordered_set<int> whileTrueTopTest;
    int whileTopTestPendingExit = -1;
    std::unordered_set<int> topTestWhileExit;
    std::unordered_set<int> topTestWhileFallBreak;
    std::unordered_set<int> topTestWhileFallContinue;
    std::unordered_map<int, int> loopBreakElse;
    {
        struct Back { int instr; int end; int target; };
        std::vector<Back> backs;
        std::vector<std::pair<int,int>> condBacks;
        std::unordered_set<int> forIters;
        std::unordered_map<int,int> fwdCondAt;
        PycBuffer scan(code->code()->value(), code->code()->length());
        int sop, sarg, spos = 0;
        while (!scan.atEof()) {
            int ioff = spos;
            bc_next(scan, mod, sop, sarg, spos);
            int sa = sarg;
            if (mod->verCompare(3, 10) >= 0)
                sa *= sizeof(uint16_t);
            if (sop == Pyc::JUMP_BACKWARD_A)
                backs.push_back({ ioff, spos, spos - sa });
            else if (sop == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A
                    || sop == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A
                    || sop == Pyc::POP_JUMP_BACKWARD_IF_NONE_A
                    || sop == Pyc::POP_JUMP_BACKWARD_IF_NOT_NONE_A)
                condBacks.push_back({ ioff, spos - sa });
            else if (sop == Pyc::FOR_ITER_A)
                forIters.insert(ioff);
            if (sop == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                    || sop == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                    || sop == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                    || sop == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A)
                fwdCondAt[spos] = ioff;
        }
        for (const auto& cb : condBacks) {
            auto it = fwdCondAt.find(cb.second);
            if (it != fwdCondAt.end())
                rotWhileGuard.insert(it->second);
        }
        std::unordered_map<int, Back> bottom;
        for (const auto& b : backs) {
            auto it = bottom.find(b.target);
            if (it == bottom.end() || b.instr > it->second.instr)
                bottom[b.target] = b;
        }
        for (const auto& kv : bottom) {
            int H = kv.first;
            const Back& b = kv.second;
            if (forIters.count(H))
                continue;
            bool exc_overlap = false;
            for (const auto& e : exception_entries) {
                if (e.start_offset < b.instr && e.end_offset > H) {
                    bool contained = (e.start_offset >= H
                            && e.end_offset <= b.instr);
                    bool loopInsideExc = (e.start_offset <= H
                            && e.end_offset >= b.instr);
                    if (!contained && !loopInsideExc) {
                        int uStart = e.start_offset, uEnd = e.end_offset;
                        for (const auto& e2 : exception_entries)
                            if (e2.target == e.target) {
                                uStart = std::min(uStart, e2.start_offset);
                                uEnd = std::max(uEnd, e2.end_offset);
                            }
                        if (uStart <= H && uEnd >= b.instr)
                            loopInsideExc = true;
                    }
                    if (contained || loopInsideExc)
                        continue;
                    bool isWith = false;
                    if (mod->verCompare(3, 11) >= 0 && e.push_lasti) {
                        PycBuffer hs(code->code()->value(),
                                     code->code()->length());
                        hs.setPos(e.target);
                        int ho, ha, hp;
                        for (int i = 0; i < 4 && !hs.atEof(); ++i) {
                            bc_next(hs, mod, ho, ha, hp);
                            if (ho == Pyc::WITH_EXCEPT_START) {
                                isWith = true; break;
                            }
                            if (ho != Pyc::PUSH_EXC_INFO && ho != Pyc::NOP)
                                break;
                        }
                    }
                    if (!isWith) { exc_overlap = true; break; }
                }
            }
            if (exc_overlap)
                continue;
            bool skip = false;
            for (const auto& cb : condBacks) {
                if (cb.second == H) { skip = true; break; }
                if (cb.first > b.instr && cb.second > H && cb.second < b.instr) {
                    skip = true; break;
                }
            }
            if (skip)
                continue;
            for (const auto& kv2 : bottom) {
                if (kv2.first > H && kv2.second.instr > b.instr
                        && kv2.first <= b.instr && b.instr < kv2.second.instr) {
                    skip = true; break;
                }
            }
            if (skip)
                continue;
            int topTestExit = -1;
            {
                PycBuffer hs(code->code()->value(), code->code()->length());
                hs.setPos(H);
                int ho, ha, hp = H;
                while (!hs.atEof() && hp < b.instr) {
                    bc_next(hs, mod, ho, ha, hp);
                    if (ho == Pyc::CACHE || ho == Pyc::NOP)
                        continue;
                    if (ho == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                            || ho == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                            || ho == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                            || ho == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A) {
                        int ht = ha;
                        if (mod->verCompare(3, 10) >= 0)
                            ht *= sizeof(uint16_t);
                        if (hp + ht >= b.end)
                            topTestExit = hp + ht;
                        break;
                    }
                    if (ho == Pyc::STORE_FAST_A || ho == Pyc::STORE_NAME_A
                            || ho == Pyc::STORE_GLOBAL_A || ho == Pyc::STORE_DEREF_A
                            || ho == Pyc::STORE_SUBSCR || ho == Pyc::STORE_ATTR_A
                            || ho == Pyc::JUMP_FORWARD_A || ho == Pyc::JUMP_BACKWARD_A
                            || ho == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A
                            || ho == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A
                            || ho == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A
                            || ho == Pyc::FOR_ITER_A || ho == Pyc::GET_ITER
                            || ho == Pyc::RETURN_VALUE || ho == Pyc::RAISE_VARARGS_A
                            || ho == Pyc::RERAISE_A || ho == Pyc::PUSH_EXC_INFO
                            || ho == Pyc::SEND_A || ho == Pyc::YIELD_VALUE)
                        break;
                }
            }
            if (topTestExit >= 0) {
                whileTrueHdr[H] = topTestExit;
                whileTrueTopTest.insert(H);
                topTestWhileExit.insert(topTestExit);
                PycBuffer fs(code->code()->value(), code->code()->length());
                int fo, fa, fp = 0, prevOp = -1, prevStart = -1, prevArg = 0;
                while (!fs.atEof() && fp < topTestExit) {
                    int fstart = fp;
                    bc_next(fs, mod, fo, fa, fp);
                    if (fo == Pyc::CACHE) continue;
                    if (fp == topTestExit) { prevOp = fo; prevStart = fstart; prevArg = fa; }
                }
                if (prevOp != -1 && prevStart >= H
                        && prevOp != Pyc::JUMP_BACKWARD_A
                        && prevOp != Pyc::JUMP_BACKWARD_NO_INTERRUPT_A
                        && prevOp != Pyc::JUMP_FORWARD_A
                        && prevOp != Pyc::RETURN_VALUE
                        && prevOp != Pyc::RAISE_VARARGS_A
                        && prevOp != Pyc::RERAISE_A)
                    topTestWhileFallBreak.insert(topTestExit);
                if ((prevOp == Pyc::JUMP_BACKWARD_A
                            || prevOp == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A)
                        && prevStart >= H
                        && topTestExit - prevArg * (int)sizeof(uint16_t) == H)
                    topTestWhileFallContinue.insert(topTestExit);
            } else {
                int whileBreakExit = -1;
                int wbeFall = -1;
                PycBuffer bs(code->code()->value(), code->code()->length());
                bs.setPos(H);
                int bo, ba, bp = H;
                while (!bs.atEof() && bp < b.instr) {
                    bc_next(bs, mod, bo, ba, bp);
                    if (bo == Pyc::CACHE) continue;
                    if (bo == Pyc::JUMP_FORWARD_A) {
                        int bt = ba;
                        if (mod->verCompare(3, 10) >= 0)
                            bt *= sizeof(uint16_t);
                        bt += bp;
                        if (bt > b.end && bt > whileBreakExit) {
                            whileBreakExit = bt;
                            wbeFall = bp;
                        }
                    }
                }
                if (whileBreakExit >= 0) {
                    PycBuffer gs(code->code()->value(), code->code()->length());
                    gs.setPos(b.end);
                    int go, ga, gp = b.end;
                    const int W = (int)sizeof(uint16_t);
                    bool pureTail = true;
                    while (!gs.atEof() && gp < whileBreakExit) {
                        int gip = gp;
                        bc_next(gs, mod, go, ga, gp);
                        if (go == Pyc::CACHE || go == Pyc::NOP
                                || go == Pyc::RERAISE_A || go == Pyc::COPY_A
                                || go == Pyc::POP_EXCEPT || go == Pyc::PUSH_EXC_INFO
                                || go == Pyc::RAISE_VARARGS_A || go == Pyc::POP_TOP)
                            continue;
                        bool inProt = false;
                        for (const auto& e : exception_entries)
                            if (gip >= e.start_offset && gip < e.end_offset) { inProt = true; break; }
                        bool nestedLoopWith = (go == Pyc::FOR_ITER_A
                                || go == Pyc::INSTRUMENTED_FOR_ITER_A
                                || go == Pyc::BEFORE_WITH || go == Pyc::SETUP_WITH_A
                                || go == Pyc::SETUP_FINALLY_A);
                        if (inProt && !nestedLoopWith) continue;
                        if (!inProt && (go == Pyc::LOAD_CONST_A
                                || go == Pyc::STORE_FAST_A || go == Pyc::DELETE_FAST_A))
                            continue;
                        if (!inProt && go == Pyc::JUMP_FORWARD_A
                                && gp + ga * W == whileBreakExit)
                            continue;
                        if (!inProt && (go == Pyc::JUMP_BACKWARD_A
                                || go == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A)
                                && gp - ga * W == H)
                            continue;
                        pureTail = false;
                        break;
                    }
                    if (!pureTail)
                        whileBreakExit = -1;
                }
                bool chased = false;
                int chaseStart = -1;
                const int W226 = (int)sizeof(uint16_t);
                auto opAt226 = [&](int off) -> int {
                    PycBuffer ps(code->code()->value(), code->code()->length());
                    ps.setPos(off);
                    int po, pa, pp = off;
                    while (!ps.atEof()) {
                        bc_next(ps, mod, po, pa, pp);
                        if (po == Pyc::CACHE) continue;
                        return po;
                    }
                    return -1;
                };
                if (whileBreakExit >= 0 && wbeFall >= 0
                        && opAt226(wbeFall) == Pyc::PUSH_EXC_INFO) {
                    chaseStart = whileBreakExit;
                } else if (whileBreakExit < 0) {
                    int sop = opAt226(b.end);
                    bool hdrInTry = false;
                    for (const auto& e : exception_entries)
                        if (H >= e.start_offset && H < e.end_offset) { hdrInTry = true; break; }
                    if (!hdrInTry && sop != Pyc::RERAISE_A && sop != Pyc::COPY_A
                            && sop != Pyc::POP_EXCEPT && sop != Pyc::PUSH_EXC_INFO
                            && sop != Pyc::RAISE_VARARGS_A && sop != Pyc::CHECK_EXC_MATCH
                            && sop != -1)
                        chaseStart = b.end;
                }
                bool winChase = (chaseStart >= 0 && chaseStart == b.end);
                if (chaseStart >= 0) {
                    int breakInstrEnd = -1, breakTgt = -1, skipCount = 0;
                    PycBuffer cs(code->code()->value(), code->code()->length());
                    cs.setPos(chaseStart);
                    int co, ca, cp = chaseStart;
                    bool ok = true;
                    while (!cs.atEof()) {
                        bc_next(cs, mod, co, ca, cp);
                        if (co == Pyc::CACHE || co == Pyc::NOP) continue;
                        if (co == Pyc::JUMP_FORWARD_A) {
                            int t = ca * W226 + cp;
                            if (t > cp && opAt226(cp) == Pyc::PUSH_EXC_INFO) {
                                cs.setPos(t); cp = t; skipCount++; continue;
                            }
                            if (t > chaseStart) { breakInstrEnd = cp; breakTgt = t; }
                            break;
                        }
                        if (co == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                                || co == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                                || co == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                                || co == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A
                                || co == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A
                                || co == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A
                                || co == Pyc::JUMP_BACKWARD_A
                                || co == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A
                                || co == Pyc::FOR_ITER_A
                                || co == Pyc::INSTRUMENTED_FOR_ITER_A
                                || co == Pyc::BEFORE_WITH || co == Pyc::SETUP_WITH_A
                                || co == Pyc::SETUP_FINALLY_A || co == Pyc::PUSH_EXC_INFO
                                || co == Pyc::RETURN_VALUE || co == Pyc::RETURN_CONST_A
                                || co == Pyc::RAISE_VARARGS_A || co == Pyc::RERAISE_A) {
                            ok = false; break;
                        }
                    }
                    if (winChase && skipCount < 1)
                        ok = false;
                    bool enclElse = false;
                    if (ok && breakInstrEnd > 0 && breakTgt > breakInstrEnd) {
                        PycBuffer is(code->code()->value(), code->code()->length());
                        int io, ia, ip = 0;
                        while (!is.atEof() && ip < H) {
                            bc_next(is, mod, io, ia, ip);
                            if ((io == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                                        || io == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                                        || io == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                                        || io == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A)
                                    && ia * W226 + ip == breakInstrEnd) {
                                enclElse = true; break;
                            }
                        }
                    }
                    if (enclElse) {
                        whileTrueHdr[H] = breakInstrEnd;
                        topTestWhileExit.insert(breakInstrEnd);
                        loopRanges.push_back({ H, breakInstrEnd, breakTgt });
                        loopBreakElse[breakInstrEnd] = breakTgt;
                        chased = true;
                    }
                }
                if (chased) {
                } else if (whileBreakExit >= 0) {
                    whileTrueHdr[H] = whileBreakExit;
                    topTestWhileExit.insert(whileBreakExit);
                    topTestWhileFallBreak.insert(whileBreakExit);
                } else {
                    /* Rotated while-True whose LAST back-edge is a `continue`
                       (not the structural bottom): a nested if-arm jumps forward
                       OVER the continue-bearing else to a merge that equals b.end
                       (the offset right after that back-edge), then real loop-body
                       tail code follows before the loop falls through to its true
                       exit.  b.end therefore truncates the loop mid-body — the tail
                       renders as dead code after the `continue` and the jump-over-
                       else is mis-read as a `break`.  Detect this by a JUMP_FORWARD
                       inside the loop that targets exactly b.end, then extend the
                       loop end to the target of the innermost enclosing forward
                       conditional that jumps past b.end (the arm-guard's merge =
                       the true loop exit).  Tightly gated: only when such a
                       jump-over-continue-merge exists. */
                    int loopEnd = b.end;
                    bool contMerge = false;
                    {
                        PycBuffer js(code->code()->value(), code->code()->length());
                        js.setPos(H);
                        int jo, ja, jp = H;
                        const int JW = (int)sizeof(uint16_t);
                        while (!js.atEof() && jp < b.instr) {
                            bc_next(js, mod, jo, ja, jp);
                            if (jo == Pyc::JUMP_FORWARD_A
                                    && jp + ja * JW == b.end) {
                                contMerge = true;
                                break;
                            }
                        }
                    }
                    if (contMerge) {
                        int encTgt = -1;
                        PycBuffer es(code->code()->value(),
                                     code->code()->length());
                        es.setPos(H);
                        int eo, ea, ep = H;
                        const int EW = (int)sizeof(uint16_t);
                        while (!es.atEof() && ep < b.instr) {
                            int eip = ep;
                            bc_next(es, mod, eo, ea, ep);
                            if ((eo == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                                        || eo == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                                        || eo == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                                        || eo == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A)) {
                                int et = ep + ea * EW;
                                if (et > b.end && eip < b.instr
                                        && (encTgt < 0 || et < encTgt))
                                    encTgt = et;
                            }
                        }
                        if (encTgt > b.end) {
                            loopEnd = encTgt;
                            loopRanges.push_back({ H, b.instr, encTgt });
                            whileContMerge.insert(b.end);
                        } else {
                            contMerge = false;
                        }
                    }
                    whileTrueHdr[H] = loopEnd;
                }
            }
        }
    }
    std::unordered_set<int> whileTrueOpened;
    std::unordered_map<int, int> elifPendingElse;

    struct MCase { bool isFirst; int matchEnd; int failTarget; int bodyStart;
                   int popExtra; std::vector<PycRef<ASTNode>> caps; };
    std::unordered_map<int, MCase> matchCase;
    std::unordered_map<int, int> matchCaseEnd;
    std::unordered_set<int> matchBlockEnd;
    struct VCase { bool isFirst; bool isLast; int matchEnd; int failTarget; int bodyStart; int orNextAlt; };
    struct OrCont { int orNextAlt; int bodyStart; };
    std::unordered_map<int, OrCont> matchValueOr;
    std::unordered_map<int, VCase> matchValue;
    std::unordered_map<int, int> matchWildcardOpen;
    if (mod->verCompare(3, 11) >= 0) {
        struct Ins { int op; int arg; int off; int next; };
        std::vector<Ins> v;
        std::unordered_map<int,int> idxOf;
        {
            PycBuffer scan(code->code()->value(), code->code()->length());
            int so, sa, sp = 0;
            while (!scan.atEof()) {
                int io = sp;
                bc_next(scan, mod, so, sa, sp);
                if (so == Pyc::CACHE) continue;
                idxOf[io] = (int)v.size();
                v.push_back({ so, sa, io, sp });
            }
        }
        auto capName = [&](const Ins& s) -> PycRef<ASTNode> {
            if (s.op == Pyc::STORE_FAST_A)
                return new ASTName(code->getLocal(s.arg));
            if (s.op == Pyc::STORE_NAME_A || s.op == Pyc::STORE_GLOBAL_A)
                return new ASTName(code->getName(s.arg));
            if (s.op == Pyc::STORE_DEREF_A)
                return new ASTName(code->getCellVar(mod, s.arg));
            return nullptr;
        };
        auto parseCase = [&](size_t mi, int& failTarget, int& bodyStart,
                             int& popExtra,
                             std::vector<PycRef<ASTNode>>& caps) -> bool {
            int nPos = v[mi].arg;
            if (mi == 0) return false;
            const Ins& kw = v[mi-1];
            if (kw.op != Pyc::LOAD_CONST_A) return false;
            PycRef<PycObject> kwo = code->getConst(kw.arg);
            if (kwo == nullptr || (kwo->type() != PycObject::TYPE_TUPLE
                    && kwo->type() != PycObject::TYPE_SMALL_TUPLE)) return false;
            if (kwo.cast<PycTuple>()->values().size() != 0) return false;
            if (mi+2 >= v.size()) return false;
            if (!(v[mi+1].op == Pyc::COPY_A && v[mi+1].arg == 1)) return false;
            if (v[mi+2].op != Pyc::POP_JUMP_FORWARD_IF_NONE_A) return false;
            failTarget = v[mi+2].next + v[mi+2].arg * (int)sizeof(uint16_t);
            size_t j = mi+3;
            caps.clear();
            if (j >= v.size() || v[j].op != Pyc::UNPACK_SEQUENCE_A
                    || v[j].arg != nPos) return false;
            j++;
            for (int k = 0; k < nPos; ++k, ++j) {
                if (j >= v.size()) return false;
                if (v[j].op == Pyc::POP_TOP) {
                    PycRef<PycString> us = new PycString(); us->setValue("_");
                    caps.push_back(new ASTName(us));
                    continue;
                }
                PycRef<ASTNode> nm = capName(v[j]);
                if (nm == nullptr) return false;
                caps.push_back(nm);
            }
            popExtra = 0;
            while (j < v.size() && v[j].op == Pyc::POP_TOP) { popExtra++; j++; }
            if (j >= v.size()) return false;
            bodyStart = v[j].off;
            for (size_t k = j; k < v.size() && v[k].off < failTarget; ++k) {
                int op = v[k].op;
                bool isJump = op == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                        || op == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                        || op == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                        || op == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A
                        || op == Pyc::JUMP_FORWARD_A;
                if (isJump) {
                    int tgt = v[k].next + v[k].arg * (int)sizeof(uint16_t);
                    if (tgt == failTarget) return false;
                }
            }
            return true;
        };
        struct CaseRec { size_t mi; int failTarget; int bodyStart;
                         int popExtra; std::vector<PycRef<ASTNode>> caps; };
        std::vector<CaseRec> recs;
        for (size_t i = 0; i < v.size(); ++i) {
            if (v[i].op != Pyc::MATCH_CLASS_A) continue;
            int ft, bs, pe; std::vector<PycRef<ASTNode>> caps;
            if (parseCase(i, ft, bs, pe, caps))
                recs.push_back({ i, ft, bs, pe, caps });
        }
        std::unordered_map<int,int> caseByPopTop;
        auto firstMatchClassAtOrAfter = [&](int off) -> int {
            for (size_t k = 0; k < v.size(); ++k)
                if (v[k].off >= off && v[k].op == Pyc::MATCH_CLASS_A)
                    return (int)k;
            return -1;
        };
        std::unordered_set<size_t> isSuccessor;
        std::unordered_map<size_t,size_t> succOf;
        std::unordered_map<size_t,size_t> recByMi;
        for (size_t r = 0; r < recs.size(); ++r) recByMi[recs[r].mi] = r;
        for (size_t r = 0; r < recs.size(); ++r) {
            int ft = recs[r].failTarget;
            if (idxOf.count(ft) && v[idxOf[ft]].op == Pyc::POP_TOP) {
                int nm = firstMatchClassAtOrAfter(v[idxOf[ft]].next);
                if (nm >= 0 && recByMi.count((size_t)nm)) {
                    size_t s = recByMi[(size_t)nm];
                    succOf[r] = s; isSuccessor.insert(s);
                }
            }
        }
        for (size_t r = 0; r < recs.size(); ++r) {
            if (isSuccessor.count(r)) continue;
            size_t last = r;
            while (succOf.count(last)) last = succOf[last];
            int lastFt = recs[last].failTarget;
            int matchEnd = (idxOf.count(lastFt) && v[idxOf[lastFt]].op == Pyc::POP_TOP)
                           ? v[idxOf[lastFt]].next : lastFt;
            {
                int firstOff = v[recs[r].mi].off;
                int merge = -1; bool consistent = true;
                for (size_t k = 0; k < v.size(); ++k) {
                    if (v[k].off < firstOff || v[k].off > lastFt) continue;
                    if (v[k].op != Pyc::JUMP_FORWARD_A) continue;
                    int t = v[k].next + v[k].arg * (int)sizeof(uint16_t);
                    if (t > matchEnd) {
                        if (merge < 0) merge = t;
                        else if (merge != t) { consistent = false; break; }
                    }
                }
                bool gapUnsafe = false;
                if (consistent && merge > matchEnd) {
                    for (size_t k = 0; k < v.size(); ++k) {
                        if (v[k].off < matchEnd || v[k].off >= merge) continue;
                        int op = v[k].op;
                        if (op == Pyc::MATCH_CLASS_A || op == Pyc::MATCH_SEQUENCE
                                || op == Pyc::MATCH_MAPPING || op == Pyc::MATCH_KEYS
                                || op == Pyc::POP_JUMP_FORWARD_IF_NONE_A) { gapUnsafe = true; break; }
                    }
                    for (size_t k = 0; k < v.size(); ++k) {
                        if (v[k].off < matchEnd) continue;
                        if (v[k].op == Pyc::NOP) continue;
                        if (v[k].op == Pyc::STORE_FAST_A || v[k].op == Pyc::STORE_NAME_A
                                || v[k].op == Pyc::STORE_GLOBAL_A || v[k].op == Pyc::STORE_DEREF_A)
                            gapUnsafe = true;
                        break;
                    }
                    int lastOp = -1;
                    for (size_t k = 0; k < v.size(); ++k)
                        if (v[k].off >= matchEnd && v[k].off < merge) lastOp = v[k].op;
                    if (!(lastOp == Pyc::RAISE_VARARGS_A || lastOp == Pyc::RETURN_VALUE
                            || lastOp == Pyc::RETURN_CONST_A || lastOp == Pyc::RERAISE
                            || lastOp == Pyc::RERAISE_A))
                        gapUnsafe = true;
                }
                if (consistent && merge > matchEnd && !gapUnsafe) {
                    matchWildcardOpen[matchEnd] = merge;
                    matchCaseEnd[merge] = merge;
                    matchEnd = merge;
                }
            }
            for (size_t c = r; ; c = succOf[c]) {
                MCase mc;
                mc.isFirst = (c == r);
                mc.matchEnd = matchEnd;
                mc.failTarget = recs[c].failTarget;
                mc.bodyStart = recs[c].bodyStart;
                mc.popExtra = recs[c].popExtra;
                mc.caps = recs[c].caps;
                matchCase[v[recs[c].mi].off] = mc;
                int ft = recs[c].failTarget;
                matchCaseEnd[ft] = (idxOf.count(ft) && v[idxOf[ft]].op == Pyc::POP_TOP)
                                   ? v[idxOf[ft]].next : ft;
                if (!succOf.count(c)) break;
            }
            matchBlockEnd.insert(matchEnd);
        }
    }

    if (mod->verCompare(3, 11) >= 0 && mod->verCompare(3, 12) < 0) {
        struct Ins { int op; int arg; int off; int next; };
        std::vector<Ins> v; std::unordered_map<int,int> idxOf;
        {
            PycBuffer scan(code->code()->value(), code->code()->length());
            int so, sa, sp = 0;
            while (!scan.atEof()) {
                int io = sp;
                bc_next(scan, mod, so, sa, sp);
                if (so == Pyc::CACHE) continue;
                idxOf[io] = (int)v.size();
                v.push_back({ so, sa, io, sp });
            }
        }
        struct Test { int startOff; bool hasCopy; int compareOff; int ft; int matchedNext; int matchedBody; };
        std::vector<Test> tests;
        std::unordered_map<int,int> testByStart;
        for (size_t c = 1; c + 1 < v.size(); ++c) {
            if (!(v[c].op == Pyc::COMPARE_OP_A && v[c].arg == 2
                    && v[c+1].op == Pyc::POP_JUMP_FORWARD_IF_FALSE_A))
                continue;
            int j = (int)c - 1, attrs = 0;
            while (j >= 0 && v[j].op == Pyc::LOAD_ATTR_A) { attrs++; --j; }
            int patStart = -1;
            if (attrs == 0 && j >= 0 && v[j].op == Pyc::LOAD_CONST_A)
                patStart = j;
            else if (attrs >= 1 && j >= 0
                    && (v[j].op == Pyc::LOAD_GLOBAL_A || v[j].op == Pyc::LOAD_NAME_A
                        || v[j].op == Pyc::LOAD_DEREF_A || v[j].op == Pyc::LOAD_FAST_A))
                patStart = j;
            if (patStart < 0)
                continue;
            bool hasCopy = (patStart >= 1 && v[patStart-1].op == Pyc::COPY_A
                    && v[patStart-1].arg == 1);
            int startOff = hasCopy ? v[patStart-1].off : v[patStart].off;
            int ft = v[c+1].next + v[c+1].arg * (int)sizeof(uint16_t);
            int matchedBody = -1;
            if (idxOf.count(v[c+1].next)) {
                const Ins& mi = v[idxOf[v[c+1].next]];
                if (mi.op == Pyc::JUMP_FORWARD_A)
                    matchedBody = mi.next + mi.arg * (int)sizeof(uint16_t);
            }
            testByStart[startOff] = (int)tests.size();
            tests.push_back({ startOff, hasCopy, v[c].off, ft, v[c+1].next, matchedBody });
        }
        std::unordered_set<int> targeted;
        for (const auto& t : tests)
            if (testByStart.count(t.ft)) targeted.insert(t.ft);
        auto resolveStart = [&](int off) -> int {
            if (testByStart.count(off)) return off;
            if (idxOf.count(off) && v[idxOf[off]].op == Pyc::POP_TOP) {
                int i = idxOf[off] + 1;
                while (i < (int)v.size() && v[i].op == Pyc::EXTENDED_ARG_A) ++i;
                if (i < (int)v.size() && v[i].op == Pyc::JUMP_FORWARD_A)
                    return v[i].next + v[i].arg * (int)sizeof(uint16_t);
            }
            /* An OR-pattern group carries an EXTRA leading `COPY 1` (preserving the
               subject for the case AFTER the group) before the per-alternative
               `COPY 1; <pattern>; COMPARE` of its first alternative. The preceding case
               fails THERE — to this preserve-COPY, one op before the test that
               testByStart records. Skip a lone preserve-COPY directly followed by a
               recognised test start so the chain links across the group. */
            if (idxOf.count(off) && v[idxOf[off]].op == Pyc::COPY_A
                    && v[idxOf[off]].arg == 1) {
                size_t i = idxOf[off] + 1;
                if (i < v.size() && testByStart.count(v[i].off))
                    return v[i].off;
            }
            return off;
        };
        std::unordered_set<int> targetedR;
        for (const auto& t : tests) {
            int r = resolveStart(t.ft);
            if (testByStart.count(r)) targetedR.insert(r);
        }
        for (size_t t = 0; t < tests.size(); ++t) {
            if (!tests[t].hasCopy || targeted.count(tests[t].startOff)
                    || targetedR.count(tests[t].startOff))
                continue;
            std::vector<int> chain;
            std::vector<bool> orCont;
            std::unordered_set<int> seen;
            int cur = (int)t; bool ok = true;
            while (true) {
                if (seen.count(cur)) { ok = false; break; }
                seen.insert(cur); chain.push_back(cur); orCont.push_back(false);
                int mb = tests[cur].matchedBody;
                if (mb >= 0) {
                    while (true) {
                        int ft = tests[cur].ft;
                        if (!testByStart.count(ft)) break;
                        int nx = testByStart[ft];
                        if (seen.count(nx) || tests[nx].matchedBody != mb
                                || !tests[nx].hasCopy) break;
                        seen.insert(nx); chain.push_back(nx); orCont.push_back(true);
                        cur = nx;
                    }
                }
                int ft = tests[cur].ft;
                int ns = resolveStart(ft);
                if (!testByStart.count(ns)) break;
                int nx = testByStart[ns];
                if (seen.count(nx)) { ok = false; break; }
                if (!tests[nx].hasCopy && tests[nx].matchedBody < 0) {
                    chain.push_back(nx); orCont.push_back(false); break;
                }
                cur = nx;
            }
            if (!ok || chain.size() < 2) continue;
            int lastT = chain.back();
            int lastFt = tests[lastT].ft;
            /* A `case _:` wildcard whose body is RELOCATED: the last typed case fails to
               a bare subject discard `POP_TOP; [EXTENDED_ARG;] JUMP_FORWARD -> body`
               stub that jumps over intervening code (an inlined finally copy) to the
               wildcard body. */
            bool relocWild = false;
            int relocBody = -1;
            if (idxOf.count(lastFt) && v[idxOf[lastFt]].op == Pyc::POP_TOP) {
                int nx = v[idxOf[lastFt]].next;
                while (idxOf.count(nx) && v[idxOf[nx]].op == Pyc::EXTENDED_ARG_A)
                    nx = v[idxOf[nx]].next;
                if (idxOf.count(nx) && v[idxOf[nx]].op == Pyc::JUMP_FORWARD_A) {
                    relocWild = true;
                    relocBody = v[idxOf[nx]].next + v[idxOf[nx]].arg * (int)sizeof(uint16_t);
                }
            }
            /* When the last case still preserves the subject (`hasCopy`) yet does NOT
               fail to a bare subject-discard `POP_TOP`, the fail path is a capture
               pattern / guard chain (`case r if … in r:`) the value-match
               reconstruction can't represent — leave the whole chain to the faithful
               if/elif fallback. */
            if (tests[lastT].hasCopy
                    && (!idxOf.count(lastFt) || v[idxOf[lastFt]].op != Pyc::POP_TOP))
                continue;
            bool hasOr = false;
            for (bool b : orCont) if (b) { hasOr = true; break; }
            auto caseBodyStart = [&](const Test& tc) -> int {
                if (tc.hasCopy) {
                    int mn = tc.matchedNext;
                    return (idxOf.count(mn) && v[idxOf[mn]].op == Pyc::POP_TOP)
                           ? v[idxOf[mn]].next : mn;
                }
                return tc.matchedNext;
            };
            auto orBodyStart = [&](const Test& tc) -> int { return tc.matchedBody; };
            /* An OR group's shared body (matchedBody) is reached by the alternatives'
               dispatch jumps — it is never the post-match merge, so ignore those
               targets when finding the convergence point. Only genuine OR alternatives
               contribute: a single case with an empty body legitimately JUMP_FORWARDs
               straight to the merge, and that target IS the merge. */
            std::unordered_set<int> orBodies;
            for (size_t k = 0; k < chain.size(); ++k) {
                bool isOrAlt = orCont[k] || (k + 1 < chain.size() && orCont[k + 1]);
                if (isOrAlt && tests[chain[k]].matchedBody >= 0)
                    orBodies.insert(tests[chain[k]].matchedBody);
            }
            int conv = -1; bool convOk = true;
            for (size_t k = 0; k < chain.size(); ++k) {
                const Test& tc = tests[chain[k]];
                int bs = caseBodyStart(tc);
                if (!idxOf.count(bs)) continue;
                for (int ii = idxOf[bs]; ii < (int)v.size() && v[ii].off < tc.ft; ++ii) {
                    if (v[ii].op == Pyc::JUMP_FORWARD_A) {
                        int t = v[ii].next + v[ii].arg * (int)sizeof(uint16_t);
                        if (t > lastFt && !orBodies.count(t)) {
                            if (conv < 0) conv = t;
                            else if (conv != t) convOk = false;
                        }
                    }
                }
            }
            /* When the last typed case fails to a relocated `case _:` body, the wildcard
               opens where that body renders, not at the bare discard stub. */
            int wildBodyAt = lastFt;
            bool suppressWild = false;
            if (relocWild) {
                if (convOk && conv > relocBody) {
                    /* some case bodies fall through to a real merge PAST the relocated
                       `case _:` body (a match inside a try/finally). */
                    wildBodyAt = relocBody;
                } else if ((!convOk || conv < 0) && !v.empty()) {
                    /* every case body is terminal (return/raise) — no shared merge; the
                       relocated `case _:` body is the last block and the match ends at
                       end-of-code. */
                    conv = (int)v.back().next; convOk = true; wildBodyAt = relocBody;
                } else {
                    /* the discard jumps straight to the merge — a plain no-match discard
                       with no `case _:` body; register the cases without a wildcard. */
                    suppressWild = true;
                }
            }
            int matchEnd = lastFt;
            int wildcardStart = -1;
            if (convOk && conv > lastFt) {
                bool clean = true;
                for (const auto& tt : tests)
                    if (tt.startOff >= lastFt && tt.startOff < conv) { clean = false; break; }
                bool crossesExc = false;
                for (int ii = 0; ii < (int)v.size(); ++ii)
                    if (v[ii].off >= lastFt && v[ii].off < conv
                            && v[ii].op == Pyc::PUSH_EXC_INFO) { crossesExc = true; break; }
                if (crossesExc) continue;
                if (clean) {
                    matchEnd = conv;
                    if (!suppressWild) wildcardStart = wildBodyAt;
                }
            }
            {
                int spanStart = tests[chain[0]].startOff;
                bool loopBack = false;
                for (const auto& iv : v)
                    if (iv.off >= spanStart && iv.off < matchEnd
                            && iv.op == Pyc::JUMP_BACKWARD_A) {
                        int bt = iv.next - iv.arg * (int)sizeof(uint16_t);
                        if (!hasOr || bt < spanStart) { loopBack = true; break; }
                    }
                if (loopBack) continue;
            }
            for (size_t k = 0; k < chain.size(); ++k) {
                const Test& tc = tests[chain[k]];
                bool isOrAlt = orCont[k] || (k + 1 < chain.size() && orCont[k + 1]);
                int bodyStart2 = isOrAlt ? orBodyStart(tc) : caseBodyStart(tc);
                if (orCont[k]) {
                    OrCont oc;
                    oc.orNextAlt = (k + 1 < chain.size() && orCont[k + 1])
                                   ? tests[chain[k + 1]].startOff : -1;
                    oc.bodyStart = bodyStart2;
                    matchValueOr[tc.compareOff] = oc;
                    continue;
                }
                size_t gEnd = k;
                while (gEnd + 1 < chain.size() && orCont[gEnd + 1]) ++gEnd;
                int ftResolved = resolveStart(tests[chain[gEnd]].ft);
                VCase vc;
                vc.isFirst = (k == 0);
                vc.isLast  = (gEnd + 1 == chain.size());
                vc.matchEnd = matchEnd;
                vc.failTarget = ftResolved;
                vc.bodyStart = bodyStart2;
                vc.orNextAlt = (k + 1 < chain.size() && orCont[k + 1])
                               ? tests[chain[k + 1]].startOff : -1;
                matchValue[tc.compareOff] = vc;
                matchCaseEnd[ftResolved] = ftResolved;
            }
            if (wildcardStart >= 0) {
                matchWildcardOpen[wildcardStart] = matchEnd;
                matchCaseEnd[matchEnd] = matchEnd;
            }
            matchBlockEnd.insert(matchEnd);
        }
    }

    std::unordered_map<int, int> compTernTest, compTernThenBwd, compTernElseBwd;
    std::unordered_map<int, int> compTernTestOp;
    {
        const char* cn = code->name() ? code->name()->value() : "";
        bool isComp = cn && (strcmp(cn, "<listcomp>") == 0 || strcmp(cn, "<setcomp>") == 0
                || strcmp(cn, "<dictcomp>") == 0 || strcmp(cn, "<genexpr>") == 0);
        if (isComp) {
            struct In { int op, arg, off, next; };
            std::vector<In> v; std::unordered_map<int,int> ix;
            { PycBuffer s(code->code()->value(), code->code()->length());
              int o, a, p = 0;
              while (!s.atEof()) {
                  int io = p; bc_next(s, mod, o, a, p);
                  if (o == Pyc::CACHE) continue;
                  ix[io] = (int)v.size(); v.push_back({o,a,io,p});
              } }
            int forOff = -1;
            for (auto& in : v) if (in.op == Pyc::FOR_ITER_A) { forOff = in.off; break; }
            auto isFwdTest = [](int o){ return o == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                    || o == Pyc::POP_JUMP_FORWARD_IF_TRUE_A || o == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                    || o == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A; };
            auto isBwdFilt = [](int o){ return o == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A
                    || o == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A; };
            const int W = (int)sizeof(uint16_t);
            for (size_t i = 0; forOff >= 0 && i < v.size(); ++i) {
                if (!isFwdTest(v[i].op)) continue;
                int elseOff = v[i].next + v[i].arg * W;
                int j = (int)i + 1; int thenBwd = -1;
                while (j < (int)v.size() && v[j].off < elseOff) {
                    if (isBwdFilt(v[j].op) && v[j].next - v[j].arg * W == forOff) { thenBwd = j; break; }
                    if (v[j].op == Pyc::FOR_ITER_A || isFwdTest(v[j].op)) break;
                    j++;
                }
                if (thenBwd < 0 || thenBwd + 1 >= (int)v.size()
                        || v[thenBwd+1].op != Pyc::JUMP_FORWARD_A) continue;
                int keepOff = v[thenBwd+1].next + v[thenBwd+1].arg * W;
                if (!ix.count(elseOff)) continue;
                int k = ix[elseOff]; int elseBwd = -1;
                while (k < (int)v.size() && v[k].off < keepOff) {
                    if (isBwdFilt(v[k].op) && v[k].next - v[k].arg * W == forOff) { elseBwd = k; break; }
                    if (v[k].op == Pyc::FOR_ITER_A || isFwdTest(v[k].op)) break;
                    k++;
                }
                if (elseBwd < 0 || v[elseBwd].next != keepOff) continue;
                compTernTest[v[i].off] = v[i].next;
                compTernTestOp[v[i].off] = v[i].op;
                compTernThenBwd[v[thenBwd].off] = elseOff;
                compTernElseBwd[v[elseBwd].off] = keepOff;
            }
        }
    }

    /* Pre-scan: an ALL-TERMINAL try/except (every clause returns/raises — no clause
       jumps forward to a post-handler merge, and the try body itself returns/raises
       with no normal-exit jump) with NO else, whose handler chain is followed by its
       lasti cleanup stub and then REAL enclosing-level code (a function-final return /
       an elif false-target). The terminal clauses have no jump-to-merge, so the LAST
       clause's end cannot be inferred and it absorbs that trailing code as dead body
       (unittest/loader._find_test_path dir branch). Bound it: terminalExceptMerge[eT]=M
       (M = handler-chain end past the cleanup stub) so the single-clause except open
       and the bare-end bump close it at M. Loop try/excepts are left alone (a back-edge
       makes M a loop exit, not a terminal merge). */
    if (mod->verCompare(3, 11) >= 0 && !exception_entries.empty()) {
        const int W = (int)sizeof(uint16_t);
        auto opAt = [&](int off) -> int {
            PycBuffer b(code->code()->value(), code->code()->length());
            b.setPos(off);
            int o = Pyc::PYC_INVALID_OPCODE, a = 0, np = off;
            if (!b.atEof()) bc_next(b, mod, o, a, np);
            return o; };
        for (const auto& e0 : exception_entries) {
            if (e0.push_lasti) continue;
            int eT = e0.target;
            if (terminalExceptMerge.count(eT)) continue;
            if (opAt(eT) != Pyc::PUSH_EXC_INFO) continue;
            int hmax = eT; bool grew = true;
            while (grew) {
                grew = false;
                for (const auto& he : exception_entries)
                    if (he.start_offset >= eT && he.start_offset <= hmax) {
                        if (he.end_offset > hmax) { hmax = he.end_offset; grew = true; }
                        if (he.target > hmax) { hmax = he.target; grew = true; }
                    }
            }
            int M = hmax;
            {
                PycBuffer mb(code->code()->value(), code->code()->length());
                mb.setPos(hmax);
                int mo, ma, mp = hmax;
                while (!mb.atEof()) {
                    int mip = mp;
                    bc_next(mb, mod, mo, ma, mp);
                    if (mo == Pyc::COPY_A || mo == Pyc::POP_EXCEPT
                            || mo == Pyc::RERAISE_A || mo == Pyc::NOP
                            || mo == Pyc::CACHE || mo == Pyc::EXTENDED_ARG_A) {
                        M = mp;
                        if (mp <= mip) break;
                        continue;
                    }
                    M = mip;
                    break;
                }
            }
            if (M <= eT) continue;
            int mOp = opAt(M);
            if (mOp == Pyc::PYC_INVALID_OPCODE || mOp == Pyc::RERAISE_A
                    || mOp == Pyc::PUSH_EXC_INFO)
                continue;
            bool allTerminal = true, hasBare = false, multiClause = false;
            int nChecks = 0;
            int bodyStart = eT;
            for (const auto& be : exception_entries)
                if (be.target == eT && be.start_offset < bodyStart)
                    bodyStart = be.start_offset;
            {
                PycBuffer bb(code->code()->value(), code->code()->length());
                bb.setPos(bodyStart);
                int bo, ba, bp = bodyStart;
                while (bp < hmax && !bb.atEof()) {
                    int bip = bp;
                    bc_next(bb, mod, bo, ba, bp);
                    if (bp < eT && bo == Pyc::JUMP_FORWARD_A && bp + ba * W >= hmax) {
                        allTerminal = false; break;
                    }
                    /* a back-edge in the span = in/around a loop -> M is a loop exit */
                    if (bo == Pyc::JUMP_BACKWARD_A
                            || bo == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A
                            || bo == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A
                            || bo == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A
                            || bo == Pyc::FOR_ITER_A) {
                        allTerminal = false; break;
                    }
                    if (bp <= bip) break;
                }
            }
            if (allTerminal) {
                PycBuffer hb(code->code()->value(), code->code()->length());
                hb.setPos(eT);
                int ho, ha, hp = eT; bool prevCheck = false;
                while (hp < hmax && !hb.atEof()) {
                    int hip = hp;
                    bc_next(hb, mod, ho, ha, hp);
                    if (ho == Pyc::JUMP_FORWARD_A && hp + ha * W >= M) {
                        allTerminal = false; break;
                    }
                    if (ho == Pyc::CHECK_EXC_MATCH) nChecks++;
                    if (prevCheck
                            && (ho == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                                || ho == Pyc::POP_JUMP_FORWARD_IF_TRUE_A)) {
                        int t = hp + ha * W;
                        if (t > eT && t < hmax) {
                            multiClause = true;
                            if (opAt(t) == Pyc::POP_TOP) hasBare = true;
                        }
                    }
                    if (ho != Pyc::CACHE && ho != Pyc::NOP && ho != Pyc::EXTENDED_ARG_A)
                        prevCheck = (ho == Pyc::CHECK_EXC_MATCH);
                    if (hp <= hip) break;
                }
            }
            if (nChecks == 0 && opAt(e0.target + 2) == Pyc::POP_TOP) hasBare = true;
            if (!allTerminal) continue;
            if (!(hasBare || nChecks <= 1)) continue;
            bool nestedInTry = false;
            for (const auto& oe : exception_entries)
                if (!oe.push_lasti && oe.target != eT
                        && oe.start_offset < eT && oe.target > M) {
                    nestedInTry = true; break;
                }
            if (nestedInTry) continue;
            terminalExceptMerge[eT] = M;
            terminalExceptMulti[eT] = multiClause;
        }
    }
    PycRef<ASTNode> ternCondBlk, ternThenVal;
    try {
    while (!source.atEof()) {
#if defined(BLOCK_DEBUG) || defined(STACK_DEBUG)
        fprintf(stderr, "%-7d", pos);
    #ifdef STACK_DEBUG
        fprintf(stderr, "%-5d", (unsigned int)stack_hist.size() + 1);
    #endif
    #ifdef BLOCK_DEBUG
        for (unsigned int i = 0; i < blocks.size(); i++)
            fprintf(stderr, "    ");
        fprintf(stderr, "%s (%d)", curblock->type_str(), curblock->end());
    #endif
        fprintf(stderr, "\n");
#endif

        if (!dupActiveSkip.empty()) {
            bool dskip = false;
            for (int T : dupActiveSkip)
                if (pos >= T && pos < dupHandlerEnd[T]) {
                    source.setPos(dupHandlerEnd[T]); pos = dupHandlerEnd[T];
                    dskip = true; break;
                }
            if (dskip) continue;
        }

        if (chainedBareExcept.count(pos)
                && curblock->blktype() == ASTBlock::BLK_EXCEPT
                && curblock->end() == pos
                && blocks.size() > 1) {
            PycRef<ASTBlock> prevExc = curblock;
            blocks.pop();
            curblock = blocks.top();
            if (prevExc->size() != 0)
                curblock->append(prevExc.cast<ASTNode>());
            if (!stack_hist.empty()) {
                stack = stack_hist.top();
                stack_hist.pop();
            }
            stack_hist.push(stack);
            int bareEnd = chainedBareExcept[pos];
            /* Terminal try/except last (bare) clause: its handlers all return/raise, so
               the protected-range end stops before the post-cleanup merge — bound it at
               the handler-chain merge M so the code after the try/except[/else] is not
               absorbed. */
            for (const auto& tm : terminalExceptMerge)
                if (tm.first <= pos && pos < tm.second && tm.second > bareEnd)
                    bareEnd = tm.second;
            PycRef<ASTBlock> bareExc = new ASTCondBlock(
                    ASTBlock::BLK_EXCEPT, bareEnd, NULL, false);
            bareExc->init();
            blocks.push(bareExc);
            curblock = blocks.top();
        }

        if (finElseAt.count(pos)
                && curblock->blktype() == ASTBlock::BLK_IF
                && curblock->end() == pos
                && blocks.size() > 1) {
            PycRef<ASTBlock> ifb = curblock;
            blocks.pop();
            curblock = blocks.top();
            curblock->append(ifb.cast<ASTNode>());
            PycRef<ASTBlock> elseb = new ASTBlock(ASTBlock::BLK_ELSE, finElseAt[pos]);
            elseb->init();
            blocks.push(elseb);
            curblock = blocks.top();
        }

        /* Open the `else: raise` arm of an `if C: … else: raise` whose if-body's
           jump-over-else was NESTED (so the standard JUMP_FORWARD-keyed else-open
           missed it and the BLK_IF would close here, rendering the raise
           unconditionally — see the ifElseRaise pre-scan). Close the finished
           BLK_IF and open a real BLK_ELSE[pos, M) so the raise renders inside it
           and the enclosing block's fall-through (e.g. a loop continue) stays
           reachable. */
        if (ifElseRaise.count(pos)
                && curblock->blktype() == ASTBlock::BLK_IF
                && curblock->end() == pos
                && blocks.size() > 1) {
            PycRef<ASTBlock> ifb = curblock;
            blocks.pop();
            curblock = blocks.top();
            curblock->append(ifb.cast<ASTNode>());
            PycRef<ASTBlock> elseb = new ASTBlock(ASTBlock::BLK_ELSE, ifElseRaise[pos]);
            elseb->init();
            blocks.push(elseb);
            curblock = blocks.top();
        }

        if (loopTailElseAt.count(pos)
                && curblock->blktype() == ASTBlock::BLK_IF
                && curblock->end() == pos
                && blocks.size() > 1) {
            PycRef<ASTBlock> ifb = curblock;
            blocks.pop();
            curblock = blocks.top();
            curblock->append(ifb.cast<ASTNode>());
            PycRef<ASTBlock> elseb = new ASTBlock(ASTBlock::BLK_ELSE, loopTailElseAt[pos]);
            elseb->init();
            blocks.push(elseb);
            curblock = blocks.top();
        }

        /* Open the `else:` arm of an `if C: … else: …` that is the final statement
           of a loop body, where the if-true branch never falls through to the
           else-target (every true-path exits via break/continue/return/raise).
           Without a JUMP_FORWARD over the else the standard else_pop machinery
           never fires, so the BLK_IF would close here and the else body would
           flatten to the loop-body level, dropping the loop continuation that
           follows it (see the loopBodyElseAt pre-scan). */
        if (loopBodyElseAt.count(pos)
                && curblock->blktype() == ASTBlock::BLK_IF
                && curblock->end() == pos
                && blocks.size() > 1) {
            PycRef<ASTBlock> ifb = curblock;
            blocks.pop();
            curblock = blocks.top();
            curblock->append(ifb.cast<ASTNode>());
            PycRef<ASTBlock> elseb = new ASTBlock(ASTBlock::BLK_ELSE, loopBodyElseAt[pos]);
            elseb->init();
            blocks.push(elseb);
            curblock = blocks.top();
        }

        if (elifPendingElse.count(pos)
                && curblock->blktype() == ASTBlock::BLK_ELIF
                && curblock->end() == pos
                && blocks.size() > 1) {
            int cap = elifPendingElse[pos];
            bool safe = false; int elseEnd = -1;
            bool contElif = false;
            bool brokeMisc = false;
            bool skippedFinallyCopy = false;
            {
                PycBuffer es(code->code()->value(), code->code()->length());
                es.setPos(pos);
                int eo, ea, ep = pos;
                while (ep < cap && !es.atEof()) {
                    /* This else-of-elif arm is an `except`-clause arm inside a
                       try/finally whose finally is coalesced (rendered once from the
                       handler copy). CPython inlines the finally normal copy after the
                       arm's own POP_EXCEPT and `del <name>`, before its terminal
                       `return`; that inlined copy is a finallyCopySkip region carrying
                       the finally's own conditionals, which this scan would otherwise
                       mis-read as a continuation-elif test and refuse to open the else.
                       Skip the copy (as the renderer does) and continue at the arm's
                       real terminal. Gated to a finallyCopySkip region start, so a plain
                       `try/except: continue` retry loop with no interposed finally is
                       untouched. */
                    if (finallyCopySkip.count(ep)) {
                        int skipTo = ep + finallyCopySkip[ep];
                        if (skipTo > ep && skipTo <= cap) {
                            skippedFinallyCopy = true;
                            es.setPos(skipTo); ep = skipTo;
                            continue;
                        }
                    }
                    bc_next(es, mod, eo, ea, ep);
                    if (eo == Pyc::RAISE_VARARGS_A || eo == Pyc::RETURN_VALUE
                            || eo == Pyc::RETURN_CONST_A || eo == Pyc::RERAISE
                            || eo == Pyc::RERAISE_A) {
                        safe = true; elseEnd = ep; break;
                    }
                    if (eo == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                            || eo == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                            || eo == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                            || eo == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A) {
                        contElif = true; break;
                    }
                    if (eo == Pyc::COMPARE_OP_A
                            || eo == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A
                            || eo == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A
                            || eo == Pyc::JUMP_FORWARD_A || eo == Pyc::JUMP_BACKWARD_A
                            || eo == Pyc::FOR_ITER_A || eo == Pyc::PUSH_EXC_INFO) {
                        brokeMisc = true;
                        break;
                    }
                }
            }
            /* The candidate is a genuine `else:` ONLY when the op PHYSICALLY ending
               right before `pos` is TERMINAL (return/raise/reraise) — then nothing
               FALLS THROUGH into `pos`, so it is reached solely via the last elif's
               false-jump (the no-match path) = the else. If a prior arm or a
               non-terminal last elif falls through into `pos`, it is UNCONDITIONAL
               post-chain code, not an else. */
            bool elifTerminal = false;
            if (curblock->blktype() == ASTBlock::BLK_ELIF
                    && opEndingBefore.count(pos)) {
                int pe = opEndingBefore[pos];
                elifTerminal = (pe == Pyc::RETURN_VALUE || pe == Pyc::RETURN_CONST_A
                        || pe == Pyc::RAISE_VARARGS_A || pe == Pyc::RERAISE
                        || pe == Pyc::RERAISE_A);
            }
            /* The else body is PLAIN code falling through to the chain merge `cap`
               (terminal-elif whose sibling arms JF past it to `cap`, then shared
               code). Open the BLK_ELSE[pos, cap) IFF `cap` is NOT the implicit
               `return None` epilogue — when cap is real shared code the original IS
               the else (only the skeleton differs); when cap is `LOAD_CONST None;
               RETURN_VALUE` the else-render adds a return None and the original is
               genuinely unconditional post-chain code. */
            bool plainFallElse = false;
            if (elifTerminal && !safe && !contElif && !brokeMisc && cap > pos && stack.empty()) {
                PycBuffer cs(code->code()->value(), code->code()->length());
                cs.setPos(cap);
                int c1o, c1a, cp = cap; bool capEpilogue = false;
                if (!cs.atEof()) {
                    bc_next(cs, mod, c1o, c1a, cp);
                    if (c1o == Pyc::LOAD_CONST_A
                            && code->getConst(c1a).type() == PycObject::TYPE_NONE
                            && !cs.atEof()) {
                        int c2o, c2a, cp2 = cp;
                        bc_next(cs, mod, c2o, c2a, cp2);
                        if (c2o == Pyc::RETURN_VALUE) capEpilogue = true;
                    }
                }
                plainFallElse = !capEpilogue;
            }
            /* The safe terminal else was reachable here only by skipping an interposed
               finally-normal-copy, so end the else at the chain merge (the shared
               POP_EXCEPT): the arm's own trailing return (which follows the skipped
               copy) renders inside the else, and the preceding fall-through arm is not
               captured by it. */
            if (safe && skippedFinallyCopy && elseEnd < cap)
                elseEnd = cap;
            if (safe && elseEnd <= cap && stack.empty()) {
                PycRef<ASTBlock> elifb = curblock;
                blocks.pop();
                curblock = blocks.top();
                curblock->append(elifb.cast<ASTNode>());
                stack_hist.push(stack);
                PycRef<ASTBlock> elseb = new ASTBlock(ASTBlock::BLK_ELSE, elseEnd);
                elseb->init();
                blocks.push(elseb);
                curblock = blocks.top();
            } else if (contElif && stack.empty() && cap > pos
                       && opEndingBefore.count(pos)
                       && opEndingBefore[pos] == Pyc::JUMP_BACKWARD_A) {
                PycRef<ASTBlock> elifb = curblock;
                blocks.pop();
                curblock = blocks.top();
                curblock->append(elifb.cast<ASTNode>());
                stack_hist.push(stack);
                PycRef<ASTBlock> elseb = new ASTBlock(ASTBlock::BLK_ELSE, cap);
                elseb->init();
                blocks.push(elseb);
                curblock = blocks.top();
            } else if (plainFallElse) {
                /* terminal-elif `else:` whose body is plain code falling to a
                   non-epilogue merge `cap` — open BLK_ELSE[pos, cap). */
                PycRef<ASTBlock> elifb = curblock;
                blocks.pop();
                curblock = blocks.top();
                curblock->append(elifb.cast<ASTNode>());
                stack_hist.push(stack);
                PycRef<ASTBlock> elseb = new ASTBlock(ASTBlock::BLK_ELSE, cap);
                elseb->init();
                blocks.push(elseb);
                curblock = blocks.top();
            }
        }

        if ((curblock->blktype() == ASTBlock::BLK_ELSE
                || curblock->blktype() == ASTBlock::BLK_ELIF)
                && curblock->end() > pos
                && stack.empty()
                && blocks.size() > 2
                && !curblock->nodes().empty()
                && curblock->nodes().back().type() == ASTNode::NODE_KEYWORD
                && curblock->nodes().back().cast<ASTKeyword>()->key() == ASTKeyword::KW_BREAK) {
            int innerEnd = curblock->end();
            PycRef<ASTBlock> inner = curblock;
            blocks.pop();
            PycRef<ASTBlock> encl = blocks.top();
            if ((encl->blktype() == ASTBlock::BLK_IF
                    || encl->blktype() == ASTBlock::BLK_ELIF)
                    && encl->end() == pos) {
                blocks.pop();
                encl->append(inner.cast<ASTNode>());
                curblock = blocks.top();
                curblock->append(encl.cast<ASTNode>());
                stack_hist.push(stack);
                PycRef<ASTBlock> elseb = new ASTBlock(ASTBlock::BLK_ELSE, innerEnd);
                elseb->init();
                blocks.push(elseb);
                curblock = blocks.top();
            } else {
                blocks.push(inner);
                curblock = inner;
            }
        }

        if ((curblock->blktype() == ASTBlock::BLK_FOR
                || curblock->blktype() == ASTBlock::BLK_ASYNCFOR)
                && curblock->inited()
                && !curblock.cast<ASTIterBlock>()->isComprehension()
                && blocks.size() > 1) {
            auto ec = forEarlyClose.find(curblock.cast<ASTIterBlock>()->start());
            if (ec != forEarlyClose.end() && ec->second == pos) {
                PycRef<ASTBlock> fb = curblock;
                blocks.pop();
                curblock = blocks.top();
                curblock->append(fb.cast<ASTNode>());
            }
        }

        if ((curblock->blktype() == ASTBlock::BLK_FOR
                || curblock->blktype() == ASTBlock::BLK_ASYNCFOR)
                && curblock->inited()
                && !curblock.cast<ASTIterBlock>()->isComprehension()
                && curblock->end() == pos && pos > 0
                && blocks.size() > 1) {
            int fstart = curblock.cast<ASTIterBlock>()->start();
            bool hasExc = false;
            if (forStartToExit.count(fstart) && forStartToExit[fstart] == pos
                    && !forHasBreak.count(fstart) && !forBreakBeyondExit.count(fstart)
                    && !forExitFallThrough.count(fstart)) {
                PycBuffer es(code->code()->value(), code->code()->length());
                es.setPos(fstart);
                int eo, ea, ep = fstart;
                while (ep < pos && !es.atEof()) {
                    bc_next(es, mod, eo, ea, ep);
                    if (eo == Pyc::PUSH_EXC_INFO) { hasExc = true; break; }
                }
            }
            if (hasExc) {
                PycRef<ASTBlock> fb = curblock;
                blocks.pop();
                curblock = blocks.top();
                curblock->append(fb.cast<ASTNode>());
            }
        }

        /* break-escape finally: an inlined finally copy at a loop-body `break` (all
           exits jump to the loop exit).  The finally renders once as the coalesced
           `finally:`; here we emit a single `break` in its place and skip the copy
           (the break runs the finally then exits the loop).  Covers both the
           exceptional (`except E: …; break`) and the guarded (`if C: …; break`)
           loop-body breaks that share the keepalive teardown. */
        {
            auto bcs = breakEscCopySkip.find(pos);
            if (bcs != breakEscCopySkip.end()) {
                curblock->append(new ASTKeyword(ASTKeyword::KW_BREAK));
                int np = pos + bcs->second;
                source.setPos(np);
                pos = np;
                while (next_exception_entry < exception_entries.size()
                        && exception_entries[next_exception_entry].start_offset < np)
                    next_exception_entry++;
                /* the skip bypasses else_pop: close any enclosing if/elif/else whose
                   merge is at/before the resume so the post-break code renders at the
                   right level (the guarded `if C: …; break` merge). */
                while (blocks.size() > 1
                        && (curblock->blktype() == ASTBlock::BLK_IF
                            || curblock->blktype() == ASTBlock::BLK_ELIF
                            || curblock->blktype() == ASTBlock::BLK_ELSE)
                        && curblock->end() > 0 && curblock->end() <= np) {
                    PycRef<ASTBlock> tmp = curblock;
                    blocks.pop();
                    curblock = blocks.top();
                    curblock->append(tmp.cast<ASTNode>());
                }
                if (source.atEof())
                    break;
                continue;
            }
        }

        {
            auto fcs = finallyCopySkip.find(pos);
            if (fcs != finallyCopySkip.end()) {
                while (curblock->blktype() == ASTBlock::BLK_CASE
                        && curblock->end() <= pos && blocks.size() > 1) {
                    PycRef<ASTBlock> cs = curblock;
                    blocks.pop();
                    curblock = blocks.top();
                    curblock->append(cs.cast<ASTNode>());
                }
                if (curblock->blktype() == ASTBlock::BLK_MATCH
                        && curblock->end() <= pos && blocks.size() > 1) {
                    PycRef<ASTBlock> ms = curblock;
                    blocks.pop();
                    curblock = blocks.top();
                    curblock->append(ms.cast<ASTNode>());
                }
                if (blocks.size() > 2 && curblock->blktype() == ASTBlock::BLK_EXCEPT
                        && curblock->size() > 0
                        && curblock->nodes().back()->type() == ASTNode::NODE_RAISE) {
                    std::stack<PycRef<ASTBlock> > pk = blocks; pk.pop();
                    bool fsh = false;
                    if (!pk.empty() && pk.top()->blktype() == ASTBlock::BLK_CONTAINER
                            && pk.top().cast<ASTContainerBlock>()->hasExcept()
                            && !pk.top().cast<ASTContainerBlock>()->hasFinally()) {
                        pk.pop();
                        if (!pk.empty() && pk.top()->blktype() == ASTBlock::BLK_TRY
                                && openFinallyTargets.count(pk.top()->end()))
                            fsh = true;
                    }
                    if (fsh) {
                        PycRef<ASTBlock> exc = curblock;
                        blocks.pop();
                        if (exc->size() != 0)
                            blocks.top()->append(exc.cast<ASTNode>());
                        curblock = blocks.top();
                        if (!stack_hist.empty())
                            stack_hist.pop();
                        PycRef<ASTBlock> cont = curblock;
                        blocks.pop();
                        blocks.top()->append(cont.cast<ASTNode>());
                        curblock = blocks.top();
                    }
                }
                auto closeableInner = [&](const PycRef<ASTBlock>& b) -> bool {
                    int bt = b->blktype();
                    if ((bt == ASTBlock::BLK_IF || bt == ASTBlock::BLK_ELIF
                            || bt == ASTBlock::BLK_ELSE || bt == ASTBlock::BLK_EXCEPT)
                            && b->end() == pos)
                        return true;
                    if (bt == ASTBlock::BLK_CONTAINER
                            && b.cast<ASTContainerBlock>()->hasExcept()
                            && !b.cast<ASTContainerBlock>()->hasFinally())
                        return true;
                    return false;
                };
                if (blocks.size() > 1 && closeableInner(curblock)) {
                    std::stack<PycRef<ASTBlock> > peek = blocks;
                    int nif = 0; bool shape = false, sawExcept = false, sawIf = false;
                    while (peek.size() > 1 && closeableInner(peek.top())) {
                        int pt = peek.top()->blktype();
                        if (pt == ASTBlock::BLK_EXCEPT || pt == ASTBlock::BLK_CONTAINER) sawExcept = true;
                        if (pt == ASTBlock::BLK_IF || pt == ASTBlock::BLK_ELIF || pt == ASTBlock::BLK_ELSE) sawIf = true;
                        peek.pop(); nif++;
                        if (peek.top()->blktype() == ASTBlock::BLK_TRY) { shape = true; break; }
                    }
                    if (shape && sawExcept && !sawIf) shape = false;
                    if (shape) {
                        for (int k = 0; k < nif; ++k) {
                            PycRef<ASTBlock> ib = curblock;
                            blocks.pop();
                            if (!stack_hist.empty()) stack_hist.pop();
                            curblock = blocks.top();
                            curblock->append(ib.cast<ASTNode>());
                        }
                    }
                }
                /* Inner-try deferred finally-return (see finTryReturnRet): this
                   finally normal-copy is the inlined finally of a `return X` that is
                   the LAST statement of the inner try body (the try body's normal
                   exit runs the finally then loads+returns X, past the loop exit).
                   The inner BLK_TRY is still curblock at this copy-skip point, so
                   append the synthesized `return X` into it now — it renders as the
                   try's last statement, before the except handler opens. */
                if (!finTryReturnRet.empty()) {
                    auto ftr = finTryReturnRet.find(pos);
                    if (ftr != finTryReturnRet.end()
                            && curblock->blktype() == ASTBlock::BLK_TRY) {
                        curblock->append(ftr->second);
                        finTryReturnRet.erase(ftr);
                    }
                }
                int np = pos + fcs->second;
                source.setPos(np);
                pos = np;
                if (source.atEof())
                    break;
                continue;
            }
        }

        {
            auto hsk = withHandlerSkip.find(pos);
            if (hsk != withHandlerSkip.end()) {
                source.setPos(hsk->second);
                pos = hsk->second;
                if (source.atEof())
                    break;
            }
        }

        if (withOpen.count(pos)) {
            PycRef<ASTBlock> wblk = new ASTWithBlock(withOpen[pos]);
            if (asyncWithBody.count(pos))
                wblk.cast<ASTWithBlock>()->setAsync(true);
            blocks.push(wblk);
            curblock = blocks.top();
        }

        if (matchCaseEnd.count(pos) && blocks.size() > 1
                && curblock->blktype() != ASTBlock::BLK_CASE) {
            std::stack<PycRef<ASTBlock> > peek = blocks;
            bool shape = false;
            while (peek.size() > 1) {
                ASTBlock::BlkType bt = peek.top()->blktype();
                if (bt == ASTBlock::BLK_CASE) { shape = true; break; }
                if (bt != ASTBlock::BLK_EXCEPT && bt != ASTBlock::BLK_CONTAINER
                        && bt != ASTBlock::BLK_TRY && bt != ASTBlock::BLK_FINALLY
                        && bt != ASTBlock::BLK_IF && bt != ASTBlock::BLK_ELSE
                        && bt != ASTBlock::BLK_ELIF) break;
                peek.pop();
            }
            if (shape) {
                while (curblock->blktype() != ASTBlock::BLK_CASE && blocks.size() > 1) {
                    PycRef<ASTBlock> inner = curblock;
                    ASTBlock::BlkType bt = inner->blktype();
                    blocks.pop();
                    if ((bt == ASTBlock::BLK_TRY || bt == ASTBlock::BLK_CONTAINER
                            || bt == ASTBlock::BLK_FINALLY) && !stack_hist.empty())
                        stack_hist.pop();
                    curblock = blocks.top();
                    if (bt == ASTBlock::BLK_EXCEPT && inner->size() == 0
                            && inner.cast<ASTCondBlock>()->cond() == NULL)
                        continue;
                    curblock->append(inner.cast<ASTNode>());
                }
            }
        }

        if (matchCaseEnd.count(pos) && blocks.size() > 1
                && (curblock->blktype() == ASTBlock::BLK_IF
                    || curblock->blktype() == ASTBlock::BLK_ELSE
                    || curblock->blktype() == ASTBlock::BLK_ELIF)) {
            std::stack<PycRef<ASTBlock> > peek = blocks;
            int nif = 0; bool shape = false;
            while (peek.size() > 1
                    && (peek.top()->blktype() == ASTBlock::BLK_IF
                        || peek.top()->blktype() == ASTBlock::BLK_ELSE
                        || peek.top()->blktype() == ASTBlock::BLK_ELIF)) {
                peek.pop(); nif++;
                if (peek.top()->blktype() == ASTBlock::BLK_CASE) { shape = true; break; }
            }
            if (shape) {
                for (int k = 0; k < nif; ++k) {
                    PycRef<ASTBlock> inner = curblock;
                    blocks.pop();
                    if (!stack_hist.empty())
                        stack_hist.pop();
                    curblock = blocks.top();
                    curblock->append(inner.cast<ASTNode>());
                }
            }
        }

        if (curblock->blktype() == ASTBlock::BLK_CASE && matchCaseEnd.count(pos)) {
            PycRef<ASTBlock> cs = curblock;
            blocks.pop();
            curblock = blocks.top();
            curblock->append(cs.cast<ASTNode>());
            int after = matchCaseEnd[pos];
            source.setPos(after);
            pos = after;
            while (next_exception_entry < exception_entries.size()
                    && exception_entries[next_exception_entry].start_offset < pos)
                next_exception_entry++;
            continue;
        }
        if (curblock->blktype() == ASTBlock::BLK_MATCH && matchWildcardOpen.count(pos)) {
            PycRef<PycString> us = new PycString();
            us->setValue("_");
            blocks.push(new ASTCaseBlock(matchWildcardOpen[pos], new ASTName(us)));
            curblock = blocks.top();
            curblock->init();
        }

        if (nestedElseHandlerVals.count(pos) && blocks.size() > 1
                && !(curblock->blktype() == ASTBlock::BLK_ELSE
                     && curblock->end() == pos)) {
            bool pendingElse = false;
            {
                std::stack<PycRef<ASTBlock> > pk = blocks;
                while (!pk.empty()) {
                    ASTBlock::BlkType bt = pk.top()->blktype();
                    if (bt == ASTBlock::BLK_ELSE && pk.top()->end() == pos) {
                        pendingElse = true; break;
                    }
                    if (bt != ASTBlock::BLK_CONTAINER && bt != ASTBlock::BLK_EXCEPT
                            && bt != ASTBlock::BLK_TRY && bt != ASTBlock::BLK_FINALLY)
                        break;
                    pk.pop();
                }
            }
            if (pendingElse) {
                while (blocks.size() > 1
                        && !(curblock->blktype() == ASTBlock::BLK_ELSE
                             && curblock->end() == pos)) {
                    PycRef<ASTBlock> inner = curblock;
                    if (inner->blktype() == ASTBlock::BLK_TRY
                            || inner->blktype() == ASTBlock::BLK_CONTAINER) {
                        if (!stack_hist.empty())
                            stack_hist.pop();
                    }
                    blocks.pop();
                    curblock = blocks.top();
                    if (inner->blktype() == ASTBlock::BLK_EXCEPT
                            && inner->size() == 0
                            && inner.cast<ASTCondBlock>()->cond() == NULL)
                        continue;
                    curblock->append(inner.cast<ASTNode>());
                }
            }
        }

        if (curblock->blktype() == ASTBlock::BLK_ELSE
                && curblock->end() == pos
                && nestedElseHandlerVals.count(pos)
                && blocks.size() > 1) {
            PycRef<ASTBlock> elseb = curblock;
            blocks.pop();
            curblock = blocks.top();
            if (curblock->blktype() == ASTBlock::BLK_CONTAINER
                    && curblock.cast<ASTContainerBlock>()->hasExcept()) {
                if (!stack_hist.empty()) {
                    stack = stack_hist.top();
                    stack_hist.pop();
                }
                curblock->append(elseb.cast<ASTNode>());
                stack_hist.push(stack);
                int teoEnd = teoExceptMerge.count(pos) ? teoExceptMerge[pos] : 0;
                PycRef<ASTBlock> except = new ASTCondBlock(ASTBlock::BLK_EXCEPT, teoEnd, NULL, false);
                except->init();
                blocks.push(except.cast<ASTBlock>());
                curblock = blocks.top();
            } else {
                blocks.push(elseb.cast<ASTBlock>());
                curblock = elseb;
            }
        }

        if (curblock->blktype() == ASTBlock::BLK_MATCH && matchBlockEnd.count(pos)) {
            PycRef<ASTBlock> ms = curblock;
            blocks.pop();
            curblock = blocks.top();
            curblock->append(ms.cast<ASTNode>());
        }

        while (next_exception_entry < exception_entries.size()
                && exception_entries[next_exception_entry].start_offset < pos) {
            next_exception_entry++;
        }

        while (next_exception_entry < exception_entries.size()
                && exception_entries[next_exception_entry].start_offset == pos
                && !exception_entries[next_exception_entry].push_lasti
                && openFinallyTargets.count(exception_entries[next_exception_entry].target)) {
            next_exception_entry++;
        }

        while (next_exception_entry < exception_entries.size()
                && exception_entries[next_exception_entry].start_offset == pos
                && !exception_entries[next_exception_entry].push_lasti
                && openExceptTargets.count(exception_entries[next_exception_entry].target)) {
            next_exception_entry++;
        }

        /* A try-body that contains a mid-body construct exiting the function (an
           `if C: ...; return` arm) is split by the compiler into two protected
           fragments that share one handler: [body_start, gap) and [gap_end, end),
           with the unprotected return path in between. The first fragment already
           opened the enclosing BLK_TRY; the second fragment must NOT re-open a
           nested try for the same handler — doing so would push a spurious
           container/try on top of any block (e.g. a rotated `while` whose bottom
           test lives in the second fragment), tearing the loop apart. If we are
           still physically inside a BLK_TRY whose handler is this entry's target,
           and an earlier fragment with the same target exists, skip the
           continuation; the original try stays open and closes at its handler. */
        while (next_exception_entry < exception_entries.size()
                && exception_entries[next_exception_entry].start_offset == pos
                && !exception_entries[next_exception_entry].push_lasti
                && exception_entries[next_exception_entry].stack_depth == 0) {
            int T = exception_entries[next_exception_entry].target;
            bool insideSameTry = false;
            { std::stack<PycRef<ASTBlock> > bs = blocks;
              while (!bs.empty()) {
                  if (bs.top()->blktype() == ASTBlock::BLK_TRY
                          && bs.top()->end() == T) { insideSameTry = true; break; }
                  bs.pop();
              } }
            bool hasEarlierFrag = false;
            for (const auto& e2 : exception_entries)
                if (e2.target == T && !e2.push_lasti
                        && e2.start_offset < pos) { hasEarlierFrag = true; break; }
            if (insideSameTry && hasEarlierFrag)
                next_exception_entry++;
            else
                break;
        }

        auto isEmptyReraiseHandler = [&](int T) -> bool {
            PycBuffer hb(code->code()->value(), code->code()->length());
            hb.setPos(T);
            int o, a, p = T;
            if (hb.atEof()) return false;
            bc_next(hb, mod, o, a, p);
            if (o != Pyc::PUSH_EXC_INFO) return false;
            while (!hb.atEof()) {
                int ip = p;
                bc_next(hb, mod, o, a, p);
                if (o != Pyc::CACHE && o != Pyc::NOP) break;
                if (p <= ip) return false;
            }
            return o == Pyc::RERAISE || o == Pyc::RERAISE_A;
        };
        while (next_exception_entry < exception_entries.size()
                && exception_entries[next_exception_entry].start_offset == pos
                && !exception_entries[next_exception_entry].push_lasti
                && isEmptyReraiseHandler(exception_entries[next_exception_entry].target)) {
            int tgt = exception_entries[next_exception_entry].target;
            int nfrag = 0;
            for (const auto& e2 : exception_entries)
                if (e2.target == tgt && !e2.push_lasti) nfrag++;
            if (nfrag < 2) break;
            next_exception_entry++;
        }

        /* A loop-tail `if C:` whose true-arm ends in the loop back-edge (a `continue`,
           JUMP_BACKWARD) gets no else_pop (there is no forward jump-over-else), so it
           does NOT auto-close at its false-target. When that false-target EXACTLY
           coincides with an exception-entry (try) start, the try would otherwise open
           NESTED inside the still-open BLK_IF — absorbing the else's try/except as dead
           code after the rendered `continue`. Close the finished BLK_IF into its parent
           FIRST; the try then opens at the parent level and the else renders flattened,
           byte-identical to `if C: A; continue` + B. A genuine if/ELSE (true-arm forward
           jump-over-else) closes via the normal else_pop path and never reaches here. */
        if ((finallyOpenAt.count(pos) || exceptOpenAt.count(pos)
                || (next_exception_entry < exception_entries.size()
                    && exception_entries[next_exception_entry].start_offset == pos
                    && !exception_entries[next_exception_entry].push_lasti))
                && curblock->blktype() == ASTBlock::BLK_IF
                && curblock->end() == pos
                && blocks.size() > 1) {
            int prevOp = -1;
            { PycBuffer s(code->code()->value(), code->code()->length());
              int o, a, p = 0;
              while (!s.atEof()) {
                  bc_next(s, mod, o, a, p);
                  if (p == pos && o != Pyc::CACHE) { prevOp = o; break; }
                  if (p > pos) break;
              } }
            if (prevOp == Pyc::JUMP_BACKWARD_A
                    || prevOp == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A) {
                PycRef<ASTBlock> ifb = curblock;
                blocks.pop();
                curblock = blocks.top();
                curblock->append(ifb.cast<ASTNode>());
            }
        }

        int finMaxTarget = -1;
        if (finallyOpenAt.count(pos))
            for (const auto& fp : finallyOpenAt[pos])
                if (fp.target > finMaxTarget) finMaxTarget = fp.target;
        bool exceptOuter = finMaxTarget >= 0 && exceptOpenAt.count(pos)
                && !openExceptTargets.count(exceptOpenAt[pos])
                && exceptOpenAt[pos] > finMaxTarget;
        if (exceptOuter) {
            int T = exceptOpenAt[pos];
            PycRef<ASTBlock> cont = new ASTContainerBlock(0, T);
            blocks.push(cont.cast<ASTBlock>());
            curblock = blocks.top();
            stack_hist.push(stack);
            PycRef<ASTBlock> tryblock = new ASTBlock(ASTBlock::BLK_TRY, T, true);
            blocks.push(tryblock.cast<ASTBlock>());
            curblock = blocks.top();
            openExceptTargets.insert(T);
            while (next_exception_entry < exception_entries.size()
                    && exception_entries[next_exception_entry].start_offset == pos
                    && !exception_entries[next_exception_entry].push_lasti
                    && exception_entries[next_exception_entry].target == T)
                next_exception_entry++;
        }

        if (finallyOpenAt.count(pos)) {
            std::vector<FinallyCoalesce> fplans = finallyOpenAt[pos];
            for (size_t a = 1; a < fplans.size(); ++a)
                for (size_t b = a; b > 0 && fplans[b-1].target < fplans[b].target; --b) {
                    FinallyCoalesce tmp = fplans[b-1];
                    fplans[b-1] = fplans[b]; fplans[b] = tmp;
                }
            /* When a try/except (exceptOpenAt) ALSO begins at this offset and it is
               NESTED BETWEEN two coalesced finallys sharing the body start — an
               OUTER finally (target > except) wrapping the `try/except` whose body
               in turn wraps an INNER finally (target < except): `try: (try: (try: B
               finally: F_in) except E: H) finally: F_out`.  The default order opens
               BOTH finallys first, so the inner finally wrongly wraps the except.
               Split the opens around the except: open the outer finally(s) (target >
               except), then the try/except, then the inner finally(s).  Tightly
               gated: fires only when exceptOpenAt[pos] exists, is not already open,
               and plans exist on BOTH sides of its target. */
            int splitExcT = -1;
            if (exceptOpenAt.count(pos) && !openExceptTargets.count(exceptOpenAt[pos])) {
                int et = exceptOpenAt[pos];
                bool above = false, below = false;
                for (const auto& p : fplans) {
                    if (p.target > et) above = true;
                    else if (p.target < et) below = true;
                }
                if (above && below) splitExcT = et;
            }
            for (const FinallyCoalesce& plan : fplans) {
                if (splitExcT >= 0 && plan.target < splitExcT
                        && !openExceptTargets.count(splitExcT)) {
                    /* open the intervening try/except before the inner finally(s). */
                    PycRef<ASTContainerBlock> econt = new ASTContainerBlock(0, splitExcT);
                    blocks.push(econt.cast<ASTBlock>());
                    curblock = blocks.top();
                    stack_hist.push(stack);
                    PycRef<ASTBlock> etry = new ASTBlock(ASTBlock::BLK_TRY, splitExcT, true);
                    blocks.push(etry.cast<ASTBlock>());
                    curblock = blocks.top();
                    openExceptTargets.insert(splitExcT);
                    while (next_exception_entry < exception_entries.size()
                            && exception_entries[next_exception_entry].start_offset == pos
                            && !exception_entries[next_exception_entry].push_lasti
                            && exception_entries[next_exception_entry].target == splitExcT)
                        next_exception_entry++;
                }
                PycRef<ASTContainerBlock> cont = new ASTContainerBlock(plan.target, 0);
                cont->setFinallyRange(plan.finBodyEnd, plan.after);
                blocks.push(cont.cast<ASTBlock>());
                curblock = blocks.top();
                stack_hist.push(stack);
                PycRef<ASTBlock> tryblock = new ASTBlock(ASTBlock::BLK_TRY,
                        plan.tryEnd ? plan.tryEnd : plan.target, true);
                blocks.push(tryblock.cast<ASTBlock>());
                curblock = blocks.top();
                openFinallyTargets.insert(plan.target);
                while (next_exception_entry < exception_entries.size()
                        && exception_entries[next_exception_entry].start_offset == pos
                        && !exception_entries[next_exception_entry].push_lasti
                        && exception_entries[next_exception_entry].target == plan.target)
                    next_exception_entry++;
            }
        }

        /* When several try/except blocks begin at this same offset, open the outer
           (larger-target) ones first so they wrap the primary (inner) try opened
           below. Their handler regions live entirely past the inner body, and the
           tiny protected fragments they own are skipped via openExceptTargets as
           the walk reaches them. */
        if (exceptOpenAtExtra.count(pos)) {
            std::vector<int> extras = exceptOpenAtExtra[pos];
            /* open outermost (largest target) first */
            for (size_t i = 0; i + 1 < extras.size(); ++i)
                for (size_t j = i + 1; j < extras.size(); ++j)
                    if (extras[j] > extras[i]) {
                        int t = extras[i]; extras[i] = extras[j]; extras[j] = t;
                    }
            for (int T : extras) {
                if (openExceptTargets.count(T))
                    continue;
                PycRef<ASTBlock> cont = new ASTContainerBlock(0, T);
                blocks.push(cont.cast<ASTBlock>());
                curblock = blocks.top();
                stack_hist.push(stack);
                PycRef<ASTBlock> tryblock = new ASTBlock(ASTBlock::BLK_TRY, T, true);
                blocks.push(tryblock.cast<ASTBlock>());
                curblock = blocks.top();
                openExceptTargets.insert(T);
            }
        }

        if (exceptOpenAt.count(pos) && !openExceptTargets.count(exceptOpenAt[pos])) {
            int T = exceptOpenAt[pos];
            PycRef<ASTBlock> cont = new ASTContainerBlock(0, T);
            blocks.push(cont.cast<ASTBlock>());
            curblock = blocks.top();
            stack_hist.push(stack);
            PycRef<ASTBlock> tryblock = new ASTBlock(ASTBlock::BLK_TRY, T, true);
            blocks.push(tryblock.cast<ASTBlock>());
            curblock = blocks.top();
            openExceptTargets.insert(T);
            while (next_exception_entry < exception_entries.size()
                    && exception_entries[next_exception_entry].start_offset == pos
                    && !exception_entries[next_exception_entry].push_lasti
                    && exception_entries[next_exception_entry].target == T)
                next_exception_entry++;
        }

        while (next_exception_entry < exception_entries.size()
                && exception_entries[next_exception_entry].start_offset == pos
                && !exception_entries[next_exception_entry].push_lasti
                && dupHandlerEnd.count(exception_entries[next_exception_entry].target)) {
            dupActiveSkip.insert(exception_entries[next_exception_entry].target);
            next_exception_entry++;
        }

        if (next_exception_entry < exception_entries.size()
                && exception_entries[next_exception_entry].start_offset == pos
                && !exception_entries[next_exception_entry].push_lasti
                && !(withResume.count(pos)
                     && curblock->blktype() == ASTBlock::BLK_WITH
                     && curblock->end() == pos)) {
            const auto& entry = exception_entries[next_exception_entry];
            int htype = 0;
            {
                PycBuffer hscan(code->code()->value(), code->code()->length());
                hscan.setPos(entry.target);
                int hop, harg, hp, prevop = -1;
                for (int hi = 0; hi < 40 && !hscan.atEof(); ++hi) {
                    bc_next(hscan, mod, hop, harg, hp);
                    if (prevop == Pyc::PUSH_EXC_INFO && hop == Pyc::POP_TOP) {
                        htype = 1; break;
                    }
                    if (hop == Pyc::CHECK_EXC_MATCH
                            || hop == Pyc::JUMP_FORWARD_A) { htype = 1; break; }
                    if (hop == Pyc::RERAISE || hop == Pyc::RERAISE_A) { htype = 2; break; }
                    prevop = hop;
                }
            }
            bool isFinallyHandler = false;
            {
                int hEnd = -1;
                for (const auto& e2 : exception_entries)
                    if (e2.start_offset == entry.target) { hEnd = e2.end_offset; break; }
                if (hEnd > entry.target) {
                    PycBuffer hs(code->code()->value(), code->code()->length());
                    hs.setPos(entry.target);
                    int ho, ha, hp = entry.target;
                    while (hp < hEnd && !hs.atEof()) {
                        int hip = hp;
                        bc_next(hs, mod, ho, ha, hp);
                        if (hip == entry.target && ho == Pyc::PUSH_EXC_INFO)
                            continue;
                        if (ho == Pyc::POP_EXCEPT || ho == Pyc::CHECK_EXC_MATCH
                                || ho == Pyc::JUMP_FORWARD_A)
                            break;
                        if (ho == Pyc::RERAISE || ho == Pyc::RERAISE_A) {
                            isFinallyHandler = true;
                            break;
                        }
                    }
                }
            }
            int htypeFin = (htype == 2 || (htype == 0 && isFinallyHandler)) ? 2 : htype;

            bool did_finally = false;
            bool inNestedFin = false;
            int nestedFinX = -1;
            for (const auto& r : nestedFinRegions)
                if (entry.start_offset >= r.first && entry.target <= r.second) { inNestedFin = true; nestedFinX = r.second; break; }
            bool enclosingWith = false;
            bool enclosingExcept = false;
            bool enclosingFor = false;
            PycRef<ASTBlock> enclForBlk;
            if (entry.stack_depth == 1) {
                std::stack<PycRef<ASTBlock> > ws = blocks;
                while (!ws.empty()) {
                    if (ws.top()->blktype() == ASTBlock::BLK_WITH) { enclosingWith = true; break; }
                    ws.pop();
                }
                std::stack<PycRef<ASTBlock> > ls = blocks;
                while (!ls.empty()) {
                    if (ls.top()->blktype() == ASTBlock::BLK_FOR) {
                        enclosingFor = true; enclForBlk = ls.top(); break;
                    }
                    ls.pop();
                }
                std::stack<PycRef<ASTBlock> > es = blocks;
                while (!es.empty()) {
                    if (es.top()->blktype() == ASTBlock::BLK_EXCEPT) { enclosingExcept = true; break; }
                    es.pop();
                }
            }
            bool forwardAllowed = (entry.stack_depth == 0
                    || inNestedFin
                    || (entry.stack_depth == 1
                        && (curblock->blktype() == ASTBlock::BLK_WITH || enclosingWith
                            || enclosingExcept)));
            bool backEdgeAllowed = (entry.stack_depth == 1 && enclosingFor);
            int enclConstructs = 0;
            {
                std::stack<PycRef<ASTBlock> > bs = blocks;
                while (!bs.empty()) {
                    ASTBlock::BlkType bt = bs.top()->blktype();
                    if (bt == ASTBlock::BLK_FOR || bt == ASTBlock::BLK_ASYNCFOR
                            || bt == ASTBlock::BLK_WITH || bt == ASTBlock::BLK_EXCEPT)
                        enclConstructs++;
                    bs.pop();
                }
            }
            bool deepNested = (entry.stack_depth >= 2
                    && entry.stack_depth == enclConstructs);
            if (htypeFin == 2 && (forwardAllowed || backEdgeAllowed || deepNested)) {
                int jump_off = -1, after_off = -1, ret_off = -1, raise_off = -1;
                int back_off = -1, back_target = -1;
                PycBuffer s(code->code()->value(), code->code()->length());
                s.setPos(entry.end_offset);
                int sop, sarg, sp = entry.end_offset;
                while (!s.atEof()) {
                    int ip = sp;
                    bc_next(s, mod, sop, sarg, sp);
                    if (sp == entry.target) {
                        if (sop == Pyc::JUMP_FORWARD_A) {
                            jump_off = ip;
                            after_off = sp + sarg * 2;
                        } else if (sop == Pyc::RETURN_VALUE
                                || sop == Pyc::RETURN_CONST_A) {
                            ret_off = ip;
                        } else if (sop == Pyc::JUMP_BACKWARD_A
                                || sop == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A) {
                            back_off = ip;
                            back_target = sp - sarg * 2;
                        } else if (sop == Pyc::RAISE_VARARGS_A
                                || sop == Pyc::RERAISE
                                || sop == Pyc::RERAISE_A) {
                            raise_off = ip;
                        }
                        break;
                    }
                    if (sp > entry.target) break;
                }
                /* A try/finally whose body contains its own try/except blocks is
                   split in the exception table into several fragments that all
                   share this finally's target: the inner handlers punch holes in
                   the outer protected range. Each fragment would otherwise open a
                   fresh outer try, and because a fragment's end falls in the middle
                   of an enclosed loop/handler the try cannot close where expected,
                   so the following fragments nest ever deeper. Span the BLK_TRY over
                   the WHOLE body — to the last same-target fragment's end — and
                   consume the intermediate fragments here so they don't re-open. The
                   inner try/except entries have their own (different) targets and are
                   left untouched, opening normally inside the body. */
                int bodyFragEnd = entry.end_offset;
                int bodyFragCount = 0;
                for (const auto& e2 : exception_entries) {
                    if (e2.push_lasti || e2.target != entry.target)
                        continue;
                    if (e2.start_offset < entry.start_offset
                            || e2.end_offset > entry.target)
                        continue;
                    bodyFragCount++;
                    if (e2.end_offset > bodyFragEnd)
                        bodyFragEnd = e2.end_offset;
                }
                bool multiFragBody = bodyFragCount > 1 && bodyFragEnd > entry.end_offset
                        && bodyFragEnd <= entry.target;
                int finNormEnd = (jump_off > entry.end_offset) ? jump_off
                               : (back_off > entry.end_offset) ? back_off : -1;
                bool depthLoopFin = false;
                if (entry.stack_depth == 1 && enclosingFor && !forwardAllowed
                        && finNormEnd > entry.end_offset) {
                    auto nextReal = [&](PycBuffer& b, int& op, int& arg, int& p) -> int {
                        while (!b.atEof()) {
                            int ip = p; bc_next(b, mod, op, arg, p);
                            if (op != Pyc::NOP && op != Pyc::CACHE) return ip;
                        }
                        op = -1; return -1;
                    };
                    PycBuffer hs(code->code()->value(), code->code()->length());
                    hs.setPos(entry.target);
                    int ho = -1, ha = 0, hp = entry.target;
                    nextReal(hs, ho, ha, hp);
                    if (ho == Pyc::PUSH_EXC_INFO) {
                        PycBuffer ns(code->code()->value(), code->code()->length());
                        ns.setPos(entry.end_offset);
                        int no, na, np = entry.end_offset;
                        int ho2, ha2, hp2 = hp;
                        bool same = true;
                        while (true) {
                            int nip = nextReal(ns, no, na, np);
                            if (nip < 0 || nip >= finNormEnd) break;
                            nextReal(hs, ho2, ha2, hp2);
                            if (no != ho2) { same = false; break; }
                            if (na != ha2) {
                                const int W = (int)sizeof(uint16_t);
                                bool fwdCond = (no == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                                        || no == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                                        || no == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                                        || no == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A);
                                int tgt = np + na * W;
                                if (!(fwdCond && tgt > nip
                                        && tgt >= finNormEnd - 2 * W
                                        && tgt <= finNormEnd + 4 * W)) {
                                    same = false; break;
                                }
                            }
                        }
                        if (same) {
                            int ro = -1, ra = 0, rp = hp2;
                            nextReal(hs, ro, ra, rp);
                            if (ro == Pyc::RERAISE || ro == Pyc::RERAISE_A)
                                depthLoopFin = true;
                        }
                    }
                }
                if ((forwardAllowed || depthLoopFin || deepNested) && jump_off > entry.end_offset && after_off > entry.target) {
                    PycRef<ASTContainerBlock> cont = new ASTContainerBlock(entry.target, 0);
                    cont->setFinallyRange(jump_off, after_off);
                    blocks.push(cont.cast<ASTBlock>());
                    curblock = blocks.top();
                    stack_hist.push(stack);
                    bool spanBody = multiFragBody && bodyFragEnd <= jump_off;
                    int tryEndOff = spanBody ? bodyFragEnd : entry.end_offset;
                    PycRef<ASTBlock> tryblock = new ASTBlock(ASTBlock::BLK_TRY, tryEndOff, true);
                    blocks.push(tryblock.cast<ASTBlock>());
                    curblock = blocks.top();
                    next_exception_entry++;
                    /* The remaining body fragments (the holes the enclosed try/except
                       blocks punch in this finally's protected range) all share this
                       target; mark it open so they are skipped instead of re-opening a
                       fresh outer try as we walk into the body. */
                    if (spanBody)
                        openFinallyTargets.insert(entry.target);
                    did_finally = true;
                } else if (depthLoopFin && back_off > entry.end_offset
                        && jump_off < 0 && ret_off < 0 && raise_off < 0
                        && back_target == enclForBlk.cast<ASTIterBlock>()->start()
                        && enclForBlk->end() > back_off) {
                    int loopExit = enclForBlk->end();
                    PycRef<ASTContainerBlock> cont = new ASTContainerBlock(entry.target, 0);
                    cont->setFinallyRange(back_off, loopExit);
                    blocks.push(cont.cast<ASTBlock>());
                    curblock = blocks.top();
                    stack_hist.push(stack);
                    PycRef<ASTBlock> tryblock = new ASTBlock(ASTBlock::BLK_TRY, entry.end_offset, true);
                    blocks.push(tryblock.cast<ASTBlock>());
                    curblock = blocks.top();
                    next_exception_entry++;
                    loopCloseAtExit.insert(loopExit);
                    loopTailFinClose.insert(back_off);
                    did_finally = true;
                } else if (forwardAllowed && inNestedFin && nestedFinX > entry.target
                        && raise_off > entry.end_offset
                        && jump_off < 0 && ret_off < 0) {
                    PycRef<ASTContainerBlock> cont = new ASTContainerBlock(entry.target, 0);
                    cont->setFinallyRange(raise_off, nestedFinX);
                    blocks.push(cont.cast<ASTBlock>());
                    curblock = blocks.top();
                    stack_hist.push(stack);
                    PycRef<ASTBlock> tryblock = new ASTBlock(ASTBlock::BLK_TRY, entry.end_offset, true);
                    blocks.push(tryblock.cast<ASTBlock>());
                    curblock = blocks.top();
                    next_exception_entry++;
                    did_finally = true;
                } else if (forwardAllowed && ret_off > entry.end_offset) {
                    bool handlerIsFinally = false;
                    {
                        int hEnd = -1;
                        for (const auto& e2 : exception_entries)
                            if (e2.start_offset == entry.target && e2.push_lasti) {
                                hEnd = e2.end_offset; break;
                            }
                        if (hEnd > entry.target) {
                            PycBuffer hs(code->code()->value(), code->code()->length());
                            hs.setPos(entry.target);
                            int ho, ha, hp = entry.target;
                            while (hp < hEnd && !hs.atEof()) {
                                int hip = hp;
                                bc_next(hs, mod, ho, ha, hp);
                                if (hip == entry.target && ho == Pyc::PUSH_EXC_INFO)
                                    continue;
                                if (ho == Pyc::POP_EXCEPT || ho == Pyc::CHECK_EXC_MATCH)
                                    break;
                                if (ho == Pyc::RERAISE || ho == Pyc::RERAISE_A) {
                                    handlerIsFinally = true;
                                    break;
                                }
                            }
                        }
                    }
                    if (!handlerIsFinally)
                        ret_off = -1;
                }
                int epiFinEnd = -1;
                if (ret_off > entry.end_offset) {
                    int ap = -1;
                    {
                        PycBuffer ps(code->code()->value(), code->code()->length());
                        ps.setPos(entry.target);
                        int po, pa, pp = entry.target;
                        if (!ps.atEof()) {
                            bc_next(ps, mod, po, pa, pp);
                            if (po == Pyc::PUSH_EXC_INFO) ap = pp;
                        }
                    }
                    if (ap > 0) {
                        PycBuffer ns(code->code()->value(), code->code()->length());
                        PycBuffer hs(code->code()->value(), code->code()->length());
                        ns.setPos(entry.end_offset);
                        hs.setPos(ap);
                        int no, na, np = entry.end_offset;
                        int ho2, ha2, hp2 = ap;
                        while (np < entry.target && !ns.atEof() && !hs.atEof()) {
                            int nip = np, hip = hp2;
                            bc_next(ns, mod, no, na, np);
                            bc_next(hs, mod, ho2, ha2, hp2);
                            if (no == ho2) {
                                if (np <= nip || hp2 <= hip) break;
                                continue;
                            }
                            if ((ho2 == Pyc::RERAISE || ho2 == Pyc::RERAISE_A)
                                    && no == Pyc::LOAD_CONST_A
                                    && code->getConst(na).type() == PycObject::TYPE_NONE) {
                                PycBuffer n2(code->code()->value(), code->code()->length());
                                n2.setPos(np);
                                int n2o, n2a, n2p = np;
                                if (!n2.atEof()) {
                                    bc_next(n2, mod, n2o, n2a, n2p);
                                    if (n2o == Pyc::RETURN_VALUE
                                            || n2o == Pyc::INSTRUMENTED_RETURN_VALUE_A)
                                        epiFinEnd = nip;
                                }
                            }
                            break;
                        }
                    }
                }
                if (ret_off > entry.end_offset) {
                    int aft = -1;
                    for (const auto& e2 : exception_entries) {
                        if (e2.start_offset == entry.target) {
                            PycBuffer cs(code->code()->value(), code->code()->length());
                            cs.setPos(e2.target);
                            int co, ca, cp = e2.target;
                            while (!cs.atEof()) {
                                bc_next(cs, mod, co, ca, cp);
                                if (co == Pyc::RERAISE || co == Pyc::RERAISE_A) {
                                    aft = cp;
                                    break;
                                }
                                if (co == Pyc::PUSH_EXC_INFO) break;
                            }
                            break;
                        }
                    }
                    if (aft > entry.target) {
                        bool epi = (epiFinEnd > entry.end_offset && epiFinEnd < ret_off);
                        /* CONST deferred return (`try: B; return <const> finally: F`):
                           the const is loaded AFTER the inlined finally copy, so it is
                           NOT on the stack at the try-close — finallyReturnExit (which
                           appends stack.top()) would silently DROP it. End the finally
                           body before the const load and synthesise `return <const>`
                           into the try via nestedFinReturnConst (asyncio Event.wait:
                           `try: await fut; return True finally: self._waiters.remove`). */
                        int retConstIdx = -1, constFinEnd = -1;
                        if (!epi) {
                            PycBuffer cs(code->code()->value(), code->code()->length());
                            cs.setPos(ret_off);
                            int co, ca, cp = ret_off;
                            if (!cs.atEof()) {
                                bc_next(cs, mod, co, ca, cp);
                                if (co == Pyc::RETURN_CONST_A) { retConstIdx = ca; constFinEnd = ret_off; }
                            }
                            if (retConstIdx < 0) {
                                int prevStart = -1, prevOp = -1, prevArg = -1;
                                PycBuffer ps(code->code()->value(), code->code()->length());
                                ps.setPos(entry.end_offset);
                                int po, pa, pp = entry.end_offset;
                                while (pp < ret_off && !ps.atEof()) {
                                    int pip = pp;
                                    bc_next(ps, mod, po, pa, pp);
                                    if (po != Pyc::CACHE && po != Pyc::NOP) { prevStart = pip; prevOp = po; prevArg = pa; }
                                    if (pp <= pip) break;
                                }
                                if (prevOp == Pyc::LOAD_CONST_A) { retConstIdx = prevArg; constFinEnd = prevStart; }
                            }
                        }
                        PycRef<ASTContainerBlock> cont = new ASTContainerBlock(entry.target, 0);
                        cont->setFinallyRange(epi ? epiFinEnd : (retConstIdx >= 0 ? constFinEnd : ret_off), aft);
                        blocks.push(cont.cast<ASTBlock>());
                        curblock = blocks.top();
                        stack_hist.push(stack);
                        PycRef<ASTBlock> tryblock = new ASTBlock(ASTBlock::BLK_TRY, entry.end_offset, true);
                        blocks.push(tryblock.cast<ASTBlock>());
                        curblock = blocks.top();
                        if (epi)
                            finallyEpilogueCut.insert(epiFinEnd);
                        else if (retConstIdx >= 0)
                            nestedFinReturnConst[entry.end_offset] = retConstIdx;
                        else {
                            finallyReturnExit[entry.end_offset] = ret_off;
                            PycBuffer rs(code->code()->value(), code->code()->length());
                            rs.setPos(entry.end_offset);
                            int ro, ra, rp = entry.end_offset, prevop = -1;
                            while (rp < ret_off && !rs.atEof()) {
                                int rip = rp;
                                bc_next(rs, mod, ro, ra, rp);
                                if ((ro == Pyc::RETURN_VALUE
                                        || ro == Pyc::INSTRUMENTED_RETURN_VALUE_A)
                                        && (prevop == Pyc::STORE_FAST_A
                                            || prevop == Pyc::STORE_NAME_A
                                            || prevop == Pyc::STORE_GLOBAL_A
                                            || prevop == Pyc::STORE_DEREF_A
                                            || prevop == Pyc::STORE_ATTR_A
                                            || prevop == Pyc::STORE_SUBSCR
                                            || prevop == Pyc::POP_TOP
                                            /* A finally whose body is conditional
                                               replicates the deferred return at each
                                               branch leaf, so the merge is a RUN of
                                               consecutive RETURN_VALUEs (one per
                                               false-branch fall-through). They are all
                                               the same deferred return already emitted
                                               in the try; skip the whole run, not just
                                               the first. */
                                            || prevop == Pyc::RETURN_VALUE
                                            || prevop == Pyc::INSTRUMENTED_RETURN_VALUE_A))
                                    finallyReturnSkip.insert(rip);
                                if (ro != Pyc::CACHE && ro != Pyc::NOP)
                                    prevop = ro;
                                if (rp <= rip) break;
                            }
                        }
                        next_exception_entry++;
                        did_finally = true;
                    }
                }

                if (raise_off < 0 && jump_off < 0 && ret_off < 0
                        && entry.end_offset == entry.target) {
                    PycBuffer ls(code->code()->value(), code->code()->length());
                    ls.setPos(entry.start_offset);
                    int lo, la, lp = entry.start_offset, lastOp = -1, lastIp = -1;
                    while (lp < entry.end_offset && !ls.atEof()) {
                        int lip = lp;
                        bc_next(ls, mod, lo, la, lp);
                        if (lp <= lip) break;
                        lastOp = lo; lastIp = lip;
                    }
                    if (lastOp == Pyc::RAISE_VARARGS_A || lastOp == Pyc::RERAISE
                            || lastOp == Pyc::RERAISE_A)
                        raise_off = lastIp;
                    else if (lastOp == Pyc::JUMP_BACKWARD_A) {
                        const int W = (int)sizeof(uint16_t);
                        PycBuffer bs(code->code()->value(), code->code()->length());
                        bs.setPos(lastIp);
                        int bo, ba, bp = lastIp;
                        if (!bs.atEof()) bc_next(bs, mod, bo, ba, bp);
                        int bt = bp - ba * W;
                        if (bt >= entry.start_offset && bt < entry.end_offset)
                            raise_off = lastIp;
                    }
                }
                if (forwardAllowed && !did_finally && raise_off >= 0
                        && jump_off < 0 && ret_off < 0
                        && !finallyPlanByTarget.count(entry.target)) {
                    int cleanupTgt = -1;
                    for (const auto& e2 : exception_entries)
                        if (e2.start_offset == entry.target && e2.push_lasti) {
                            cleanupTgt = e2.target; break;
                        }
                    int afterPush = -1;
                    {
                        PycBuffer ps(code->code()->value(), code->code()->length());
                        ps.setPos(entry.target);
                        int po, pa, pp = entry.target;
                        if (!ps.atEof()) {
                            bc_next(ps, mod, po, pa, pp);
                            if (po == Pyc::PUSH_EXC_INFO) afterPush = pp;
                        }
                    }
                    int rerOff = -1;
                    if (afterPush > 0) {
                        PycBuffer rs(code->code()->value(), code->code()->length());
                        rs.setPos(afterPush);
                        int ro, ra, rp = afterPush;
                        while (!rs.atEof()) {
                            int rip = rp;
                            bc_next(rs, mod, ro, ra, rp);
                            if (ro == Pyc::POP_EXCEPT || ro == Pyc::CHECK_EXC_MATCH)
                                break;
                            if (ro == Pyc::RERAISE || ro == Pyc::RERAISE_A) {
                                rerOff = rip; break;
                            }
                            if (rp <= rip) break;
                        }
                    }
                    bool cf = false;
                    if (afterPush > 0 && rerOff > afterPush) {
                        PycBuffer cs(code->code()->value(), code->code()->length());
                        cs.setPos(afterPush);
                        const int W = (int)sizeof(uint16_t);
                        int co, ca, cp = afterPush;
                        while (cp < rerOff && !cs.atEof()) {
                            int cip = cp;
                            bc_next(cs, mod, co, ca, cp);
                            if (co == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A) {
                                int bt = cp - ca * W, bo = -1, ba, bp = bt;
                                PycBuffer bs(code->code()->value(), code->code()->length());
                                bs.setPos(bt);
                                if (!bs.atEof()) bc_next(bs, mod, bo, ba, bp);
                                if (bo != Pyc::SEND_A) { cf = true; break; }
                            } else if (co == Pyc::FOR_ITER_A || co == Pyc::JUMP_BACKWARD_A
                                    || co == Pyc::PUSH_EXC_INFO
                                    || co == Pyc::SETUP_FINALLY_A
                                    || co == Pyc::BEFORE_WITH
                                    || co == Pyc::SETUP_WITH_A) { cf = true; break; }
                            if (cp <= cip) break;
                        }
                    }
                    int aft = -1;
                    if (cleanupTgt > 0) {
                        PycBuffer xs(code->code()->value(), code->code()->length());
                        xs.setPos(cleanupTgt);
                        int xo, xa, xp = cleanupTgt;
                        while (!xs.atEof()) {
                            bc_next(xs, mod, xo, xa, xp);
                            if (xo == Pyc::RERAISE || xo == Pyc::RERAISE_A) { aft = xp; break; }
                            if (xo == Pyc::PUSH_EXC_INFO) break;
                        }
                    }
                    if (!cf && afterPush > 0 && rerOff > afterPush && aft > rerOff) {
                        PycRef<ASTContainerBlock> cont = new ASTContainerBlock(entry.target, 0);
                        cont->setFinallyRange(rerOff, aft);
                        blocks.push(cont.cast<ASTBlock>());
                        curblock = blocks.top();
                        stack_hist.push(stack);
                        PycRef<ASTBlock> tryblock = new ASTBlock(ASTBlock::BLK_TRY, entry.target, true);
                        blocks.push(tryblock.cast<ASTBlock>());
                        curblock = blocks.top();
                        FinallyCoalesce plan = { entry.target, afterPush, rerOff, aft, 0 };
                        finallyPlanByTarget[entry.target] = plan;
                        raiseFinallyClose.insert(rerOff);
                        next_exception_entry++;
                        did_finally = true;
                    }
                }
            }

            bool depth_ok = entry.stack_depth == 0 || htype == 1;
            if (!did_finally && depth_ok) {
                if (curblock->blktype() == ASTBlock::BLK_CONTAINER) {
                    curblock.cast<ASTContainerBlock>()->setExcept(entry.target);
                } else {
                    PycRef<ASTBlock> next = new ASTContainerBlock(0, entry.target);
                    blocks.push(next.cast<ASTBlock>());
                    curblock = blocks.top();
                }

                {
                    int eS = entry.end_offset, eT = entry.target;
                    if (eS < eT && !elseStartOffsets.count(eS)) {
                        bool straight = true, lastJump = false; int realN = 0;
                        int lastReal = 0;
                        int lastJumpTgt = -1;               // JUMP_FORWARD target of else exit
                        int exitOps = 0;                    // RETURN/RAISE/JUMP count in region
                        PycBuffer eb(code->code()->value(), code->code()->length());
                        eb.setPos(eS);
                        int eo, ea, ep = eS;
                        while (ep < eT && !eb.atEof()) {
                            int eip = ep;
                            bc_next(eb, mod, eo, ea, ep);
                            if (eo == Pyc::NOP || eo == Pyc::CACHE || eo == Pyc::RESUME_A) {
                                if (ep <= eip) break;
                                continue;
                            }
                            if (eo == Pyc::PUSH_EXC_INFO || eo == Pyc::FOR_ITER_A
                                    || eo == Pyc::BEFORE_WITH
                                    || eo == Pyc::BEFORE_ASYNC_WITH
                                    || eo == Pyc::SETUP_FINALLY_A) {
                                straight = false; break;
                            }
                            realN++;
                            lastReal = eo;
                            lastJump = (eo == Pyc::JUMP_FORWARD_A
                                    || eo == Pyc::JUMP_BACKWARD_A);
                            lastJumpTgt = (eo == Pyc::JUMP_FORWARD_A)
                                    ? (ep + ea * (int)sizeof(uint16_t)) : -1;
                            if (eo == Pyc::RETURN_VALUE || eo == Pyc::RETURN_CONST_A
                                    || eo == Pyc::RAISE_VARARGS_A || eo == Pyc::RERAISE_A
                                    || eo == Pyc::JUMP_FORWARD_A || eo == Pyc::JUMP_BACKWARD_A
                                    || eo == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                                    || eo == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                                    || eo == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                                    || eo == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A
                                    || eo == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A
                                    || eo == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A)
                                exitOps++;
                            if (ep <= eip) break;
                        }
                        bool nestedEnt = false;
                        for (const auto& ne : exception_entries)
                            if (ne.start_offset > eS && ne.start_offset < eT) {
                                nestedEnt = true; break;
                            }
                        if (straight && lastJump && realN >= 2 && !nestedEnt) {
                            elseStartOffsets.insert(eS);
                            /* Shared-tail `return X` absorb (see teoElseAbsorbRet): the
                               else's terminal JUMP_FORWARD (at lastJumpTgt) lands PAST
                               the handler on a `LOAD_x; RETURN_VALUE` reached by no other
                               predecessor. Synthesize `return X` for the BLK_ELSE and
                               suppress the tail-level emit. Tightly gated: single JF
                               predecessor, target strictly after the handler, exactly a
                               one-op load feeding RETURN_VALUE, and the else body's ONLY
                               exit is that terminal JF (exitOps==1) — so a region that is
                               really the try body's tail (an inner-if `return`/`raise`
                               arm followed by the merge jump) is NOT mistaken for an else
                               clause. Additionally the JF must jump PAST an enclosing
                               loop's exit (a loopRange containing the try whose exit lies
                               in (eT, lastJumpTgt)): only then is the return placed as a
                               shared post-loop tail block that pycdc would otherwise
                               strand as dead code. A try/except/else with a shared
                               post-try merge and NO enclosing loop (whose bare
                               `except: return` must survive) is excluded. */
                            bool pastLoopExit = false;
                            for (const auto& lr : loopRanges)
                                if (lr.exit > eT && lr.exit < lastJumpTgt
                                        && eS >= lr.start && eS < lr.exit) {
                                    pastLoopExit = true; break;
                                }
                            if (lastJumpTgt > eT && exitOps == 1 && pastLoopExit) {
                                PycBuffer rb(code->code()->value(),
                                        code->code()->length());
                                rb.setPos(lastJumpTgt);
                                int ro, ra, rp = lastJumpTgt;
                                PycRef<ASTNode> rval(nullptr);
                                if (!rb.atEof()) {
                                    bc_next(rb, mod, ro, ra, rp);
                                    if (ro == Pyc::LOAD_FAST_A)
                                        rval = new ASTName(code->getLocal(ra));
                                    else if (ro == Pyc::LOAD_NAME_A)
                                        rval = new ASTName(code->getName(ra));
                                    else if (ro == Pyc::LOAD_DEREF_A
                                            || ro == Pyc::LOAD_CLASSDEREF_A)
                                        rval = new ASTName(code->getCellVar(mod, ra));
                                    else if (ro == Pyc::LOAD_CONST_A)
                                        rval = new ASTObject(code->getConst(ra));
                                }
                                int no = Pyc::PYC_INVALID_OPCODE, na = 0, npp = rp;
                                if (rval != nullptr && !rb.atEof())
                                    bc_next(rb, mod, no, na, npp);
                                /* count predecessors of lastJumpTgt: any *_JUMP_*
                                   whose target equals it (other than our JF). */
                                int preds = 0;
                                {
                                    PycBuffer pb(code->code()->value(),
                                            code->code()->length());
                                    pb.setPos(0);
                                    int po, pa, pp = 0;
                                    while (!pb.atEof()) {
                                        int pip = pp;
                                        bc_next(pb, mod, po, pa, pp);
                                        int pt = -1;
                                        if (po == Pyc::JUMP_FORWARD_A
                                                || po == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                                                || po == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                                                || po == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                                                || po == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A)
                                            pt = pp + pa * (int)sizeof(uint16_t);
                                        else if (po == Pyc::JUMP_BACKWARD_A
                                                || po == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A
                                                || po == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A
                                                || po == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A)
                                            pt = pp - pa * (int)sizeof(uint16_t);
                                        if (pt == lastJumpTgt) preds++;
                                        if (pp <= pip) break;
                                    }
                                }
                                if (rval != nullptr && no == Pyc::RETURN_VALUE
                                        && preds == 1) {
                                    teoElseAbsorbRet[eT] = new ASTReturn(rval);
                                    teoElseAbsorbSuppress.insert(lastJumpTgt);
                                }
                            }
                        }
                        else if (nestedEnt && realN >= 1
                                && !nestedElseHandler.count(eS)) {
                            bool contained = true;
                            for (const auto& ne : exception_entries)
                                if (ne.start_offset >= eS && ne.start_offset < eT
                                        && ne.target > eT) {
                                    contained = false; break;
                                }
                            const int neW = (int)sizeof(uint16_t);
                            PycBuffer nb(code->code()->value(),
                                    code->code()->length());
                            nb.setPos(eS);
                            int no, na, np = eS;
                            bool exitJump = false, elseCond = false, prevCheck = false;
                            bool backEdgeExit = false, sawWith = false;
                            int mergeM = 0;
                            while (np < eT && !nb.atEof()) {
                                int nip = np;
                                bc_next(nb, mod, no, na, np);
                                if (no == Pyc::JUMP_FORWARD_A
                                        && (np + na * neW) >= eT) {
                                    exitJump = true;
                                    int t = np + na * neW;
                                    if (mergeM == 0 || t < mergeM) mergeM = t;
                                }
                                if ((no == Pyc::JUMP_BACKWARD_A
                                        || no == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A)
                                        && (np - na * neW) <= eS)
                                    backEdgeExit = true;
                                if (no == Pyc::BEFORE_WITH || no == Pyc::BEFORE_ASYNC_WITH
                                        || no == Pyc::FOR_ITER_A)
                                    sawWith = true;
                                /* A conditional inside a NESTED try-body (a nested
                                   exception entry's range within [eS,eT)) belongs to
                                   that inner construct, not the else level, so it does
                                   not disqualify the nested-else; only an else-level
                                   conditional does. Admits a multi-level
                                   try/except/else whose inner try-body carries `if`s. */
                                if ((no == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                                        || no == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                                        || no == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A
                                        || no == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A)
                                        && !prevCheck) {
                                    bool inNestedTry = false;
                                    for (const auto& ne2 : exception_entries)
                                        if (!ne2.push_lasti
                                                && ne2.start_offset > eS
                                                && ne2.start_offset < eT
                                                && nip >= ne2.start_offset
                                                && nip < ne2.end_offset) {
                                            inNestedTry = true; break;
                                        }
                                    if (!inNestedTry)
                                        elseCond = true;
                                }
                                if (no != Pyc::CACHE && no != Pyc::NOP
                                        && no != Pyc::EXTENDED_ARG_A)
                                    prevCheck = (no == Pyc::CHECK_EXC_MATCH);
                                if (np <= nip) break;
                            }
                            if (contained && exitJump && !elseCond) {
                                nestedElseHandler[eS] = eT;
                                if (mergeM > eT) teoExceptMerge[eT] = mergeM;
                            } else if (contained && backEdgeExit && !exitJump && !sawWith) {
                                nestedElseHandler[eS] = eT;
                            }
                        }
                        else if (straight && !lastJump && realN >= 2 && !nestedEnt
                                && !nestedElseHandler.count(eS)
                                && (lastReal == Pyc::RETURN_VALUE
                                    || lastReal == Pyc::RAISE_VARARGS_A
                                    || lastReal == Pyc::RERAISE_A)) {
                            /* TERMINAL else: the success region [eS,eT) — UNPROTECTED
                               code between the try body end and the handler — is a real
                               `else:` whose every path ends in return/raise (no forward
                               jump over the handler to infer it from). The original left
                               it unprotected (eS = entry.end_offset), distinguishing a
                               true else from a return INSIDE the try (a protected
                               trailing return is not in [eS,eT)). Route via the SPLIT
                               path (elseStartOffsets + teoSplitElse), NOT a real
                               BLK_ELSE: the try/except renders normally (a bare except
                               still opens via the handler machinery) and the trailing
                               else nodes — including any inner `if`, a complete node by
                               then — split off at the try-close. M bounds the terminal
                               excepts (they return/raise, no jump-to-merge). */
                            bool selfContained = true;
                            const int cW = (int)sizeof(uint16_t);
                            PycBuffer cb(code->code()->value(),
                                    code->code()->length());
                            cb.setPos(eS);
                            int co, ca, cp = eS;
                            while (cp < eT && !cb.atEof()) {
                                int cip = cp;
                                bc_next(cb, mod, co, ca, cp);
                                if (co == Pyc::JUMP_FORWARD_A
                                        && cp + ca * cW >= eT) { selfContained = false; break; }
                                if ((co == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                                        || co == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                                        || co == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                                        || co == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A)
                                        && cp + ca * cW > eT) { selfContained = false; break; }
                                if (cp <= cip) break;
                            }
                            /* Bail if NESTED inside an enclosing try whose handler is
                               past eT — reconstructing this try's terminal else then
                               disrupts the enclosing try/except rendering. */
                            bool nestedInTry = false;
                            for (const auto& oe : exception_entries)
                                if (oe.start_offset < eS && oe.target > eT) {
                                    nestedInTry = true; break;
                                }
                            if (selfContained && !nestedInTry) {
                                int hmax = eT; bool grew = true;
                                while (grew) {
                                    grew = false;
                                    for (const auto& he : exception_entries) {
                                        if (he.start_offset >= eT
                                                && he.start_offset <= hmax) {
                                            if (he.end_offset > hmax) { hmax = he.end_offset; grew = true; }
                                            if (he.target > hmax) { hmax = he.target; grew = true; }
                                        }
                                    }
                                }
                                int M = hmax;
                                PycBuffer mb(code->code()->value(),
                                        code->code()->length());
                                mb.setPos(hmax);
                                int mo, ma, mp = hmax;
                                while (!mb.atEof()) {
                                    int mip = mp;
                                    bc_next(mb, mod, mo, ma, mp);
                                    /* skip ONLY the lasti cleanup stub; a PUSH_EXC_INFO
                                       is the next handler — don't cross it. */
                                    if (mo == Pyc::COPY_A || mo == Pyc::POP_EXCEPT
                                            || mo == Pyc::RERAISE_A || mo == Pyc::NOP
                                            || mo == Pyc::CACHE || mo == Pyc::EXTENDED_ARG_A) {
                                        M = mp;
                                        if (mp <= mip) break;
                                        continue;
                                    }
                                    M = mip;
                                    break;
                                }
                                /* multiClause = the typed first clause's CHECK jumps to
                                   another clause; nChecks = typed clause count. The
                                   else-reorder past >1 typed clause is not modelled, and
                                   such files are already co_code-clean as the equivalent
                                   return-in-try form, so restrict to <=1 typed clause. */
                                bool multiClause = false;
                                int nChecks = 0;
                                /* topLevelChecks counts CHECK_EXC_MATCH belonging to the
                                   handler chain's OWN sibling clauses, excluding any inside
                                   a NESTED exception entry within the handler body — a bare
                                   `except:` whose body has its own try/except would
                                   otherwise inflate the count. firstClauseTyped: the handler
                                   OPENS with a typed clause (a CHECK_EXC_MATCH reached
                                   before any POP_TOP), i.e. not a bare `except:`. */
                                int topLevelChecks = 0;
                                bool firstClauseTyped = false, sawFirstReal = false;
                                {
                                    PycBuffer hb(code->code()->value(),
                                            code->code()->length());
                                    hb.setPos(eT);
                                    int ho, ha, hp = eT; bool prevCheck = false;
                                    while (hp < hmax && !hb.atEof()) {
                                        int hip = hp;
                                        bc_next(hb, mod, ho, ha, hp);
                                        if (ho == Pyc::CHECK_EXC_MATCH) {
                                            nChecks++;
                                            bool inNested = false;
                                            for (const auto& ne2 : exception_entries)
                                                if (!ne2.push_lasti
                                                        && ne2.start_offset > eT
                                                        && ne2.start_offset < hmax
                                                        && hip >= ne2.start_offset
                                                        && hip < ne2.end_offset) {
                                                    inNested = true; break;
                                                }
                                            if (!inNested) topLevelChecks++;
                                        }
                                        if (!sawFirstReal && ho != Pyc::CACHE
                                                && ho != Pyc::NOP && ho != Pyc::EXTENDED_ARG_A
                                                && ho != Pyc::PUSH_EXC_INFO) {
                                            sawFirstReal = true;
                                            /* a typed clause loads the exception type before
                                               CHECK_EXC_MATCH; a bare clause discards with
                                               POP_TOP first. */
                                            firstClauseTyped = (ho != Pyc::POP_TOP);
                                        }
                                        if (prevCheck
                                                && (ho == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                                                    || ho == Pyc::POP_JUMP_FORWARD_IF_TRUE_A)) {
                                            int t = hp + ha * cW;
                                            if (t > eT && t < hmax) { multiClause = true; }
                                        }
                                        if (ho != Pyc::CACHE && ho != Pyc::NOP
                                                && ho != Pyc::EXTENDED_ARG_A)
                                            prevCheck = (ho == Pyc::CHECK_EXC_MATCH);
                                        if (hp <= hip) break;
                                    }
                                }
                                /* A SECOND typed clause is admitted only for the CLEANEST
                                   shape, since a two-clause handler chain with a
                                   return-in-try form is co_code-identical to a real else
                                   and the else-reorder is otherwise unmodelled. The chain
                                   must be TWO TOP-LEVEL TYPED siblings (topLevelChecks == 2,
                                   firstClauseTyped) — a bare `except:` whose body contains a
                                   nested try/except would otherwise look like two clauses.
                                   Two further guards distinguish a genuine unprotected
                                   `else:` (asyncio sslproto._do_shutdown) from an
                                   unprotected try-body TAIL / an inlined finally-copy:
                                     (1) the else region [eS,eT) must be STRICTLY
                                         straight-line — no conditional branch inside it
                                         (an inlined finally-copy carries its own `if`);
                                     (2) it must not overlap ANY other protection entry
                                         (a protected tail belongs to an enclosing
                                         try/finally, not a sibling else). */
                                bool multiOk = (nChecks <= 1);
                                bool twoTypedSiblings =
                                        (topLevelChecks == 2 && firstClauseTyped
                                         && multiClause);
                                if (nChecks >= 2 && !multiOk && twoTypedSiblings) {
                                    bool elseStraight = true;
                                    /* The else body must OPEN with a fresh value-producing
                                       op (a LOAD), not a stack-continuation (SWAP/POP_TOP):
                                       `return <call>` whose call lives inside the protected
                                       range spills its `SWAP; POP_TOP; RETURN_VALUE` just
                                       past the range end — that is the try body's own tail,
                                       not an else (xmlrpc client Transport.request). */
                                    {
                                        PycBuffer fb(code->code()->value(),
                                                code->code()->length());
                                        fb.setPos(eS);
                                        int fo = Pyc::PYC_INVALID_OPCODE, fa, fp = eS;
                                        while (fp < eT && !fb.atEof()) {
                                            int fip = fp;
                                            bc_next(fb, mod, fo, fa, fp);
                                            if (fo != Pyc::NOP && fo != Pyc::CACHE
                                                    && fo != Pyc::RESUME_A
                                                    && fo != Pyc::EXTENDED_ARG_A)
                                                break;
                                            if (fp <= fip) break;
                                        }
                                        if (fo == Pyc::SWAP_A || fo == Pyc::POP_TOP
                                                || fo == Pyc::POP_EXCEPT)
                                            elseStraight = false;
                                    }
                                    PycBuffer sb(code->code()->value(),
                                            code->code()->length());
                                    sb.setPos(eS);
                                    int so2, sa2, sp2 = eS;
                                    while (elseStraight && sp2 < eT && !sb.atEof()) {
                                        int sip = sp2;
                                        bc_next(sb, mod, so2, sa2, sp2);
                                        if (so2 == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                                                || so2 == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                                                || so2 == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                                                || so2 == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A
                                                || so2 == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A
                                                || so2 == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A
                                                || so2 == Pyc::JUMP_FORWARD_A
                                                || so2 == Pyc::JUMP_BACKWARD_A) {
                                            elseStraight = false; break;
                                        }
                                        if (sp2 <= sip) break;
                                    }
                                    bool elseProtected = false;
                                    for (const auto& pe : exception_entries)
                                        if (pe.start_offset < eT && pe.end_offset > eS) {
                                            elseProtected = true; break;
                                        }
                                    /* The else must be a REAL body, not the implicit
                                       function epilogue: a lone `return <const>` /
                                       `return None` at the end of a try/except-terminated
                                       function is co_code-identical whether written as an
                                       else or left implicit, and the source has neither —
                                       so require the else region to perform work (a CALL /
                                       STORE / attribute set), not just load-and-return
                                       (subprocess Popen._stdin_write). */
                                    bool elseHasWork = false;
                                    {
                                        PycBuffer wb(code->code()->value(),
                                                code->code()->length());
                                        wb.setPos(eS);
                                        int wo, wa, wp = eS;
                                        while (wp < eT && !wb.atEof()) {
                                            int wip = wp;
                                            bc_next(wb, mod, wo, wa, wp);
                                            if (wo != Pyc::NOP && wo != Pyc::CACHE
                                                    && wo != Pyc::RESUME_A
                                                    && wo != Pyc::EXTENDED_ARG_A
                                                    && wo != Pyc::LOAD_CONST_A
                                                    && wo != Pyc::LOAD_FAST_A
                                                    && wo != Pyc::LOAD_NAME_A
                                                    && wo != Pyc::LOAD_GLOBAL_A
                                                    && wo != Pyc::LOAD_DEREF_A
                                                    && wo != Pyc::RETURN_VALUE
                                                    && wo != Pyc::RETURN_CONST_A) {
                                                elseHasWork = true; break;
                                            }
                                            if (wp <= wip) break;
                                        }
                                    }
                                    multiOk = (elseStraight && !elseProtected
                                            && elseHasWork);
                                }
                                if (multiOk) {
                                    elseStartOffsets.insert(eS);
                                    if (twoTypedSiblings)
                                        elseStartMultiClause.insert(eS);
                                    if (M > eT) {
                                        terminalExceptMerge[eT] = M;
                                        terminalExceptMulti[eT] = multiClause;
                                    }
                                }
                            }
                        }
                        else if (!straight && realN >= 1 && !nestedEnt
                                && !nestedElseHandler.count(eS)) {
                            /* try/except[/else] whose else body is NOT straight-line —
                               it contains a `for`/`while`/`if` (a FOR_ITER or forward
                               branch made the flat scan give up). The original protects
                               ONLY the true try body [entry.start,eS); the trailing
                               unprotected region [eS,eT) that falls through from the try
                               is the `else:`. Without this the whole else body is
                               absorbed into the try, over-scoping the handler. Route via
                               the SPLIT path (elseStartOffsets + teoSplitElse): the
                               try/except opens normally and the fall-through nodes split
                               off at the try-close. Tightly gated below so a return/if
                               that is genuinely INSIDE the try body is never mistaken for
                               an else. */
                            const int gW = (int)sizeof(uint16_t);
                            /* No nested exception entry (typed or lasti) begins inside
                               the region — a nested try in the else needs different
                               handling and is left to the existing machinery. */
                            bool hasNestedEntry = false;
                            for (const auto& ne : exception_entries)
                                if (ne.start_offset >= eS && ne.start_offset < eT) {
                                    hasNestedEntry = true; break;
                                }
                            /* No block-opening exception op inside the region. */
                            bool hasExcOp = false;
                            /* Every forward jump either stays inside [eS,eT) or lands on
                               a single merge M >= eT (past the handler); a jump that
                               lands strictly inside the handler interior (eT,M) or that
                               has more than one distinct past-handler target disqualifies
                               the region. */
                            int mergeJ = -1; bool jumpOk = true; int lastOp = 0;
                            int lastOpTgt = -1;
                            {
                                PycBuffer gb(code->code()->value(),
                                        code->code()->length());
                                gb.setPos(eS);
                                int go, ga, gp = eS;
                                while (gp < eT && !gb.atEof()) {
                                    int gip = gp;
                                    bc_next(gb, mod, go, ga, gp);
                                    if (go == Pyc::NOP || go == Pyc::CACHE
                                            || go == Pyc::RESUME_A
                                            || go == Pyc::EXTENDED_ARG_A) {
                                        if (gp <= gip) break;
                                        continue;
                                    }
                                    if (go == Pyc::PUSH_EXC_INFO
                                            || go == Pyc::SETUP_FINALLY_A) {
                                        hasExcOp = true; break;
                                    }
                                    lastOp = go;
                                    lastOpTgt = -1;
                                    if (go == Pyc::JUMP_BACKWARD_A
                                            || go == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A)
                                        lastOpTgt = gp - ga * gW;
                                    int tgt = -1;
                                    if (go == Pyc::JUMP_FORWARD_A
                                            || go == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                                            || go == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                                            || go == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                                            || go == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A
                                            || go == Pyc::FOR_ITER_A)
                                        tgt = gp + ga * gW;
                                    if (tgt >= 0 && tgt > eS) {
                                        if (tgt >= eT) {
                                            /* forward exit past the try body: must reach
                                               the merge, never the handler interior. */
                                            if (mergeJ < 0) mergeJ = tgt;
                                            else if (mergeJ != tgt) { jumpOk = false; break; }
                                        }
                                    }
                                    if (gp <= gip) break;
                                }
                            }
                            /* No jump from OUTSIDE [eS,eT) lands INSIDE the region — the
                               else must be entered solely by falling through from the try
                               body (a code region jumped into from elsewhere is not an
                               else clause). */
                            bool externalPred = false;
                            {
                                PycBuffer pb(code->code()->value(),
                                        code->code()->length());
                                pb.setPos(0);
                                int po, pa, pp = 0;
                                while (!pb.atEof()) {
                                    int pip = pp;
                                    bc_next(pb, mod, po, pa, pp);
                                    int pt = -1;
                                    if (po == Pyc::JUMP_FORWARD_A
                                            || po == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                                            || po == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                                            || po == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                                            || po == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A
                                            || po == Pyc::FOR_ITER_A)
                                        pt = pp + pa * gW;
                                    else if (po == Pyc::JUMP_BACKWARD_A
                                            || po == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A
                                            || po == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A
                                            || po == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A)
                                        pt = pp - pa * gW;
                                    if (pt > eS && pt < eT
                                            && (pip < eS || pip >= eT))
                                        externalPred = true;
                                    if (pp <= pip) break;
                                }
                            }
                            /* The region must terminate: either it ends by returning /
                               raising (a for/while-else that falls off the end into a
                               return), its last op is a forward jump to the merge, or it
                               ends on a back-edge to an enclosing loop header (a `continue`
                               / loop-tail: the else body is the tail of a loop iteration,
                               so control leaves the region backward to the loop, at or
                               before the region start — not a jump into the region). */
                            bool endsCleanly = (lastOp == Pyc::RETURN_VALUE
                                    || lastOp == Pyc::RETURN_CONST_A
                                    || lastOp == Pyc::RAISE_VARARGS_A
                                    || lastOp == Pyc::RERAISE_A
                                    || lastOp == Pyc::JUMP_FORWARD_A
                                    || ((lastOp == Pyc::JUMP_BACKWARD_A
                                            || lastOp == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A)
                                        && lastOpTgt >= 0 && lastOpTgt <= eS));
                            /* Not nested inside an enclosing try whose handler is past eT
                               (reconstructing this else would disrupt the outer try). */
                            bool nestedInTry = false;
                            for (const auto& oe : exception_entries)
                                if (!oe.push_lasti
                                        && oe.start_offset < eS && oe.target > eT) {
                                    nestedInTry = true; break;
                                }
                            if (!hasNestedEntry && !hasExcOp && jumpOk
                                    && !externalPred && endsCleanly && !nestedInTry) {
                                /* Bound the handler chain to find the merge M past the
                                   except (skip only the lasti-cleanup stub, never cross a
                                   sibling PUSH_EXC_INFO). */
                                int hmax = eT; bool grew = true;
                                while (grew) {
                                    grew = false;
                                    for (const auto& he : exception_entries)
                                        if (he.start_offset >= eT
                                                && he.start_offset <= hmax) {
                                            if (he.end_offset > hmax) { hmax = he.end_offset; grew = true; }
                                            if (he.target > hmax) { hmax = he.target; grew = true; }
                                        }
                                }
                                int M = hmax;
                                PycBuffer mb(code->code()->value(),
                                        code->code()->length());
                                mb.setPos(hmax);
                                int mo, ma, mp = hmax;
                                while (!mb.atEof()) {
                                    int mip = mp;
                                    bc_next(mb, mod, mo, ma, mp);
                                    if (mo == Pyc::COPY_A || mo == Pyc::POP_EXCEPT
                                            || mo == Pyc::RERAISE_A || mo == Pyc::NOP
                                            || mo == Pyc::CACHE || mo == Pyc::EXTENDED_ARG_A) {
                                        M = mp;
                                        if (mp <= mip) break;
                                        continue;
                                    }
                                    M = mip;
                                    break;
                                }
                                /* An else-exit forward jump (mergeJ) marks the true merge
                                   when it lands at/after the bounded handler end. */
                                if (mergeJ >= eT && mergeJ >= M) M = mergeJ;
                                /* Loop-tail else guard: when the region ends on a back-edge
                                   to an enclosing loop header H (a `continue`), the whole
                                   try/except/else lives inside that loop. The bytecode the
                                   loop falls through to on EXIT (H's FOR_ITER / while-guard
                                   exit target) is post-loop code that runs unconditionally
                                   — it is NOT the else. If the bounded except end M reaches
                                   that loop-exit target, splitting here would drag the
                                   post-loop tail into the else, so decline (leave the shape
                                   to the existing machinery). H's loop exit is the FOR_ITER
                                   exit at H, or the while-guard's forward exit. */
                                bool backEdgeSafe = true;
                                if (mergeJ < 0
                                        && (lastOp == Pyc::JUMP_BACKWARD_A
                                            || lastOp == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A)
                                        && lastOpTgt >= 0) {
                                    int loopExit = -1;
                                    PycBuffer lb(code->code()->value(),
                                            code->code()->length());
                                    lb.setPos(lastOpTgt);
                                    int lo, la, lp = lastOpTgt;
                                    if (!lb.atEof()) {
                                        bc_next(lb, mod, lo, la, lp);
                                        if (lo == Pyc::FOR_ITER_A)
                                            loopExit = lp + la * gW;
                                    }
                                    if (loopExit >= 0 && M >= loopExit)
                                        backEdgeSafe = false;
                                }
                                if (backEdgeSafe) {
                                /* Count typed clauses in the handler chain. A single
                                   clause (or a bare except) opens with its end bounded at
                                   M directly (terminalExceptMulti=false). MULTIPLE typed
                                   clauses (`except E1: … except E2: …`) chain through the
                                   existing multi-clause machinery: the first clause opens
                                   OPEN (s215End stays 0), each following clause opens via
                                   chainedBareExcept and the LAST clause's end is bumped to
                                   M by the terminalExceptMerge scan (marked multi so the
                                   first clause is not prematurely bounded at M). */
                                int nChecks2 = 0;
                                {
                                    PycBuffer hb(code->code()->value(),
                                            code->code()->length());
                                    hb.setPos(eT);
                                    int ho, ha, hp = eT;
                                    while (hp < hmax && !hb.atEof()) {
                                        int hip = hp;
                                        bc_next(hb, mod, ho, ha, hp);
                                        if (ho == Pyc::CHECK_EXC_MATCH) nChecks2++;
                                        if (hp <= hip) break;
                                    }
                                }
                                elseStartOffsets.insert(eS);
                                if (M > eT) {
                                    terminalExceptMerge[eT] = M;
                                    terminalExceptMulti[eT] = (nChecks2 > 1);
                                }
                                }
                            }
                        }
                    }
                }

                stack_hist.push(stack);
                int teoTryEnd = entry.target;
                if (nestedElseHandler.count(entry.end_offset))
                    teoTryEnd = entry.end_offset;
                PycRef<ASTBlock> tryblock = new ASTBlock(ASTBlock::BLK_TRY, teoTryEnd, true);
                blocks.push(tryblock.cast<ASTBlock>());
                curblock = blocks.top();
                next_exception_entry++;
            }
        }

        if (blocks.size() > 1
                && (curblock->blktype() == ASTBlock::BLK_IF
                    || curblock->blktype() == ASTBlock::BLK_ELIF
                    || curblock->blktype() == ASTBlock::BLK_ELSE)
                && curblock->end() == pos) {
            std::stack<PycRef<ASTBlock> > peek = blocks;
            int nif = 0; bool shape = false;
            while (peek.size() > 1
                    && (peek.top()->blktype() == ASTBlock::BLK_IF
                        || peek.top()->blktype() == ASTBlock::BLK_ELIF
                        || peek.top()->blktype() == ASTBlock::BLK_ELSE)
                    && peek.top()->end() == pos) {
                peek.pop(); nif++;
                if (peek.top()->blktype() == ASTBlock::BLK_TRY
                        && peek.top()->end() == pos) { shape = true; break; }
            }
            if (shape) {
                for (int k = 0; k < nif; ++k) {
                    PycRef<ASTBlock> innerif = curblock;
                    blocks.pop();
                    if (!stack_hist.empty()) stack_hist.pop();
                    curblock = blocks.top();
                    curblock->append(innerif.cast<ASTNode>());
                }
            }
        }

        if ((finallyP1Wrap.count(pos) || emptyFinNestedMerge.count(pos))
                && curblock->blktype() == ASTBlock::BLK_EXCEPT
                && blocks.size() > 2) {
            PycRef<ASTBlock> exc = curblock;
            blocks.pop();
            curblock = blocks.top();
            if (!stack_hist.empty()) {
                stack = stack_hist.top();
                stack_hist.pop();
            }
            if (exc->size() != 0)
                curblock->append(exc.cast<ASTNode>());
            if (curblock->blktype() == ASTBlock::BLK_CONTAINER
                    && !curblock.cast<ASTContainerBlock>()->hasFinally()
                    && blocks.size() > 1) {
                PycRef<ASTBlock> cont = curblock;
                blocks.pop();
                curblock = blocks.top();
                curblock->append(cont.cast<ASTNode>());
            }
        }

        if (blocks.size() > 2
                && (curblock->blktype() == ASTBlock::BLK_IF
                    || curblock->blktype() == ASTBlock::BLK_ELIF
                    || curblock->blktype() == ASTBlock::BLK_ELSE)
                && curblock->end() > 0 && curblock->end() <= pos) {
            std::stack<PycRef<ASTBlock> > peek = blocks;
            while (!peek.empty()
                    && (peek.top()->blktype() == ASTBlock::BLK_IF
                        || peek.top()->blktype() == ASTBlock::BLK_ELIF
                        || peek.top()->blktype() == ASTBlock::BLK_ELSE)
                    && peek.top()->end() > 0 && peek.top()->end() <= pos)
                peek.pop();
            if (!peek.empty() && peek.top()->blktype() == ASTBlock::BLK_TRY
                    && peek.top()->end() == pos) {
                while (blocks.size() > 1
                        && (curblock->blktype() == ASTBlock::BLK_IF
                            || curblock->blktype() == ASTBlock::BLK_ELIF
                            || curblock->blktype() == ASTBlock::BLK_ELSE)
                        && curblock->end() > 0 && curblock->end() <= pos) {
                    PycRef<ASTBlock> sb = curblock;
                    if (!stack_hist.empty())
                        stack_hist.pop();
                    blocks.pop();
                    curblock = blocks.top();
                    curblock->append(sb.cast<ASTNode>());
                }
            }
        }

        auto isTerminalLast = [](const PycRef<ASTBlock>& b) -> bool {
            if (b->nodes().empty()) return false;
            PycRef<ASTNode> last = b->nodes().back();
            if (last == NULL) return false;
            if (last->type() == ASTNode::NODE_KEYWORD) {
                int k = last.cast<ASTKeyword>()->key();
                return k == ASTKeyword::KW_BREAK || k == ASTKeyword::KW_CONTINUE;
            }
            return last->type() == ASTNode::NODE_RETURN
                || last->type() == ASTNode::NODE_RAISE;
        };
        if (blocks.size() > 2
                && (curblock->blktype() == ASTBlock::BLK_IF
                    || curblock->blktype() == ASTBlock::BLK_ELIF
                    || curblock->blktype() == ASTBlock::BLK_ELSE)
                && curblock->end() > pos
                && isTerminalLast(curblock)) {
            std::stack<PycRef<ASTBlock> > peek = blocks;
            while (!peek.empty()
                    && (peek.top()->blktype() == ASTBlock::BLK_IF
                        || peek.top()->blktype() == ASTBlock::BLK_ELIF
                        || peek.top()->blktype() == ASTBlock::BLK_ELSE)
                    && peek.top()->end() > pos
                    && isTerminalLast(peek.top()))
                peek.pop();
            if (!peek.empty() && peek.top()->blktype() == ASTBlock::BLK_TRY
                    && peek.top()->end() == pos) {
                while (blocks.size() > 1
                        && (curblock->blktype() == ASTBlock::BLK_IF
                            || curblock->blktype() == ASTBlock::BLK_ELIF
                            || curblock->blktype() == ASTBlock::BLK_ELSE)
                        && curblock->end() > pos
                        && isTerminalLast(curblock)) {
                    PycRef<ASTBlock> sb = curblock;
                    if (!stack_hist.empty())
                        stack_hist.pop();
                    blocks.pop();
                    curblock = blocks.top();
                    curblock->append(sb.cast<ASTNode>());
                }
                if (curblock->blktype() == ASTBlock::BLK_TRY
                        && curblock->end() == pos) {
                    PycBuffer hs(code->code()->value(), code->code()->length());
                    hs.setPos(pos);
                    int ho, ha, hp = pos; bool sawCheck = false;
                    for (int hi = 0; hi < 40 && !hs.atEof(); ++hi) {
                        bc_next(hs, mod, ho, ha, hp);
                        if (ho == Pyc::CHECK_EXC_MATCH) { sawCheck = true; continue; }
                        if (sawCheck && (ho == Pyc::POP_JUMP_FORWARD_IF_FALSE_A)) {
                            ship215ExceptEnd[pos] = hp + ha * (int)sizeof(uint16_t);
                            break;
                        }
                        if (ho == Pyc::RERAISE || ho == Pyc::RERAISE_A) break;
                    }
                }
            }
        }

        if (blocks.size() > 2
                && (curblock->blktype() == ASTBlock::BLK_IF
                    || curblock->blktype() == ASTBlock::BLK_ELIF
                    || curblock->blktype() == ASTBlock::BLK_ELSE)
                && curblock->end() > pos) {
            int opHere = -1;
            {
                const char* eb = code->code()->value();
                int el = (int)code->code()->length();
                if (pos >= 0 && pos < el) {
                    PycBuffer s(eb, el); s.setPos(pos);
                    int o, a, p = pos; bc_next(s, mod, o, a, p); opHere = o;
                }
            }
            if (opHere == Pyc::PUSH_EXC_INFO) {
                std::stack<PycRef<ASTBlock> > peek = blocks;
                while (!peek.empty()
                        && (peek.top()->blktype() == ASTBlock::BLK_IF
                            || peek.top()->blktype() == ASTBlock::BLK_ELIF
                            || peek.top()->blktype() == ASTBlock::BLK_ELSE)
                        && peek.top()->end() > pos)
                    peek.pop();
                bool tryRevealed = false;
                if (peek.size() >= 2 && peek.top()->blktype() == ASTBlock::BLK_TRY
                        && peek.top()->end() == pos) {
                    peek.pop();
                    tryRevealed = peek.top()->blktype() == ASTBlock::BLK_CONTAINER
                        && peek.top().cast<ASTContainerBlock>()->hasExcept();
                }
                if (tryRevealed) {
                    while (blocks.size() > 1
                            && (curblock->blktype() == ASTBlock::BLK_IF
                                || curblock->blktype() == ASTBlock::BLK_ELIF
                                || curblock->blktype() == ASTBlock::BLK_ELSE)
                            && curblock->end() > pos) {
                        PycRef<ASTBlock> sb = curblock;
                        if (!stack_hist.empty())
                            stack_hist.pop();
                        blocks.pop();
                        curblock = blocks.top();
                        curblock->append(sb.cast<ASTNode>());
                    }
                    if (curblock->blktype() == ASTBlock::BLK_TRY
                            && curblock->end() == pos
                            && !ship215ExceptEnd.count(pos)) {
                        PycBuffer hs(code->code()->value(), code->code()->length());
                        hs.setPos(pos);
                        int ho, ha, hp = pos; bool sawCheck = false;
                        for (int hi = 0; hi < 40 && !hs.atEof(); ++hi) {
                            bc_next(hs, mod, ho, ha, hp);
                            if (ho == Pyc::CHECK_EXC_MATCH) { sawCheck = true; continue; }
                            if (sawCheck && (ho == Pyc::POP_JUMP_FORWARD_IF_FALSE_A)) {
                                ship215ExceptEnd[pos] = hp + ha * (int)sizeof(uint16_t);
                                break;
                            }
                            if (ho == Pyc::RERAISE || ho == Pyc::RERAISE_A) break;
                        }
                    }
                }
            }
        }

        if (elseStartOffsets.count(pos)
                && curblock->blktype() == ASTBlock::BLK_TRY
                && !elseBoundaryByTryEnd.count(curblock->end()))
            elseBoundaryByTryEnd[curblock->end()] = (int)curblock->size();
        /* The try body's final statement can be an `if`/`elif`/`else` chain whose
           merge point coincides with the else-start (eS): that inner if-block is
           still OPEN on top of the BLK_TRY at eS, so the plain BLK_TRY-on-top check
           above misses the boundary. Walk down through the open if-chain to the
           enclosing BLK_TRY and record the boundary there. The whole open if-chain
           collapses into exactly ONE node appended to the try when it closes, so the
           else body begins right after it: boundary = try->size() + 1. */
        if (elseStartMultiClause.count(pos)
                && (curblock->blktype() == ASTBlock::BLK_IF
                    || curblock->blktype() == ASTBlock::BLK_ELIF
                    || curblock->blktype() == ASTBlock::BLK_ELSE)
                && curblock->end() == pos) {
            std::stack<PycRef<ASTBlock> > pk = blocks;
            while (!pk.empty()
                    && (pk.top()->blktype() == ASTBlock::BLK_IF
                        || pk.top()->blktype() == ASTBlock::BLK_ELIF
                        || pk.top()->blktype() == ASTBlock::BLK_ELSE)
                    && pk.top()->end() == pos)
                pk.pop();
            if (!pk.empty() && pk.top()->blktype() == ASTBlock::BLK_TRY
                    && !elseBoundaryByTryEnd.count(pk.top()->end()))
                elseBoundaryByTryEnd[pk.top()->end()] = (int)pk.top()->size() + 1;
        }

        if (curblock->blktype() == ASTBlock::BLK_TRY
                && curblock->end() == pos
                && blocks.size() > 1) {
            PycRef<ASTBlock> prev = curblock;
            blocks.pop();
            curblock = blocks.top();

            if (finallyReturnExit.count(pos) && !stack.empty()) {
                prev->append(new ASTReturn(stack.top()));
            }
            /* SHIP-263: a CONST deferred return (`return True`) crossing two finally
               levels — its value is loaded AFTER both finally copies, so it is not on
               the stack here; synthesise it from the recorded const into the inner try. */
            else if (nestedFinReturnConst.count(pos)) {
                PycRef<ASTNode> rv = new ASTObject(code->getConst(nestedFinReturnConst[pos]));
                prev->append(new ASTReturn(rv));
            }

            if (curblock->blktype() == ASTBlock::BLK_CONTAINER
                    && curblock.cast<ASTContainerBlock>()->hasExcept()
                    && nestedElseHandler.count(pos)) {
                curblock->append(prev.cast<ASTNode>());
                int h = nestedElseHandler[pos];
                PycRef<ASTBlock> elseb = new ASTBlock(ASTBlock::BLK_ELSE, h, 1);
                elseb->init();
                nestedElseHandlerVals.insert(h);
                blocks.push(elseb.cast<ASTBlock>());
                curblock = blocks.top();
            } else if (curblock->blktype() == ASTBlock::BLK_CONTAINER
                    && curblock.cast<ASTContainerBlock>()->hasExcept()) {
                if (!stack_hist.empty()) {
                    stack = stack_hist.top();
                    stack_hist.pop();
                }

                PycRef<ASTBlock> teoElse = teoSplitElse(prev);
                curblock->append(prev.cast<ASTNode>());
                if (teoElse != nullptr)
                    curblock->append(teoElse.cast<ASTNode>());
                stack_hist.push(stack);

                int s215End = ship215ExceptEnd.count(pos) ? ship215ExceptEnd[pos] : 0;
                if (s215End == 0 && terminalExceptMerge.count(pos)
                        && !terminalExceptMulti[pos])
                    s215End = terminalExceptMerge[pos];
                /* xlrd/sheet put_cell_unragged: a NESTED bare `except:` handler whose
                   body re-raises (`except: …; raise`) has no normal-exit merge, so it
                   opens with end 0 and stays OPEN. When it is nested INSIDE an outer
                   `except E:` clause that is itself followed by a sibling catch-all
                   `except:` (a chainedBareExcept already recorded at an offset PAST this
                   handler's own lasti-cleanup), the still-open nested handler absorbs
                   the outer sibling's body as duplicate dead code — the outer `except:`
                   is dropped. Bound the nested handler at its own lasti-cleanup end C
                   (the COPY/POP_EXCEPT/RERAISE stub start) so it closes there, exposing
                   the outer typed except (which then closes and lets chainedBareExcept
                   open the sibling clause). Tightly gated: pos must be a bare handler
                   (PUSH_EXC_INFO; POP_TOP), its lasti-cleanup entry must be [pos..C)->C
                   with a re-raising body, and a chainedBareExcept sibling must start at
                   or after C (a genuine sibling clause follows). */
                if (s215End == 0) {
                    int hC = -1;
                    for (const auto& ce : exception_entries)
                        if (ce.push_lasti && ce.start_offset == pos
                                && ce.target == ce.end_offset && ce.target > pos) {
                            hC = ce.target; break;
                        }
                    if (hC > pos) {
                        /* bare handler: PUSH_EXC_INFO immediately followed by POP_TOP */
                        PycBuffer hs(code->code()->value(), code->code()->length());
                        hs.setPos(pos);
                        int ho, ha, hp = pos; bool bare = false, reraises = false;
                        int firstReal = -1, lastReal = -1;
                        while (hp < hC && !hs.atEof()) {
                            int hip = hp;
                            bc_next(hs, mod, ho, ha, hp);
                            if (ho == Pyc::CACHE || ho == Pyc::NOP
                                    || ho == Pyc::EXTENDED_ARG_A
                                    || ho == Pyc::PUSH_EXC_INFO) {
                                if (hp <= hip) break;
                                continue;
                            }
                            if (firstReal < 0) firstReal = ho;
                            lastReal = ho;
                            if (hp <= hip) break;
                        }
                        bare = (firstReal == Pyc::POP_TOP);
                        reraises = (lastReal == Pyc::RAISE_VARARGS_A
                                || lastReal == Pyc::RERAISE
                                || lastReal == Pyc::RERAISE_A);
                        bool siblingFollows = false;
                        for (const auto& kv : chainedBareExcept)
                            if (kv.first >= hC) { siblingFollows = true; break; }
                        if (bare && reraises && siblingFollows)
                            s215End = hC;
                    }
                }
                PycRef<ASTBlock> except = new ASTCondBlock(ASTBlock::BLK_EXCEPT, s215End, NULL, false);
                except->init();
                blocks.push(except);
                curblock = blocks.top();
            } else if (curblock->blktype() == ASTBlock::BLK_CONTAINER
                    && curblock.cast<ASTContainerBlock>()->hasFinally()
                    && returnInFinallyAfter.count(
                           curblock.cast<ASTContainerBlock>()->finally())) {
                if (!stack_hist.empty()) {
                    stack = stack_hist.top();
                    stack_hist.pop();
                }
                int rifAfter = returnInFinallyAfter[
                        curblock.cast<ASTContainerBlock>()->finally()];
                PycRef<ASTBlock> finblk = new ASTBlock(ASTBlock::BLK_FINALLY, pos, true);
                finblk->init();
                if (prev->blktype() == ASTBlock::BLK_TRY && prev->size() > 0) {
                    PycRef<ASTNode> back = prev.cast<ASTBlock>()->nodes().back();
                    bool isCondReturn = back->type() == ASTNode::NODE_BLOCK
                            && back.cast<ASTBlock>()->blktype() == ASTBlock::BLK_IF;
                    if (back->type() == ASTNode::NODE_RETURN || isCondReturn) {
                        prev.cast<ASTBlock>()->removeLast();
                        finblk->append(back);
                    }
                }
                curblock->append(prev.cast<ASTNode>());
                curblock->append(finblk.cast<ASTNode>());
                PycRef<ASTBlock> cont = curblock;
                blocks.pop();
                curblock = blocks.top();
                curblock->append(cont.cast<ASTNode>());
                source.setPos(rifAfter);
                pos = rifAfter;
                while (next_exception_entry < exception_entries.size()
                        && exception_entries[next_exception_entry].start_offset < rifAfter)
                    next_exception_entry++;
                continue;
            } else if (curblock->blktype() == ASTBlock::BLK_CONTAINER
                    && curblock.cast<ASTContainerBlock>()->hasFinally()
                    && emptyFinallyTargets.count(
                           curblock.cast<ASTContainerBlock>()->finally())) {
                /* EMPTY finally (`try: B finally: pass`): the BLK_TRY just closed at
                   its body-exit jump. Emit an empty BLK_FINALLY (renders `pass`),
                   close the container, and resume at `after` — past the dead handler
                   (and, for a loop body, past the back-edge: the enclosing loop is
                   modelled from its own exit). Never jump to a finally body. */
                PycRef<ASTContainerBlock> cont = curblock.cast<ASTContainerBlock>();
                if (!stack_hist.empty()) {
                    stack = stack_hist.top();
                    stack_hist.pop();
                }
                cont->append(prev.cast<ASTNode>());
                PycRef<ASTBlock> finblk = new ASTBlock(ASTBlock::BLK_FINALLY, pos, true);
                finblk->init();
                cont->append(finblk.cast<ASTNode>());
                int after = cont->finallyAfter();
                blocks.pop();
                curblock = blocks.top();
                curblock->append(cont.cast<ASTNode>());
                while (next_exception_entry < exception_entries.size()
                        && exception_entries[next_exception_entry].start_offset < after)
                    next_exception_entry++;
                source.setPos(after);
                pos = after;
                continue;
            } else if (curblock->blktype() == ASTBlock::BLK_CONTAINER
                    && curblock.cast<ASTContainerBlock>()->hasFinally()
                    && raiseFinallyTargets.count(
                           curblock.cast<ASTContainerBlock>()->finally())) {
                /* `try: B finally: raise`: same shape as the empty finally, but the
                   finally body is a single bare `raise`. Emit a BLK_FINALLY holding a
                   `raise`, close the container, and resume past the dead handler. */
                PycRef<ASTContainerBlock> cont = curblock.cast<ASTContainerBlock>();
                if (!stack_hist.empty()) {
                    stack = stack_hist.top();
                    stack_hist.pop();
                }
                cont->append(prev.cast<ASTNode>());
                PycRef<ASTBlock> finblk = new ASTBlock(ASTBlock::BLK_FINALLY, pos, true);
                finblk->init();
                finblk->append(new ASTRaise(ASTRaise::param_t()));
                cont->append(finblk.cast<ASTNode>());
                int after = cont->finallyAfter();
                blocks.pop();
                curblock = blocks.top();
                curblock->append(cont.cast<ASTNode>());
                while (next_exception_entry < exception_entries.size()
                        && exception_entries[next_exception_entry].start_offset < after)
                    next_exception_entry++;
                source.setPos(after);
                pos = after;
                continue;
            } else if (curblock->blktype() == ASTBlock::BLK_CONTAINER
                    && curblock.cast<ASTContainerBlock>()->hasFinally()
                    && curblock.cast<ASTContainerBlock>()->finallyBodyEnd() > 0) {
                if (!stack_hist.empty()) {
                    stack = stack_hist.top();
                    stack_hist.pop();
                }
                if (prev->blktype() == ASTBlock::BLK_TRY) {
                    PycRef<ASTBlock> tb = prev.cast<ASTBlock>();
                    auto isRetNone = [](const PycRef<ASTNode>& n) -> bool {
                        if (n == nullptr || n->type() != ASTNode::NODE_RETURN)
                            return false;
                        if (n.cast<ASTReturn>()->rettype() != ASTReturn::RETURN)
                            return false;
                        PycRef<ASTNode> v = n.cast<ASTReturn>()->value();
                        return v == nullptr
                            || (v->type() == ASTNode::NODE_OBJECT
                                && v.cast<ASTObject>()->object() == Pyc_None);
                    };
                    while (tb->size() > 1 && isRetNone(tb->nodes().back()))
                        tb->removeLast();
                }
                curblock->append(prev.cast<ASTNode>());
                PycRef<ASTBlock> finblk = new ASTBlock(ASTBlock::BLK_FINALLY,
                        curblock.cast<ASTContainerBlock>()->finallyBodyEnd(), true);
                finblk->init();
                blocks.push(finblk);
                curblock = blocks.top();
                auto fpb = finallyPlanByTarget.find(pos);
                if (fpb != finallyPlanByTarget.end()) {
                    source.setPos(fpb->second.finBodyStart);
                    pos = fpb->second.finBodyStart;
                }
            } else {
                blocks.push(prev);
                curblock = prev;
            }
        }

        if (finallyEpilogueCut.count(pos) && blocks.size() > 1
                && (curblock->blktype() == ASTBlock::BLK_IF
                    || curblock->blktype() == ASTBlock::BLK_ELSE)) {
            std::stack<PycRef<ASTBlock> > peek = blocks;
            int nif = 0;
            bool shape = false;
            while (peek.size() > 1
                    && (peek.top()->blktype() == ASTBlock::BLK_IF
                        || peek.top()->blktype() == ASTBlock::BLK_ELSE)
                    && peek.top()->end() >= pos) {
                peek.pop();
                nif++;
                if (peek.top()->blktype() == ASTBlock::BLK_FINALLY
                        && peek.top()->end() == pos) {
                    shape = true;
                    break;
                }
            }
            if (shape) {
                for (int k = 0; k < nif; ++k) {
                    PycRef<ASTBlock> innerif = curblock;
                    blocks.pop();
                    if (!stack_hist.empty())
                        stack_hist.pop();
                    curblock = blocks.top();
                    curblock->append(innerif.cast<ASTNode>());
                }
            }
        }

        /* break-escape finally: the coalesced finally body ends in a terminal
           `else: break` whose recorded end OVERSHOOTS the BLK_FINALLY end (it is the
           loop bottom-test offset the if-merge points at) — so neither the end==pos
           nor the [pos-8,pos) drains below fire, and the BLK_FINALLY can't close (its
           exceptional copy then re-renders as a duplicate).  At the registered
           finBodyEnd drain the run of inner BLK_IF/BLK_ELIF/BLK_ELSE (end > pos)
           sitting directly on a BLK_FINALLY whose end==pos so the close fires +
           resumes at the bottom test.  Non-mutating peek first; gated to the armed
           finBodyEnd offset so no other finally is affected. */
        if (breakEscFinBodyEnd.count(pos) && blocks.size() > 1
                && (curblock->blktype() == ASTBlock::BLK_IF
                    || curblock->blktype() == ASTBlock::BLK_ELIF
                    || curblock->blktype() == ASTBlock::BLK_ELSE)
                && curblock->end() > pos) {
            std::stack<PycRef<ASTBlock> > peek = blocks;
            int nif = 0; bool shape = false;
            while (peek.size() > 1
                    && (peek.top()->blktype() == ASTBlock::BLK_IF
                        || peek.top()->blktype() == ASTBlock::BLK_ELIF
                        || peek.top()->blktype() == ASTBlock::BLK_ELSE)
                    && peek.top()->end() > pos) {
                peek.pop();
                nif++;
                if (peek.top()->blktype() == ASTBlock::BLK_FINALLY
                        && peek.top()->end() == pos) {
                    shape = true;
                    break;
                }
            }
            if (shape) {
                for (int k = 0; k < nif; ++k) {
                    PycRef<ASTBlock> innerif = curblock;
                    blocks.pop();
                    if (!stack_hist.empty())
                        stack_hist.pop();
                    curblock = blocks.top();
                    curblock->append(innerif.cast<ASTNode>());
                }
            }
        }

        if (blocks.size() > 1
                && (curblock->blktype() == ASTBlock::BLK_IF
                    || curblock->blktype() == ASTBlock::BLK_ELSE)
                && curblock->end() == pos) {
            std::stack<PycRef<ASTBlock> > peek = blocks;
            int nif = 0;
            bool shape = false;
            while (peek.size() > 1
                    && (peek.top()->blktype() == ASTBlock::BLK_IF
                        || peek.top()->blktype() == ASTBlock::BLK_ELSE)
                    && peek.top()->end() == pos) {
                peek.pop();
                nif++;
                if (peek.top()->blktype() == ASTBlock::BLK_FINALLY
                        && peek.top()->end() == pos) {
                    shape = true;
                    break;
                }
            }
            if (shape) {
                for (int k = 0; k < nif; ++k) {
                    PycRef<ASTBlock> innerif = curblock;
                    blocks.pop();
                    if (!stack_hist.empty())
                        stack_hist.pop();
                    curblock = blocks.top();
                    curblock->append(innerif.cast<ASTNode>());
                }
            }
        }

        if (blocks.size() > 1
                && (curblock->blktype() == ASTBlock::BLK_IF
                    || curblock->blktype() == ASTBlock::BLK_ELSE)
                && curblock->end() < pos && curblock->end() >= pos - 8) {
            bool padOnly = true;
            { PycBuffer gs(code->code()->value(), code->code()->length());
              int go, ga, gp = curblock->end();
              gs.setPos(gp);
              while (gp < pos && !gs.atEof()) {
                  bc_next(gs, mod, go, ga, gp);
                  if (go == Pyc::CACHE) continue;
                  if (go != Pyc::RERAISE && go != Pyc::RERAISE_A
                          && go != Pyc::COPY_A && go != Pyc::POP_EXCEPT
                          && go != Pyc::EXTENDED_ARG_A) { padOnly = false; break; }
              } }
            if (padOnly) {
                std::stack<PycRef<ASTBlock> > peek = blocks;
                int nif = 0;
                bool shape = false;
                while (peek.size() > 1
                        && (peek.top()->blktype() == ASTBlock::BLK_IF
                            || peek.top()->blktype() == ASTBlock::BLK_ELSE)
                        && peek.top()->end() < pos && peek.top()->end() >= pos - 8) {
                    peek.pop();
                    nif++;
                    if (peek.top()->blktype() == ASTBlock::BLK_FINALLY
                            && peek.top()->end() == pos) {
                        shape = true;
                        break;
                    }
                }
                if (shape) {
                    for (int k = 0; k < nif; ++k) {
                        PycRef<ASTBlock> innerif = curblock;
                        blocks.pop();
                        if (!stack_hist.empty())
                            stack_hist.pop();
                        curblock = blocks.top();
                        curblock->append(innerif.cast<ASTNode>());
                    }
                }
            }
        }

        if (curblock->blktype() == ASTBlock::BLK_FINALLY
                && curblock->end() == pos
                && blocks.size() > 1) {
            bool raiseFin = raiseFinallyClose.count(pos) != 0;
            bool loopTailFin = loopTailFinClose.count(pos) != 0;
            PycRef<ASTBlock> finblk = curblock;
            blocks.pop();
            curblock = blocks.top();
            if (curblock->blktype() == ASTBlock::BLK_CONTAINER
                    && curblock.cast<ASTContainerBlock>()->hasFinally()
                    && curblock.cast<ASTContainerBlock>()->finallyAfter() > 0) {
                PycRef<ASTContainerBlock> cont = curblock.cast<ASTContainerBlock>();
                cont->append(finblk.cast<ASTNode>());
                int after = cont->finallyAfter();
                blocks.pop();
                curblock = blocks.top();
                curblock->append(cont.cast<ASTNode>());
                {
                    std::stack<PycRef<ASTBlock> > nsc = blocks;
                    if (curblock->blktype() == ASTBlock::BLK_ELSE
                            && nestedElseHandlerVals.count(curblock->end())
                            && curblock->end() > pos && curblock->end() < after)
                        after = curblock->end();
                    while (!nsc.empty()) {
                        PycRef<ASTBlock> nb = nsc.top(); nsc.pop();
                        if (nb->blktype() == ASTBlock::BLK_ELSE
                                && nestedElseHandlerVals.count(nb->end())
                                && nb->end() > pos && nb->end() < after)
                            after = nb->end();
                    }
                }
                int ifElseMerge = -1;
                if ((curblock->blktype() == ASTBlock::BLK_IF
                        || curblock->blktype() == ASTBlock::BLK_ELIF)
                        && curblock->end() > pos && curblock->end() < after) {
                    PycBuffer pk(code->code()->value(), code->code()->length());
                    pk.setPos(curblock->end());
                    int po = -1, pa = 0, pp = curblock->end();
                    bc_next(pk, mod, po, pa, pp);
                    bool backEdge = (po == Pyc::JUMP_BACKWARD_A
                            || po == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A
                            || po == Pyc::INSTRUMENTED_JUMP_BACKWARD_A);
                    /* (Historical: a `compoundCond` guard here used to bail when the
                       chain start was a short-circuit boolean `elif A and B:` / nested
                       `else: if A: if B:` — the else->elif collapse once crashed the
                       boolean recon. The recon now folds that collapse cleanly, so the
                       guard was removed: `else: if A: if B is None: pass` now renders
                       as `elif A and B is None:` byte-faithfully. */
                    bool loopBodyElse = false;
                    if (!backEdge) {
                        PycBuffer ms(code->code()->value(), code->code()->length());
                        ms.setPos(curblock->end());
                        int mo, ma, mp = curblock->end(), lastOp = -1;
                        const int W3 = (int)sizeof(uint16_t);
                        bool mergeFwdReached = false;
                        for (int i = 0; i < 4096 && mp < after; ++i) {
                            int prev = mp;
                            if (ms.atEof()) break;
                            bc_next(ms, mod, mo, ma, mp);
                            if (mp <= prev) break;
                            if (mo == Pyc::CACHE) continue;
                            lastOp = mo;
                            bool fwd = (mo == Pyc::JUMP_FORWARD_A || mo == Pyc::FOR_ITER_A
                                    || mo == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                                    || mo == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                                    || mo == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                                    || mo == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A);
                            if (fwd && mp + ma * W3 == after) mergeFwdReached = true;
                        }
                        bool lastIsBackEdge = (lastOp == Pyc::JUMP_BACKWARD_A
                                || lastOp == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A
                                || lastOp == Pyc::INSTRUMENTED_JUMP_BACKWARD_A);
                        loopBodyElse = lastIsBackEdge && !mergeFwdReached;
                    }
                    if (!backEdge && !loopBodyElse) {
                        ifElseMerge = after;
                        after = curblock->end();
                    }
                }
                source.setPos(after);
                pos = after;
                if (ifElseMerge >= 0) {
                    PycRef<ASTBlock> ifb = curblock;
                    if (!stack_hist.empty())
                        stack_hist.pop();
                    blocks.pop();
                    curblock = blocks.top();
                    curblock->append(ifb.cast<ASTNode>());
                    PycRef<ASTBlock> elseb = new ASTBlock(ASTBlock::BLK_ELSE, ifElseMerge);
                    blocks.push(elseb.cast<ASTBlock>());
                    curblock = elseb;
                }
                while (next_exception_entry < exception_entries.size()
                        && exception_entries[next_exception_entry].start_offset < after) {
                    next_exception_entry++;
                }
                bool retFinClose = (curblock->blktype() == ASTBlock::BLK_IF
                            || curblock->blktype() == ASTBlock::BLK_ELIF
                            || curblock->blktype() == ASTBlock::BLK_ELSE)
                        && curblock->end() == pos;
                if (raiseFin || loopTailFin || retFinClose) {
                    while (blocks.size() > 1
                            && curblock->end() > 0
                            && curblock->end() <= pos
                            && (curblock->blktype() == ASTBlock::BLK_IF
                                || curblock->blktype() == ASTBlock::BLK_ELIF
                                || curblock->blktype() == ASTBlock::BLK_ELSE
                                || curblock->blktype() == ASTBlock::BLK_WHILE
                                || curblock->blktype() == ASTBlock::BLK_FOR)) {
                        bool isLoop = curblock->blktype() == ASTBlock::BLK_WHILE
                                || curblock->blktype() == ASTBlock::BLK_FOR;
                        PycRef<ASTBlock> b = curblock;
                        if (!isLoop && !stack_hist.empty())
                            stack_hist.pop();
                        blocks.pop();
                        curblock = blocks.top();
                        curblock->append(b.cast<ASTNode>());
                    }
                }
                if (finallyCopySkip.count(pos)) {
                    if (source.atEof())
                        break;
                    continue;
                }
                if (curblock->blktype() == ASTBlock::BLK_TRY
                        && curblock->end() == pos) {
                    if (source.atEof())
                        break;
                    continue;
                }
            } else {
                blocks.push(finblk);
                curblock = finblk;
            }
        }

        if (curblock->blktype() == ASTBlock::BLK_EXCEPT
                && curblock->end() == pos
                && blocks.size() > 1) {
            PycRef<ASTBlock> prev = curblock;
            blocks.pop();
            curblock = blocks.top();

            if (!stack_hist.empty()) {
                stack = stack_hist.top();
                stack_hist.pop();
            }

            if (prev->size() != 0) {
                curblock->append(prev.cast<ASTNode>());
            }

            if (curblock->blktype() == ASTBlock::BLK_CONTAINER
                    && !curblock.cast<ASTContainerBlock>()->hasFinally()) {
                PycRef<ASTBlock> cont = curblock;
                blocks.pop();
                curblock = blocks.top();
                curblock->append(cont.cast<ASTNode>());

                if (exceptElseAt.count(pos)
                        && (curblock->blktype() == ASTBlock::BLK_IF
                            || curblock->blktype() == ASTBlock::BLK_ELIF)
                        && curblock->end() == pos
                        && blocks.size() > 1) {
                    PycRef<ASTBlock> ifb = curblock;
                    blocks.pop();
                    curblock = blocks.top();
                    curblock->append(ifb.cast<ASTNode>());
                    PycRef<ASTBlock> elseb = new ASTBlock(ASTBlock::BLK_ELSE, exceptElseAt[pos]);
                    elseb->init();
                    { int eAt = exceptElseAt[pos];
                      bool nestedHandler = false;
                      for (const auto& ee : exception_entries)
                          if (ee.start_offset >= pos && ee.start_offset < eAt) {
                              nestedHandler = true; break;
                          }
                      int toMergeE = 0; bool bodyE = false;
                      PycBuffer ps(code->code()->value(), code->code()->length());
                      ps.setPos(pos);
                      int po, pa, pp = pos;
                      while (pp < eAt && !ps.atEof()) {
                          int pip = pp;
                          bc_next(ps, mod, po, pa, pp);
                          int pt = -1;
                          if (po == Pyc::POP_JUMP_FORWARD_IF_FALSE_A || po == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                                  || po == Pyc::POP_JUMP_FORWARD_IF_NONE_A || po == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A)
                              pt = pp + pa * (int)sizeof(uint16_t);
                          if (pt == eAt && !bodyE) ++toMergeE;
                          if (po == Pyc::POP_TOP || po == Pyc::STORE_FAST_A || po == Pyc::STORE_NAME_A
                                  || po == Pyc::STORE_GLOBAL_A || po == Pyc::STORE_DEREF_A
                                  || po == Pyc::STORE_ATTR_A || po == Pyc::STORE_SUBSCR)
                              bodyE = true;
                          if (pp <= pip) break;
                      }
                      if (nestedHandler || toMergeE >= 2) exceptElseNoCollapse.insert((ASTBlock *)elseb); }
                    blocks.push(elseb);
                    curblock = blocks.top();
                }
                else if (!exceptElseAt.count(pos)
                        && (curblock->blktype() == ASTBlock::BLK_IF
                            || curblock->blktype() == ASTBlock::BLK_ELSE
                            || curblock->blktype() == ASTBlock::BLK_ELIF)
                        && curblock->end() == pos) {
                    else_pop = true;
                }
            }
        }

        if (blocks.size() > 3
                && curblock->blktype() == ASTBlock::BLK_EXCEPT
                && curblock->end() == 0) {
            std::stack<PycRef<ASTBlock> > savedBlocks = blocks;
            stackhist_t savedHist = stack_hist;
            FastStack savedStack = stack;
            PycRef<ASTBlock> exc = curblock;
            blocks.pop();
            PycRef<ASTBlock> cont = blocks.top();
            blocks.pop();
            PycRef<ASTBlock> box = blocks.top();
            ASTBlock::BlkType bt = box->blktype();
            bool ok = false;
            if (cont->blktype() == ASTBlock::BLK_CONTAINER
                    && cont->end() == 0
                    && box->end() == pos
                    && (bt == ASTBlock::BLK_FOR || bt == ASTBlock::BLK_WHILE)) {
                if (!stack_hist.empty()) { stack = stack_hist.top(); stack_hist.pop(); }
                if (exc->size() != 0)
                    cont->append(exc.cast<ASTNode>());
                box->append(cont.cast<ASTNode>());
                curblock = box;
                loopCloseAtExit.insert(pos);
                ok = true;
            } else if (!loopCloseAtExit.count(pos)
                    && cont->blktype() == ASTBlock::BLK_CONTAINER
                    && cont->end() == 0
                    && box->end() == pos
                    && (bt == ASTBlock::BLK_IF || bt == ASTBlock::BLK_ELIF
                        || bt == ASTBlock::BLK_ELSE)) {
                blocks.pop();
                PycRef<ASTBlock> parent = blocks.top();
                ASTBlock::BlkType pt = parent->blktype();
                bool parentClean = (pt == ASTBlock::BLK_MAIN
                        || pt == ASTBlock::BLK_IF || pt == ASTBlock::BLK_ELIF
                        || pt == ASTBlock::BLK_ELSE || pt == ASTBlock::BLK_TRY
                        || pt == ASTBlock::BLK_FOR || pt == ASTBlock::BLK_WHILE)
                        && (parent->end() == 0 || parent->end() > pos);
                if (parentClean) {
                    if (!stack_hist.empty()) { stack = stack_hist.top(); stack_hist.pop(); }
                    if (!stack_hist.empty()) stack_hist.pop();
                    if (exc->size() != 0)
                        cont->append(exc.cast<ASTNode>());
                    box->append(cont.cast<ASTNode>());
                    parent->append(box.cast<ASTNode>());
                    curblock = parent;
                    ok = true;
                }
            } else if (!loopCloseAtExit.count(pos)
                    && cont->blktype() == ASTBlock::BLK_CONTAINER
                    && cont->end() == 0
                    && box->end() == pos
                    && bt == ASTBlock::BLK_TRY) {
                blocks.pop();
                PycRef<ASTBlock> ocont = blocks.top();
                if (ocont->blktype() == ASTBlock::BLK_CONTAINER
                        && ocont.cast<ASTContainerBlock>()->hasExcept()
                        && !ocont.cast<ASTContainerBlock>()->hasFinally()) {
                    if (!stack_hist.empty()) { stack = stack_hist.top(); stack_hist.pop(); }
                    if (exc->size() != 0)
                        cont->append(exc.cast<ASTNode>());
                    box->append(cont.cast<ASTNode>());
                    if (!stack_hist.empty()) { stack = stack_hist.top(); stack_hist.pop(); }
                    ocont->append(box.cast<ASTNode>());
                    stack_hist.push(stack);
                    PycRef<ASTBlock> oexc = new ASTCondBlock(ASTBlock::BLK_EXCEPT, 0, NULL, false);
                    oexc->init();
                    blocks.push(oexc);
                    curblock = blocks.top();
                    ok = true;
                }
            } else if (!loopCloseAtExit.count(pos)
                    && cont->blktype() == ASTBlock::BLK_CONTAINER
                    && cont->end() == 0
                    && (bt == ASTBlock::BLK_IF || bt == ASTBlock::BLK_ELIF
                        || bt == ASTBlock::BLK_ELSE)
                    && box->end() > 0 && box->end() < pos
                    && backwardJumpOffsets.count(box->end())) {
                blocks.pop();
                PycRef<ASTBlock> loopb = blocks.top();
                ASTBlock::BlkType lt = loopb->blktype();
                if ((lt == ASTBlock::BLK_FOR || lt == ASTBlock::BLK_WHILE)
                        && loopb->end() == pos) {
                    if (!stack_hist.empty()) { stack = stack_hist.top(); stack_hist.pop(); }
                    if (!stack_hist.empty()) stack_hist.pop();
                    if (exc->size() != 0)
                        cont->append(exc.cast<ASTNode>());
                    box->append(cont.cast<ASTNode>());
                    loopb->append(box.cast<ASTNode>());
                    curblock = loopb;
                    loopCloseAtExit.insert(pos);
                    ok = true;
                }
            }
            if (!ok) {
                blocks = savedBlocks;
                stack_hist = savedHist;
                stack = savedStack;
                curblock = blocks.top();
            }
        }

        bool degenerateForReturn = false;
        if (forStuckExit.count(pos)
                && blocks.size() > 1
                && curblock->blktype() == ASTBlock::BLK_FOR
                && curblock->end() == pos) {
            int fstart = curblock.cast<ASTIterBlock>()->start();
            degenerateForReturn = (backedgeCount[fstart] == 0
                    && !forHasBreak.count(fstart)
                    && !forBreakBeyondExit.count(fstart));
        }
        bool fallThroughBreak = false;
        if (blocks.size() > 1
                && curblock->blktype() == ASTBlock::BLK_FOR
                && curblock->end() == pos
                && !loopCloseAtExit.count(pos)
                && !forStuckExit.count(pos)) {
            int fstart = curblock.cast<ASTIterBlock>()->start();
            fallThroughBreak = forExitFallThrough.count(fstart)
                    && (backedgeCount[fstart] >= 1
                        || doubleBreakContLoop.count(fstart))
                    && (!forHasBreak.count(fstart)
                        || doubleBreakContLoop.count(fstart))
                    && !forElseMerge.count(pos);
            if (!fallThroughBreak && forUncondBreak.count(pos)
                    && !forElseMerge.count(pos))
                fallThroughBreak = true;
        }
        if (topTestWhileExit.count(pos) && blocks.size() > 2
                && (curblock->blktype() == ASTBlock::BLK_EXCEPT
                    || curblock->blktype() == ASTBlock::BLK_CONTAINER)) {
            std::stack<PycRef<ASTBlock> > pk = blocks;
            int ndrain = 0; bool shape = false;
            while (pk.size() > 1
                    && (pk.top()->blktype() == ASTBlock::BLK_EXCEPT
                        || pk.top()->blktype() == ASTBlock::BLK_CONTAINER)) {
                pk.pop(); ndrain++;
                if (!pk.empty() && pk.top()->blktype() == ASTBlock::BLK_WHILE
                        && pk.top()->end() == pos) { shape = true; break; }
            }
            if (shape) {
                for (int k = 0; k < ndrain; ++k) {
                    PycRef<ASTBlock> ib = curblock;
                    blocks.pop();
                    curblock = blocks.top();
                    if (ib->size() != 0)
                        curblock->append(ib.cast<ASTNode>());
                    if (!stack_hist.empty())
                        stack_hist.pop();
                }
            }
        }
        if (topTestWhileExit.count(pos)
                && blocks.size() > 1
                && curblock->blktype() == ASTBlock::BLK_WHILE
                && curblock->end() == pos) {
            PycRef<ASTBlock> loopb = curblock;
            blocks.pop();
            PycRef<ASTBlock> parent = blocks.top();
            ASTBlock::BlkType pt = parent->blktype();
            bool clean_parent = (pt == ASTBlock::BLK_MAIN
                    || pt == ASTBlock::BLK_IF || pt == ASTBlock::BLK_ELIF
                    || pt == ASTBlock::BLK_ELSE || pt == ASTBlock::BLK_TRY
                    || pt == ASTBlock::BLK_EXCEPT || pt == ASTBlock::BLK_FINALLY
                    || pt == ASTBlock::BLK_WITH
                    || pt == ASTBlock::BLK_FOR || pt == ASTBlock::BLK_WHILE)
                    && (parent->end() == 0 || parent->end() >= pos);
            if (clean_parent) {
                if (topTestWhileFallBreak.count(pos))
                    loopb->append(new ASTKeyword(ASTKeyword::KW_BREAK));
                else if (topTestWhileFallContinue.count(pos))
                    loopb->append(new ASTKeyword(ASTKeyword::KW_CONTINUE));
                if (!stack_hist.empty())
                    stack_hist.pop();
                curblock = parent;
                curblock->append(loopb.cast<ASTNode>());
                if (loopBreakElse.count(pos)
                        && (curblock->blktype() == ASTBlock::BLK_IF
                            || curblock->blktype() == ASTBlock::BLK_ELIF)
                        && curblock->end() == pos
                        && blocks.size() > 1) {
                    PycRef<ASTBlock> ifb = curblock;
                    if (!stack_hist.empty())
                        stack_hist.pop();
                    blocks.pop();
                    curblock = blocks.top();
                    curblock->append(ifb.cast<ASTNode>());
                    PycRef<ASTBlock> elseb = new ASTBlock(
                            ASTBlock::BLK_ELSE, loopBreakElse[pos]);
                    blocks.push(elseb.cast<ASTBlock>());
                    curblock = elseb;
                }
            } else {
                blocks.push(loopb);
            }
        }

        if (loopCloseAtExit.count(pos)
                || degenerateForReturn
                || fallThroughBreak
                || (forStuckExit.count(pos)
                    && blocks.size() > 1
                    && curblock->blktype() == ASTBlock::BLK_FOR
                    && curblock->end() == pos
                    && backedgeCount[curblock.cast<ASTIterBlock>()->start()] >= 1)) {
            if (blocks.size() > 1
                    && curblock->blktype() == ASTBlock::BLK_FOR
                    && curblock->end() == pos) {
                PycRef<ASTBlock> loopb = curblock;
                if (fallThroughBreak)
                    loopb->append(new ASTKeyword(ASTKeyword::KW_BREAK));
                blocks.pop();
                PycRef<ASTBlock> parent = blocks.top();
                ASTBlock::BlkType pt = parent->blktype();
                bool clean_parent = (pt == ASTBlock::BLK_MAIN
                        || pt == ASTBlock::BLK_IF || pt == ASTBlock::BLK_ELIF
                        || pt == ASTBlock::BLK_ELSE
                        || pt == ASTBlock::BLK_TRY
                        || pt == ASTBlock::BLK_EXCEPT
                        || pt == ASTBlock::BLK_FOR
                        || pt == ASTBlock::BLK_WHILE
                        || pt == ASTBlock::BLK_WITH)
                        && (parent->end() == 0 || parent->end() > pos);
                bool staleIf = (pt == ASTBlock::BLK_IF || pt == ASTBlock::BLK_ELIF
                        || pt == ASTBlock::BLK_ELSE)
                        && parent->end() > 0 && parent->end() <= pos;
                if (clean_parent) {
                    curblock = parent;
                    curblock->append(loopb.cast<ASTNode>());
                    if (loopb->blktype() == ASTBlock::BLK_FOR
                            && forElseBreakLoop.count(loopb.cast<ASTIterBlock>()->start())
                            && forElseMerge.count(pos)) {
                        stack_hist.push(stack);
                        PycRef<ASTBlock> elseblk = new ASTBlock(
                                ASTBlock::BLK_ELSE, forElseMerge[pos]);
                        elseblk->init();
                        loopElseBlocks.insert((ASTBlock *)elseblk);
                        blocks.push(elseblk);
                        curblock = blocks.top();
                    }
                } else if (staleIf && blocks.size() > 1) {
                    std::vector<PycRef<ASTBlock>> chain;
                    chain.push_back(parent);
                    while (blocks.size() > 1) {
                        PycRef<ASTBlock> top = blocks.top();
                        ASTBlock::BlkType tt = top->blktype();
                        bool topStale = (tt == ASTBlock::BLK_IF
                                || tt == ASTBlock::BLK_ELIF
                                || tt == ASTBlock::BLK_ELSE)
                                && top->end() > 0 && top->end() <= pos;
                        if (!topStale)
                            break;
                        blocks.pop();
                        chain.push_back(top);
                    }
                    PycRef<ASTBlock> anc = blocks.top();
                    ASTBlock::BlkType at = anc->blktype();
                    bool ancClean = (at == ASTBlock::BLK_MAIN
                            || at == ASTBlock::BLK_IF || at == ASTBlock::BLK_ELIF
                            || at == ASTBlock::BLK_ELSE || at == ASTBlock::BLK_TRY
                            || at == ASTBlock::BLK_EXCEPT || at == ASTBlock::BLK_WITH
                            || at == ASTBlock::BLK_FOR || at == ASTBlock::BLK_WHILE)
                            && (anc->end() == 0 || anc->end() > pos);
                    if (ancClean) {
                        chain.front()->append(loopb.cast<ASTNode>());
                        for (size_t k = 1; k < chain.size(); ++k)
                            chain[k]->append(chain[k-1].cast<ASTNode>());
                        anc->append(chain.back().cast<ASTNode>());
                        for (size_t k = 0; k < chain.size(); ++k)
                            if (!stack_hist.empty())
                                stack_hist.pop();
                        curblock = anc;
                    } else {
                        for (size_t k = chain.size(); k-- > 0; )
                            blocks.push(chain[k]);
                        blocks.push(loopb);
                    }
                } else {
                    blocks.push(loopb);
                }
            }
            loopCloseAtExit.erase(pos);
        }

        if (withResume.count(pos)) {
            while (blocks.size() > 1
                    && curblock->end() == pos
                    && (curblock->blktype() == ASTBlock::BLK_IF
                        || curblock->blktype() == ASTBlock::BLK_ELIF
                        || curblock->blktype() == ASTBlock::BLK_ELSE
                        || curblock->blktype() == ASTBlock::BLK_WHILE
                        || curblock->blktype() == ASTBlock::BLK_FOR)) {
                bool isLoop = curblock->blktype() == ASTBlock::BLK_WHILE
                        || curblock->blktype() == ASTBlock::BLK_FOR;
                PycRef<ASTBlock> b = curblock;
                if (!isLoop && !stack_hist.empty())
                    stack_hist.pop();
                blocks.pop();
                curblock = blocks.top();
                curblock->append(b.cast<ASTNode>());
            }
        }

        bool inTryWithReopen = false;
        bool withResumeToFcs = false;
        while (curblock->blktype() == ASTBlock::BLK_WITH
                && curblock->end() == pos
                && blocks.size() > 1
                && withResume.count(pos)) {
            PycRef<ASTBlock> w = curblock;
            bool isInTry = withInTryResume.count(pos) > 0;
            blocks.pop();
            curblock = blocks.top();
            curblock->append(w.cast<ASTNode>());
            int resume = withResume[pos];
            source.setPos(resume);
            pos = resume;
            while (next_exception_entry < exception_entries.size()
                    && exception_entries[next_exception_entry].start_offset < pos) {
                next_exception_entry++;
            }
            if (finallyCopySkip.count(pos)) { withResumeToFcs = true; break; }
            if (isInTry) { inTryWithReopen = true; break; }
            while (blocks.size() > 1
                    && curblock->end() > 0
                    && curblock->end() <= pos
                    && (curblock->blktype() == ASTBlock::BLK_IF
                        || curblock->blktype() == ASTBlock::BLK_ELIF
                        || curblock->blktype() == ASTBlock::BLK_ELSE
                        || curblock->blktype() == ASTBlock::BLK_WHILE
                        || curblock->blktype() == ASTBlock::BLK_FOR)) {
                bool isLoop = curblock->blktype() == ASTBlock::BLK_WHILE
                        || curblock->blktype() == ASTBlock::BLK_FOR;
                PycRef<ASTBlock> b = curblock;
                if (!isLoop && !stack_hist.empty())
                    stack_hist.pop();
                blocks.pop();
                curblock = blocks.top();
                curblock->append(b.cast<ASTNode>());
            }
            auto hsk = withHandlerSkip.find(pos);
            if (hsk != withHandlerSkip.end()) {
                source.setPos(hsk->second);
                pos = hsk->second;
            }
        }
        if (inTryWithReopen)
            continue;
        if (withResumeToFcs)
            continue;

        if (source.atEof())
            break;

        bool deferBoolFoldForTernary = false;
        if (else_pop && curblock->blktype() == ASTBlock::BLK_ELSE
                && curblock->end() == pos && blocks.size() > 1
                && !boolPending.empty()
                && (boolPending.back().target == pos
                    || canonPop(boolPending.back().target) == canonPop(pos))) {
            std::stack<PycRef<ASTBlock> > pk = blocks;
            pk.pop();
            PycRef<ASTBlock> parent = pk.top();
            if (!parent->nodes().empty()) {
                PycRef<ASTNode> lastn = parent->nodes().back();
                if (lastn->type() == ASTNode::NODE_BLOCK
                        && lastn.cast<ASTBlock>()->blktype() == ASTBlock::BLK_IF
                        && boolPending.back().off < lastn.cast<ASTBlock>()->end())
                    deferBoolFoldForTernary = true;
            }
        }
        while (!deferBoolFoldForTernary && !boolPending.empty() && !stack.empty()
                && (boolPending.back().target == pos
                    || canonPop(boolPending.back().target) == canonPop(pos))) {
            BoolShortCircuit sc = boolPending.back();
            boolPending.pop_back();
            PycRef<ASTNode> rhs = stack.top();
            stack.pop();
            PycRef<ASTBinary> bin = new ASTBinary(
                    recoverFoldedAndOperand(sc.left, sc.isOr, code, mod), rhs,
                    sc.isOr ? ASTBinary::BIN_LOG_OR : ASTBinary::BIN_LOG_AND);
            int lo = code->lineForOffset(sc.off), lt = code->lineForOffset(sc.target);
            bin->setScLine(lo);
            if (lo >= 0 && lt >= 0 && lo != lt)
                bin->setBreakBefore(true);
            stack.push(bin.cast<ASTNode>());
        }

        {
            auto es = withExitSkip.find(pos);
            if (es != withExitSkip.end()) {
                source.setPos(es->second);
                pos = es->second;
                while (next_exception_entry < exception_entries.size()
                        && exception_entries[next_exception_entry].start_offset < pos) {
                    next_exception_entry++;
                }
                continue;
            }
        }

        curpos = pos;
        g_stmtOff = curpos;   // tag nodes built for this instruction (see ASTNode)
        g_stmtLine = code->lineForOffset(curpos);
        g_stmtCol = code->colForOffset(curpos);
        bc_next(source, mod, opcode, operand, pos);

        /* Track the run of NOP offsets ending just before this instruction (used to
           recover multi-line const-default signature anchors at MAKE_FUNCTION). */
        std::vector<int> nopsBeforeCur = pendingNopOffs;
        if (opcode == Pyc::NOP)
            pendingNopOffs.push_back(curpos);
        else
            pendingNopOffs.clear();

        /* Shared-tail return absorb (see teoElseAbsorbRet): this offset is a
           `LOAD_x; RETURN_VALUE` already emitted into the BLK_ELSE. Skip both ops so
           it is not re-rendered as dead code after the loop-exit statement. */
        if (teoElseAbsorbSuppress.count(curpos)) {
            int no = Pyc::PYC_INVALID_OPCODE, na = 0, npos = pos;
            if (!source.atEof())
                bc_next(source, mod, no, na, npos);
            if (no == Pyc::RETURN_VALUE) {
                source.setPos(npos);
                pos = npos;
                continue;
            }
            source.setPos(pos);   /* not the expected shape: restore, fall through */
        }

        if (compAndOrOp.count(curpos) && !stack.empty()
                && curblock->blktype() == ASTBlock::BLK_FOR
                && curblock.cast<ASTIterBlock>()->isComprehension()) {
            const CompAndOrOp& r = compAndOrOp[curpos];
            PycRef<ASTNode> c = stack.top(); stack.pop();
            PycRef<ASTNode> oper;
            switch (r.sense) {
            case 0: oper = c; break;
            case 1: oper = new ASTUnary(c, ASTUnary::UN_NOT); break;
            case 2: oper = new ASTCompare(c, new ASTObject(Pyc_None), ASTCompare::CMP_IS); break;
            default: oper = new ASTCompare(c, new ASTObject(Pyc_None), ASTCompare::CMP_IS_NOT); break;
            }
            int inOp = compAndOrTopAnd ? ASTBinary::BIN_LOG_OR : ASTBinary::BIN_LOG_AND;
            int btwOp = compAndOrTopAnd ? ASTBinary::BIN_LOG_AND : ASTBinary::BIN_LOG_OR;
            compAndOrGroup = (compAndOrGroup == nullptr) ? oper
                    : new ASTBinary(compAndOrGroup, oper, inOp);
            if (r.groupEnd) {
                compAndOrAcc = (compAndOrAcc == nullptr) ? compAndOrGroup
                        : new ASTBinary(compAndOrAcc, compAndOrGroup, btwOp);
                compAndOrGroup = nullptr;
            }
            if (r.isLast) {
                curblock.cast<ASTIterBlock>()->setCondition(compAndOrAcc);
                source.setPos(compAndOrKeep); pos = compAndOrKeep;
            }
            continue;
        }

        if (compTernTest.count(curpos) && !stack.empty()) {
            PycRef<ASTNode> cval = stack.top(); stack.pop();
            int top = compTernTestOp[curpos];
            PycRef<ASTNode> cond; bool neg = false;
            if (top == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A)
                cond = new ASTCompare(cval, new ASTObject(Pyc_None), ASTCompare::CMP_IS);
            else if (top == Pyc::POP_JUMP_FORWARD_IF_NONE_A)
                cond = new ASTCompare(cval, new ASTObject(Pyc_None), ASTCompare::CMP_IS_NOT);
            else { cond = cval; neg = (top == Pyc::POP_JUMP_FORWARD_IF_TRUE_A); }
            ternCondBlk = new ASTCondBlock(ASTBlock::BLK_IF, 0, cond, neg);
            source.setPos(compTernTest[curpos]);
            pos = compTernTest[curpos];
            continue;
        }
        if (compTernThenBwd.count(curpos) && !stack.empty()) {
            ternThenVal = stack.top(); stack.pop();
            source.setPos(compTernThenBwd[curpos]);
            pos = compTernThenBwd[curpos];
            continue;
        }
        if (compTernElseBwd.count(curpos) && !stack.empty()
                && ternCondBlk != nullptr && ternThenVal != nullptr
                && curblock->blktype() == ASTBlock::BLK_FOR
                && curblock.cast<ASTIterBlock>()->isComprehension()) {
            PycRef<ASTNode> elseVal = stack.top(); stack.pop();
            PycRef<ASTNode> tern = new ASTTernary(ternCondBlk, ternThenVal, elseVal);
            PycRef<ASTNode> existing = curblock.cast<ASTIterBlock>()->condition();
            if (existing != nullptr)
                tern = new ASTBinary(existing, tern, ASTBinary::BIN_LOG_AND);
            curblock.cast<ASTIterBlock>()->setCondition(tern);
            ternCondBlk = nullptr; ternThenVal = nullptr;
            source.setPos(compTernElseBwd[curpos]);
            pos = compTernElseBwd[curpos];
            continue;
        }

        if (exceptAsSkip.count(curpos))
            continue;
        if (exceptAsBind.count(curpos)) {
            if (curblock->blktype() == ASTBlock::BLK_EXCEPT)
                curblock.cast<ASTCondBlock>()->setExceptVar(
                        new ASTName(exceptAsBind[curpos]));
            continue;
        }
        if (chainIfTrue.count(curpos)) {
            int t = chainIfTrue[curpos];
            source.setPos(t);
            pos = t;
            continue;
        }

        bool ternaryElseBeforeOrPop = false;
        if (else_pop
                && (opcode == Pyc::JUMP_IF_TRUE_OR_POP_A || opcode == Pyc::JUMP_IF_FALSE_OR_POP_A)
                && curblock->blktype() == ASTBlock::BLK_ELSE
                && curblock->end() == curpos && curblock->size() == 0
                && blocks.size() > 1) {
            std::stack<PycRef<ASTBlock> > pk = blocks;
            pk.pop();
            PycRef<ASTBlock> parent = pk.top();
            if (!parent->nodes().empty()) {
                PycRef<ASTNode> lastn = parent->nodes().back();
                if (lastn->type() == ASTNode::NODE_BLOCK
                        && lastn.cast<ASTBlock>()->blktype() == ASTBlock::BLK_IF
                        && lastn.cast<ASTBlock>()->size() == 0)
                    ternaryElseBeforeOrPop = true;
            }
        }

        if (need_try && opcode != Pyc::SETUP_EXCEPT_A) {
            need_try = false;

            /* Store the current stack for the except/finally statement(s) */
            stack_hist.push(stack);
            PycRef<ASTBlock> tryblock = new ASTBlock(ASTBlock::BLK_TRY, curblock->end(), true);
            blocks.push(tryblock);
            curblock = blocks.top();
        } else if (else_pop
                && (ternaryElseBeforeOrPop
                    || (opcode != Pyc::JUMP_FORWARD_A
                && opcode != Pyc::JUMP_IF_FALSE_A
                && opcode != Pyc::JUMP_IF_FALSE_OR_POP_A
                && opcode != Pyc::POP_JUMP_IF_FALSE_A
                && opcode != Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                && opcode != Pyc::JUMP_IF_TRUE_A
                && opcode != Pyc::JUMP_IF_TRUE_OR_POP_A
                && opcode != Pyc::POP_JUMP_IF_TRUE_A
                && opcode != Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                && opcode != Pyc::POP_BLOCK))) {
            else_pop = false;

            PycRef<ASTBlock> prev = curblock;
            while (prev->end() < pos
                    && prev->blktype() != ASTBlock::BLK_MAIN) {
                if (prev->blktype() != ASTBlock::BLK_CONTAINER) {
                    if (prev->end() == 0) {
                        break;
                    }

                    /* We want to keep the stack the same, but we need to pop
                     * a level off the history. */
                    //stack = stack_hist.top();
                    if (!stack_hist.empty())
                        stack_hist.pop();
                }
                if (prev->blktype() == ASTBlock::BLK_WHILE) {
                    PycRef<ASTNode> wc = prev.cast<ASTCondBlock>()->cond();
                    if (wc != NULL && wc.type() == ASTNode::NODE_OBJECT
                            && wc.cast<ASTObject>()->object() == Pyc_True)
                        prev->append(new ASTKeyword(ASTKeyword::KW_BREAK));
                }
                blocks.pop();

                if (blocks.empty())
                    break;

                curblock = blocks.top();
                curblock->append(prev.cast<ASTNode>());

                prev = curblock;

                CheckIfExpr(stack, curblock);
            }
        }

        if (deferBoolFoldForTernary) {
            while (!boolPending.empty() && !stack.empty()
                    && (boolPending.back().target == curpos
                        || canonPop(boolPending.back().target) == canonPop(curpos))) {
                BoolShortCircuit sc = boolPending.back();
                boolPending.pop_back();
                PycRef<ASTNode> rhs = stack.top();
                stack.pop();
                PycRef<ASTBinary> bin = new ASTBinary(
                        recoverFoldedAndOperand(sc.left, sc.isOr, code, mod), rhs,
                        sc.isOr ? ASTBinary::BIN_LOG_OR : ASTBinary::BIN_LOG_AND);
                int lo = code->lineForOffset(sc.off), lt = code->lineForOffset(sc.target);
                bin->setScLine(lo);
                if (lo >= 0 && lt >= 0 && lo != lt)
                    bin->setBreakBefore(true);
                stack.push(bin.cast<ASTNode>());
            }
        }

        if (whileTrueHdr.count(curpos) && !whileTrueOpened.count(curpos)
                && curblock->blktype() != ASTBlock::BLK_CONTAINER) {
            whileTrueOpened.insert(curpos);
            PycRef<ASTCondBlock> whb = new ASTCondBlock(
                    ASTBlock::BLK_WHILE, whileTrueHdr[curpos],
                    new ASTObject(Pyc_True), false);
            /* An unconditional loop the source wrote as `while <truthy const>:` has
               its test optimized away, so the literal survives only as a dangling
               entry in co_consts. `while True:` leaves a bool `True` there; the
               common `while 1:` idiom leaves none. The condition object stays
               Pyc_True (the whole while-True machinery keys off that), but when no
               bool `True` const is present the source was `while 1:`, so flag the
               block to RENDER its condition as `1`. A genuine `while True:` always
               retains the const (even under -OO), so a matching loop is never
               reflagged. */
            {
                bool hasTrueConst = false;
                if (code->consts() != nullptr) {
                    for (int ci = 0; ci < code->consts()->size(); ci++) {
                        PycRef<PycObject> cc = code->consts()->get(ci);
                        if (cc != nullptr && cc->type() == PycObject::TYPE_TRUE) {
                            hasTrueConst = true;
                            break;
                        }
                    }
                }
                if (!hasTrueConst && !whileTrueTopTest.count(curpos))
                    whb->setCondRenderAsOne(true);
            }
            whb->init();
            blocks.push(whb.cast<ASTBlock>());
            curblock = blocks.top();
            if (whileTrueTopTest.count(curpos))
                whileTopTestPendingExit = whileTrueHdr[curpos];
        }

        /* --- main opcode dispatch ---------------------------------------
         * One case per byte-code instruction. Each updates the operand stack
         * `stack` and/or the block stack `blocks`/`curblock`. (Everything above
         * in this loop iteration is the control-flow matching that opens/closes
         * blocks before the instruction itself is handled.) */
        switch (opcode) {
        case Pyc::BINARY_OP_A:
        case Pyc::BINARY_ADD:
        case Pyc::BINARY_AND:
        case Pyc::BINARY_DIVIDE:
        case Pyc::BINARY_FLOOR_DIVIDE:
        case Pyc::BINARY_LSHIFT:
        case Pyc::BINARY_MODULO:
        case Pyc::BINARY_MULTIPLY:
        case Pyc::BINARY_OR:
        case Pyc::BINARY_POWER:
        case Pyc::BINARY_RSHIFT:
        case Pyc::BINARY_SUBTRACT:
        case Pyc::BINARY_TRUE_DIVIDE:
        case Pyc::BINARY_XOR:
        case Pyc::BINARY_MATRIX_MULTIPLY:
        case Pyc::INPLACE_ADD:
        case Pyc::INPLACE_AND:
        case Pyc::INPLACE_DIVIDE:
        case Pyc::INPLACE_FLOOR_DIVIDE:
        case Pyc::INPLACE_LSHIFT:
        case Pyc::INPLACE_MODULO:
        case Pyc::INPLACE_MULTIPLY:
        case Pyc::INPLACE_OR:
        case Pyc::INPLACE_POWER:
        case Pyc::INPLACE_RSHIFT:
        case Pyc::INPLACE_SUBTRACT:
        case Pyc::INPLACE_TRUE_DIVIDE:
        case Pyc::INPLACE_XOR:
        case Pyc::INPLACE_MATRIX_MULTIPLY:
            handleBinaryOp(opcode, operand);
            break;
        case Pyc::BINARY_SUBSCR:
            handleSubscript(opcode);
            break;
        case Pyc::BREAK_LOOP:
            curblock->append(new ASTKeyword(ASTKeyword::KW_BREAK));
            break;
        case Pyc::BUILD_CLASS:
            {
                PycRef<ASTNode> class_code = stack.top();
                stack.pop();
                PycRef<ASTNode> bases = stack.top();
                stack.pop();
                PycRef<ASTNode> name = stack.top();
                stack.pop();
                stack.push(new ASTClass(class_code, bases, name));
            }
            break;
        case Pyc::BUILD_FUNCTION:
            {
                PycRef<ASTNode> fun_code = stack.top();
                stack.pop();
                stack.push(new ASTFunction(fun_code, {}, {}));
            }
            break;
        case Pyc::BUILD_LIST_A:
        case Pyc::BUILD_SET_A:
        case Pyc::BUILD_MAP_A:
        case Pyc::BUILD_CONST_KEY_MAP_A:
            handleBuildCollection(opcode, operand);
            break;
        case Pyc::STORE_MAP:
            {
                PycRef<ASTNode> key = stack.top();
                stack.pop();
                PycRef<ASTNode> value = stack.top();
                stack.pop();
                PycRef<ASTMap> map = stack.top().cast<ASTMap>();
                map->add(key, value);
            }
            break;
        case Pyc::BUILD_SLICE_A:
            handleBuildCollection(opcode, operand);
            break;
        case Pyc::BUILD_STRING_A:
        case Pyc::BUILD_TUPLE_A:
            handleBuildCollection(opcode, operand);
            break;
        case Pyc::KW_NAMES_A:
            {

                int kwparams = code->getConst(operand).cast<PycTuple>()->size();
                ASTKwNamesMap kwparamList;
                std::vector<PycRef<PycObject>> keys = code->getConst(operand).cast<PycSimpleSequence>()->values();
                for (int i = 0; i < kwparams; i++) {
                    kwparamList.add(new ASTObject(keys[kwparams - i - 1]), stack.top());
                    stack.pop();
                }
                stack.push(new ASTKwNamesMap(kwparamList));
            }
            break;
        case Pyc::CALL_A:
        case Pyc::CALL_FUNCTION_A:
        case Pyc::INSTRUMENTED_CALL_A:
            {
                if ((operand & 0xFFFF) == 0 && lastSubstantialOp == Pyc::GET_ITER) {
                    PycRef<ASTNode> mfunc = stack.top(2);
                    if (mfunc != nullptr && mfunc.type() == ASTNode::NODE_FUNCTION) {
                        PycRef<PycCode> cc = mfunc.cast<ASTFunction>()->code()
                                .cast<ASTObject>()->object().cast<PycCode>();
                        const char* nm = cc->name()->value();
                        if (strcmp(nm, "<listcomp>") == 0 || strcmp(nm, "<setcomp>") == 0
                                || strcmp(nm, "<dictcomp>") == 0
                                || strcmp(nm, "<genexpr>") == 0) {
                            PycRef<ASTNode> inlined =
                                    InlineComprehension(cc, mod, stack.top(1));
                            if (inlined != nullptr) {
                                stack.pop();
                                stack.pop();
                                stack.push(inlined);
                                break;
                            }
                        }
                    }
                }

                int kwparams = (operand & 0xFF00) >> 8;
                int pparams = (operand & 0xFF);
                ASTCall::kwparam_t kwparamList;
                ASTCall::pparam_t pparamList;

                stack_hist.push(stack);
                PycRef<ASTNode> classKwMap = nullptr;
                if (mod->verCompare(3, 11) >= 0 && !stack.empty()
                        && stack.top() != nullptr
                        && stack.top().type() == ASTNode::NODE_KW_NAMES_MAP) {
                    classKwMap = stack.top();
                    stack.pop();
                }
                std::vector<PycRef<ASTNode>> aboveFunc;
                while (!stack.empty() && stack.top() != nullptr
                        && stack.top().type() != ASTNode::NODE_FUNCTION) {
                    aboveFunc.push_back(stack.top());
                    stack.pop();
                }
                PycRef<ASTNode> function = (!stack.empty()) ? stack.top() : nullptr;
                if (function != nullptr)
                    stack.pop();
                PycRef<ASTNode> loadbuild = (!stack.empty()) ? stack.top() : nullptr;
                if (loadbuild != nullptr)
                    stack.pop();
                if (loadbuild != nullptr
                        && loadbuild.type() == ASTNode::NODE_LOADBUILDCLASS
                        && function != nullptr && !aboveFunc.empty()) {
                    if (!stack.empty() && stack.top() == nullptr)
                        stack.pop();
                    PycRef<ASTNode> name = aboveFunc.back();
                    ASTTuple::value_t bases;
                    for (size_t i = aboveFunc.size() - 1; i-- > 0; )
                        bases.push_back(aboveFunc[i]);
                    PycRef<ASTNode> call = new ASTCall(function, pparamList, kwparamList);
                    stack.push(new ASTClass(call, new ASTTuple(bases), name, classKwMap));
                    stack_hist.pop();
                    break;
                }
                else
                {
                    stack = stack_hist.top();
                    stack_hist.pop();
                }

                /*
                KW_NAMES(i)
                    Stores a reference to co_consts[consti] into an internal variable for use by CALL.
                    co_consts[consti] must be a tuple of strings.
                    New in version 3.11.
                */
                if (mod->verCompare(3, 11) >= 0) {
                    PycRef<ASTNode> object_or_map = stack.top();
                    if (object_or_map.type() == ASTNode::NODE_KW_NAMES_MAP) {
                        stack.pop();
                        PycRef<ASTKwNamesMap> kwparams_map = object_or_map.cast<ASTKwNamesMap>();
                        for (ASTKwNamesMap::map_t::const_iterator it = kwparams_map->values().begin(); it != kwparams_map->values().end(); it++) {
                            kwparamList.push_front(std::make_pair(it->first, it->second));
                            pparams -= 1;
                        }
                    }
                }
                else {
                    for (int i = 0; i < kwparams; i++) {
                        PycRef<ASTNode> val = stack.top();
                        stack.pop();
                        PycRef<ASTNode> key = stack.top();
                        stack.pop();
                        kwparamList.push_front(std::make_pair(key, val));
                    }
                }
                for (int i=0; i<pparams; i++) {
                    PycRef<ASTNode> param = stack.top();
                    stack.pop();
                    if (param.type() == ASTNode::NODE_FUNCTION) {
                        PycRef<ASTNode> fun_code = param.cast<ASTFunction>()->code();
                        PycRef<PycCode> code_src = fun_code.cast<ASTObject>()->object().cast<PycCode>();
                        PycRef<PycString> function_name = code_src->name();
                        if (function_name->isEqual("<lambda>")) {
                            pparamList.push_front(param);
                        } else {
                            // Decorator used
                            PycRef<ASTNode> decor_name = new ASTName(function_name);
                            curblock->append(new ASTStore(param, decor_name));

                            pparamList.push_front(decor_name);
                        }
                    } else {
                        pparamList.push_front(param);
                    }
                }
                PycRef<ASTNode> func = stack.top();
                stack.pop();

                if ((opcode == Pyc::CALL_A || opcode == Pyc::INSTRUMENTED_CALL_A)
                        && pparams == 0 && kwparams == 0
                        && !stack.empty() && stack.top() != nullptr) {
                    PycRef<ASTNode> decorated = func;
                    bool is_decorator = false;
                    if (decorated.type() == ASTNode::NODE_FUNCTION) {
                        PycRef<PycCode> code_src = decorated.cast<ASTFunction>()
                                ->code().cast<ASTObject>()->object().cast<PycCode>();
                        if (!code_src->name()->isEqual("<lambda>")) {
                            PycRef<ASTNode> decor_name = new ASTName(code_src->name());
                            curblock->append(new ASTStore(decorated, decor_name));
                            decorated = decor_name;
                            is_decorator = true;
                        }
                    } else if (decorated.type() == ASTNode::NODE_CLASS) {
                        PycRef<ASTNode> nm = decorated.cast<ASTClass>()->name();
                        PycRef<PycString> cn = (nm != nullptr
                                && nm.type() == ASTNode::NODE_OBJECT)
                            ? nm.cast<ASTObject>()->object().try_cast<PycString>()
                            : nullptr;
                        if (cn != nullptr) {
                            PycRef<ASTNode> decor_name = new ASTName(cn);
                            curblock->append(new ASTStore(decorated, decor_name));
                            decorated = decor_name;
                            is_decorator = true;
                        }
                    } else if (decorated.type() == ASTNode::NODE_CALL) {
                        is_decorator = true;
                    }
                    if (is_decorator) {
                        PycRef<ASTNode> decorator = stack.top();
                        stack.pop();
                        ASTCall::pparam_t decParams;
                        decParams.push_front(decorated);
                        stack.push(new ASTCall(decorator, decParams,
                                               ASTCall::kwparam_t()));
                        break;
                    }
                }

                if ((opcode == Pyc::CALL_A || opcode == Pyc::INSTRUMENTED_CALL_A) &&
                        stack.top() == nullptr) {
                    stack.pop();
                }

                stack.push(new ASTCall(func, pparamList, kwparamList));
            }
            break;
        case Pyc::CALL_FUNCTION_VAR_A:
            {
                PycRef<ASTNode> var = stack.top();
                stack.pop();
                int kwparams = (operand & 0xFF00) >> 8;
                int pparams = (operand & 0xFF);
                ASTCall::kwparam_t kwparamList;
                ASTCall::pparam_t pparamList;
                for (int i=0; i<kwparams; i++) {
                    PycRef<ASTNode> val = stack.top();
                    stack.pop();
                    PycRef<ASTNode> key = stack.top();
                    stack.pop();
                    kwparamList.push_front(std::make_pair(key, val));
                }
                for (int i=0; i<pparams; i++) {
                    pparamList.push_front(stack.top());
                    stack.pop();
                }
                PycRef<ASTNode> func = stack.top();
                stack.pop();

                PycRef<ASTNode> call = new ASTCall(func, pparamList, kwparamList);
                call.cast<ASTCall>()->setVar(var);
                stack.push(call);
            }
            break;
        case Pyc::CALL_FUNCTION_KW_A:
            {
                PycRef<ASTNode> kw = stack.top();
                stack.pop();
                int kwparams = (operand & 0xFF00) >> 8;
                int pparams = (operand & 0xFF);
                ASTCall::kwparam_t kwparamList;
                ASTCall::pparam_t pparamList;
                for (int i=0; i<kwparams; i++) {
                    PycRef<ASTNode> val = stack.top();
                    stack.pop();
                    PycRef<ASTNode> key = stack.top();
                    stack.pop();
                    kwparamList.push_front(std::make_pair(key, val));
                }
                for (int i=0; i<pparams; i++) {
                    pparamList.push_front(stack.top());
                    stack.pop();
                }
                PycRef<ASTNode> func = stack.top();
                stack.pop();

                PycRef<ASTNode> call = new ASTCall(func, pparamList, kwparamList);
                call.cast<ASTCall>()->setKW(kw);
                stack.push(call);
            }
            break;
        case Pyc::CALL_FUNCTION_VAR_KW_A:
            {
                PycRef<ASTNode> kw = stack.top();
                stack.pop();
                PycRef<ASTNode> var = stack.top();
                stack.pop();
                int kwparams = (operand & 0xFF00) >> 8;
                int pparams = (operand & 0xFF);
                ASTCall::kwparam_t kwparamList;
                ASTCall::pparam_t pparamList;
                for (int i=0; i<kwparams; i++) {
                    PycRef<ASTNode> val = stack.top();
                    stack.pop();
                    PycRef<ASTNode> key = stack.top();
                    stack.pop();
                    kwparamList.push_front(std::make_pair(key, val));
                }
                for (int i=0; i<pparams; i++) {
                    pparamList.push_front(stack.top());
                    stack.pop();
                }
                PycRef<ASTNode> func = stack.top();
                stack.pop();

                PycRef<ASTNode> call = new ASTCall(func, pparamList, kwparamList);
                call.cast<ASTCall>()->setKW(kw);
                call.cast<ASTCall>()->setVar(var);
                stack.push(call);
            }
            break;
        case Pyc::CALL_FUNCTION_EX_A:
            {
                PycRef<ASTNode> kw = nullptr;
                if (operand & 0x01) {
                    kw = stack.top();
                    stack.pop();
                }
                PycRef<ASTNode> var = stack.top();
                stack.pop();
                PycRef<ASTNode> func = stack.top();
                stack.pop();
                if (!stack.empty() && stack.top() == nullptr)
                    stack.pop();
                /* `class X(*bases):` — dynamic/star-unpacked bases compile the
                   __build_class__ call through CALL_FUNCTION_EX: the args
                   (classbody fn, name, *bases) are assembled as a list
                   (BUILD_LIST [fn, name]; LIST_EXTEND bases; LIST_TO_TUPLE) and
                   star-called. The fixed-arg build_class path (CALL) never sees it,
                   leaving a raw NODE_LOADBUILDCLASS. Recover the class: fn = var[0],
                   name = var[1], bases = var[2:] (a `*bases` spread renders via its
                   ASTUnary UN_STAR element). */
                if (func != nullptr && func.type() == ASTNode::NODE_LOADBUILDCLASS
                        && kw == nullptr && var != nullptr
                        && var.type() == ASTNode::NODE_TUPLE) {
                    const ASTTuple::value_t& tv = var.cast<ASTTuple>()->values();
                    if (tv.size() >= 2 && tv[0] != nullptr
                            && tv[0].type() == ASTNode::NODE_FUNCTION) {
                        ASTTuple::value_t bases(tv.begin() + 2, tv.end());
                        PycRef<ASTNode> classcall = new ASTCall(tv[0],
                                ASTCall::pparam_t(), ASTCall::kwparam_t());
                        stack.push(new ASTClass(classcall,
                                new ASTTuple(bases), tv[1]));
                        break;
                    }
                }
                if (func == nullptr) {
                    func = var;
                    var = nullptr;
                }
                PycRef<ASTCall> call = new ASTCall(func, ASTCall::pparam_t(),
                                                   ASTCall::kwparam_t());
                if (var != nullptr)
                    call->setVar(var);
                if (kw != nullptr)
                    call->setKW(kw);
                stack.push(call.cast<ASTNode>());
            }
            break;
        case Pyc::CALL_METHOD_A:
            {
                ASTCall::pparam_t pparamList;
                for (int i = 0; i < operand; i++) {
                    PycRef<ASTNode> param = stack.top();
                    stack.pop();
                    if (param.type() == ASTNode::NODE_FUNCTION) {
                        PycRef<ASTNode> fun_code = param.cast<ASTFunction>()->code();
                        PycRef<PycCode> code_src = fun_code.cast<ASTObject>()->object().cast<PycCode>();
                        PycRef<PycString> function_name = code_src->name();
                        if (function_name->isEqual("<lambda>")) {
                            pparamList.push_front(param);
                        } else {
                            // Decorator used
                            PycRef<ASTNode> decor_name = new ASTName(function_name);
                            curblock->append(new ASTStore(param, decor_name));

                            pparamList.push_front(decor_name);
                        }
                    } else {
                        pparamList.push_front(param);
                    }
                }
                PycRef<ASTNode> func = stack.top();
                stack.pop();
                stack.push(new ASTCall(func, pparamList, ASTCall::kwparam_t()));
            }
            break;
        case Pyc::CONTINUE_LOOP_A:
            curblock->append(new ASTKeyword(ASTKeyword::KW_CONTINUE));
            break;
        case Pyc::COMPARE_OP_A:
            {
                auto oci = matchValueOr.find(curpos);
                if (oci != matchValueOr.end()) {
                    const OrCont& oc = oci->second;
                    PycRef<ASTNode> pattern = stack.top(); stack.pop();
                    stack.pop();
                    if (curblock->blktype() == ASTBlock::BLK_CASE) {
                        PycRef<ASTNode> prev = curblock.cast<ASTCaseBlock>()->pattern();
                        curblock.cast<ASTCaseBlock>()->setPattern(
                                new ASTBinary(prev, pattern, ASTBinary::BIN_OR));
                    }
                    int dst = (oc.orNextAlt >= 0) ? oc.orNextAlt : oc.bodyStart;
                    source.setPos(dst);
                    pos = dst;
                    while (next_exception_entry < exception_entries.size()
                            && exception_entries[next_exception_entry].start_offset < pos)
                        next_exception_entry++;
                    break;
                }
                auto vmi = matchValue.find(curpos);
                if (vmi != matchValue.end()) {
                    const VCase& vc = vmi->second;
                    PycRef<ASTNode> pattern = stack.top(); stack.pop();
                    if (vc.isLast) {
                        stack.pop();
                    } else {
                        PycRef<ASTNode> copy = stack.top(); stack.pop();
                        if (vc.isFirst) {
                            PycRef<ASTNode> subject = stack.top();
                            blocks.push(new ASTMatchBlock(vc.matchEnd, subject));
                            curblock = blocks.top();
                        }
                    }
                    blocks.push(new ASTCaseBlock(vc.failTarget, pattern));
                    curblock = blocks.top();
                    curblock->init();
                    int dst = (vc.orNextAlt >= 0) ? vc.orNextAlt : vc.bodyStart;
                    source.setPos(dst);
                    pos = dst;
                    while (next_exception_entry < exception_entries.size()
                            && exception_entries[next_exception_entry].start_offset < pos)
                        next_exception_entry++;
                    break;
                }
                PycRef<ASTNode> right = stack.top();
                stack.pop();
                PycRef<ASTNode> left = stack.top();
                stack.pop();
                auto arg = operand;
                if (mod->verCompare(3, 12) == 0)
                    arg >>= 4; // changed under GH-100923
                else if (mod->verCompare(3, 13) >= 0)
                    arg >>= 5;
                if (left != nullptr && left.type() == ASTNode::NODE_CHAINCOMPARE) {
                    left.cast<ASTChainCompare>()->extend(arg, right);
                    stack.push(left);
                } else if (chainCmp) {
                    stack.push(new ASTChainCompare(left, right, arg));
                } else {
                    stack.push(new ASTCompare(left, right, arg));
                }
                chainCmp = 0;
            }
            break;
        case Pyc::CONTAINS_OP_A:
            handleIsContainsOp(opcode, operand);
            break;
        case Pyc::DELETE_ATTR_A:
        case Pyc::DELETE_GLOBAL_A:
        case Pyc::DELETE_NAME_A:
        case Pyc::DELETE_FAST_A:
        case Pyc::DELETE_SLICE_0:
        case Pyc::DELETE_SLICE_1:
        case Pyc::DELETE_SLICE_2:
        case Pyc::DELETE_SUBSCR:
        case Pyc::DELETE_SLICE_3:
            handleDelete(opcode, operand);
            break;
        case Pyc::DUP_TOP:
        case Pyc::DUP_TOP_TWO:
        case Pyc::DUP_TOPX_A:
            handleStackManip(opcode, operand);
            break;
        case Pyc::END_FINALLY:
            {
                bool isFinally = false;
                if (curblock->blktype() == ASTBlock::BLK_FINALLY) {
                    PycRef<ASTBlock> final = curblock;
                    blocks.pop();

                    stack = stack_hist.top();
                    stack_hist.pop();

                    curblock = blocks.top();
                    curblock->append(final.cast<ASTNode>());
                    isFinally = true;
                } else if (curblock->blktype() == ASTBlock::BLK_EXCEPT) {
                    blocks.pop();
                    PycRef<ASTBlock> prev = curblock;

                    bool isUninitAsyncFor = false;
                    if (blocks.top()->blktype() == ASTBlock::BLK_CONTAINER) {
                        auto container = blocks.top();
                        blocks.pop();
                        auto asyncForBlock = blocks.top();
                        isUninitAsyncFor = asyncForBlock->blktype() == ASTBlock::BLK_ASYNCFOR && !asyncForBlock->inited();
                        if (isUninitAsyncFor) {
                            auto tryBlock = container->nodes().front().cast<ASTBlock>();
                            if (!tryBlock->nodes().empty() && tryBlock->blktype() == ASTBlock::BLK_TRY) {
                                auto store = tryBlock->nodes().front().try_cast<ASTStore>();
                                if (store) {
                                    asyncForBlock.cast<ASTIterBlock>()->setIndex(store->dest());
                                }
                            }
                            curblock = blocks.top();
                            stack = stack_hist.top();
                            stack_hist.pop();
                            if (!curblock->inited())
                                fprintf(stderr, "Error when decompiling 'async for'.\n");
                        } else {
                            blocks.push(container);
                        }
                    }

                    if (!isUninitAsyncFor) {
                        if (curblock->size() != 0) {
                            blocks.top()->append(curblock.cast<ASTNode>());
                        }

                        curblock = blocks.top();

                        /* Turn it into an else statement. */
                        if (curblock->end() != pos || curblock.cast<ASTContainerBlock>()->hasFinally()) {
                            PycRef<ASTBlock> elseblk = new ASTBlock(ASTBlock::BLK_ELSE, prev->end());
                            elseblk->init();
                            blocks.push(elseblk);
                            curblock = blocks.top();
                        }
                        else {
                            stack = stack_hist.top();
                            stack_hist.pop();
                        }
                    }
                }

                if (curblock->blktype() == ASTBlock::BLK_CONTAINER) {
                    /* This marks the end of the except block(s). */
                    PycRef<ASTContainerBlock> cont = curblock.cast<ASTContainerBlock>();
                    if (!cont->hasFinally() || isFinally) {
                        /* If there's no finally block, pop the container. */
                        blocks.pop();
                        curblock = blocks.top();
                        curblock->append(cont.cast<ASTNode>());
                    }
                }
            }
            break;
        case Pyc::EXEC_STMT:
            {
                if (stack.top().type() == ASTNode::NODE_CHAINSTORE) {
                    stack.pop();
                }
                PycRef<ASTNode> loc = stack.top();
                stack.pop();
                PycRef<ASTNode> glob = stack.top();
                stack.pop();
                PycRef<ASTNode> stmt = stack.top();
                stack.pop();

                curblock->append(new ASTExec(stmt, glob, loc));
            }
            break;
        case Pyc::FOR_ITER_A:
        case Pyc::INSTRUMENTED_FOR_ITER_A:
        case Pyc::FOR_LOOP_A:
            handleForIter(opcode, operand);
            break;
        case Pyc::GET_AITER:
            {
                // Logic similar to FOR_ITER_A
                PycRef<ASTNode> iter = stack.top(); // Iterable
                stack.pop();

                PycRef<ASTBlock> top = blocks.top();
                auto afi = asyncForInfo.find(curpos);
                if (afi != asyncForInfo.end()) {
                    const AsyncForRec& r = afi->second;
                    PycRef<ASTIterBlock> forblk = new ASTIterBlock(
                            ASTBlock::BLK_ASYNCFOR, r.loopTop, r.loopEnd, iter);
                    blocks.push(forblk.cast<ASTBlock>());
                    curblock = blocks.top();
                    stack.push(nullptr);
                    source.setPos(r.storeAt);
                    pos = r.storeAt;
                } else if (top->blktype() == ASTBlock::BLK_WHILE) {
                    blocks.pop();
                    PycRef<ASTIterBlock> forblk = new ASTIterBlock(ASTBlock::BLK_ASYNCFOR, curpos, top->end(), iter);
                    blocks.push(forblk.cast<ASTBlock>());
                    curblock = blocks.top();
                    stack.push(nullptr);
                } else {
                     fprintf(stderr, "Unsupported use of GET_AITER outside of SETUP_LOOP\n");
                }
            }
            break;
        case Pyc::GET_ANEXT:
            break;
        case Pyc::END_ASYNC_FOR:
            break;
        case Pyc::FORMAT_VALUE_A:
        case Pyc::GET_AWAITABLE:
        case Pyc::GET_AWAITABLE_A:
            handleExprWrap(opcode, operand);
            break;
        case Pyc::SEND_A:
            if (!handleSend())
                return new ASTNodeList(defblock->nodes());
            break;
        case Pyc::GET_ITER:
        case Pyc::GET_YIELD_FROM_ITER:
            /* We just entirely ignore this */
            break;
        case Pyc::IMPORT_NAME_A:
        case Pyc::IMPORT_FROM_A:
        case Pyc::IMPORT_STAR:
            handleImport(opcode, operand);
            break;
        case Pyc::IS_OP_A:
            handleIsContainsOp(opcode, operand);
            break;
        case Pyc::POP_JUMP_BACKWARD_IF_FALSE_A:
        case Pyc::POP_JUMP_BACKWARD_IF_TRUE_A:
        case Pyc::POP_JUMP_BACKWARD_IF_NONE_A:
        case Pyc::POP_JUMP_BACKWARD_IF_NOT_NONE_A:
            if (whileBottomSkip.count(curpos)) {
                if (!stack.empty())
                    stack.pop();
                break;
            }
            while (blocks.size() > 1
                    && curblock->end() > 0 && curblock->end() <= curpos
                    && curblock->blktype() != ASTBlock::BLK_MAIN
                    && !(curblock->blktype() == ASTBlock::BLK_FOR
                         && curblock.cast<ASTIterBlock>()->isComprehension())) {
                bool isLoop = curblock->blktype() == ASTBlock::BLK_WHILE
                        || curblock->blktype() == ASTBlock::BLK_FOR;
                PycRef<ASTBlock> b = curblock;
                if (!isLoop && !stack_hist.empty())
                    stack_hist.pop();
                blocks.pop();
                curblock = blocks.top();
                curblock->append(b.cast<ASTNode>());
            }
            if (curblock->blktype() == ASTBlock::BLK_FOR
                    && curblock.cast<ASTIterBlock>()->isComprehension()) {
                PycRef<ASTNode> cond = stack.top();
                stack.pop();
                if (opcode == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A)
                    cond = new ASTUnary(cond, ASTUnary::UN_NOT);
                else if (opcode == Pyc::POP_JUMP_BACKWARD_IF_NONE_A)
                    cond = new ASTCompare(cond, new ASTObject(Pyc_None), ASTCompare::CMP_IS_NOT);
                else if (opcode == Pyc::POP_JUMP_BACKWARD_IF_NOT_NONE_A)
                    cond = new ASTCompare(cond, new ASTObject(Pyc_None), ASTCompare::CMP_IS);
                PycRef<ASTNode> existing = curblock.cast<ASTIterBlock>()->condition();
                if (existing != nullptr) {
                    if (compPureOr) {
                        cond = new ASTBinary(existing, cond, ASTBinary::BIN_LOG_OR);
                    } else if (compFilterFwd) {
                        cleanBuild = false;
                        return new ASTNodeList(defblock->nodes());
                    } else {
                        cond = new ASTBinary(existing, cond, ASTBinary::BIN_LOG_AND);
                    }
                }
                curblock.cast<ASTIterBlock>()->setCondition(cond);
            } else if (curblock->blktype() == ASTBlock::BLK_IF
                    || curblock->blktype() == ASTBlock::BLK_ELIF) {
                bool wasElif = curblock->blktype() == ASTBlock::BLK_ELIF;
                if (stack.empty() || blocks.size() < 2)
                    throw std::runtime_error("while-conversion stack/block underflow");
                stack.pop();
                PycRef<ASTCondBlock> ifb = curblock.cast<ASTCondBlock>();
                PycRef<ASTCondBlock> whb = new ASTCondBlock(
                        ASTBlock::BLK_WHILE, ifb->end(), ifb->cond(), ifb->negative());
                for (const auto& n : ifb->nodes())
                    whb->append(n);
                blocks.pop();
                if (stack_hist.size())
                    stack_hist.pop();
                curblock = blocks.top();
                if (compoundAndLeadGuard.count(whb->end())
                        && whb->cond() != nullptr
                        && curblock->blktype() == ASTBlock::BLK_IF
                        && curblock->size() == 0
                        && curblock.cast<ASTCondBlock>()->cond() != nullptr
                        && curblock->end() >= whb->end()
                        && blocks.size() >= 2) {
                    PycRef<ASTCondBlock> outerIf = curblock.cast<ASTCondBlock>();
                    PycRef<ASTNode> aCond = outerIf->negative()
                            ? NegateCond(outerIf->cond()) : outerIf->cond();
                    PycRef<ASTNode> merged = new ASTBinary(
                            aCond, whb->cond(), ASTBinary::BIN_LOG_AND);
                    PycRef<ASTCondBlock> whb2 = new ASTCondBlock(
                            ASTBlock::BLK_WHILE, whb->end(), merged, false);
                    for (const auto& n : whb->nodes())
                        whb2->append(n);
                    whb = whb2;
                    blocks.pop();
                    if (stack_hist.size())
                        stack_hist.pop();
                    curblock = blocks.top();
                }
                if (wasElif) {
                    PycRef<ASTBlock> elsew = new ASTBlock(ASTBlock::BLK_ELSE, whb->end());
                    elsew->init();
                    elsew->append(whb.cast<ASTNode>());
                    curblock->append(elsew.cast<ASTNode>());
                } else {
                    curblock->append(whb.cast<ASTNode>());
                }
                if (forElseMerge.count(whb->end())) {
                    stack_hist.push(stack);
                    PycRef<ASTBlock> elseblk = new ASTBlock(
                            ASTBlock::BLK_ELSE, forElseMerge[whb->end()]);
                    elseblk->init();
                    loopElseBlocks.insert((ASTBlock *)elseblk);
                    blocks.push(elseblk);
                    curblock = blocks.top();
                }
                if (chainWhileBottomExit.count(curpos)) {
                    int e = chainWhileBottomExit[curpos];
                    if (chainWhileBwdEnd.count(curpos))
                        e = whb->end();
                    source.setPos(e);
                    pos = e;
                }
            } else {
                fprintf(stderr, "Unsupported opcode: %s (%d)\n",
                        Pyc::OpcodeName(opcode), opcode);
                cleanBuild = false;
                return new ASTNodeList(defblock->nodes());
            }
            break;
        case Pyc::JUMP_IF_FALSE_A:
        case Pyc::JUMP_IF_TRUE_A:
        case Pyc::JUMP_IF_FALSE_OR_POP_A:
        case Pyc::JUMP_IF_TRUE_OR_POP_A:
        case Pyc::POP_JUMP_IF_FALSE_A:
        case Pyc::POP_JUMP_IF_TRUE_A:
        case Pyc::POP_JUMP_FORWARD_IF_FALSE_A:
        case Pyc::POP_JUMP_FORWARD_IF_TRUE_A:
        case Pyc::POP_JUMP_FORWARD_IF_NONE_A:
        case Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A:
        case Pyc::INSTRUMENTED_POP_JUMP_IF_FALSE_A:
        case Pyc::INSTRUMENTED_POP_JUMP_IF_TRUE_A:
            {
                while (blocks.size() > 1
                        && (curblock->blktype() == ASTBlock::BLK_IF
                            || curblock->blktype() == ASTBlock::BLK_ELIF
                            || curblock->blktype() == ASTBlock::BLK_ELSE)
                        && curblock->end() > 0 && curblock->end() < curpos) {
                    PycRef<ASTBlock> sb = curblock;
                    if (!stack_hist.empty())
                        stack_hist.pop();
                    blocks.pop();
                    curblock = blocks.top();
                    curblock->append(sb.cast<ASTNode>());
                }
                if (whileTopTestPendingExit >= 0
                        && curblock->blktype() == ASTBlock::BLK_WHILE
                        && curblock->size() == 0
                        && !stack.empty()) {
                    int t = operand;
                    if (mod->verCompare(3, 10) >= 0)
                        t *= sizeof(uint16_t);
                    t += pos;
                    if (t == whileTopTestPendingExit) {
                        PycRef<ASTNode> wcond = stack.top();
                        stack.pop();
                        if (opcode == Pyc::POP_JUMP_FORWARD_IF_TRUE_A)
                            wcond = NegateCond(wcond);
                        else if (opcode == Pyc::POP_JUMP_FORWARD_IF_NONE_A)
                            wcond = new ASTCompare(wcond, new ASTObject(Pyc_None),
                                                   ASTCompare::CMP_IS_NOT);
                        else if (opcode == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A)
                            wcond = new ASTCompare(wcond, new ASTObject(Pyc_None),
                                                   ASTCompare::CMP_IS);
                        curblock.cast<ASTCondBlock>()->setCondition(wcond);
                        whileTopTestPendingExit = -1;
                        break;
                    }
                }
                if (whileBottomSkip.count(curpos)) {
                    if (!stack.empty())
                        stack.pop();
                    break;
                }
                if (chainLeadSkip.count(curpos)) {
                    if (!stack.empty())
                        stack.pop();
                    break;
                }
                if (opcode == Pyc::JUMP_IF_FALSE_OR_POP_A
                        && !stack.empty() && stack.top() != nullptr
                        && stack.top().type() == ASTNode::NODE_CHAINCOMPARE
                        && !fwdJumpTargets.count(curpos))
                    break;
                if (chainIfGlue.count(curpos))
                    break;

                if (orReconStep.count(curpos)) {
                    const OrReconStep& st = orReconStep[curpos];
                    if (!stack.empty()) {
                        PycRef<ASTNode> oper = stack.top();
                        stack.pop();
                        if (opcode == Pyc::POP_JUMP_FORWARD_IF_NONE_A)
                            oper = new ASTCompare(oper, new ASTObject(Pyc_None),
                                                  ASTCompare::CMP_IS_NOT);
                        else if (opcode == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A)
                            oper = new ASTCompare(oper, new ASTObject(Pyc_None),
                                                  ASTCompare::CMP_IS);
                        if (st.neg)
                            oper = NegateCond(oper);
                        orReconAcc.push_back(oper);
                    }
                    if (!st.isFinal)
                        break;
                    ASTBinary::BinOp inOp = st.andOfOrs ? ASTBinary::BIN_LOG_OR
                                                        : ASTBinary::BIN_LOG_AND;
                    ASTBinary::BinOp outOp = st.andOfOrs ? ASTBinary::BIN_LOG_AND
                                                         : ASTBinary::BIN_LOG_OR;
                    PycRef<ASTNode> condNode;
                    size_t idx = 0;
                    for (size_t gi = 0; gi < st.groupSizes.size(); ++gi) {
                        PycRef<ASTNode> grp;
                        for (int k = 0; k < st.groupSizes[gi]
                                        && idx < orReconAcc.size(); ++k) {
                            PycRef<ASTNode> oper = orReconAcc[idx++];
                            grp = (grp == NULL) ? oper
                                    : new ASTBinary(grp, oper, inOp);
                        }
                        condNode = (condNode == NULL) ? grp
                                : new ASTBinary(condNode, grp, outOp);
                    }
                    orReconAcc.clear();
                    if (condNode != NULL)
                        stack.push(condNode);
                }

                if (opcode == Pyc::JUMP_IF_FALSE_OR_POP_A
                        || opcode == Pyc::JUMP_IF_TRUE_OR_POP_A) {
                    int t = operand;
                    if (mod->verCompare(3, 10) >= 0)
                        t *= sizeof(uint16_t);
                    if (mod->verCompare(3, 11) >= 0)
                        t += pos;
                    if ((curblock->blktype() == ASTBlock::BLK_IF
                            || curblock->blktype() == ASTBlock::BLK_ELIF)
                            && t > curblock->end()) {
                        PycRef<ASTCondBlock> fakeIf = curblock.cast<ASTCondBlock>();
                        bool outerOr = (opcode == Pyc::JUMP_IF_TRUE_OR_POP_A);
                        if (curblock->size() == 0
                                && fakeIf->cond() != nullptr
                                && !stack.empty()
                                && blocks.size() >= 2
                                && outerOr == !fakeIf->negative()) {
                            PycRef<ASTNode> bOperand = stack.top();
                            stack.pop();
                            blocks.pop();
                            if (!stack_hist.empty())
                                stack_hist.pop();
                            curblock = blocks.top();
                            PycRef<ASTNode> innerNode = new ASTBinary(
                                    fakeIf->cond(), bOperand,
                                    fakeIf->negative() ? ASTBinary::BIN_LOG_OR
                                                       : ASTBinary::BIN_LOG_AND);
                            boolPending.push_back({ innerNode, outerOr, t, curpos });
                            break;
                        }
                        cleanBuild = false;
                        return new ASTNodeList(defblock->nodes());
                    }
                    PycRef<ASTNode> left = stack.top();
                    stack.pop();
                    boolPending.push_back({ left,
                            opcode == Pyc::JUMP_IF_TRUE_OR_POP_A, t, curpos });
                    break;
                }

                PycRef<ASTNode> cond = stack.top();
                PycRef<ASTCondBlock> ifblk;
                int popped = ASTCondBlock::UNINITED;

                if (opcode == Pyc::POP_JUMP_IF_FALSE_A
                        || opcode == Pyc::POP_JUMP_IF_TRUE_A
                        || opcode == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                        || opcode == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                        || opcode == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                        || opcode == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A
                        || opcode == Pyc::INSTRUMENTED_POP_JUMP_IF_FALSE_A
                        || opcode == Pyc::INSTRUMENTED_POP_JUMP_IF_TRUE_A) {
                    /* Pop condition before the jump */
                    stack.pop();
                    popped = ASTCondBlock::PRE_POPPED;
                }

                if (opcode == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                        || opcode == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A) {
                    bool jump_if_none = opcode == Pyc::POP_JUMP_FORWARD_IF_NONE_A;
                    int noneOp = jump_if_none ? ASTCompare::CMP_IS_NOT : ASTCompare::CMP_IS;
                    if (cond != nullptr && cond.type() == ASTNode::NODE_CHAINCOMPARE
                            && chainIfNoneFinal.count(curpos)) {
                        cond.cast<ASTChainCompare>()->extend(noneOp,
                                new ASTObject(Pyc_None));
                    } else {
                        cond = new ASTCompare(cond, new ASTObject(Pyc_None), noneOp);
                    }
                }

                /* Store the current stack for the else statement(s) */
                stack_hist.push(stack);

                if (opcode == Pyc::JUMP_IF_FALSE_OR_POP_A
                        || opcode == Pyc::JUMP_IF_TRUE_OR_POP_A) {
                    /* Pop condition only if condition is met */
                    stack.pop();
                    popped = ASTCondBlock::POPPED;
                }

                bool neg = opcode == Pyc::JUMP_IF_TRUE_A
                        || opcode == Pyc::JUMP_IF_TRUE_OR_POP_A
                        || opcode == Pyc::POP_JUMP_IF_TRUE_A
                        || opcode == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                        || opcode == Pyc::INSTRUMENTED_POP_JUMP_IF_TRUE_A;
                /* tail-negated OR-of-ANDs final operand: condNode assembled above is
                   already the correct positive condition (the `not D` conjunct was
                   folded in by orReconStep), so don't re-negate on the IF_TRUE opcode.
                   The IF_TRUE->exit target still correctly makes the block skip its
                   body on the jump (a positive `if cond:` open). */
                if (orReconFinalNoNeg.count(curpos))
                    neg = false;

                int offs = operand;
                if (mod->verCompare(3, 10) >= 0)
                    offs *= sizeof(uint16_t); // // BPO-27129
                if (mod->verCompare(3, 12) >= 0
                        || opcode == Pyc::JUMP_IF_FALSE_A
                        || opcode == Pyc::JUMP_IF_TRUE_A
                        || opcode == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                        || opcode == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                        || opcode == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                        || opcode == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A
                        || (mod->verCompare(3, 11) >= 0
                            && (opcode == Pyc::JUMP_IF_FALSE_OR_POP_A
                                || opcode == Pyc::JUMP_IF_TRUE_OR_POP_A))) {
                    /* Offset is relative in these cases */
                    offs += pos;
                }

                bool elseNestedIfContinues = false;
                if (curblock->blktype() == ASTBlock::BLK_ELSE
                        && curblock->size() == 0
                        && offs > pos && offs < curblock->end()) {
                    PycBuffer sb(code->code()->value(), code->code()->length());
                    sb.setPos(pos);
                    int so, sa, sp = pos;
                    while (sp < offs && !sb.atEof()) {
                        int ioff = sp;
                        bc_next(sb, mod, so, sa, sp);
                        int tgt = -1;
                        if (so == Pyc::FOR_ITER_A)
                            tgt = sp + sa * (int)sizeof(uint16_t);
                        if (tgt == offs) { elseNestedIfContinues = true; break; }
                        if (sp <= ioff) break;
                    }
                }
                /* The else body AFTER the nested `if` (whose false-target is offs)
                   reaches a loop `continue` (a JUMP_BACKWARD to an enclosing loop
                   header) before any further conditional test: this is
                   `else: if C: …; continue`, NOT an `elif C:`. Collapsing it to
                   `elif` drops the trailing continue and — because the merge sits
                   PAST that back-edge — mis-routes the enclosing loop's tail
                   back-edge to the function level (a stray `continue` →
                   "'continue' not properly in loop"). Scan [offs, elseEnd): skip
                   straight-line ops; if the first BRANCH op is a JUMP_BACKWARD to a
                   loopRange start → keep the BLK_ELSE; a real `elif` instead hits a
                   forward conditional test first (the next arm) or has the nested
                   if's true-arm JUMP over the else to the merge. */
                if (!elseNestedIfContinues
                        && curblock->blktype() == ASTBlock::BLK_ELSE
                        && curblock->size() == 0
                        && offs > pos && offs < curblock->end()) {
                    PycBuffer cb2(code->code()->value(), code->code()->length());
                    cb2.setPos(offs);
                    int co2, ca2, cp2 = offs;
                    while (cp2 < curblock->end() && !cb2.atEof()) {
                        int ioff = cp2;
                        bc_next(cb2, mod, co2, ca2, cp2);
                        if (co2 == Pyc::JUMP_BACKWARD_A
                                || co2 == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A) {
                            int jt = cp2 - ca2 * (int)sizeof(uint16_t);
                            for (const auto& lr : loopRanges)
                                if (lr.start == jt) { elseNestedIfContinues = true; break; }
                            break;
                        }
                        /* a forward conditional / loop test = the next elif arm or a
                           nested structure: a genuine elif chain continuation. */
                        if (co2 == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                                || co2 == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                                || co2 == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                                || co2 == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A
                                || co2 == Pyc::FOR_ITER_A
                                || co2 == Pyc::JUMP_FORWARD_A
                                || co2 == Pyc::RETURN_VALUE
                                || co2 == Pyc::RETURN_CONST_A
                                || co2 == Pyc::RAISE_VARARGS_A
                                || co2 == Pyc::PUSH_EXC_INFO)
                            break;
                        if (cp2 <= ioff) break;
                    }
                }

                bool ternaryElseArm = false;
                if (curblock->blktype() == ASTBlock::BLK_ELSE
                        && curblock->size() == 0
                        && !stack.empty()
                        && offs > pos && offs < curblock->end()
                        && !backwardJumpOffsets.count(offs)) {
                    for (const auto& j : fwdJumps) {
                        if (j.first > pos && j.first < offs
                                && j.second == curblock->end()) {
                            ternaryElseArm = true;
                            break;
                        }
                    }
                    if (ternaryElseArm) {
                        PycBuffer tb(code->code()->value(), code->code()->length());
                        tb.setPos(pos);
                        int to, ta, tp = pos;
                        while (tp < offs && !tb.atEof()) {
                            int ioff = tp;
                            bc_next(tb, mod, to, ta, tp);
                            if (to == Pyc::STORE_FAST_A || to == Pyc::STORE_NAME_A
                                    || to == Pyc::STORE_GLOBAL_A
                                    || to == Pyc::STORE_DEREF_A
                                    || to == Pyc::STORE_ATTR_A
                                    || to == Pyc::STORE_SUBSCR
                                    || to == Pyc::DELETE_FAST_A
                                    || to == Pyc::POP_TOP
                                    || to == Pyc::RETURN_VALUE
                                    || to == Pyc::RAISE_VARARGS_A
                                    || to == Pyc::RERAISE_A
                                    || to == Pyc::IMPORT_NAME_A) {
                                ternaryElseArm = false;
                                break;
                            }
                            if (tp <= ioff) break;
                        }
                    }
                }

                bool ternaryAssignElse = false;
                if (curblock->blktype() == ASTBlock::BLK_ELSE
                        && curblock->size() == 0
                        && stack.empty()
                        && offs > pos && offs < curblock->end()
                        && !backwardJumpOffsets.count(offs)) {
                    int joinT = -1;
                    for (const auto& j : fwdJumps) {
                        if (j.first > pos && j.first < offs
                                && j.second > offs && j.second < curblock->end()) {
                            joinT = j.second;
                            break;
                        }
                    }
                    if (joinT > 0) {
                        bool pureArms = true;
                        PycBuffer tb(code->code()->value(), code->code()->length());
                        tb.setPos(pos);
                        int to, ta, tp = pos;
                        while (tp < joinT && !tb.atEof()) {
                            int ioff = tp;
                            bc_next(tb, mod, to, ta, tp);
                            if (to == Pyc::STORE_FAST_A || to == Pyc::STORE_NAME_A
                                    || to == Pyc::STORE_GLOBAL_A
                                    || to == Pyc::STORE_DEREF_A
                                    || to == Pyc::STORE_ATTR_A
                                    || to == Pyc::STORE_SUBSCR
                                    || to == Pyc::DELETE_FAST_A
                                    || to == Pyc::POP_TOP
                                    || to == Pyc::RETURN_VALUE
                                    || to == Pyc::RAISE_VARARGS_A
                                    || to == Pyc::RERAISE_A
                                    || to == Pyc::IMPORT_NAME_A
                                    || to == Pyc::PUSH_EXC_INFO
                                    || to == Pyc::FOR_ITER_A
                                    || to == Pyc::BEFORE_WITH) {
                                pureArms = false;
                                break;
                            }
                            if (tp <= ioff) break;
                        }
                        ternaryAssignElse = pureArms;
                    }
                }

                bool compTernaryElem = false;
                if (curblock->blktype() == ASTBlock::BLK_FOR
                        && curblock.cast<ASTIterBlock>()->isComprehension()
                        && !compFilterFwd && offs > pos
                        && !backwardJumpOffsets.count(offs)) {
                    for (const auto& j : fwdJumps) {
                        if (j.first > curpos && j.first < offs && j.second > offs) {
                            compTernaryElem = true;
                            break;
                        }
                    }
                }

                if (cond.type() == ASTNode::NODE_COMPARE
                        && cond.cast<ASTCompare>()->op() == ASTCompare::CMP_EXCEPTION) {
                    int except_end = offs;
                    if (curblock->blktype() == ASTBlock::BLK_EXCEPT
                            && curblock.cast<ASTCondBlock>()->cond() == NULL) {
                        if (recoverThisExcept) {
                            PycBuffer pk(code->code()->value(), code->code()->length());
                            pk.setPos(pos);
                            if (!pk.atEof()) {
                                int po, pa, pp;
                                bc_next(pk, mod, po, pa, pp);
                                if (po != Pyc::POP_TOP)
                                    recoverThisExcept = false;
                            }
                        }
                        bool insideOpenSplitExcept = false;
                        for (int t : openExceptTargets)
                            if (offs < t) { insideOpenSplitExcept = true; break; }
                        bool crossesEnclosingTry = false;
                        if (curblock->end() > 0) {
                            std::stack<PycRef<ASTBlock> > sc = blocks;
                            while (!sc.empty()) {
                                PycRef<ASTBlock> b = sc.top(); sc.pop();
                                if (b->blktype() == ASTBlock::BLK_TRY
                                        && b->end() >= offs
                                        && b->end() < curblock->end()) {
                                    crossesEnclosingTry = true; break;
                                }
                            }
                        }
                        bool danglingIntoEnclosingTry = false;
                        if (curblock->end() == 0) {
                            std::stack<PycRef<ASTBlock> > sc = blocks;
                            while (!sc.empty()) {
                                PycRef<ASTBlock> b = sc.top(); sc.pop();
                                if (b->blktype() == ASTBlock::BLK_FOR
                                        || b->blktype() == ASTBlock::BLK_WHILE)
                                    break;
                                if (b->blktype() == ASTBlock::BLK_TRY && b->end() > offs
                                        && openFinallyTargets.count(b->end())) {
                                    danglingIntoEnclosingTry = true; break;
                                }
                            }
                        }
                        bool chainedBare = false;
                        {
                            PycBuffer pk(code->code()->value(), code->code()->length());
                            pk.setPos(offs);
                            if (!pk.atEof()) {
                                int po, pa, pp;
                                bc_next(pk, mod, po, pa, pp);
                                if (po == Pyc::POP_TOP) {
                                    int bestEnd = -1, bestSpan = 0x7fffffff;
                                    for (const auto& e : exception_entries) {
                                        if (e.start_offset <= offs && offs < e.end_offset
                                                && e.end_offset - e.start_offset < bestSpan) {
                                            bestSpan = e.end_offset - e.start_offset;
                                            bestEnd = e.end_offset;
                                        }
                                    }
                                    if (bestEnd > offs) {
                                        int outerRethrow = -1;
                                        for (const auto& e : exception_entries)
                                            if (e.start_offset <= offs && offs < e.end_offset
                                                    && e.end_offset == bestEnd) {
                                                outerRethrow = e.target; break;
                                            }
                                        if (outerRethrow > bestEnd) {
                                            bool nestedHandler = false;
                                            for (const auto& e : exception_entries)
                                                if (e.start_offset > offs
                                                        && e.start_offset < outerRethrow
                                                        && e.target > offs
                                                        && e.target < outerRethrow) {
                                                    nestedHandler = true; break;
                                                }
                                            if (nestedHandler) {
                                                int ext = -1;
                                                for (const auto& e : exception_entries)
                                                    if (e.target == outerRethrow
                                                            && e.start_offset > offs
                                                            && e.start_offset < outerRethrow
                                                            && e.start_offset > ext)
                                                        ext = e.start_offset;
                                                if (ext > bestEnd) bestEnd = ext;
                                            }
                                        }
                                    }
                                    bool fireBare = bestEnd > offs;
                                    int bareEnd = bestEnd;
                                    if (fireBare && curblock->end() != 0) {
                                        int lastOp = -1;
                                        PycBuffer rs(code->code()->value(),
                                                code->code()->length());
                                        rs.setPos(offs);
                                        int ro, ra, rp = offs;
                                        while (rp < bestEnd && !rs.atEof()) {
                                            int io = rp;
                                            bc_next(rs, mod, ro, ra, rp);
                                            if (ro != Pyc::CACHE && ro != Pyc::NOP)
                                                lastOp = ro;
                                            if (rp <= io) break;
                                        }
                                        fireBare = (lastOp == Pyc::RAISE_VARARGS_A
                                                || lastOp == Pyc::RERAISE
                                                || lastOp == Pyc::RERAISE_A);
                                        if (!fireBare) {
                                            int opBefore = -1;
                                            PycBuffer ps(code->code()->value(),
                                                    code->code()->length());
                                            int po2, pa2, pp2 = 0;
                                            while (pp2 < offs && !ps.atEof()) {
                                                int io = pp2;
                                                bc_next(ps, mod, po2, pa2, pp2);
                                                if (pp2 == offs) { opBefore = po2; break; }
                                                if (pp2 <= io) break;
                                            }
                                            bool prevTerminal = (opBefore == Pyc::RAISE_VARARGS_A
                                                    || opBefore == Pyc::RERAISE
                                                    || opBefore == Pyc::RERAISE_A
                                                    || opBefore == Pyc::RETURN_VALUE
                                                    || opBefore == Pyc::RETURN_CONST_A);
                                            if (prevTerminal) {
                                                PycBuffer xs(code->code()->value(),
                                                        code->code()->length());
                                                int xo, xa, xp = bestEnd;
                                                const int W = (int)sizeof(uint16_t);
                                                if (xp < (int)code->code()->length()) {
                                                    xs.setPos(bestEnd);
                                                    bc_next(xs, mod, xo, xa, xp);
                                                    if (xo == Pyc::POP_EXCEPT && !xs.atEof()) {
                                                        int jo, ja, jp = xp;
                                                        bc_next(xs, mod, jo, ja, jp);
                                                        if (jo == Pyc::JUMP_FORWARD_A) {
                                                            int merge = jp + ja * W;
                                                            bool cleanTail = merge > jp;
                                                            { int co, ca, cp = jp;
                                                              PycBuffer cs(code->code()->value(),
                                                                      code->code()->length());
                                                              cs.setPos(jp);
                                                              while (cp < merge && !cs.atEof()) {
                                                                  int io = cp;
                                                                  bc_next(cs, mod, co, ca, cp);
                                                                  if (co != Pyc::COPY_A && co != Pyc::POP_EXCEPT
                                                                          && co != Pyc::RERAISE && co != Pyc::RERAISE_A
                                                                          && co != Pyc::EXTENDED_ARG_A
                                                                          && co != Pyc::CACHE && co != Pyc::NOP) {
                                                                      cleanTail = false; break;
                                                                  }
                                                                  if (cp <= io) break;
                                                              } }
                                                            if (merge > bestEnd && cleanTail) {
                                                                fireBare = true;
                                                                bareEnd = merge;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    if (fireBare) {
                                        chainedBareExcept[offs] = bareEnd;
                                        chainedBare = true;
                                    }
                                }
                            }
                        }
                        /* A typed `except T:` whose enclosing try/except container never
                           received a normal-exit merge (container end == 0) because the
                           try body falls straight through into its own else/handler
                           chain — and whose handler body is terminal (it breaks out of an
                           enclosing loop rather than merging) — must still be bounded, or
                           it stays open to EOF and swallows everything after the loop. This
                           shape arises for a `try/except` nested INSIDE another except whose
                           handler `break`s the loop: the break carries a doubled
                           POP_EXCEPT (it unwinds both exception scopes) and a JUMP_FORWARD
                           to the loop exit, with no fall-through merge. Detect it and keep
                           the handler's natural end (offs, the start of the cleanup
                           RERAISE) so the block closes at the cleanup boundary. */
                        bool nestedTermExcept = false;
                        if (curblock->end() == 0 && !chainedBare
                                && !crossesEnclosingTry && !danglingIntoEnclosingTry) {
                            bool enclLoop = false;
                            std::stack<PycRef<ASTBlock> > sc = blocks;
                            if (!sc.empty()) sc.pop(); /* skip the BLK_EXCEPT itself */
                            while (!sc.empty()) {
                                ASTBlock::BlkType bt = sc.top()->blktype();
                                if ((bt == ASTBlock::BLK_FOR || bt == ASTBlock::BLK_WHILE)
                                        && sc.top()->end() > 0) {
                                    enclLoop = true; break;
                                }
                                sc.pop();
                            }
                            if (enclLoop) {
                                /* The container received no normal-exit merge, which means
                                   the protected try body never jumps to a fall-through
                                   point. The shape that derails decompilation is a
                                   try/except NESTED inside another except handler whose body
                                   breaks the enclosing loop: that break unwinds BOTH
                                   exception scopes, so it emits two consecutive POP_EXCEPT
                                   before its JUMP. A plain single-level except-in-loop (one
                                   POP_EXCEPT) is bounded correctly already, so gate on the
                                   doubled POP_EXCEPT to avoid disturbing it. With no merge
                                   to fall through to, the handler ends at offs (the cleanup
                                   RERAISE boundary). */
                                PycBuffer bs(code->code()->value(),
                                        code->code()->length());
                                bs.setPos(pos);
                                int bo, ba, bp = pos, prevReal = -1;
                                bool doublePop = false;
                                while (bp < offs && !bs.atEof()) {
                                    int bip = bp;
                                    bc_next(bs, mod, bo, ba, bp);
                                    if (bo == Pyc::CACHE || bo == Pyc::NOP
                                            || bo == Pyc::EXTENDED_ARG_A) {
                                        if (bp <= bip) break;
                                        continue;
                                    }
                                    if (bo == Pyc::POP_EXCEPT && prevReal == Pyc::POP_EXCEPT) {
                                        doublePop = true; break;
                                    }
                                    prevReal = bo;
                                    if (bp <= bip) break;
                                }
                                if (doublePop)
                                    nestedTermExcept = true;
                            }
                        }
                        if (!nestedTermExcept
                                && !chainedBare && !crossesEnclosingTry && !danglingIntoEnclosingTry
                                && (curblock->end() > 0
                                    || (!recoverThisExcept && !insideOpenSplitExcept)))
                            except_end = curblock->end();
                        recoverThisExcept = false;
                        blocks.pop();
                        curblock = blocks.top();

                        stack_hist.pop();
                    } else if (curblock->blktype() == ASTBlock::BLK_EXCEPT) {
                        if (!chainedBareExcept.count(offs)) {
                            PycBuffer pk(code->code()->value(), code->code()->length());
                            pk.setPos(offs);
                            if (!pk.atEof()) {
                                int po, pa, pp;
                                bc_next(pk, mod, po, pa, pp);
                                if (po == Pyc::POP_TOP) {
                                    int bestEnd = -1, bestSpan = 0x7fffffff;
                                    for (const auto& e : exception_entries) {
                                        if (e.start_offset <= offs && offs < e.end_offset
                                                && e.end_offset - e.start_offset < bestSpan) {
                                            bestSpan = e.end_offset - e.start_offset;
                                            bestEnd = e.end_offset;
                                        }
                                    }
                                    if (bestEnd > offs)
                                        chainedBareExcept[offs] = bestEnd;
                                }
                            }
                        }
                        PycRef<ASTBlock> prevExc = curblock;
                        blocks.pop();
                        curblock = blocks.top();
                        if (prevExc->size() != 0)
                            curblock->append(prevExc.cast<ASTNode>());
                        if (!stack_hist.empty()) {
                            stack = stack_hist.top();
                            stack_hist.pop();
                        }
                    }

                    for (const auto& kv : forStartToExit) {
                        if (forElseMerge.count(kv.second)
                                && forElseMerge[kv.second] == except_end
                                && curpos >= kv.first && curpos < kv.second) {
                            except_end = kv.second;
                            forElseBreakLoop.insert(kv.first);
                            loopCloseAtExit.insert(kv.second);
                            break;
                        }
                    }
                    {
                        int loopClamp = except_end;
                        for (const auto& kv : forStartToExit) {
                            if (curpos >= kv.first && curpos < kv.second
                                    && kv.second > offs && kv.second < loopClamp)
                                loopClamp = kv.second;
                        }
                        if (loopClamp < except_end)
                            except_end = loopClamp;
                    }
                    if (except_end > offs && finallyCopySkip.count(except_end)) {
                        PycRef<ASTBlock> enclIf = curblock;
                        { std::stack<PycRef<ASTBlock> > sc = blocks;
                          if (enclIf->blktype() == ASTBlock::BLK_CONTAINER && sc.size() > 1) {
                              sc.pop(); if (!sc.empty()) enclIf = sc.top();
                          } }
                        bool inIfBranch = (enclIf->blktype() == ASTBlock::BLK_IF
                                || enclIf->blktype() == ASTBlock::BLK_ELIF
                                || enclIf->blktype() == ASTBlock::BLK_ELSE)
                                && enclIf->end() == except_end;
                        bool clauseTerminal = true;
                        if (inIfBranch) {
                            PycBuffer ts(code->code()->value(), code->code()->length());
                            ts.setPos(pos);
                            int to_, ta_, tp = pos;
                            while (tp < offs && !ts.atEof()) {
                                int tip = tp;
                                bc_next(ts, mod, to_, ta_, tp);
                                if (to_ == Pyc::JUMP_FORWARD_A
                                        && tp + ta_ * (int)sizeof(uint16_t) >= offs) {
                                    clauseTerminal = false; break;
                                }
                                if (tp <= tip) break;
                            }
                        }
                        if (inIfBranch && clauseTerminal)
                            except_end = offs;
                    }
                    if (except_end > offs) {
                        PycRef<ASTBlock> encl = curblock;
                        { std::stack<PycRef<ASTBlock> > sc = blocks;
                          if (encl->blktype() == ASTBlock::BLK_CONTAINER && sc.size() > 1) {
                              sc.pop();
                              if (!sc.empty()) encl = sc.top();
                          } }
                        if ((encl->blktype() == ASTBlock::BLK_IF
                                || encl->blktype() == ASTBlock::BLK_ELIF)
                                && encl->end() > offs && encl->end() < except_end
                                && !exceptElseAt.count(encl->end())) {
                            int elseT = encl->end(), merge = except_end;
                            bool simple = true;
                            int toMerge = 0;
                            bool exceptTerminal = true;
                            {
                                PycBuffer ts(code->code()->value(), code->code()->length());
                                ts.setPos(pos);
                                int to_, ta_, tp = pos;
                                while (tp < offs && !ts.atEof()) {
                                    int tip = tp;
                                    bc_next(ts, mod, to_, ta_, tp);
                                    if (to_ == Pyc::JUMP_FORWARD_A
                                            && tp + ta_ * (int)sizeof(uint16_t) == merge) {
                                        exceptTerminal = false; break;
                                    }
                                    if (tp <= tip) break;
                                }
                            }
                            bool nestedOk = exceptTerminal, hasNested = false;
                            if (nestedOk) {
                                for (const auto& ee : exception_entries) {
                                    if (ee.start_offset >= elseT && ee.start_offset < merge) {
                                        hasNested = true;
                                        if (ee.target < elseT || ee.target > merge) {
                                            nestedOk = false; break;
                                        }
                                    }
                                }
                                nestedOk = nestedOk && hasNested;
                            }
                            PycBuffer es(code->code()->value(), code->code()->length());
                            es.setPos(elseT);
                            int eo, ea, ep = elseT;
                            while (ep < merge && !es.atEof()) {
                                int eip = ep;
                                bc_next(es, mod, eo, ea, ep);
                                int tgt = -1;
                                if (eo == Pyc::JUMP_FORWARD_A || eo == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                                        || eo == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                                        || eo == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                                        || eo == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A)
                                    tgt = ep + ea * (int)sizeof(uint16_t);
                                else if (eo == Pyc::JUMP_BACKWARD_A
                                        || eo == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A
                                        || eo == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A)
                                    tgt = ep - ea * (int)sizeof(uint16_t);
                                if ((eo == Pyc::PUSH_EXC_INFO && !nestedOk)
                                        || (eo == Pyc::BEFORE_WITH && !nestedOk)
                                        || eo == Pyc::SETUP_FINALLY_A) {
                                    simple = false; break;
                                }
                                if (eo == Pyc::FOR_ITER_A) {
                                    int fex = ep + ea * (int)sizeof(uint16_t);
                                    if (fex > merge) { simple = false; break; }
                                    if (ep <= eip) break;
                                    continue;
                                }
                                bool backEdge = (eo == Pyc::JUMP_BACKWARD_A
                                        || eo == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A
                                        || eo == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A
                                        || eo == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A);
                                if (backEdge && tgt >= elseT && tgt < merge) {
                                    if (ep <= eip) break;
                                    continue;
                                }
                                if (tgt >= 0 && (tgt > merge || tgt <= eip)) {
                                    simple = false; break;
                                }
                                if (tgt == merge) ++toMerge;
                                if (ep <= eip) break;
                            }
                            if (simple) {
                                exceptElseAt[elseT] = merge;
                                except_end = elseT;
                            }
                        }
                    }
                    ifblk = new ASTCondBlock(ASTBlock::BLK_EXCEPT, except_end, cond.cast<ASTCompare>()->right(), false);
                } else if (curblock->blktype() == ASTBlock::BLK_ELSE
                           && curblock->size() == 0
                           && !rotWhileGuard.count(curpos)
                           && !elseNestedIfContinues
                           && !ternaryElseArm
                           && !ternaryAssignElse
                           && !loopElseBlocks.count((ASTBlock *)curblock)
                           && !exceptElseNoCollapse.count((ASTBlock *)curblock)
                           && !stack_hist.empty()) {
                    int _outerElseEnd = curblock->end();
                    blocks.pop();
                    stack = stack_hist.top();
                    stack_hist.pop();
                    ifblk = new ASTCondBlock(ASTBlock::BLK_ELIF, offs, cond, neg);
                    if (_outerElseEnd > offs)
                        elifPendingElse[offs] = _outerElseEnd;
                } else if (curblock->size() == 0 && !curblock->inited()
                           && curblock->blktype() == ASTBlock::BLK_WHILE) {
                    /* The condition for a while loop */
                    PycRef<ASTBlock> top = blocks.top();
                    blocks.pop();
                    ifblk = new ASTCondBlock(top->blktype(), offs, cond, neg);

                    /* We don't store the stack for loops! Pop it! */
                    stack_hist.pop();
                } else if (curblock->size() == 0 && curblock->end() <= offs
                           && (curblock->blktype() == ASTBlock::BLK_IF
                           || curblock->blktype() == ASTBlock::BLK_ELIF
                           || curblock->blktype() == ASTBlock::BLK_WHILE)
                           && !(rotWhileGuard.count(curpos)
                                && !compoundWhileBody.count(pos))) {
                    PycRef<ASTNode> newcond;
                    PycRef<ASTCondBlock> top = curblock.cast<ASTCondBlock>();
                    PycRef<ASTNode> cond1 = top->cond();
                    blocks.pop();

                    if (curblock->blktype() == ASTBlock::BLK_WHILE) {
                        if (!stack_hist.empty())
                            stack_hist.pop();
                    } else {
                        FastStack s_top = stack_hist.top();
                        stack_hist.pop();
                        if (!stack_hist.empty())
                            stack_hist.pop();
                        stack_hist.push(s_top);
                    }

                    bool isAnd = (canonTarget(curblock->end()) == canonTarget(offs)
                            || (curblock->end() == curpos && !top->negative()));
                    bool firstNeg = isAnd ? top->negative() : !top->negative();
                    PycRef<ASTNode> left = firstNeg
                            ? NegateCond(cond1) : cond1;
                    PycRef<ASTNode> right = neg
                            ? NegateCond(cond) : cond;
                    bool leftIsTrue = top->blktype() == ASTBlock::BLK_WHILE
                            && isAnd && !firstNeg
                            && cond1.type() == ASTNode::NODE_OBJECT
                            && cond1.cast<ASTObject>()->object() == Pyc_True;
                    if (leftIsTrue)
                        newcond = right;
                    else if (!isAnd && firstNeg && neg
                             && (cond1->type() == ASTNode::NODE_COMPARE
                                 || cond->type() == ASTNode::NODE_COMPARE)) {
                        /* de Morgan: an OR whose BOTH operands are negated comes from
                           a source `not (cond1 and cond)`. The compiler emits the
                           operands UN-negated (cond1 with its IF_FALSE-to-body jump,
                           cond a `==`/`is`/… with IF_TRUE-to-skip); rendering the
                           distributed `not cond1 or not cond` re-inverts a COMPARISON
                           operand (`==`->`!=`) and recompiles to the OPPOSITE jump
                           sense (original `==`+IF_TRUE vs distributed `!=`+IF_FALSE).
                           Fold back to `not (cond1 and cond)` so the comparison keeps
                           its original operator. Gated on a comparison operand: with
                           only bare operands the two forms compile identically, so
                           folding there would be gratuitous render churn. A genuine
                           `not A or B != C` has its second operand POSITIVE (neg=false)
                           and never reaches here. */
                        newcond = new ASTUnary(
                                new ASTBinary(cond1, cond, ASTBinary::BIN_LOG_AND),
                                ASTUnary::UN_NOT);
                    }
                    else if (isAnd && firstNeg && neg
                             && (cond1->type() == ASTNode::NODE_COMPARE
                                 || cond->type() == ASTNode::NODE_COMPARE)) {
                        /* de Morgan (dual of the OR case above): an AND whose BOTH
                           operands are negated comes from a source `not (cond1 or
                           cond)`. The compiler emits the operands un-negated (each a
                           `==`/`is`/… with an IF_TRUE-to-skip jump — any one true skips
                           the guarded body); distributing the negation to `not cond1
                           and not cond` re-inverts a COMPARISON operand (`==`->`!=`) and
                           recompiles to the opposite jump sense. Fold to `not (cond1 or
                           cond)` so the comparison keeps its original operator. Gated on
                           a comparison operand, like the OR case: a genuine `A != B and
                           C != D` has POSITIVE operands (firstNeg/neg false) and never
                           reaches here. */
                        newcond = new ASTUnary(
                                new ASTBinary(cond1, cond, ASTBinary::BIN_LOG_OR),
                                ASTUnary::UN_NOT);
                    }
                    else if (isAnd && neg
                             && cond->type() == ASTNode::NODE_COMPARE
                             && cond1->type() == ASTNode::NODE_UNARY
                             && cond1.cast<ASTUnary>()->op() == ASTUnary::UN_NOT
                             && cond1.cast<ASTUnary>()->operand()->type() == ASTNode::NODE_BINARY
                             && cond1.cast<ASTUnary>()->operand().cast<ASTBinary>()->op()
                                    == ASTBinary::BIN_LOG_OR) {
                        /* Extend an already-folded `not (… or …)` chain (see the dual
                           case) with a further negated comparison, so `not(A or B or C)`
                           builds up left-to-right rather than degrading to `not(A or B)
                           and not C` at the third operand (where the folded left operand
                           is no longer flagged negative). */
                        PycRef<ASTNode> innerOr = cond1.cast<ASTUnary>()->operand();
                        newcond = new ASTUnary(
                                new ASTBinary(innerOr, cond, ASTBinary::BIN_LOG_OR),
                                ASTUnary::UN_NOT);
                    }
                    else
                        newcond = new ASTBinary(left, right,
                                isAnd ? ASTBinary::BIN_LOG_AND : ASTBinary::BIN_LOG_OR);
                    ifblk = new ASTCondBlock(top->blktype(), offs, newcond, false);
                } else if (curblock->blktype() == ASTBlock::BLK_FOR
                            && curblock.cast<ASTIterBlock>()->isComprehension()
                            && !compTernaryElem
                            && mod->verCompare(2, 7) >= 0) {
                    if ((opcode == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                                || opcode == Pyc::POP_JUMP_FORWARD_IF_TRUE_A)
                            && backwardJumpOffsets.count(offs)) {
                        PycRef<ASTNode> fcond = neg
                                ? new ASTUnary(cond, ASTUnary::UN_NOT) : cond;
                        PycRef<ASTNode> existing =
                                curblock.cast<ASTIterBlock>()->condition();
                        if (existing != nullptr) {
                            if (compFilterFwd) {
                                cleanBuild = false;
                                return new ASTNodeList(defblock->nodes());
                            }
                            fcond = new ASTBinary(existing, fcond, ASTBinary::BIN_LOG_AND);
                        }
                        curblock.cast<ASTIterBlock>()->setCondition(fcond);
                        stack_hist.pop();
                        break;
                    }
                    if (compPureOr
                            && (opcode == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                                || opcode == Pyc::POP_JUMP_FORWARD_IF_FALSE_A)) {
                        PycRef<ASTNode> oper = neg ? cond : NegateCond(cond);
                        PycRef<ASTNode> existing =
                                curblock.cast<ASTIterBlock>()->condition();
                        if (existing != nullptr)
                            oper = new ASTBinary(existing, oper, ASTBinary::BIN_LOG_OR);
                        curblock.cast<ASTIterBlock>()->setCondition(oper);
                        stack_hist.pop();
                        break;
                    }
                    compFilterFwd = true;
                    curblock.cast<ASTIterBlock>()->setCondition(cond);
                    stack_hist.pop();
                    // TODO: Handle older python versions, where condition
                    // is laid out a little differently.
                    break;
                } else {
                    /* Plain old if statement */
                    ifblk = new ASTCondBlock(ASTBlock::BLK_IF, offs, cond, neg);

                    if ((curblock->blktype() == ASTBlock::BLK_FOR
                            || curblock->blktype() == ASTBlock::BLK_WHILE)
                            && curblock->end() > offs && offs > pos
                            && !loopTailElseAt.count(offs)) {
                        const int W = (int)sizeof(uint16_t);
                        PycBuffer bs(code->code()->value(), code->code()->length());
                        bs.setPos(pos);
                        int bo, ba, bp = pos, lastReal = -1, lastEnd = pos, realN = 0;
                        int backEdgeN = 0; bool sawExc = false, reachesOffs = false;
                        while (bp < offs && !bs.atEof()) {
                            int bip = bp;
                            bc_next(bs, mod, bo, ba, bp);
                            if (bp <= bip) break;
                            if (bo == Pyc::CACHE || bo == Pyc::NOP
                                    || bo == Pyc::EXTENDED_ARG_A) continue;
                            lastReal = bo; lastEnd = bp; realN++;
                            if (bo == Pyc::CHECK_EXC_MATCH) sawExc = true;
                            if ((bo == Pyc::JUMP_BACKWARD_A
                                    || bo == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A)
                                    && (bp - ba * W) < pos)
                                backEdgeN++;
                            if ((bo == Pyc::JUMP_FORWARD_A
                                    || bo == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                                    || bo == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                                    || bo == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                                    || bo == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A)
                                    && (bp + ba * W) == offs)
                                reachesOffs = true;
                        }
                        bool fallThrough = (lastEnd == offs
                                && lastReal != Pyc::JUMP_BACKWARD_A
                                && lastReal != Pyc::JUMP_BACKWARD_NO_INTERRUPT_A
                                && lastReal != Pyc::JUMP_FORWARD_A
                                && lastReal != Pyc::RETURN_VALUE
                                && lastReal != Pyc::RAISE_VARARGS_A
                                && lastReal != Pyc::RERAISE && lastReal != Pyc::RERAISE_A);
                        bool teoElseInBody = false;
                        for (const auto& ee : exception_entries) {
                            if (ee.start_offset < pos || ee.start_offset >= offs) continue;
                            if (ee.target > offs || ee.target - ee.end_offset < 8) continue;
                            PycBuffer ts(code->code()->value(), code->code()->length());
                            ts.setPos(ee.end_offset);
                            int to, ta, tp = ee.end_offset;
                            while (tp < ee.target && !ts.atEof()) {
                                int tip = tp;
                                bc_next(ts, mod, to, ta, tp);
                                if (tp <= tip) break;
                                if ((to == Pyc::JUMP_BACKWARD_A
                                        || to == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A)
                                        && (tp - ta * W) < pos) { teoElseInBody = true; break; }
                            }
                            if (teoElseInBody) break;
                        }
                        int elseEnd = -1;
                        if (realN >= 2 && backEdgeN >= 2 && sawExc && teoElseInBody
                                && !reachesOffs && !fallThrough) {
                            PycBuffer es(code->code()->value(), code->code()->length());
                            es.setPos(offs);
                            int eo, ea, ep = offs, maxBE = -1;
                            while (ep < curblock->end() && !es.atEof()) {
                                int eip = ep;
                                bc_next(es, mod, eo, ea, ep);
                                if (ep <= eip) break;
                                if ((eo == Pyc::JUMP_BACKWARD_A
                                        || eo == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A)
                                        && (ep - ea * W) < pos && ep > maxBE)
                                    maxBE = ep;
                            }
                            if (maxBE > 0) {
                                PycBuffer ps(code->code()->value(), code->code()->length());
                                ps.setPos(maxBE);
                                int po, pa, pp = maxBE;
                                while (pp < curblock->end() && !ps.atEof()) {
                                    int pip = pp;
                                    bc_next(ps, mod, po, pa, pp);
                                    if (pp <= pip) break;
                                    if (po == Pyc::RERAISE || po == Pyc::RERAISE_A
                                            || po == Pyc::COPY_A || po == Pyc::POP_EXCEPT
                                            || po == Pyc::CACHE
                                            || po == Pyc::EXTENDED_ARG_A) { maxBE = pp; continue; }
                                    break;
                                }
                                elseEnd = maxBE;
                            }
                        }
                        if (elseEnd > offs)
                            loopTailElseAt[offs] = elseEnd;
                    }

                    /* A plain `if C:` that is the last statement of a loop body
                       and DOES carry a real `else:` clause, but whose if-true
                       branch never falls through to (nor forward-jumps to) the
                       else-target — every true path leaves via break/continue/
                       return/raise. The standard else opens off a JUMP_FORWARD
                       over the else; here there is none, so the BLK_IF would
                       close at `offs` and the else body would flatten to the
                       loop-body level, swallowing the loop's continue back-edge
                       that physically follows it. Detect the shape and arm a
                       BLK_ELSE[offs, loopBodyEnd) so the else renders and the
                       trailing back-edge stays the loop continuation. */
                    if (!loopTailElseAt.count(offs)
                            && (curblock->blktype() == ASTBlock::BLK_WHILE
                                || curblock->blktype() == ASTBlock::BLK_FOR)
                            && curblock->end() > offs && offs > pos
                            && !backwardJumpOffsets.count(offs)) {
                        const int W = (int)sizeof(uint16_t);
                        /* the enclosing loop whose body span [start, bodyEnd)
                           contains both `pos` and `offs`, and whose merge is this
                           loop block's end (`exit`). */
                        int bodyEnd = -1;
                        for (const auto& lr : loopRanges) {
                            if (lr.start < pos && lr.end >= offs
                                    && lr.exit == curblock->end()) {
                                if (lr.end > bodyEnd) bodyEnd = lr.end;
                            }
                        }
                        if (bodyEnd > offs) {
                            /* the if-true branch [pos, offs): must NOT fall
                               through to `offs` and must NOT forward-jump to it,
                               so `offs` is reached solely via the if's
                               false-jump = a genuine else. Reject anything with
                               an exception handler (handled elsewhere). */
                            PycBuffer bs(code->code()->value(), code->code()->length());
                            bs.setPos(pos);
                            int bo, ba, bp = pos;
                            bool reachesOffs = false, sawExc = false;
                            bool convergesInElse = false;
                            while (bp < offs && !bs.atEof()) {
                                int bip = bp;
                                bc_next(bs, mod, bo, ba, bp);
                                if (bp <= bip) break;
                                if (bo == Pyc::CACHE || bo == Pyc::NOP
                                        || bo == Pyc::EXTENDED_ARG_A) continue;
                                if (bo == Pyc::PUSH_EXC_INFO || bo == Pyc::CHECK_EXC_MATCH
                                        || bo == Pyc::SETUP_FINALLY_A) sawExc = true;
                                if (bo == Pyc::JUMP_FORWARD_A
                                        || bo == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                                        || bo == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                                        || bo == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                                        || bo == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A) {
                                    int jt = bp + ba * W;
                                    if (jt == offs)
                                        reachesOffs = true;
                                    /* a forward jump from the if-true branch
                                       landing INSIDE the candidate else body means
                                       the two paths CONVERGE before the loop
                                       bottom — `offs` is a partial/inner merge, not
                                       a loop-tail else (e.g. an if/elif/else whose
                                       else: raises, then shared unconditional loop
                                       body). */
                                    if (jt > offs && jt < bodyEnd)
                                        convergesInElse = true;
                                }
                            }
                            /* the if-true branch falls through into `offs` unless
                               the op physically ending right before `offs` is a
                               jump/return/raise — if it falls through, `offs` is a
                               plain merge, not an else. opEndingBefore is robust
                               against inline-cache counting. */
                            int prevOp = opEndingBefore.count(offs)
                                    ? opEndingBefore[offs] : -1;
                            /* A guard (`if C: continue` / `if C: break`) ends its
                               true-branch with a JUMP_BACKWARD (continue) or a
                               JUMP_FORWARD to the loop merge (break) right before
                               `offs`; the code after it is the rest of the loop
                               body, NOT an else — and break/continue guards
                               compile identically to their if/else form, so the
                               guard render is the canonical one. Only a true
                               branch that LEAVES the function (return/raise)
                               before `offs` makes `offs` a genuine else arm whose
                               loss would drop the loop continuation. */
                            bool fallThrough = !(prevOp == Pyc::RETURN_VALUE
                                    || prevOp == Pyc::RETURN_CONST_A
                                    || prevOp == Pyc::RAISE_VARARGS_A
                                    || prevOp == Pyc::RERAISE
                                    || prevOp == Pyc::RERAISE_A);
                            /* the else body [offs, bodyEnd) must itself be free of
                               nested exception handlers and not be re-entered by a
                               back-edge that targets BEFORE offs (which would mean
                               offs is loop-internal, not a clean else). */
                            bool elseClean = !sawExc && !reachesOffs && !fallThrough
                                    && !convergesInElse;
                            if (elseClean) {
                                PycBuffer es(code->code()->value(),
                                        code->code()->length());
                                es.setPos(offs);
                                int eo, ea, ep = offs;
                                while (ep < bodyEnd && !es.atEof()) {
                                    int eip = ep;
                                    bc_next(es, mod, eo, ea, ep);
                                    if (ep <= eip) break;
                                    if (eo == Pyc::PUSH_EXC_INFO
                                            || eo == Pyc::SETUP_FINALLY_A) {
                                        elseClean = false; break;
                                    }
                                    if ((eo == Pyc::JUMP_BACKWARD_A
                                            || eo == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A)
                                            && (ep - ea * W) < offs
                                            && (ep - ea * W) >= pos) {
                                        elseClean = false; break;
                                    }
                                }
                            }
                            if (elseClean)
                                loopBodyElseAt[offs] = bodyEnd;
                        }
                    }
                }

                if (popped)
                    ifblk->init(popped);

                blocks.push(ifblk.cast<ASTBlock>());
                curblock = blocks.top();
            }
            break;
        case Pyc::JUMP_ABSOLUTE_A:
        // bpo-47120: Replaced JUMP_ABSOLUTE by the relative jump JUMP_BACKWARD.
        case Pyc::JUMP_BACKWARD_A:
        case Pyc::JUMP_BACKWARD_NO_INTERRUPT_A:
            {
                int offs = operand;
                if (mod->verCompare(3, 10) >= 0)
                    offs *= sizeof(uint16_t);
                if (opcode == Pyc::JUMP_BACKWARD_A
                        || opcode == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A)
                    offs = pos - offs;

                if (offs < pos) {
                    if (opcode == Pyc::JUMP_BACKWARD_A
                            && whileTrueHdr.count(offs)
                            && pos == whileTrueHdr[offs]
                            && curblock->blktype() == ASTBlock::BLK_WHILE) {
                        PycRef<ASTBlock> whb = curblock;
                        if (topTestWhileFallContinue.count(pos))
                            whb->append(new ASTKeyword(ASTKeyword::KW_CONTINUE));
                        blocks.pop();
                        curblock = blocks.top();
                        curblock->append(whb.cast<ASTNode>());
                        break;
                    }
                    while (curblock->blktype() == ASTBlock::BLK_FOR
                            && curblock->end() == curpos
                            && curblock.cast<ASTIterBlock>()->start() != offs
                            && curblock.cast<ASTIterBlock>()->start() > offs
                            && blocks.size() > 1) {
                        PycRef<ASTBlock> innerLoop = curblock;
                        blocks.pop();
                        curblock = blocks.top();
                        curblock->append(innerLoop.cast<ASTNode>());
                    }
                    if (curblock->blktype() == ASTBlock::BLK_FOR
                            || curblock->blktype() == ASTBlock::BLK_ASYNCFOR) {
                        bool is_jump_to_start = offs == curblock.cast<ASTIterBlock>()->start();
                        bool should_pop_for_block = curblock.cast<ASTIterBlock>()->isComprehension();
                        // in v3.8, SETUP_LOOP is deprecated and for blocks aren't terminated by POP_BLOCK, so we add them here
                        bool should_add_for_block = mod->majorVer() == 3 && mod->minorVer() >= 8 && is_jump_to_start && !curblock.cast<ASTIterBlock>()->isComprehension();

                        if (should_pop_for_block || should_add_for_block) {
                            PycRef<ASTNode> top = stack.top();

                            if (top.type() == ASTNode::NODE_COMPREHENSION) {
                                PycRef<ASTComprehension> comp = top.cast<ASTComprehension>();

                                comp->addGenerator(curblock.cast<ASTIterBlock>());
                            }

                            PycRef<ASTBlock> tmp = curblock;
                            int forEnd = tmp->end();
                            bool forTrailCont = false;
                            if (should_add_for_block) {
                                int forH = tmp.cast<ASTIterBlock>()->start();
                                for (const auto& fj : fwdJumps) {
                                    if (fj.second != curpos) continue;
                                    int O = fj.first;
                                    if (O <= forH || O >= curpos) continue;
                                    bool inNested = false;
                                    for (const auto& fe : forStartToExit)
                                        if (fe.first > forH && O >= fe.first
                                                && O < fe.second) { inNested = true; break; }
                                    for (const auto& lr : loopRanges)
                                        if (!inNested && lr.start > forH && O >= lr.start
                                                && O < lr.end) { inNested = true; break; }
                                    if (!inNested) { forTrailCont = true; break; }
                                }
                            }
                            if (forTrailCont)
                                tmp->append(new ASTKeyword(ASTKeyword::KW_CONTINUE));
                            blocks.pop();
                            curblock = blocks.top();
                            if (should_add_for_block) {
                                curblock->append(tmp.cast<ASTNode>());
                            }
                            if (should_add_for_block
                                    && forElseMerge.count(forEnd)) {
                                stack_hist.push(stack);
                                PycRef<ASTBlock> elseblk = new ASTBlock(
                                        ASTBlock::BLK_ELSE, forElseMerge[forEnd]);
                                elseblk->init();
                                loopElseBlocks.insert((ASTBlock *)elseblk);
                                blocks.push(elseblk);
                                curblock = blocks.top();
                            }
                        }
                    } else if (curblock->blktype() == ASTBlock::BLK_ELSE) {
                        bool isContinueBE = false;
                        for (const auto& lr : loopRanges)
                            if (lr.start == offs && lr.end > pos) { isContinueBE = true; break; }
                        {
                            auto fe = forStartToExit.find(offs);
                            if (fe == forStartToExit.end()
                                    || curblock->end() >= fe->second)
                                isContinueBE = false;
                        }
                        if (!isContinueBE && whileTrueHdr.count(offs)
                                && loopElseBlocks.count((ASTBlock *)curblock)) {
                            isContinueBE = true;
                            topTestWhileExit.insert(whileTrueHdr[offs]);
                            topTestWhileFallBreak.insert(whileTrueHdr[offs]);
                        }
                        if (!isContinueBE
                                && loopElseBlocks.count((ASTBlock *)curblock)
                                && !forBreakBeyondExit.count(offs)) {
                            auto fe = forStartToExit.find(offs);
                            if (fe != forStartToExit.end()
                                    && curblock->end() < fe->second)
                                isContinueBE = true;
                        }
                        if (!isContinueBE
                                && forStartToExit.find(offs) == forStartToExit.end()
                                && !loopElseBlocks.count((ASTBlock *)curblock)) {
                            for (const auto& lr : loopRanges) {
                                if (whileTrueHdr.count(lr.start)
                                        && offs <= lr.start && lr.start - offs <= 4
                                        && lr.start <= pos && pos < lr.end) {
                                    isContinueBE = true;
                                    break;
                                }
                            }
                        }
                        /* Plain `while <cond>:` (top-test, NOT while-True) else-continue:
                           the `else:` body ends in `continue` — a JUMP_BACKWARD to the
                           while header `offs`. whileTrueHdr doesn't cover a conditional
                           while and the for-scoping cleared isContinueBE (offs is not a
                           FOR_ITER start), so the continue was dropped (aiohttp/multipart
                           parse_content_disposition). Fire when `offs` is a loop header
                           with a LATER back-edge to it (strictly after pos = a live
                           iteration edge, so a real continue not the loop's sole/closing
                           edge), and pos is inside that range. */
                        if (!isContinueBE
                                && forStartToExit.find(offs) == forStartToExit.end()
                                && !loopElseBlocks.count((ASTBlock *)curblock)
                                && !whileTrueHdr.count(offs)) {
                            bool laterBE = false;
                            for (const auto& lr : loopRanges)
                                if (lr.start == offs && lr.end > pos
                                        && offs <= pos) {
                                    laterBE = true; break;
                                }
                            if (laterBE)
                                isContinueBE = true;
                        }
                        /* Rotated `while cond:` CONTINUE whose loop CLOSES via a
                           CONDITIONAL bottom re-test (not a later unconditional
                           back-edge): this `else:` body's JUMP_BACKWARD to the header
                           `offs` is the loop's ONLY unconditional back-edge (a
                           `continue`), and the natural fall-off-end re-tests the
                           condition and jumps back to the BODY (targeting the guard's
                           fall-through, not the header). The laterBE test above finds
                           no sibling back-edge to `offs` (the bottom re-test targets
                           the body, a different offset), so the continue was dropped
                           — leaving the loop with only its bottom re-test on recompile
                           (character `an observed function`: outermost of three nested
                           `while x is None:`). Detect via the recorded
                           continue-exit (rotWhileContinueExit holds the offset right
                           after this JUMP_BACKWARD, since the bottom re-test follows). */
                        if (!isContinueBE
                                && forStartToExit.find(offs) == forStartToExit.end()
                                && !loopElseBlocks.count((ASTBlock *)curblock)
                                && rotWhileContinueExit.count(pos)) {
                            for (const auto& lr : loopRanges)
                                if (lr.start == offs && lr.exit == pos) {
                                    isContinueBE = true; break;
                                }
                        }
                        if (!isContinueBE
                                && loopElseBlocks.count((ASTBlock *)curblock)
                                && forElseFinBreak.count(offs)) {
                            isContinueBE = true;
                        }
                        /* Nested for-else search-loop idiom: the inner for-else's
                           `else:` body is `continue` of the ENCLOSING for, which is
                           itself an armed for-else+break (forElseBreakLoop) — the
                           back-edge to `offs` IS the enclosing loop's only edge so its
                           loopRange end == pos and the for-scoping/SHIP-196 paths
                           exclude it (the outer DOES break past its exit). The outer
                           for-else is reconstructed (the nested arming above), so its
                           inner else-continue must emit. Mirrors the forElseFinBreak
                           gate just above. */
                        if (!isContinueBE
                                && loopElseBlocks.count((ASTBlock *)curblock)
                                && forElseBreakLoop.count(offs)) {
                            isContinueBE = true;
                        }
                        /* Rotated while-True continue-else: the `else:` arm of a
                           nested if inside the loop ends in `continue` (a
                           JUMP_BACKWARD to the while-True header `offs`) whose own
                           back-edge is the loop's LAST edge (loopRange end == pos),
                           and the arm merges at a whileContMerge point (the offset
                           right after this back-edge, with loop-body tail beyond).
                           The laterBE/for-scoping paths all require lr.end > pos, so
                           the continue was dropped as dead.  Fire when offs is a
                           while-True header and this else block ends at that
                           continue-merge. */
                        if (!isContinueBE
                                && whileTrueHdr.count(offs)
                                && whileContMerge.count(curblock->end())
                                && offs <= curpos && curpos < curblock->end()) {
                            isContinueBE = true;
                        }
                        if (isContinueBE)
                            curblock->append(new ASTKeyword(ASTKeyword::KW_CONTINUE));
                        if (stack_hist.empty() || blocks.size() < 2)
                            throw std::runtime_error("else back-edge stack/block underflow");
                        stack = stack_hist.top();
                        stack_hist.pop();

                        blocks.pop();
                        blocks.top()->append(curblock.cast<ASTNode>());
                        curblock = blocks.top();

                        if (curblock->blktype() == ASTBlock::BLK_CONTAINER
                                && !curblock.cast<ASTContainerBlock>()->hasFinally()) {
                            blocks.pop();
                            blocks.top()->append(curblock.cast<ASTNode>());
                            curblock = blocks.top();
                        }
                    } else {
                        curblock->append(new ASTKeyword(ASTKeyword::KW_CONTINUE));
                        if (curblock->blktype() == ASTBlock::BLK_TRY
                                && !forHasBreak.count(offs)
                                && backedgeCount[offs] == 1) {
                            auto fe = forStartToExit.find(offs);
                            if (fe != forStartToExit.end()) {
                                loopCloseAtExit.insert(fe->second);
                                recoverThisExcept = true;
                            }
                        }
                    }

                    /* We're in a loop, this jumps back to the start */
                    /* I think we'll just ignore this case... */
                    break; // Bad idea? Probably!
                }

                if (curblock->blktype() == ASTBlock::BLK_CONTAINER) {
                    PycRef<ASTContainerBlock> cont = curblock.cast<ASTContainerBlock>();
                    if (cont->hasExcept() && pos < cont->except()) {
                        PycRef<ASTBlock> except = new ASTCondBlock(ASTBlock::BLK_EXCEPT, 0, NULL, false);
                        except->init();
                        blocks.push(except);
                        curblock = blocks.top();
                    }
                    break;
                }

                if (!stack_hist.empty()) {
                    stack = stack_hist.top();
                    stack_hist.pop();
                } else {
                    fprintf(stderr, "Warning: Stack history is empty, something wrong might have happened\n");
                }

                PycRef<ASTBlock> prev = curblock;
                PycRef<ASTBlock> nil;
                bool push = true;

                do {
                    blocks.pop();

                    blocks.top()->append(prev.cast<ASTNode>());

                    if (prev->blktype() == ASTBlock::BLK_IF
                            || prev->blktype() == ASTBlock::BLK_ELIF) {
                        if (push) {
                            stack_hist.push(stack);
                        }
                        PycRef<ASTBlock> next = new ASTBlock(ASTBlock::BLK_ELSE, blocks.top()->end());
                        if (prev->inited() == ASTCondBlock::PRE_POPPED) {
                            next->init(ASTCondBlock::PRE_POPPED);
                        }

                        blocks.push(next.cast<ASTBlock>());
                        prev = nil;
                    } else if (prev->blktype() == ASTBlock::BLK_EXCEPT) {
                        if (push) {
                            stack_hist.push(stack);
                        }
                        PycRef<ASTBlock> next = new ASTCondBlock(ASTBlock::BLK_EXCEPT, blocks.top()->end(), NULL, false);
                        next->init();

                        blocks.push(next.cast<ASTBlock>());
                        prev = nil;
                    } else if (prev->blktype() == ASTBlock::BLK_ELSE) {
                        /* Special case */
                        prev = blocks.top();
                        if (!push) {
                            stack = stack_hist.top();
                            stack_hist.pop();
                        }
                        push = false;
                    } else {
                        prev = nil;
                    }

                } while (prev != nil);

                curblock = blocks.top();
            }
            break;
        case Pyc::JUMP_FORWARD_A:
        case Pyc::INSTRUMENTED_JUMP_FORWARD_A:
            {
                int offs = operand;
                if (mod->verCompare(3, 10) >= 0)
                    offs *= sizeof(uint16_t); // // BPO-27129

                if (dupForBreakJF.count(curpos)
                        && curblock->blktype() == ASTBlock::BLK_IF
                        && curblock->end() == dupForBreakIfEnd[curpos]
                        && blocks.size() > 1) {
                    int E = dupForBreakJF[curpos];
                    PycRef<ASTBlock> ifb = curblock;
                    if (!stack_hist.empty())
                        stack_hist.pop();
                    blocks.pop();
                    curblock = blocks.top();
                    curblock->append(ifb.cast<ASTNode>());
                    curblock->append(new ASTKeyword(ASTKeyword::KW_BREAK));
                    source.setPos(E);
                    pos = E;
                    break;
                }

                if ((curblock->blktype() == ASTBlock::BLK_IF
                            || curblock->blktype() == ASTBlock::BLK_ELIF)
                        && curblock->end() == curpos
                        && (pos + offs) >= curblock->end()
                        && blocks.size() > 2) {
                    std::stack<PycRef<ASTBlock> > pk = blocks;
                    pk.pop();
                    PycRef<ASTBlock> par = pk.top();
                    if (par->blktype() == ASTBlock::BLK_ELSE
                            && nestedElseHandlerVals.count(par->end())
                            && (pos + offs) >= par->end()) {
                        PycRef<ASTBlock> ifb = curblock;
                        if (!stack_hist.empty())
                            stack_hist.pop();
                        blocks.pop();
                        curblock = blocks.top();
                        curblock->append(ifb.cast<ASTNode>());
                    }
                }

                if (curblock->blktype() == ASTBlock::BLK_ELSE
                        && nestedElseHandlerVals.count(curblock->end())
                        && (pos + offs) >= curblock->end()
                        && blocks.size() > 1) {
                    PycRef<ASTBlock> elseb = curblock;
                    blocks.pop();
                    curblock = blocks.top();
                    if (curblock->blktype() == ASTBlock::BLK_CONTAINER
                            && curblock.cast<ASTContainerBlock>()->hasExcept()) {
                        if (!stack_hist.empty()) {
                            stack = stack_hist.top();
                            stack_hist.pop();
                        }
                        curblock->append(elseb.cast<ASTNode>());
                        stack_hist.push(stack);
                        PycRef<ASTBlock> except = new ASTCondBlock(ASTBlock::BLK_EXCEPT, 0, NULL, false);
                        except->init();
                        blocks.push(except.cast<ASTBlock>());
                        curblock = blocks.top();
                        break;
                    }
                    blocks.push(elseb.cast<ASTBlock>());
                    curblock = elseb;
                }

                if (!stack.empty() && stack.top() != nullptr
                        && stack.top().type() == ASTNode::NODE_CHAINCOMPARE) {
                    int target = pos + offs;
                    source.setPos(target);
                    pos = target;
                    break;
                }

                if (curblock->blktype() != ASTBlock::BLK_CONTAINER) {
                    int btarget = pos + offs;
                    bool isBreak = false;
                    for (const auto& lr : loopRanges) {
                        /* WHILE-loop continue back-edge: when a LATER back-edge to the
                           same header exists, this earlier back-edge is a `continue` and
                           its recorded exit (right after it) is an if/elif MERGE inside
                           the loop, not the loop exit — so a forward jump-over-else
                           landing there must NOT be read as a `break` (aiohttp/multipart
                           parse_content_disposition). The for-loop analogue is the
                           contBE handling below; this covers while loops (lr.start not a
                           FOR_ITER start). Scoped to ranges with a later sibling
                           back-edge so a single-back-edge while still detects real
                           breaks. */
                        if (!forStartToExit.count(lr.start)) {
                            bool laterBE = false;
                            for (const auto& lr2 : loopRanges)
                                if (lr2.start == lr.start && lr2.end > lr.end) {
                                    laterBE = true; break;
                                }
                            if (laterBE)
                                continue;
                        }
                        bool contBE = forStartToExit.count(lr.start)
                                && forStartToExit.at(lr.start) != lr.exit;
                        if (contBE) {
                            bool hasTailBE = false;
                            for (const auto& lr2 : loopRanges)
                                if (lr2.start == lr.start && lr2.end > lr.end) {
                                    hasTailBE = true; break;
                                }
                            if (hasTailBE
                                    || (forExitFallThrough.count(lr.start)
                                        && !doubleBreakContLoop.count(lr.start)))
                                continue;
                        }
                        bool toExit = (lr.exit == btarget)
                                || (lr.exit < btarget
                                    && onlyPaddingBetween(lr.exit, btarget))
                                || (forElseMerge.count(lr.exit)
                                    && forElseMerge[lr.exit] == btarget)
                                || (contBE
                                    && forStartToExit.at(lr.start) == btarget)
                                || (chainWhileBottomExit.count(lr.end)
                                    && chainWhileBottomExit.at(lr.end) == btarget);
                        /* A jump landing on a rotated-while continue-merge (the
                           offset right after a mid-body `continue` back-edge, with
                           loop-body tail following) is a jump-over-else, NOT a
                           break — see whileContMerge in the while-True end scan. */
                        if (whileContMerge.count(btarget))
                            toExit = false;
                        /* This loopRange is a rotated-`while cond:` CONTINUE back-edge
                           (unconditional JUMP_BACKWARD to the header) whose `exit`
                           (right after it) is the loop's CONDITIONAL bottom re-test,
                           not the loop exit (see rotWhileContinueExit). A forward jump
                           landing there is the body's normal completion falling through
                           INTO the bottom re-test — NOT a break. */
                        if (rotWhileContinueExit.count(lr.exit) && lr.exit == btarget)
                            toExit = false;
                        if (toExit && curpos >= lr.start && curpos < lr.end) {
                            isBreak = true;
                            break;
                        }
                    }
                    if (!isBreak) {
                        int btarget2 = pos + offs;
                        for (const auto& kv : forStartToExit) {
                            if (btarget2 == kv.second
                                    && curpos >= kv.first && curpos < kv.second
                                    && !forElseMerge.count(kv.second)
                                    && ((!forExitFallThrough.count(kv.first)
                                         && !forBreakBeyondExit.count(kv.first))
                                        || doubleBreakContLoop.count(kv.first))) {
                                isBreak = true;
                                break;
                            }
                        }
                    }
                    if (!isBreak) {
                        int btarget2b = pos + offs;
                        for (const auto& kv : forStartToExit) {
                            int s = kv.first, E = kv.second;
                            if (forElseBreakLoop.count(s)
                                    && forElseMerge.count(E) && forElseMerge[E] == btarget2b
                                    && curpos >= s && curpos < E) {
                                isBreak = true;
                                break;
                            }
                        }
                    }
                    if (!isBreak && curblock->blktype() == ASTBlock::BLK_TRY) {
                        int btarget3 = pos + offs;
                        for (const auto& kv : forStartToExit) {
                            if (forElseMerge.count(kv.second)
                                    && forElseMerge[kv.second] == btarget3
                                    && curpos >= kv.first && curpos < kv.second) {
                                curblock->append(new ASTKeyword(ASTKeyword::KW_BREAK));
                                break;
                            }
                        }
                    }
                    if (isBreak) {
                        while (blocks.size() > 1
                                && (curblock->blktype() == ASTBlock::BLK_IF
                                    || curblock->blktype() == ASTBlock::BLK_ELSE)
                                && curblock->end() == curpos) {
                            PycRef<ASTBlock> ib = curblock;
                            if (!stack_hist.empty())
                                stack_hist.pop();
                            blocks.pop();
                            curblock = blocks.top();
                            curblock->append(ib.cast<ASTNode>());
                        }
                        curblock->append(new ASTKeyword(ASTKeyword::KW_BREAK));
                        break;
                    }
                    {
                        int btC = pos + offs;
                        bool isWtContinue = false;
                        /* The continue recovery above must NOT fire when this
                           JUMP_FORWARD is the mandatory jump the last elif of an
                           if/elif/else chain makes to skip its OWN else clause(s). That
                           case is locally identical to a trailing `continue` (an arm
                           whose body ends in a forward jump over a run of terminal
                           blocks to the loop's shared back-edge), but the skipped region
                           is the chain's else-arms, not trailing loop-body code, so a
                           `continue` is wrong and recompiles to a per-arm back-edge.
                           Tell them apart by a SIBLING arm: in an if/elif/else chain the
                           earlier arms' bodies each end in an unconditional JUMP_FORWARD
                           to the SAME merge (here the loop's shared back-edge btC), so a
                           preceding JUMP_FORWARD->btC marks a chain merge. A genuine
                           lone `continue` has no preceding arm jumping to btC. Paired
                           with the nTerm>=2 gate (a one-armed if/else has a single
                           terminal block and never reaches here) this is reliable. */
                        {
                            PycBuffer es(code->code()->value(), code->code()->length());
                            es.setPos(0);
                            int eo, ea, ep = 0;
                            while (ep < curpos && !es.atEof()) {
                                bc_next(es, mod, eo, ea, ep);
                                if (eo == Pyc::JUMP_FORWARD_A
                                        && ep + ea * (int)sizeof(uint16_t) == btC) {
                                    isWtContinue = false;
                                    goto wtDone;
                                }
                            }
                        }
                        for (const auto& lr : loopRanges) {
                            if (lr.end != btC || !whileTrueHdr.count(lr.start)
                                    || curpos < lr.start || curpos >= lr.end)
                                continue;
                            int nBack = 0;
                            for (const auto& lr2 : loopRanges)
                                if (lr2.start == lr.start) nBack++;
                            if (nBack != 1)
                                continue;
                            PycBuffer rs(code->code()->value(), code->code()->length());
                            rs.setPos(pos);
                            int ro, ra, rp = pos, nTerm = 0;
                            bool clean = true;
                            while (rp < btC && !rs.atEof()) {
                                bc_next(rs, mod, ro, ra, rp);
                                if (ro == Pyc::RETURN_VALUE || ro == Pyc::RETURN_CONST_A
                                        || ro == Pyc::RAISE_VARARGS_A) { nTerm++; continue; }
                                if (ro == Pyc::POP_JUMP_FORWARD_IF_FALSE_A || ro == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                                        || ro == Pyc::POP_JUMP_FORWARD_IF_NONE_A || ro == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A
                                        || ro == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A || ro == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A
                                        || ro == Pyc::JUMP_FORWARD_A || ro == Pyc::JUMP_BACKWARD_A
                                        || ro == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A || ro == Pyc::JUMP_ABSOLUTE_A
                                        || ro == Pyc::JUMP_IF_TRUE_OR_POP_A || ro == Pyc::JUMP_IF_FALSE_OR_POP_A
                                        || ro == Pyc::FOR_ITER_A || ro == Pyc::SEND_A
                                        || ro == Pyc::SETUP_FINALLY_A || ro == Pyc::SETUP_EXCEPT_A
                                        || ro == Pyc::PUSH_EXC_INFO || ro == Pyc::POP_EXCEPT
                                        || ro == Pyc::CHECK_EXC_MATCH) { clean = false; break; }
                            }
                            if (clean && nTerm >= 2 && rp == btC) {
                                isWtContinue = true;
                                break;
                            }
                        }
                    wtDone:
                        if (isWtContinue) {
                            curblock->append(new ASTKeyword(ASTKeyword::KW_CONTINUE));
                            break;
                        }
                    }
                }

                if (curblock->blktype() == ASTBlock::BLK_CONTAINER) {
                    PycRef<ASTContainerBlock> cont = curblock.cast<ASTContainerBlock>();
                    if (cont->hasExcept()) {
                        stack_hist.push(stack);

                        curblock->setEnd(pos+offs);
                        PycRef<ASTBlock> except = new ASTCondBlock(ASTBlock::BLK_EXCEPT, pos+offs, NULL, false);
                        except->init();
                        blocks.push(except);
                        curblock = blocks.top();
                    }
                    break;
                }

                if (!stack_hist.empty()) {
                    if (stack.empty()) // if it's part of if-expression, TOS at the moment is the result of "if" part
                        stack = stack_hist.top();
                    stack_hist.pop();
                }

                if (curblock->blktype() == ASTBlock::BLK_TRY
                        && openFinallyTargets.count(curblock->end())
                        && offs != 0 && pos + offs > curpos
                        && pos + offs < curblock->end()) {
                    source.setPos(pos + offs);
                    pos = pos + offs;
                    break;
                }

                PycRef<ASTBlock> prev = curblock;
                PycRef<ASTBlock> nil;
                bool push = true;
                int exceptToMerge = -1;
                bool siblingExceptFollows = false;
                if (offs != 0 && finallyCopySkip.count(pos + offs)) {
                    PycBuffer sk(code->code()->value(), code->code()->length());
                    int cur = pos, target = pos + offs;
                    sk.setPos(cur);
                    while (cur < target && !sk.atEof()) {
                        int so, sa, sp = cur;
                        bc_next(sk, mod, so, sa, sp);
                        if (so == Pyc::CHECK_EXC_MATCH) { siblingExceptFollows = true; break; }
                        /* A trailing BARE `except:` sibling has no CHECK_EXC_MATCH — it
                           begins with a POP_TOP at the start of its own handler exception
                           entry. When a typed clause inside a try/finally jumps over such
                           a bare clause to the finally copy, the bare clause must still be
                           rendered, so treat it as a following sibling too (else the
                           finally-skip swallows it). */
                        if (so == Pyc::POP_TOP) {
                            for (const auto& e : exception_entries)
                                if (e.start_offset == cur && e.push_lasti) {
                                    siblingExceptFollows = true; break;
                                }
                            /* A bare `except:` that is the fall-through of a typed
                               except-chain with an `as e` binding is preceded by the typed
                               clause's own exceptional `del e` cleanup, so the shared lasti
                               protection entry starts BEFORE the bare clause's POP_TOP (at
                               the cleanup, not the POP_TOP). The entry-start test above then
                               misses it. The typed clause's fireBare scan has already
                               recorded this bare-clause start in chainedBareExcept; honour
                               that as the authoritative bare-except sibling marker so the
                               finally-copy skip does not swallow the clause. */
                            if (!siblingExceptFollows && chainedBareExcept.count(cur))
                                siblingExceptFollows = true;
                            if (siblingExceptFollows) break;
                        }
                        if (sp <= cur) break;
                        cur = sp;
                    }
                }

                do {
                    if (blocks.empty())
                        throw std::runtime_error("except-merge block underflow");
                    blocks.pop();

                    PycRef<ASTBlock> teoElse = teoSplitElse(prev);
                    if (!blocks.empty())
                        blocks.top()->append(prev.cast<ASTNode>());
                    if (teoElse != nullptr && !blocks.empty())
                        blocks.top()->append(teoElse.cast<ASTNode>());

                    if (prev->blktype() == ASTBlock::BLK_EXCEPT
                            && offs != 0 && finallyCopySkip.count(pos + offs)
                            && !siblingExceptFollows) {
                        exceptToMerge = pos + offs;
                        if (!blocks.empty()) {
                            prev = blocks.top();
                            continue;
                        }
                        prev = nil;
                    } else if (prev->blktype() == ASTBlock::BLK_IF
                            || prev->blktype() == ASTBlock::BLK_ELIF) {
                        if (offs == 0) {
                            prev = nil;
                            continue;
                        }

                        if (prev->end() == curpos && !blocks.empty()) {
                            bool qualifies = false;
                            std::stack<PycRef<ASTBlock> > sc = blocks;
                            while (!sc.empty()) {
                                PycRef<ASTBlock> b = sc.top(); sc.pop();
                                ASTBlock::BlkType tb = b->blktype();
                                if (tb == ASTBlock::BLK_TRY
                                        && b->end() >= pos
                                        && b->end() < pos + offs) {
                                    qualifies = true; break;
                                }
                                if (tb == ASTBlock::BLK_IF || tb == ASTBlock::BLK_ELIF) {
                                    if (b->end() > curpos && b->end() < pos + offs) {
                                        qualifies = true; break;
                                    }
                                    if (b->end() == curpos)
                                        continue;
                                }
                                if (tb == ASTBlock::BLK_ELSE
                                        && b->end() >= curpos && b->end() <= pos + offs)
                                    continue;
                                break;
                            }
                            if (qualifies) {
                                prev = blocks.top();
                                continue;
                            }
                        }

                        if (push) {
                            stack_hist.push(stack);
                        }
                        PycRef<ASTBlock> next = new ASTBlock(ASTBlock::BLK_ELSE, pos+offs);
                        if (prev->inited() == ASTCondBlock::PRE_POPPED) {
                            next->init(ASTCondBlock::PRE_POPPED);
                        }

                        blocks.push(next.cast<ASTBlock>());
                        prev = nil;
                    } else if (prev->blktype() == ASTBlock::BLK_EXCEPT) {
                        if (offs == 0) {
                            prev = nil;
                            continue;
                        }

                        {
                            int spanA = -1, bestT = -1;
                            for (const auto& sp : excHandlerSpan) {
                                if (sp.first <= curpos && curpos < sp.second
                                        && sp.first > bestT) {
                                    bestT = sp.first; spanA = sp.second;
                                }
                            }
                            if (spanA > 0 && spanA < pos + offs) {
                                bool siblingFollows = false;
                                int opAtA = -1, aArg = 0, aNext = spanA;
                                {
                                    PycBuffer ck(code->code()->value(), code->code()->length());
                                    ck.setPos(pos);
                                    int ko, ka, kp = pos;
                                    while (kp < spanA && !ck.atEof()) {
                                        int kip = kp;
                                        bc_next(ck, mod, ko, ka, kp);
                                        if (ko == Pyc::CHECK_EXC_MATCH) {
                                            siblingFollows = true; break;
                                        }
                                        if (kp <= kip) break;
                                    }
                                    PycBuffer ak(code->code()->value(), code->code()->length());
                                    ak.setPos(spanA);
                                    if (!ak.atEof())
                                        bc_next(ak, mod, opAtA, aArg, aNext);
                                }
                                if (!siblingFollows) {
                                    bool fire = false, openElse = false;
                                    bool dupReturnMerge = false;
                                    int dupRetSpanRet = -1, dupRetTgtRet = -1;
                                    if (opAtA == Pyc::JUMP_FORWARD_A
                                            || opAtA == Pyc::INSTRUMENTED_JUMP_FORWARD_A) {
                                        fire = true;
                                    } else {
                                        std::stack<PycRef<ASTBlock> > sc = blocks;
                                        bool encl = false;
                                        if (!sc.empty()
                                                && sc.top()->blktype() == ASTBlock::BLK_CONTAINER) {
                                            sc.pop();
                                            /* Peel through intervening BLK_ELSE arms whose
                                               body's last statement is this try/except — the
                                               else shares the resume merge so its end is
                                               >= A. The real owner of the resume (the
                                               enclosing BLK_IF/BLK_ELIF whose false-target
                                               is A) sits just below it (`elif X: if c: …
                                               else: try/except` nested in an outer `elif Y:`
                                               — the inner if-else holds the try/except and A
                                               is the outer arm's own `else`). Without
                                               peeling, the spurious bare `except:` opens and
                                               swallows the sibling arm(s). */
                                            while (!sc.empty()
                                                    && sc.top()->blktype() == ASTBlock::BLK_ELSE
                                                    && sc.top()->end() >= spanA)
                                                sc.pop();
                                            if (!sc.empty()) {
                                                if ((sc.top()->blktype() == ASTBlock::BLK_IF
                                                        || sc.top()->blktype() == ASTBlock::BLK_ELIF)
                                                        && sc.top()->end() == spanA)
                                                    encl = true;
                                            }
                                        }
                                        if (encl) {
                                            fire = true;
                                            openElse = true;
                                        }
                                    }
                                    /* No sibling except follows and the JUMP_FORWARD
                                       overshoots the inner handler's natural merge
                                       (spanA) because this whole try/except is the LAST
                                       statement of an ENCLOSING `except E as v:` handler
                                       — its normal-exit jump lands on the enclosing
                                       handler's `as v` name-cleanup rather than on spanA.
                                       spanA itself is a DIFFERENT copy of that same
                                       cleanup (the inner-try body-success path). Without
                                       this, the fall-through below opens a spurious
                                       swallowing bare `except:` over the cleanup copy
                                       (asyncio BaseEventLoop.call_exception_handler:
                                       `except BaseException as exc: try: … except
                                       (SystemExit,KeyboardInterrupt): raise except
                                       BaseException: log`). Detect the shape by the
                                       compiler-generated `as v` cleanup at spanA:
                                       POP_EXCEPT; LOAD_CONST None; STORE_FAST v;
                                       DELETE_FAST v (same v). Close the inner except at
                                       spanA so the enclosing handler renders its own
                                       implicit cleanup. Tightly gated to that exact
                                       byte sequence; no else opened. */
                                    if (!fire && opAtA == Pyc::POP_EXCEPT) {
                                        PycBuffer ns(code->code()->value(),
                                                code->code()->length());
                                        ns.setPos(spanA);
                                        int no, na, np = spanA;
                                        auto step = [&]() -> bool {
                                            int ip = np;
                                            if (ns.atEof()) return false;
                                            bc_next(ns, mod, no, na, np);
                                            return np > ip;
                                        };
                                        bool asVarCleanup = false;
                                        if (step() && no == Pyc::POP_EXCEPT
                                                && step() && no == Pyc::LOAD_CONST_A
                                                && step() && no == Pyc::STORE_FAST_A) {
                                            int storeVar = na;
                                            if (step() && no == Pyc::DELETE_FAST_A
                                                    && na == storeVar)
                                                asVarCleanup = true;
                                        }
                                        if (asVarCleanup)
                                            fire = true;
                                    }
                                    /* No sibling except follows and the except
                                       handler's normal-exit JUMP_FORWARD overshoots
                                       its own try/except's natural merge (spanA)
                                       because the try has an `else:` clause whose
                                       body already falls through to a bare
                                       `return None` AT spanA. The compiler emits a
                                       SECOND identical `return None` for the except
                                       handler's normal exit and the JUMP_FORWARD
                                       targets that duplicate (pos+offs) rather than
                                       spanA. Without this, the fall-through below
                                       opens a spurious swallowing bare `except:
                                       return None` over the else-return copy at
                                       spanA — dropping the else and adding a
                                       swallowing handler (mailbox._lock_file:
                                       `try: os.link…; dotlock_done=True except
                                       (AttributeError,PermissionError): os.rename…
                                       else: os.unlink`). Detect the shape by an
                                       identical `LOAD_CONST c; RETURN_VALUE` at BOTH
                                       spanA and the JUMP_FORWARD target. Close the
                                       except at spanA so the else renders and the
                                       duplicate return is skipped. Tightly gated to
                                       that exact byte pattern; no else opened. */
                                    if (!fire
                                            && opAtA == Pyc::LOAD_CONST_A) {
                                        int spanConst = aArg, spanOp = opAtA;
                                        /* op right after the LOAD_CONST at spanA */
                                        int so2 = -1, sa2 = 0, sp2 = aNext;
                                        {
                                            PycBuffer ss(code->code()->value(),
                                                    code->code()->length());
                                            ss.setPos(aNext);
                                            if (!ss.atEof())
                                                bc_next(ss, mod, so2, sa2, sp2);
                                        }
                                        /* LOAD_CONST c; RETURN_VALUE at the target */
                                        int to0 = -1, ta0 = 0, tp0 = pos + offs;
                                        int to1 = -1, ta1 = 0, tp1 = pos + offs;
                                        {
                                            PycBuffer ts(code->code()->value(),
                                                    code->code()->length());
                                            ts.setPos(pos + offs);
                                            if (!ts.atEof())
                                                bc_next(ts, mod, to0, ta0, tp0);
                                            ts.setPos(tp0);
                                            if (!ts.atEof())
                                                bc_next(ts, mod, to1, ta1, tp1);
                                        }
                                        if (so2 == Pyc::RETURN_VALUE
                                                && to0 == Pyc::LOAD_CONST_A
                                                && ta0 == spanConst
                                                && to1 == Pyc::RETURN_VALUE) {
                                            (void)spanOp;
                                            dupReturnMerge = true;
                                            dupRetSpanRet = aNext; /* RETURN_VALUE at spanA */
                                            dupRetTgtRet = tp0;    /* RETURN_VALUE at target */
                                        }
                                    }
                                    if (dupReturnMerge) {
                                        /* Close the except; merge at the try/except's
                                           natural merge (spanA = the else-body's
                                           normal-exit `return None` copy). Record both
                                           duplicate `return None` copies' RETURN_VALUE
                                           offsets (else-body copy at spanA and the
                                           except-handler copy at pos+offs) as
                                           suppressible implicit epilogues so neither
                                           renders as an explicit statement. */
                                        dupRetNoneSkip.insert(dupRetSpanRet);
                                        dupRetNoneSkip.insert(dupRetTgtRet);
                                        exceptToMerge = spanA;
                                        if (!blocks.empty()) {
                                            prev = blocks.top();
                                            continue;
                                        }
                                        prev = nil;
                                        continue;
                                    }
                                    if (fire) {
                                        exceptToMerge = spanA;
                                        if (openElse && !blocks.empty()
                                                && blocks.top()->blktype() == ASTBlock::BLK_CONTAINER) {
                                            PycRef<ASTBlock> cont = blocks.top();
                                            blocks.pop();
                                            if (!blocks.empty())
                                                blocks.top()->append(cont.cast<ASTNode>());
                                        }
                                        if (!blocks.empty()) {
                                            prev = blocks.top();
                                            continue;
                                        }
                                        prev = nil;
                                        continue;
                                    }
                                }
                            }
                        }

                        /* A try/except whose entire body lives on the true-branch of an
                           `if C:` that is itself the last statement before an ENCLOSING
                           try/finally's normal-exit jump: the try body's own normal-exit
                           JUMP_FORWARD skips the post-handler merge and jumps straight to
                           the finally, so the container (and hence this typed except) took
                           the finally target as its end. When control finally reaches that
                           JUMP_FORWARD instruction (curpos), the handler's natural merge —
                           the end A of its excHandlerSpan (T, A) — sits exactly AT curpos,
                           so the `curpos < A` span lookup above missed it and spanA came back
                           -1. Without this, the fall-through opens a spurious swallowing bare
                           `except:` spanning [curpos, finally) that absorbs the enclosing
                           `else:` branch (logging.config DictConfigurator.configure:
                           `if root: try: self.configure_root(root, True) except Exception as
                           e: raise …`). Detect the exact shape — a handler span ending at
                           curpos, the op at curpos is this JUMP_FORWARD, and the except
                           over-extended to the JUMP target (prev->end() == pos+offs > A) —
                           and close the except at A (== curpos) instead of opening a bare
                           except. Resuming at A re-processes the JUMP_FORWARD at the
                           enclosing level so the `if`/`else` structure closes normally. */
                        if (exceptToMerge < 0 && offs != 0) {
                            bool spanEndsAtCur = false;
                            for (const auto& sp : excHandlerSpan)
                                if (sp.second == curpos && sp.first < curpos) {
                                    spanEndsAtCur = true; break;
                                }
                            if (spanEndsAtCur
                                    && prev->end() == pos + offs
                                    && pos + offs > curpos) {
                                exceptToMerge = curpos;
                                if (!blocks.empty()) {
                                    prev = blocks.top();
                                    continue;
                                }
                                prev = nil;
                                continue;
                            }
                        }

                        bool lastNested = false;
                        if (prev->end() > 0 && !blocks.empty()) {
                            PycBuffer pk(code->code()->value(), code->code()->length());
                            pk.setPos(prev->end());
                            if (!pk.atEof()) {
                                int po, pa, pp;
                                bc_next(pk, mod, po, pa, pp);
                                if (po == Pyc::RERAISE || po == Pyc::RERAISE_A) {
                                    std::stack<PycRef<ASTBlock> > sc = blocks;
                                    while (!sc.empty()) {
                                        PycRef<ASTBlock> b = sc.top(); sc.pop();
                                        if (b->blktype() == ASTBlock::BLK_TRY
                                                && b->end() > pos
                                                && b->end() < pos + offs) {
                                            lastNested = true; break;
                                        }
                                        if (b->blktype() == ASTBlock::BLK_ELSE
                                                && nestedElseHandlerVals.count(b->end())
                                                && b->end() > pos
                                                && b->end() <= pos + offs) {
                                            lastNested = true; break;
                                        }
                                    }
                                }
                            }
                        }
                        if (lastNested) {
                            prev = blocks.top();
                            continue;
                        }

                        if (push) {
                            stack_hist.push(stack);
                        }
                        int sibEnd = pos + offs;
                        {
                            std::stack<PycRef<ASTBlock> > sc = blocks;
                            while (!sc.empty()) {
                                PycRef<ASTBlock> b = sc.top(); sc.pop();
                                if (b->blktype() == ASTBlock::BLK_ELSE
                                        && nestedElseHandlerVals.count(b->end())
                                        && b->end() > pos && b->end() < sibEnd) {
                                    sibEnd = b->end(); break;
                                }
                            }
                        }
                        PycRef<ASTBlock> next = new ASTCondBlock(ASTBlock::BLK_EXCEPT, sibEnd, NULL, false);
                        next->init();

                        blocks.push(next.cast<ASTBlock>());
                        prev = nil;
                    } else if (prev->blktype() == ASTBlock::BLK_ELSE) {
                        /* Special case */
                        if (blocks.empty() || (!push && stack_hist.empty()))
                            throw std::runtime_error("except-else block/stack underflow");
                        prev = blocks.top();
                        if (!push) {
                            stack = stack_hist.top();
                            stack_hist.pop();
                        }
                        push = false;

                        if (prev->blktype() == ASTBlock::BLK_MAIN) {
                            /* Something went out of control! */
                            prev = nil;
                        }
                    } else if (prev->blktype() == ASTBlock::BLK_TRY
                            && prev->end() < pos+offs) {
                        /* Need to add an except/finally block */
                        if (!stack_hist.empty()) {
                            stack = stack_hist.top();
                            stack_hist.pop();
                        }

                        if (blocks.top()->blktype() == ASTBlock::BLK_CONTAINER) {
                            PycRef<ASTContainerBlock> cont = blocks.top().cast<ASTContainerBlock>();
                            if (cont->hasExcept()) {
                                if (push) {
                                    stack_hist.push(stack);
                                }

                                PycRef<ASTBlock> except = new ASTCondBlock(ASTBlock::BLK_EXCEPT, pos+offs, NULL, false);
                                except->init();
                                blocks.push(except);
                            }
                        } else {
                            fprintf(stderr, "Something TERRIBLE happened!!\n");
                        }
                        prev = nil;
                    } else if ((prev->blktype() == ASTBlock::BLK_FOR
                                || prev->blktype() == ASTBlock::BLK_WHILE)
                            && offs != 0
                            && !blocks.empty()
                            && (blocks.top()->blktype() == ASTBlock::BLK_IF
                                || blocks.top()->blktype() == ASTBlock::BLK_ELIF)
                            && blocks.top()->end() == pos
                            && pos + offs > blocks.top()->end()) {
                        if (prev->blktype() == ASTBlock::BLK_FOR
                                && backedgeCount[prev.cast<ASTIterBlock>()->start()] == 0)
                            prev->append(new ASTKeyword(ASTKeyword::KW_BREAK));
                        prev = blocks.top();
                        continue;
                    } else {
                        prev = nil;
                    }

                } while (prev != nil);

                if (!blocks.empty()) {
                    curblock = blocks.top();
                    if (exceptToMerge < 0 && curblock->blktype() == ASTBlock::BLK_EXCEPT) {
                        int exEnd = pos + offs;
                        for (const auto& kv : forStartToExit) {
                            if (forElseMerge.count(kv.second)
                                    && forElseMerge[kv.second] == exEnd
                                    && curpos >= kv.first && curpos < kv.second) {
                                exEnd = kv.second;
                                forElseBreakLoop.insert(kv.first);
                                loopCloseAtExit.insert(kv.second);
                                break;
                            }
                        }
                        if (exEnd == pos + offs) {
                            std::stack<PycRef<ASTBlock> > sc = blocks;
                            sc.pop();
                            if (!sc.empty() && sc.top()->blktype() == ASTBlock::BLK_CONTAINER)
                                sc.pop();
                            if (!sc.empty()
                                    && sc.top()->blktype() == ASTBlock::BLK_IF
                                    && sc.top()->end() > pos
                                    && sc.top()->end() < pos + offs) {
                                int elseT = sc.top()->end(), merge = pos + offs;
                                bool simple = true;
                                PycBuffer es(code->code()->value(), code->code()->length());
                                es.setPos(elseT);
                                int eo, ea, ep = elseT;
                                while (ep < merge && !es.atEof()) {
                                    int eip = ep;
                                    bc_next(es, mod, eo, ea, ep);
                                    int tgt = -1;
                                    if (eo == Pyc::JUMP_FORWARD_A || eo == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                                            || eo == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                                            || eo == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                                            || eo == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A)
                                        tgt = ep + ea * (int)sizeof(uint16_t);
                                    else if (eo == Pyc::JUMP_BACKWARD_A
                                            || eo == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A
                                            || eo == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A)
                                        tgt = ep - ea * (int)sizeof(uint16_t);
                                    /* A jump escaping the region (past the merge) or a real
                                       backward branch means this is not a plain else arm.
                                       Multiple forward exits to the merge are fine, though:
                                       an else whose body holds its own if/try has each arm
                                       jump to the same merge (e.g. an `if … else: if X:
                                       try/except` branch). */
                                    if (tgt >= 0 && (tgt > merge || tgt <= eip)) {
                                        simple = false; break;
                                    }
                                    if (ep <= eip) break;
                                }
                                if (simple) {
                                    exceptElseAt[elseT] = merge;
                                    exEnd = elseT;
                                }
                            }
                        }
                        curblock->setEnd(exEnd);
                    }
                }
                if (exceptToMerge >= 0) {
                    source.setPos(exceptToMerge);
                    pos = exceptToMerge;
                }
            }
            break;
        case Pyc::LIST_APPEND:
        case Pyc::LIST_APPEND_A:
        case Pyc::SET_ADD_A:
        case Pyc::MAP_ADD_A:
        case Pyc::SET_UPDATE_A:
        case Pyc::LIST_EXTEND_A:
        case Pyc::DICT_UPDATE_A:
        case Pyc::DICT_MERGE_A:
            handleCollectionUpdate(opcode);
            break;
        case Pyc::LOAD_ATTR_A:
        case Pyc::LOAD_BUILD_CLASS:
            handleLoad(opcode, operand);
            break;
        case Pyc::LOAD_CLOSURE_A:
            if (mod->verCompare(3, 6) >= 0) {
                int sv = source.pos();
                int nop, narg, npos;
                bc_next(source, mod, nop, narg, npos);
                source.setPos(sv);
                if (nop == Pyc::BUILD_TUPLE_A || nop == Pyc::LOAD_CLOSURE_A)
                    stack.push(new ASTName(code->getCellVar(mod, operand)));
            }
            break;
        case Pyc::LOAD_CONST_A:
            {
                PycRef<ASTObject> t_ob = new ASTObject(code->getConst(operand));

                if ((t_ob->object().type() == PycObject::TYPE_TUPLE ||
                        t_ob->object().type() == PycObject::TYPE_SMALL_TUPLE)
                        && t_ob->object().cast<PycTuple>()->values().size()
                        && !nopsBeforeCur.empty()) {
                    /* A non-empty const tuple preceded by anchor NOPs: remember their
                       offsets in case this is a multi-line signature's defaults tuple. */
                    sigTupleNopOffs[curpos] = nopsBeforeCur;
                }

                if ((t_ob->object().type() == PycObject::TYPE_TUPLE ||
                        t_ob->object().type() == PycObject::TYPE_SMALL_TUPLE) &&
                        !t_ob->object().cast<PycTuple>()->values().size()) {
                    ASTTuple::value_t values;
                    stack.push(new ASTTuple(values));
                } else if (t_ob->object().type() == PycObject::TYPE_NONE) {
                    stack.push(NULL);
                } else {
                    stack.push(t_ob.cast<ASTNode>());
                }
            }
            break;
        case Pyc::LOAD_DEREF_A:
        case Pyc::LOAD_CLASSDEREF_A:
        case Pyc::LOAD_FAST_A:
        case Pyc::LOAD_FAST_LOAD_FAST_A:
        case Pyc::LOAD_GLOBAL_A:
        case Pyc::LOAD_LOCALS:
            handleLoad(opcode, operand);
            break;
        case Pyc::STORE_LOCALS:
            stack.pop();
            break;
        case Pyc::LOAD_METHOD_A:
        case Pyc::LOAD_NAME_A:
            handleLoad(opcode, operand);
            break;
        case Pyc::MAKE_CLOSURE_A:
        case Pyc::MAKE_FUNCTION_A:
            {
                PycRef<ASTNode> fun_code = stack.top();
                stack.pop();

                /* Test for the qualified name of the function (at TOS) */
                int tos_type = fun_code.cast<ASTObject>()->object().type();
                if (tos_type != PycObject::TYPE_CODE &&
                    tos_type != PycObject::TYPE_CODE2) {
                    fun_code = stack.top();
                    stack.pop();
                }

                ASTFunction::defarg_t defArgs, kwDefArgs;
                ASTFunction::annot_t funcAnnots;
                int constDefTupleOff = -1;
                if (mod->verCompare(3, 6) >= 0) {
                    if (operand & 0x08)
                        stack.pop();
                    if (operand & 0x04) {
                        PycRef<ASTNode> ann = stack.top();
                        stack.pop();
                        std::vector<PycRef<ASTNode>> flat;
                        if (ann != nullptr && ann.type() == ASTNode::NODE_TUPLE) {
                            for (const auto& v : ann.cast<ASTTuple>()->values())
                                flat.push_back(v);
                        } else if (ann != nullptr && ann.type() == ASTNode::NODE_OBJECT
                                   && (ann.cast<ASTObject>()->object()->type() == PycObject::TYPE_TUPLE
                                       || ann.cast<ASTObject>()->object()->type() == PycObject::TYPE_SMALL_TUPLE)) {
                            for (const auto& v : ann.cast<ASTObject>()->object()
                                                    .cast<PycTuple>()->values())
                                flat.push_back(new ASTObject(v));
                        }
                        for (size_t k = 0; k + 1 < flat.size(); k += 2) {
                            PycRef<ASTNode> key = flat[k];
                            if (key != nullptr && key.type() == ASTNode::NODE_OBJECT) {
                                PycRef<PycObject> ko = key.cast<ASTObject>()->object();
                                if (ko->type() == PycObject::TYPE_STRING
                                        || ko->type() == PycObject::TYPE_UNICODE
                                        || ko->type() == PycObject::TYPE_INTERNED
                                        || ko->type() == PycObject::TYPE_ASCII
                                        || ko->type() == PycObject::TYPE_ASCII_INTERNED
                                        || ko->type() == PycObject::TYPE_SHORT_ASCII
                                        || ko->type() == PycObject::TYPE_SHORT_ASCII_INTERNED) {
                                    PycRef<ASTNode> aval = flat[k + 1];
                                    if (aval == nullptr)
                                        aval = new ASTObject(Pyc_None);
                                    funcAnnots.push_back({ ko.cast<PycString>()->value(), aval });
                                }
                            }
                        }
                        if (ann != nullptr && ann.type() == ASTNode::NODE_CONST_MAP) {
                            const auto& mv = ann.cast<ASTConstMap>();
                            PycTuple::value_t keys = mv->keys().cast<ASTObject>()
                                    ->object().cast<PycTuple>()->values();
                            ASTConstMap::values_t vals = mv->values();
                            size_t vi = 0;
                            for (const auto& key : keys) {
                                if (vi < vals.size() && key->type() == PycObject::TYPE_STRING)
                                    funcAnnots.push_back({ key.cast<PycString>()->value(),
                                                           vals[vi] });
                                ++vi;
                            }
                        }
                    }
                    if (operand & 0x02) {
                        PycRef<ASTNode> kwd = stack.top();
                        stack.pop();
                        if (kwd != nullptr && kwd.type() == ASTNode::NODE_CONST_MAP) {
                            const auto& vals = kwd.cast<ASTConstMap>()->values();
                            for (auto it = vals.rbegin(); it != vals.rend(); ++it)
                                kwDefArgs.push_back(*it);
                        } else if (kwd != nullptr && kwd.type() == ASTNode::NODE_MAP) {
                            for (const auto& kv : kwd.cast<ASTMap>()->values())
                                kwDefArgs.push_back(kv.second);
                        } else if (kwd != nullptr && kwd.type() == ASTNode::NODE_OBJECT
                                   && kwd.cast<ASTObject>()->object()->type() == PycObject::TYPE_DICT) {
                            for (const auto& kv : kwd.cast<ASTObject>()->object()
                                                     .cast<PycDict>()->values())
                                kwDefArgs.push_back(new ASTObject(std::get<1>(kv)));
                        } else if (kwd != nullptr) {
                            kwDefArgs.push_back(kwd);
                        }
                    }
                    if (operand & 0x01) {
                        PycRef<ASTNode> defs = stack.top();
                        stack.pop();
                        if (defs != nullptr && defs.type() == ASTNode::NODE_TUPLE) {
                            for (const auto& v : defs.cast<ASTTuple>()->values())
                                defArgs.push_back(v);
                        } else if (defs != nullptr && defs.type() == ASTNode::NODE_OBJECT
                                   && (defs.cast<ASTObject>()->object()->type() == PycObject::TYPE_TUPLE
                                       || defs.cast<ASTObject>()->object()->type() == PycObject::TYPE_SMALL_TUPLE)) {
                            for (const auto& v : defs.cast<ASTObject>()->object()
                                                     .cast<PycTuple>()->values())
                                defArgs.push_back(new ASTObject(v));
                            constDefTupleOff = defs->srcOff();
                        } else if (defs != nullptr) {
                            defArgs.push_back(defs);
                        }
                    }
                } else {
                    const int defCount = operand & 0xFF;
                    const int kwDefCount = (operand >> 8) & 0xFF;
                    for (int i = 0; i < defCount; ++i) {
                        defArgs.push_front(stack.top());
                        stack.pop();
                    }
                    for (int i = 0; i < kwDefCount; ++i) {
                        kwDefArgs.push_front(stack.top());
                        stack.pop();
                    }
                }
                PycRef<ASTNode> fn = new ASTFunction(fun_code, defArgs, kwDefArgs);
                if (!funcAnnots.empty())
                    fn.cast<ASTFunction>()->setAnnotations(std::move(funcAnnots));
                /* If the positional defaults came from a const tuple preceded by a
                   run of K anchor NOPs (with no keyword-only defaults), the
                   signature wrapped across K source lines each carrying a constant
                   default: record K so it renders over K lines and the recompile
                   regenerates those K NOPs. Annotations are fine -- they emit real
                   code carrying their own positions and add no anchor NOPs, so the
                   count still depends only on the constant-default lines. */
                if (constDefTupleOff >= 0 && kwDefArgs.empty()
                        && fun_code.type() == ASTNode::NODE_OBJECT) {
                    auto it = sigTupleNopOffs.find(constDefTupleOff);
                    int K = (it != sigTupleNopOffs.end()) ? (int)it->second.size() : 0;
                    /* Accept the run of NOPs immediately before the defaults tuple as
                       multi-line signature anchors (one per source line carrying a
                       constant default) when there are at most as many as there are
                       defaults and EVERY NOP lies on the signature (line >= the
                       function's first line): a stripped statement before the `def`
                       also leaves a preceding NOP, but on an earlier line, and must
                       not trigger multi-line rendering. K == #defaults is the
                       one-per-line case; K < #defaults is a soft-wrapped signature
                       (some lines carry several defaults) reproduced by distributing
                       the defaults across K lines. */
                    if (K > 0 && K <= (int)defArgs.size()) {
                        int dl = fun_code.cast<ASTObject>()->object()
                                     .cast<PycCode>()->firstLine();
                        bool allInSig = true;
                        for (int noff : it->second)
                            if (code->lineForOffset(noff) < dl) { allInSig = false; break; }
                        if (allInSig) {
                            fn.cast<ASTFunction>()->setSigNopAnchors(K);
                            for (int noff : it->second)
                                g_sigAnchorNopOffs.insert(noff);
                        }
                    }
                }
                stack.push(fn);
            }
            break;
        case Pyc::NOP:
            break;
        case Pyc::POP_BLOCK:
            handlePopBlock();
            break;
        case Pyc::POP_EXCEPT:
            /* Do nothing. */
            break;
        case Pyc::PUSH_EXC_INFO:
            /* Python 3.11+: pushes exception info tuple. We ignore here to keep decompilation going. */
            break;
        case Pyc::CHECK_EXC_MATCH:
            {
                /* Python 3.11+: compares exception against handler type. */
                PycRef<ASTNode> right = stack.top();
                stack.pop();
                PycRef<ASTNode> left = stack.top();
                stack.pop();
                stack.push(new ASTCompare(left, right, ASTCompare::CMP_EXCEPTION));
            }
            break;
        case Pyc::END_FOR:
            {
                stack.pop();

                if ((opcode == Pyc::END_FOR) && (mod->majorVer() == 3) && (mod->minorVer() == 12)) {
                    // one additional pop for python 3.12
                    stack.pop();
                }

                // end for loop here
                /* TODO : Ensure that FOR loop ends here. 
                   Due to CACHE instructions at play, the end indicated in
                   the for loop by pycdas is not correct, it is off by
                   some small amount. */
                if (curblock->blktype() == ASTBlock::BLK_FOR) {
                    PycRef<ASTBlock> prev = blocks.top();
                    blocks.pop();

                    curblock = blocks.top();
                    curblock->append(prev.cast<ASTNode>());
                }
                else {
                    fprintf(stderr, "Wrong block type %i for END_FOR\n", curblock->blktype());
                }
            }
            break;
        case Pyc::POP_TOP:
            if (!handlePopTop())
                return new ASTNodeList(defblock->nodes());
            break;
        case Pyc::PRINT_ITEM:
        case Pyc::PRINT_ITEM_TO:
        case Pyc::PRINT_NEWLINE:
        case Pyc::PRINT_NEWLINE_TO:
            handlePrint(opcode);
            break;
        case Pyc::RAISE_VARARGS_A:
            handleRaiseVarargs(operand);
            break;
        case Pyc::RERAISE:
        case Pyc::RERAISE_A:
            handleReraise();
            /* Python 3.11 cleanup opcode. */
            break;
        case Pyc::RETURN_VALUE:
        case Pyc::INSTRUMENTED_RETURN_VALUE_A:
            {
                if (finallyReturnSkip.count(curpos)) {
                    if (!stack.empty())
                        stack.pop();
                    break;
                }
                /* Suppress an implicit-epilogue `return None` copy that is the
                   normal-exit of a try/except-with-else whose duplicate copies
                   were folded at the except-merge (see dupReturnMerge). Both the
                   else-body copy (spanA) and the except-handler copy (JUMP_FORWARD
                   target) map to the single implicit function epilogue and must
                   not render as explicit statements. */
                if (dupRetNoneSkip.count(curpos)) {
                    if (!stack.empty())
                        stack.pop();
                    break;
                }
                PycRef<ASTNode> value = stack.top();
                stack.pop();
                curblock->append(new ASTReturn(value));

                /* The returning except-clause arm is the final `else:` of the
                   if/elif/else chain that forms the except handler body, so curblock
                   is that BLK_ELSE, not the BLK_EXCEPT — the coalesced-finally
                   except-return resume below would not fire, the else would close at
                   the shared POP_EXCEPT, and the outer finally would be dropped. Close
                   the terminal else/elif into its enclosing BLK_EXCEPT first so the
                   existing resume runs and jumps to the finally target. Gated to a
                   return whose immediate parent block is the except handler, so
                   ordinary else-arms are untouched. */
                if (finallyExceptReturnExit.count(curpos)
                        && (curblock->blktype() == ASTBlock::BLK_ELSE
                            || (curblock->blktype() == ASTBlock::BLK_ELIF
                                && !elifPendingElse.count(curblock->end())))
                        && blocks.size() > 1) {
                    std::stack<PycRef<ASTBlock> > pk = blocks;
                    pk.pop();
                    if (!pk.empty() && pk.top()->blktype() == ASTBlock::BLK_EXCEPT) {
                        PycRef<ASTBlock> elseb = curblock;
                        blocks.pop();
                        if (!stack_hist.empty())
                            stack_hist.pop();
                        curblock = blocks.top();
                        curblock->append(elseb.cast<ASTNode>());
                    }
                }

                bool siblingExceptPending = false;
                if (finallyExceptReturnExit.count(curpos)
                        && curblock->blktype() == ASTBlock::BLK_EXCEPT) {
                    int mergeT = finallyExceptReturnExit[curpos];
                    PycBuffer sb(code->code()->value(), code->code()->length());
                    sb.setPos(pos);
                    int so, sa, sp = pos;
                    while (sp < mergeT && !sb.atEof()) {
                        int sip = sp;
                        bc_next(sb, mod, so, sa, sp);
                        if (so == Pyc::PUSH_EXC_INFO) break;
                        if (so == Pyc::CHECK_EXC_MATCH) { siblingExceptPending = true; break; }
                        if (sp <= sip) break;
                    }
                }
                if (finallyExceptReturnExit.count(curpos)
                        && curblock->blktype() == ASTBlock::BLK_EXCEPT
                        && !siblingExceptPending) {
                    int mergeT = finallyExceptReturnExit[curpos];
                    /* The recorded exit is the outer finally target T, which is only
                       the right resume point when this inner try/except is the LAST
                       statement of the outer try body. When more outer-try-body code
                       follows the inner try/except (e.g. `try: (try: B except E:
                       return) ; if C: return  finally: F` wrapped in a loop), resuming
                       at T skips that trailing code AND the finally render. The
                       BLK_EXCEPT's own end is the inner try/except merge; resume there
                       when it is a tighter, valid point so the rest of the try body
                       renders and pos still reaches T for the finally. */
                    if (curblock->end() > curpos && curblock->end() < mergeT)
                        mergeT = curblock->end();
                    PycRef<ASTBlock> exc = curblock;
                    blocks.pop();
                    if (!blocks.empty()) {
                        if (exc->size() != 0)
                            blocks.top()->append(exc.cast<ASTNode>());
                        curblock = blocks.top();
                    }
                    if (!stack_hist.empty())
                        stack_hist.pop();
                    if (!blocks.empty()
                            && curblock->blktype() == ASTBlock::BLK_CONTAINER
                            && !curblock.cast<ASTContainerBlock>()->hasFinally()) {
                        PycRef<ASTBlock> cont = curblock;
                        blocks.pop();
                        if (!blocks.empty()) {
                            blocks.top()->append(cont.cast<ASTNode>());
                            curblock = blocks.top();
                        }
                    }
                    source.setPos(mergeT);
                    pos = mergeT;
                    while (next_exception_entry < exception_entries.size()
                            && exception_entries[next_exception_entry].start_offset < mergeT)
                        next_exception_entry++;
                    break;
                }

                if (classCellRet.count(curpos)
                        && curpos != classCellLastRet
                        && classCellMerge > curpos
                        && curblock->end() < classCellMerge
                        && (curblock->blktype() == ASTBlock::BLK_IF
                            || curblock->blktype() == ASTBlock::BLK_ELIF)
                        && stack_hist.size()
                        && blocks.size() > 1) {
                    stack = stack_hist.top();
                    stack_hist.pop();

                    PycRef<ASTBlock> ifb = curblock;
                    blocks.pop();
                    curblock = blocks.top();
                    curblock->append(ifb.cast<ASTNode>());

                    stack_hist.push(stack);
                    PycRef<ASTBlock> elseb =
                            new ASTBlock(ASTBlock::BLK_ELSE, classCellMerge);
                    blocks.push(elseb);
                    curblock = elseb;
                    break;
                }

                /* Terminal `if C: A else: B` at module/class scope: the if-true arm A
                   just reached its threaded inline return-None (illegal as source at
                   this scope, so it is the compiler's shared-exit thread — see the
                   moduleElseAt pre-scan). Rather than close the BLK_IF into
                   fall-through (which would drop the else and mis-render B as always
                   executed), drop the synthetic return-None and open a real BLK_ELSE
                   for the false-target region. */
                if (moduleElseAt.count(curblock->end())
                        && curblock->blktype() == ASTBlock::BLK_IF
                        && (value == nullptr
                            || (value.type() == ASTNode::NODE_OBJECT
                                && value.cast<ASTObject>()->object() == Pyc_None))
                        && stack_hist.size()
                        && blocks.size() > 1) {
                    int elseEnd = moduleElseAt[curblock->end()];
                    curblock->removeLast();      // drop the threaded return-None
                    stack = stack_hist.top();
                    stack_hist.pop();
                    PycRef<ASTBlock> ifb = curblock;
                    blocks.pop();
                    curblock = blocks.top();
                    curblock->append(ifb.cast<ASTNode>());
                    stack_hist.push(stack);
                    PycRef<ASTBlock> elseb = new ASTBlock(ASTBlock::BLK_ELSE, elseEnd);
                    elseb->init();
                    blocks.push(elseb);
                    curblock = elseb;
                    break;
                }

                if ((curblock->blktype() == ASTBlock::BLK_IF
                        || curblock->blktype() == ASTBlock::BLK_ELSE)
                        && stack_hist.size()
                        && blocks.size() > 1
                        && (mod->verCompare(2, 6) >= 0)) {
                    stack = stack_hist.top();
                    stack_hist.pop();

                    PycRef<ASTBlock> prev = curblock;
                    blocks.pop();
                    curblock = blocks.top();
                    curblock->append(prev.cast<ASTNode>());

                    bool padParentOk =
                            curblock->blktype() == ASTBlock::BLK_MAIN
                            || ((curblock->blktype() == ASTBlock::BLK_IF
                                 || curblock->blktype() == ASTBlock::BLK_ELIF
                                 || curblock->blktype() == ASTBlock::BLK_ELSE
                                 || curblock->blktype() == ASTBlock::BLK_TRY)
                                && (curblock->end() == 0
                                    || curblock->end() > prev->end()));
                    if (mod->verCompare(3, 11) >= 0 && prev->end() > pos
                            && padParentOk) {
                        PycBuffer pscan(code->code()->value(), code->code()->length());
                        pscan.setPos(pos);
                        int pop = -1, parg = -1, ppos = pos;
                        bool allPads = true, sawPad = false;
                        while (ppos < prev->end()) {
                            bc_next(pscan, mod, pop, parg, ppos);
                            if (pop == Pyc::LOAD_CONST_A && ppos < prev->end()
                                    && code->getConst(parg).type()
                                        == PycObject::TYPE_NONE) {
                                bc_next(pscan, mod, pop, parg, ppos);
                                if (pop == Pyc::RETURN_VALUE) {
                                    sawPad = true;
                                    continue;
                                }
                            }
                            allPads = false;
                            break;
                        }
                        if (allPads && sawPad && ppos == prev->end()) {
                            source.setPos(prev->end());
                            pos = prev->end();
                            while (next_exception_entry < exception_entries.size()
                                    && exception_entries[next_exception_entry]
                                        .start_offset < pos)
                                next_exception_entry++;
                            break;
                        }
                    }

                    bool retIntoForLoop = curblock->blktype() == ASTBlock::BLK_FOR;
                    int forStart = retIntoForLoop
                            ? curblock.cast<ASTIterBlock>()->start() : -1;
                    int sv_bufpos = source.pos();
                    int sv_pos = pos;
                    if (source.atEof())
                        break;
                    bc_next(source, mod, opcode, operand, pos);
                    if ((opcode == Pyc::JUMP_BACKWARD_A
                            || opcode == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A)
                            && !retIntoForLoop
                            && blocks.size() > 2
                            && (curblock->blktype() == ASTBlock::BLK_IF
                                || curblock->blktype() == ASTBlock::BLK_ELIF
                                || curblock->blktype() == ASTBlock::BLK_ELSE)
                            && curblock->end() == sv_pos) {
                        int bo = operand;
                        if (mod->verCompare(3, 10) >= 0)
                            bo *= sizeof(uint16_t);
                        int btgt = pos - bo;
                        std::stack<PycRef<ASTBlock> > peek = blocks;
                        bool found = false;
                        while (peek.size() > 1
                                && (peek.top()->blktype() == ASTBlock::BLK_IF
                                    || peek.top()->blktype() == ASTBlock::BLK_ELIF
                                    || peek.top()->blktype() == ASTBlock::BLK_ELSE)
                                && peek.top()->end() == sv_pos) {
                            peek.pop();
                            if (peek.top()->blktype() == ASTBlock::BLK_FOR
                                    && peek.top().cast<ASTIterBlock>()->start() == btgt) {
                                found = true;
                                break;
                            }
                        }
                        if (found) {
                            while (blocks.size() > 1
                                    && (curblock->blktype() == ASTBlock::BLK_IF
                                        || curblock->blktype() == ASTBlock::BLK_ELIF
                                        || curblock->blktype() == ASTBlock::BLK_ELSE)
                                    && curblock->end() == sv_pos) {
                                PycRef<ASTBlock> ib = curblock;
                                if (!stack_hist.empty())
                                    stack_hist.pop();
                                blocks.pop();
                                curblock = blocks.top();
                                curblock->append(ib.cast<ASTNode>());
                            }
                            if (curblock->blktype() == ASTBlock::BLK_FOR) {
                                retIntoForLoop = true;
                                forStart = curblock.cast<ASTIterBlock>()->start();
                            }
                        }
                    }
                    if ((opcode == Pyc::JUMP_BACKWARD_A
                            || opcode == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A)
                            && retIntoForLoop) {
                        int boffs = operand;
                        if (mod->verCompare(3, 10) >= 0)
                            boffs *= sizeof(uint16_t);
                        if (pos - boffs == forStart
                                && backedgeCount[forStart] >= 1
                                && forBreakBeyondExit.count(forStart)) {
                            auto fe = forStartToExit.find(forStart);
                            if (fe != forStartToExit.end()
                                    && forElseMerge.count(fe->second)) {
                                loopCloseAtExit.insert(fe->second);
                                forElseBreakLoop.insert(forStart);
                            }
                        }
                        if (pos - boffs == forStart
                                && backedgeCount[forStart] >= 1
                                && !forBreakBeyondExit.count(forStart)) {
                            auto fe = forStartToExit.find(forStart);
                            if (fe != forStartToExit.end()) {
                                int loopExit = fe->second;
                                bool didNonConv = false;
                                if (blocks.size() > 2
                                        && curblock->blktype() == ASTBlock::BLK_FOR) {
                                    PycRef<ASTBlock> loopb = curblock;
                                    blocks.pop();
                                    PycRef<ASTBlock> parent = blocks.top();
                                    ASTBlock::BlkType pt = parent->blktype();
                                    bool staleIfNC = (pt == ASTBlock::BLK_IF
                                            || pt == ASTBlock::BLK_ELIF
                                            || pt == ASTBlock::BLK_ELSE)
                                            && parent->end() > 0
                                            && parent->end() < loopExit;
                                    if (staleIfNC) {
                                        blocks.pop();
                                        PycRef<ASTBlock> gp = blocks.top();
                                        ASTBlock::BlkType gt = gp->blktype();
                                        bool gpClean = (gt == ASTBlock::BLK_MAIN
                                                || gt == ASTBlock::BLK_IF
                                                || gt == ASTBlock::BLK_ELIF
                                                || gt == ASTBlock::BLK_ELSE
                                                || gt == ASTBlock::BLK_TRY)
                                                && (gp->end() == 0 || gp->end() > pos);
                                        if (gpClean) {
                                            parent->append(loopb.cast<ASTNode>());
                                            if (!stack_hist.empty())
                                                stack_hist.pop();
                                            gp->append(parent.cast<ASTNode>());
                                            curblock = gp;
                                            didNonConv = true;
                                        } else {
                                            blocks.push(parent);
                                        }
                                    }
                                    if (!didNonConv)
                                        blocks.push(loopb);
                                }
                                if (!didNonConv)
                                    loopCloseAtExit.insert(loopExit);
                            }
                        }
                    }
                    if ((opcode == Pyc::JUMP_BACKWARD_A
                                || opcode == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A)
                            && curblock->blktype() == ASTBlock::BLK_WHILE
                            && curblock->end() == pos
                            && blocks.size() > 1) {
                        int beTarget = pos - operand * (int)sizeof(uint16_t);
                        if (whileTrueHdr.count(beTarget)
                                && whileTrueHdr[beTarget] == pos) {
                            PycRef<ASTBlock> whb = curblock;
                            blocks.pop();
                            ASTBlock::BlkType ptt = blocks.top()->blktype();
                            bool okParent = (blocks.top()->end() == pos
                                    && (ptt == ASTBlock::BLK_TRY
                                        || ptt == ASTBlock::BLK_IF
                                        || ptt == ASTBlock::BLK_ELIF
                                        || ptt == ASTBlock::BLK_ELSE))
                                    || ptt == ASTBlock::BLK_MAIN;
                            if (okParent) {
                                curblock = blocks.top();
                                curblock->append(whb.cast<ASTNode>());
                            } else {
                                blocks.push(whb);
                            }
                        }
                    }
                    bool swallow;
                    if (mod->verCompare(3, 11) >= 0)
                        swallow = (opcode == Pyc::JUMP_BACKWARD_A
                                || opcode == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A);
                    else
                        swallow = (opcode == Pyc::JUMP_FORWARD_A
                                || opcode == Pyc::JUMP_ABSOLUTE_A
                                || opcode == Pyc::JUMP_BACKWARD_A
                                || opcode == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A);
                    if (swallow && mod->verCompare(3, 11) >= 0
                            && (curblock->blktype() == ASTBlock::BLK_IF
                                || curblock->blktype() == ASTBlock::BLK_ELIF)) {
                        int beTarget = pos - operand * (int)sizeof(uint16_t);
                        bool laterBackEdge = false;
                        for (const auto& lr : loopRanges)
                            if (lr.start == beTarget && lr.end > sv_pos) {
                                laterBackEdge = true;
                                break;
                            }
                        if (!laterBackEdge)
                            for (const auto& lr : loopRanges)
                                if (lr.start <= sv_pos && sv_pos < lr.end
                                        && beTarget < lr.start) {
                                    laterBackEdge = true;
                                    break;
                                }
                        if (laterBackEdge)
                            curblock->append(new ASTKeyword(ASTKeyword::KW_CONTINUE));
                    }
                    if (!swallow) {
                        source.setPos(sv_bufpos);
                        pos = sv_pos;
                    }
                }
            }
            break;
        case Pyc::RETURN_CONST_A:
        case Pyc::INSTRUMENTED_RETURN_CONST_A:
            {
                PycRef<ASTObject> value = new ASTObject(code->getConst(operand));
                curblock->append(new ASTReturn(value.cast<ASTNode>()));
            }
            break;
        case Pyc::ROT_TWO:
        case Pyc::ROT_THREE:
        case Pyc::ROT_FOUR:
            handleStackManip(opcode, operand);
            break;
        case Pyc::SET_LINENO_A:
            // Ignore
            break;
        case Pyc::SETUP_WITH_A:
        case Pyc::WITH_EXCEPT_START:
            {
                PycRef<ASTBlock> withblock = new ASTWithBlock(pos+operand);
                blocks.push(withblock);
                curblock = blocks.top();
            }
            break;
        case Pyc::BEFORE_WITH:
            /* Python 3.11: setup for with block; ignore. */
            break;
        case Pyc::BEFORE_ASYNC_WITH:
            {
                PycBuffer s(code->code()->value(), code->code()->length());
                s.setPos(pos);
                int so, sa, sp = pos, tgt = -1;
                for (int i = 0; i < 6 && !s.atEof(); ++i) {
                    bc_next(s, mod, so, sa, sp);
                    if (so == Pyc::SEND_A) {
                        tgt = sp + sa * (int)sizeof(uint16_t);
                        break;
                    }
                }
                /* skip a leading NOP at the await landing (try-wrapped async-with),
                   so tgt aligns with the body / withOpen key (see the asyncWithBody
                   pre-scan). */
                if (tgt >= 0) {
                    PycBuffer ns(code->code()->value(), code->code()->length());
                    ns.setPos(tgt);
                    int no, na, np = tgt;
                    if (!ns.atEof()) { bc_next(ns, mod, no, na, np);
                        if (no == Pyc::NOP) tgt = np; }
                }
                if (tgt >= 0 && asyncWithBody.count(tgt) && withOpen.count(tgt)) {
                    source.setPos(tgt);
                    pos = tgt;
                } else {
                    fprintf(stderr, "Unsupported opcode: %s (%d)\n",
                            Pyc::OpcodeName(opcode), opcode);
                    cleanBuild = false;
                    return new ASTNodeList(defblock->nodes());
                }
            }
            break;
        case Pyc::WITH_CLEANUP:
        case Pyc::WITH_CLEANUP_START:
            {
                // Stack top should be a None. Ignore it.
                PycRef<ASTNode> none = stack.top();
                stack.pop();

                if (none != NULL) {
                    fprintf(stderr, "Something TERRIBLE happened!\n");
                    break;
                }

                if (curblock->blktype() == ASTBlock::BLK_WITH
                        && curblock->end() == curpos) {
                    PycRef<ASTBlock> with = curblock;
                    blocks.pop();
                    curblock = blocks.top();
                    curblock->append(with.cast<ASTNode>());
                }
                else {
                    fprintf(stderr, "Something TERRIBLE happened! No matching with block found for WITH_CLEANUP at %d\n", curpos);
                }
            }
            break;
        case Pyc::WITH_CLEANUP_FINISH:
            /* Ignore this */
            break;
        case Pyc::SETUP_EXCEPT_A:
        case Pyc::SETUP_FINALLY_A:
        case Pyc::SETUP_LOOP_A:
            handleSetupBlock(opcode, operand);
            break;
        case Pyc::SLICE_0:
        case Pyc::SLICE_1:
        case Pyc::SLICE_2:
        case Pyc::SLICE_3:
            handleSubscript(opcode);
            break;
        case Pyc::STORE_ATTR_A:
        case Pyc::STORE_DEREF_A:
        case Pyc::STORE_FAST_A:
        case Pyc::STORE_GLOBAL_A:
        case Pyc::STORE_NAME_A:
            if (!handleStore(opcode, operand))
                return new ASTNodeList(defblock->nodes());
            break;
        case Pyc::STORE_SLICE_0:
        case Pyc::STORE_SLICE_1:
        case Pyc::STORE_SLICE_2:
        case Pyc::STORE_SLICE_3:
            handleStoreSlice(opcode);
            break;
        case Pyc::STORE_SUBSCR:
            handleStore(opcode, operand);
            break;
        case Pyc::UNARY_CALL:
        case Pyc::UNARY_CONVERT:
        case Pyc::UNARY_INVERT:
        case Pyc::UNARY_NEGATIVE:
        case Pyc::UNARY_NOT:
        case Pyc::UNARY_POSITIVE:
            handleUnaryOp(opcode);
            break;
        case Pyc::UNPACK_LIST_A:
        case Pyc::UNPACK_TUPLE_A:
        case Pyc::UNPACK_SEQUENCE_A:
        case Pyc::UNPACK_EX_A:
            handleUnpack(opcode, operand);
            break;
        case Pyc::MATCH_CLASS_A:
            {
                auto mci = matchCase.find(curpos);
                if (mci == matchCase.end()) {
                    fprintf(stderr, "Unsupported opcode: %s (%d)\n",
                            Pyc::OpcodeName(opcode), opcode);
                    cleanBuild = false;
                    return new ASTNodeList(defblock->nodes());
                }
                const MCase& mc = mci->second;
                stack.pop();
                PycRef<ASTNode> classnode = stack.top(); stack.pop();
                PycRef<ASTNode> subject = stack.top(); stack.pop();
                for (int k = 0; k < mc.popExtra; ++k)
                    if (!stack.empty()) stack.pop();
                ASTCall::pparam_t pparams;
                for (const auto& c : mc.caps)
                    pparams.push_back(c);
                PycRef<ASTNode> pattern = new ASTCall(classnode, pparams,
                                                      ASTCall::kwparam_t());
                if (mc.isFirst) {
                    blocks.push(new ASTMatchBlock(mc.matchEnd, subject));
                    curblock = blocks.top();
                }
                blocks.push(new ASTCaseBlock(mc.failTarget, pattern));
                curblock = blocks.top();
                curblock->init();
                source.setPos(mc.bodyStart);
                pos = mc.bodyStart;
            }
            break;
        case Pyc::YIELD_FROM:
        case Pyc::YIELD_VALUE:
        case Pyc::INSTRUMENTED_YIELD_VALUE_A:
            handleYield(opcode);
            break;
        case Pyc::SETUP_ANNOTATIONS:
            variable_annotations = true;
            break;
        case Pyc::PRECALL_A:
        case Pyc::RESUME_A:
        case Pyc::INSTRUMENTED_RESUME_A:
        case Pyc::MAKE_CELL_A:
        case Pyc::COPY_FREE_VARS_A:
            /* We just entirely ignore this / no-op */
            break;
        case Pyc::ASYNC_GEN_WRAP:
            break;
        case Pyc::CACHE:
            /* These "fake" opcodes are used as placeholders for optimizing
               certain opcodes in Python 3.11+.  Since we have no need for
               that during disassembly/decompilation, we can just treat these
               as no-ops. */
            break;
        case Pyc::PUSH_NULL:
            stack.push(nullptr);
            break;
        case Pyc::RETURN_GENERATOR:
            stack.push(nullptr);
            break;
        case Pyc::LIST_TO_TUPLE:
            handleCollectionUpdate(opcode);
            break;
        case Pyc::GEN_START_A:
            stack.pop();
            break;
        case Pyc::SWAP_A:
            {
                if (operand == 2) {
                    int sv_bufpos = source.pos();
                    int sv_pos = pos;
                    int nop = 0, narg = 0;
                    bc_next(source, mod, nop, narg, pos);
                    if (nop == Pyc::COPY_A && narg == 2) {
                        int afterCopy_bufpos = source.pos();
                        int afterCopy_pos = pos;
                        bool valueCtx = false;
                        for (int g = 0; g < 8; ++g) {
                            bc_next(source, mod, nop, narg, pos);
                            if (nop == Pyc::CACHE || nop == Pyc::COMPARE_OP_A)
                                continue;
                            valueCtx = (nop == Pyc::JUMP_IF_FALSE_OR_POP_A);
                            break;
                        }
                        if (valueCtx || chainIfSwap.count(curpos)) {
                            source.setPos(afterCopy_bufpos);
                            pos = afterCopy_pos;
                            chainCmp = 1;
                            break;
                        }
                    }
                    source.setPos(sv_bufpos);
                    pos = sv_pos;

                    {
                        int p_buf = source.pos(), p_pos = pos, nxop = 0, nxarg = 0;
                        bc_next(source, mod, nxop, nxarg, pos);
                        if ((nxop == Pyc::POP_TOP || nxop == Pyc::POP_EXCEPT)
                                && stack.top() != nullptr) {
                            PycRef<ASTNode> keep = stack.top();
                            stack.pop();
                            PycRef<ASTNode> below =
                                    stack.empty() ? nullptr : stack.top();
                            if (!stack.empty())
                                stack.pop();
                            // `import a.b.c as d`: CPython walks the dotted path
                            // with IMPORT_FROM/SWAP/POP_TOP, discarding each
                            // parent package and keeping the child module. The
                            // ASTImport already carries the full dotted name, so
                            // keep it (not the intermediate attribute name)
                            // through the walk; the trailing STORE_NAME/POP_TOP
                            // then renders the `import ... as ...`.
                            if (below != nullptr
                                    && below.type() == ASTNode::NODE_IMPORT) {
                                PycRef<ASTImport> imp = below.cast<ASTImport>();
                                PycRef<ASTNode> fl = imp->fromlist();
                                bool noFromlist = (fl == nullptr)
                                        || (fl.type() == ASTNode::NODE_OBJECT
                                            && fl.cast<ASTObject>()->object()
                                               == Pyc_None);
                                if (noFromlist && imp->level() == 0)
                                    keep = below;
                            }
                            stack.push(keep);
                            break;
                        }
                        source.setPos(p_buf);
                        pos = p_pos;
                    }
                }
                if (operand >= 2) {
                    int p_buf = source.pos(), p_pos = pos, nxop = 0, nxarg = 0;
                    bc_next(source, mod, nxop, nxarg, pos);
                    source.setPos(p_buf);
                    pos = p_pos;
                    if ((nxop == Pyc::UNPACK_SEQUENCE_A || nxop == Pyc::UNPACK_EX_A)
                            && stack.top(operand) != nullptr) {
                        std::vector<PycRef<ASTNode>> tmp(operand);
                        for (int i = operand - 1; i >= 0; i--) {
                            tmp[i] = stack.top();
                            stack.pop();
                        }
                        std::swap(tmp[0], tmp[operand - 1]);
                        for (int i = 0; i < operand; i++)
                            stack.push(tmp[i]);
                        break;
                    }
                }
                if (inplaceStore && operand >= 2) {
                    std::vector<PycRef<ASTNode>> tmp(operand);
                    for (int i = operand - 1; i >= 0; i--) {
                        tmp[i] = stack.top();
                        stack.pop();
                    }
                    std::swap(tmp[0], tmp[operand - 1]);
                    for (int i = 0; i < operand; i++)
                        stack.push(tmp[i]);
                    break;
                }
                unpack = operand;
                ASTTuple::value_t values;
                ASTTuple::value_t next_tuple;
                values.resize(operand);
                for (int i = 0; i < operand; i++) {
                    values[operand - i - 1] = stack.top();
                    stack.pop();
                }
                auto tup = new ASTTuple(values);
                tup->setRequireParens(false);
                auto next_tup = new ASTTuple(next_tuple);
                next_tup->setRequireParens(false);
                stack.push(tup);
                stack.push(next_tup);
            }
            break;
        case Pyc::BINARY_SLICE:
            handleSubscript(opcode);
            break;
        case Pyc::STORE_SLICE:
            {
                PycRef<ASTNode> end = stack.top();
                stack.pop();
                PycRef<ASTNode> start = stack.top();
                stack.pop();
                PycRef<ASTNode> dest = stack.top();
                stack.pop();
                PycRef<ASTNode> values = stack.top();
                stack.pop();

                if (start.type() == ASTNode::NODE_OBJECT
                        && start.cast<ASTObject>()->object() == Pyc_None) {
                    start = NULL;
                }

                if (end.type() == ASTNode::NODE_OBJECT
                        && end.cast<ASTObject>()->object() == Pyc_None) {
                    end = NULL;
                }

                PycRef<ASTNode> slice;
                if (start == NULL && end == NULL) {
                    slice = new ASTSlice(ASTSlice::SLICE0);
                } else if (start == NULL) {
                    slice = new ASTSlice(ASTSlice::SLICE2, start, end);
                } else if (end == NULL) {
                    slice = new ASTSlice(ASTSlice::SLICE1, start, end);
                } else {
                    slice = new ASTSlice(ASTSlice::SLICE3, start, end);
                }

                curblock->append(new ASTStore(values, new ASTSubscr(dest, slice)));
            }
            break;
        case Pyc::COPY_A:
            {
                if (operand == 1 && chainCopyOffsets.count(curpos)) {
                    if (stack.top().type() == ASTNode::NODE_CHAINSTORE) {
                        auto chainstore = stack.top();
                        stack.pop();
                        stack.push(stack.top());
                        stack.push(chainstore);
                    } else {
                        /* Open a new chained-assignment target list. The
                           duplicated value doubles as the completion sentinel:
                           append_to_chain_store stops when the stack top is
                           TYPE_NULL. A `None` value is pushed as a NULL ref (see
                           LOAD_CONST), which would trip that sentinel and split
                           `a = b = None` into separate statements. Since
                           chainCopyOffsets has confirmed this is a chained
                           assign, replace the NULL with an explicit None node
                           (renders as `None`, sentinel-safe). */
                        if (stack.top() == nullptr) {
                            stack.pop();
                            stack.push(new ASTObject(Pyc_None));
                        }
                        stack.push(stack.top());
                        ASTNodeList::list_t targets;
                        stack.push(new ASTChainStore(targets, stack.top()));
                    }
                } else if (operand == 1 && walrusCopies.count(curpos)) {
                } else {
                    PycRef<ASTNode> value = stack.top(operand);
                    stack.push(value);
                }
            }
            break;
        default:
            fprintf(stderr, "Unsupported opcode: %s (%d)\n", Pyc::OpcodeName(opcode), opcode);
            cleanBuild = false;
            return new ASTNodeList(defblock->nodes());
        }

        else_pop =  ( (curblock->blktype() == ASTBlock::BLK_ELSE)
                      || (curblock->blktype() == ASTBlock::BLK_IF)
                      || (curblock->blktype() == ASTBlock::BLK_ELIF) )
                 && (curblock->end() == pos);

        if (opcode != Pyc::PRECALL_A && opcode != Pyc::CACHE)
            lastSubstantialOp = opcode;
    }
    g_stmtOff = -1;   // decode loop done; nodes built past here are not statements
    g_stmtLine = -1;
    g_stmtCol = -1;

    if (stack_hist.size()) {
        fputs("Warning: Stack history is not empty!\n", stderr);

        while (stack_hist.size()) {
            stack_hist.pop();
        }
    }
    } catch (const std::exception&) {
        cleanBuild = false;
        return new ASTNodeList(defblock->nodes());
    }

    if (blocks.size() > 1) {
        fputs("Warning: block stack is not empty!\n", stderr);

        while (blocks.size() > 1) {
            PycRef<ASTBlock> tmp = blocks.top();
            blocks.pop();

            blocks.top()->append(tmp.cast<ASTNode>());
        }
    }

    if (!stack.empty()) {
        PycRef<ASTNode> top = stack.top();
        if (top != nullptr && top.type() == ASTNode::NODE_COMPREHENSION)
            defblock->append(top);
    }

    /* --- epilogue: the instruction stream is exhausted. Any blocks still open
     * were closed as their ends were reached during the loop; return the top
     * block's statements as the module/function body. cleanBuild=true records
     * that decompilation reached the end without bailing. */
    cleanBuild = true;
    return new ASTNodeList(defblock->nodes());
}

/* Simultaneous tuple assignment (`t1, ..., tN = v1, ..., vN`) helpers.
 * tupleAssignSafe checks that the N values on top of the stack are genuine,
 * independent right-hand-side values (not a chained-store or import node) and
 * that we are not in the middle of filling a for/with target -- i.e. that the
 * run of stores really is one parallel assignment and safe to fold. */
bool CodeBuilder::tupleAssignSafe(int K)
{
    if (!curblock->inited()
            && (curblock->blktype() == ASTBlock::BLK_FOR
                || curblock->blktype() == ASTBlock::BLK_ASYNCFOR
                || curblock->blktype() == ASTBlock::BLK_WITH))
        return false;
    for (int i = 1; i <= K; i++) {
        PycRef<ASTNode> v = stack.top(i);
        if (v == nullptr
                || v.type() == ASTNode::NODE_CHAINSTORE
                || v.type() == ASTNode::NODE_IMPORT)
            return false;
    }
    return true;
}

/* Fold one store of the active tuple-assignment run. Each call records one
 * target/value pair (they arrive in bytecode/pop order); on the final store
 * (tupleStore reaches 0) it reverses both back into source order and appends a
 * single `t1, ..., tN = v1, ..., vN` statement. */
void CodeBuilder::tupleStoreStep(PycRef<ASTNode> tname)
{
    PycRef<ASTNode> tval = stack.top();
    stack.pop();
    tupleStoreTargets.push_back(tname);
    tupleStoreValues.push_back(tval);
    if (--tupleStore == 0) {
        ASTTuple::value_t tgs, vls;
        for (auto it = tupleStoreTargets.rbegin(); it != tupleStoreTargets.rend(); ++it)
            tgs.push_back(*it);
        for (auto it = tupleStoreValues.rbegin(); it != tupleStoreValues.rend(); ++it)
            vls.push_back(*it);
        PycRef<ASTTuple> tgt = new ASTTuple(tgs);
        tgt->setRequireParens(false);
        PycRef<ASTTuple> val = new ASTTuple(vls);
        val->setRequireParens(false);
        curblock->append(new ASTStore(val.cast<ASTNode>(), tgt.cast<ASTNode>()));
        tupleStoreTargets.clear();
        tupleStoreValues.clear();
    }
}

/* The assignment-target stores. The name/local/global/closure forms
 * (STORE_FAST/NAME/GLOBAL/DEREF) share one shape and differ only in how the
 * target is looked up (getLocal / getName / getCellVar) plus a few per-opcode
 * extras; STORE_ATTR and STORE_SUBSCR store into `obj.attr` / `obj[key]` and add
 * the variable-annotation and dict-item paths. For the name forms, in order the
 * store:
 *   1. clears inplaceStore (a plain store is not an augmented assignment);
 *   2. if this offset is a walrus target, pushes an `(name := value)` store
 *      expression back onto the stack instead of emitting a statement;
 *   3. if this offset begins a foldable parallel assignment, folds it via
 *      tupleStoreStep into one `t1, ..., tN = v1, ..., vN`;
 *   4. if a sequence unpack is in progress, adds the target into the tuple
 *      pattern (handling a starred target and nested tuples), and on the last
 *      element emits the unpack -- or wires it as a for/with target;
 *   5. otherwise emits a normal `name = value` store, with special cases for a
 *      for/with target, binding to an in-progress import, extending a chained
 *      assignment, and attaching pending decorators.
 * STORE_NAME additionally drops list-comp temporaries and __classcell__ and
 * un-mangles private names; STORE_GLOBAL marks the name global. Returns false
 * (after setting cleanBuild) when a comprehension turns up as a with-manager,
 * which cannot be reconstructed -- the caller then returns the partial tree. */
bool CodeBuilder::handleStore(int opcode, int operand)
{
    switch (opcode) {
        case Pyc::STORE_DEREF_A:
            {
                inplaceStore = 0;
                if (walrusStores.count(curpos)) {
                    PycRef<ASTNode> value = stack.top(); stack.pop();
                    stack.push(new ASTStore(value, new ASTName(code->getCellVar(mod, operand)), true));
                    break;
                }
                if (tupleStore == 0 && unpack == 0 && tupleAssignStart.count(curpos)
                        && tupleAssignSafe(tupleAssignStart[curpos]))
                    tupleStore = tupleAssignStart[curpos];
                if (tupleStore > 0) {
                    tupleStoreStep(new ASTName(code->getCellVar(mod, operand)));
                    break;
                }
                if (unpack) {
                    PycRef<ASTNode> name = new ASTName(code->getCellVar(mod, operand));
                    if (unpack == unpackStar)
                        name = new ASTUnary(name, ASTUnary::UN_STAR);

                    PycRef<ASTNode> tup = stack.top();
                    if (tup.type() == ASTNode::NODE_TUPLE)
                        tup.cast<ASTTuple>()->add(name);
                    else
                        fputs("Something TERRIBLE happened!\n", stderr);

                    --unpack;
                    while (unpack <= 0 && !unpackNest.empty()) {
                        PycRef<ASTNode> inner = stack.top(); stack.pop();
                        tup = stack.top();
                        if (tup.type() == ASTNode::NODE_TUPLE)
                            tup.cast<ASTTuple>()->add(inner);
                        unpack = unpackNest.back() - 1; unpackNest.pop_back();
                    }
                    if (unpack <= 0) {
                        stack.pop();
                        PycRef<ASTNode> seq = stack.top();
                        stack.pop();

                        if (curblock->blktype() == ASTBlock::BLK_FOR
                                && !curblock->inited()) {
                            PycRef<ASTTuple> tuple = tup.try_cast<ASTTuple>();
                            if (tuple != NULL)
                                tuple->setRequireParens(false);
                            curblock.cast<ASTIterBlock>()->setIndex(tup);
                        } else if (curblock->blktype() == ASTBlock::BLK_WITH
                                   && !curblock->inited()) {
                            PycRef<ASTTuple> tuple = tup.try_cast<ASTTuple>();
                            if (tuple != NULL)
                                tuple->setRequireParens(true);
                            curblock.cast<ASTWithBlock>()->setExpr(seq);
                            curblock.cast<ASTWithBlock>()->setVar(tup);
                        } else if (seq.type() == ASTNode::NODE_CHAINSTORE) {
                            append_to_chain_store(seq, tup, stack, curblock);
                        } else {
                            curblock->append(new ASTStore(seq, tup));
                        }
                    }
                } else {
                    PycRef<ASTNode> value = stack.top();
                    stack.pop();
                    PycRef<ASTNode> name = new ASTName(code->getCellVar(mod, operand));

                    if ((curblock->blktype() == ASTBlock::BLK_FOR
                                || curblock->blktype() == ASTBlock::BLK_ASYNCFOR)
                            && !curblock->inited()) {
                        curblock.cast<ASTIterBlock>()->setIndex(name);
                    } else if (curblock->blktype() == ASTBlock::BLK_WITH
                               && !curblock->inited()) {
                        /* `with <mgr> as <cellvar>:` — the `as` target is a closure
                           cell (an inner comprehension/function captures it), stored
                           via STORE_DEREF. Set the with's expr+var like the STORE_FAST
                           target, else the manager is lost and the with mis-renders. */
                        if (value != nullptr && value.type() == ASTNode::NODE_COMPREHENSION) {
                            cleanBuild = false;
                            return false;
                        }
                        curblock.cast<ASTWithBlock>()->setExpr(value);
                        curblock.cast<ASTWithBlock>()->setVar(name);
                    } else if (value.type() == ASTNode::NODE_CHAINSTORE) {
                        append_to_chain_store(value, name, stack, curblock);
                    } else if (!tryAttachDecorators(curblock, value, name)) {
                        curblock->append(new ASTStore(value, name));
                    }
                }
            }
            break;
        case Pyc::STORE_FAST_A:
            {
                inplaceStore = 0;
                if (walrusStores.count(curpos)) {
                    PycRef<ASTNode> value = stack.top(); stack.pop();
                    PycRef<ASTNode> wname = (mod->verCompare(1, 3) < 0)
                        ? new ASTName(code->getName(operand))
                        : new ASTName(code->getLocal(operand));
                    stack.push(new ASTStore(value, wname, true));
                    break;
                }
                if (tupleStore == 0 && unpack == 0 && tupleAssignStart.count(curpos)
                        && tupleAssignSafe(tupleAssignStart[curpos]))
                    tupleStore = tupleAssignStart[curpos];
                if (tupleStore > 0) {
                    tupleStoreStep((mod->verCompare(1, 3) < 0)
                        ? new ASTName(code->getName(operand))
                        : new ASTName(code->getLocal(operand)));
                    break;
                }
                if (unpack) {
                    PycRef<ASTNode> name;

                    if (mod->verCompare(1, 3) < 0)
                        name = new ASTName(code->getName(operand));
                    else
                        name = new ASTName(code->getLocal(operand));
                    if (unpack == unpackStar)
                        name = new ASTUnary(name, ASTUnary::UN_STAR);

                    PycRef<ASTNode> tup = stack.top();
                    if (tup.type() == ASTNode::NODE_TUPLE)
                        tup.cast<ASTTuple>()->add(name);
                    else
                        fputs("Something TERRIBLE happened!\n", stderr);

                    --unpack;
                    while (unpack <= 0 && !unpackNest.empty()) {
                        PycRef<ASTNode> inner = stack.top(); stack.pop();
                        tup = stack.top();
                        if (tup.type() == ASTNode::NODE_TUPLE)
                            tup.cast<ASTTuple>()->add(inner);
                        unpack = unpackNest.back() - 1; unpackNest.pop_back();
                    }
                    if (unpack <= 0) {
                        stack.pop();
                        PycRef<ASTNode> seq = stack.top();
                        stack.pop();

                        if (curblock->blktype() == ASTBlock::BLK_FOR
                                && !curblock->inited()) {
                            PycRef<ASTTuple> tuple = tup.try_cast<ASTTuple>();
                            if (tuple != NULL)
                                tuple->setRequireParens(false);
                            curblock.cast<ASTIterBlock>()->setIndex(tup);
                        } else if (curblock->blktype() == ASTBlock::BLK_WITH
                                   && !curblock->inited()) {
                            PycRef<ASTTuple> tuple = tup.try_cast<ASTTuple>();
                            if (tuple != NULL)
                                tuple->setRequireParens(true);
                            curblock.cast<ASTWithBlock>()->setExpr(seq);
                            curblock.cast<ASTWithBlock>()->setVar(tup);
                        } else if (seq.type() == ASTNode::NODE_CHAINSTORE) {
                            append_to_chain_store(seq, tup, stack, curblock);
                        } else {
                            curblock->append(new ASTStore(seq, tup));
                        }
                    }
                } else {
                    PycRef<ASTNode> value = stack.top();
                    stack.pop();
                    PycRef<ASTNode> name;

                    if (mod->verCompare(1, 3) < 0)
                        name = new ASTName(code->getName(operand));
                    else
                        name = new ASTName(code->getLocal(operand));

                    if (name.cast<ASTName>()->name()->value()[0] == '_'
                            && name.cast<ASTName>()->name()->value()[1] == '[') {
                        /* Don't show stores of list comp append objects. */
                        break;
                    }

                    if ((curblock->blktype() == ASTBlock::BLK_FOR
                                || curblock->blktype() == ASTBlock::BLK_ASYNCFOR)
                            && !curblock->inited()) {
                        curblock.cast<ASTIterBlock>()->setIndex(name);
                    } else if (!stack.empty() && stack.top() != nullptr
                               && stack.top().type() == ASTNode::NODE_IMPORT) {
                        stack.top().cast<ASTImport>()->add_store(new ASTStore(value, name));
                    } else if (curblock->blktype() == ASTBlock::BLK_WITH
                                   && !curblock->inited()) {
                        if (value != nullptr && value.type() == ASTNode::NODE_COMPREHENSION) {
                            cleanBuild = false;
                            return false;
                        }
                        curblock.cast<ASTWithBlock>()->setExpr(value);
                        curblock.cast<ASTWithBlock>()->setVar(name);
                    } else if (value.type() == ASTNode::NODE_CHAINSTORE) {
                        append_to_chain_store(value, name, stack, curblock);
                    } else if (!tryAttachDecorators(curblock, value, name)) {
                        curblock->append(new ASTStore(value, name));
                    }
                }
            }
            break;
        case Pyc::STORE_GLOBAL_A:
            {
                inplaceStore = 0;
                if (walrusStores.count(curpos)) {
                    PycRef<ASTNode> value = stack.top(); stack.pop();
                    stack.push(new ASTStore(value, new ASTName(code->getName(operand)), true));
                    break;
                }
                if (tupleStore == 0 && unpack == 0 && tupleAssignStart.count(curpos)
                        && tupleAssignSafe(tupleAssignStart[curpos]))
                    tupleStore = tupleAssignStart[curpos];
                if (tupleStore > 0) {
                    tupleStoreStep(new ASTName(code->getName(operand)));
                    break;
                }
                PycRef<ASTNode> name = new ASTName(code->getName(operand));

                if (unpack) {
                    if (unpack == unpackStar)
                        name = new ASTUnary(name, ASTUnary::UN_STAR);
                    PycRef<ASTNode> tup = stack.top();
                    if (tup.type() == ASTNode::NODE_TUPLE)
                        tup.cast<ASTTuple>()->add(name);
                    else
                        fputs("Something TERRIBLE happened!\n", stderr);

                    --unpack;
                    while (unpack <= 0 && !unpackNest.empty()) {
                        PycRef<ASTNode> inner = stack.top(); stack.pop();
                        tup = stack.top();
                        if (tup.type() == ASTNode::NODE_TUPLE)
                            tup.cast<ASTTuple>()->add(inner);
                        unpack = unpackNest.back() - 1; unpackNest.pop_back();
                    }
                    if (unpack <= 0) {
                        stack.pop();
                        PycRef<ASTNode> seq = stack.top();
                        stack.pop();

                        if (curblock->blktype() == ASTBlock::BLK_FOR
                                && !curblock->inited()) {
                            PycRef<ASTTuple> tuple = tup.try_cast<ASTTuple>();
                            if (tuple != NULL)
                                tuple->setRequireParens(false);
                            curblock.cast<ASTIterBlock>()->setIndex(tup);
                        } else if (curblock->blktype() == ASTBlock::BLK_WITH
                                   && !curblock->inited()) {
                            PycRef<ASTTuple> tuple = tup.try_cast<ASTTuple>();
                            if (tuple != NULL)
                                tuple->setRequireParens(true);
                            curblock.cast<ASTWithBlock>()->setExpr(seq);
                            curblock.cast<ASTWithBlock>()->setVar(tup);
                        } else if (seq.type() == ASTNode::NODE_CHAINSTORE) {
                            append_to_chain_store(seq, tup, stack, curblock);
                        } else {
                            curblock->append(new ASTStore(seq, tup));
                        }
                    }
                } else {
                    PycRef<ASTNode> value = stack.top();
                    stack.pop();
                    if (!stack.empty() && stack.top() != nullptr
                            && stack.top().type() == ASTNode::NODE_IMPORT) {
                        /* `from mod import name` where `name` is declared `global`:
                           the IMPORT_FROM value binds to the import (mirrors the
                           STORE_NAME/STORE_FAST path) rather than a bare `name = name`. */
                        stack.top().cast<ASTImport>()->add_store(new ASTStore(value, name));
                    } else if (value.type() == ASTNode::NODE_CHAINSTORE) {
                        append_to_chain_store(value, name, stack, curblock);
                    } else if (!tryAttachDecorators(curblock, value, name)) {
                        curblock->append(new ASTStore(value, name));
                    }
                }

                /* Mark the global as used */
                code->markGlobal(name.cast<ASTName>()->name());
            }
            break;
        case Pyc::STORE_NAME_A:
            {
                inplaceStore = 0;
                if (walrusStores.count(curpos)) {
                    PycRef<ASTNode> value = stack.top(); stack.pop();
                    stack.push(new ASTStore(value, new ASTName(code->getName(operand)), true));
                    break;
                }
                if (tupleStore == 0 && unpack == 0 && tupleAssignStart.count(curpos)
                        && tupleAssignSafe(tupleAssignStart[curpos]))
                    tupleStore = tupleAssignStart[curpos];
                if (tupleStore > 0) {
                    tupleStoreStep(new ASTName(code->getName(operand)));
                    break;
                }
                if (unpack) {
                    PycRef<ASTNode> name = new ASTName(code->getName(operand));
                    if (unpack == unpackStar)
                        name = new ASTUnary(name, ASTUnary::UN_STAR);

                    PycRef<ASTNode> tup = stack.top();
                    if (tup.type() == ASTNode::NODE_TUPLE)
                        tup.cast<ASTTuple>()->add(name);
                    else
                        fputs("Something TERRIBLE happened!\n", stderr);

                    --unpack;
                    while (unpack <= 0 && !unpackNest.empty()) {
                        PycRef<ASTNode> inner = stack.top(); stack.pop();
                        tup = stack.top();
                        if (tup.type() == ASTNode::NODE_TUPLE)
                            tup.cast<ASTTuple>()->add(inner);
                        unpack = unpackNest.back() - 1; unpackNest.pop_back();
                    }
                    if (unpack <= 0) {
                        stack.pop();
                        PycRef<ASTNode> seq = stack.top();
                        stack.pop();

                        if (curblock->blktype() == ASTBlock::BLK_FOR
                                && !curblock->inited()) {
                            PycRef<ASTTuple> tuple = tup.try_cast<ASTTuple>();
                            if (tuple != NULL)
                                tuple->setRequireParens(false);
                            curblock.cast<ASTIterBlock>()->setIndex(tup);
                        } else if (curblock->blktype() == ASTBlock::BLK_WITH
                                   && !curblock->inited()) {
                            PycRef<ASTTuple> tuple = tup.try_cast<ASTTuple>();
                            if (tuple != NULL)
                                tuple->setRequireParens(true);
                            curblock.cast<ASTWithBlock>()->setExpr(seq);
                            curblock.cast<ASTWithBlock>()->setVar(tup);
                        } else if (seq.type() == ASTNode::NODE_CHAINSTORE) {
                            append_to_chain_store(seq, tup, stack, curblock);
                        } else {
                            curblock->append(new ASTStore(seq, tup));
                        }
                    }
                } else {
                    PycRef<ASTNode> value = stack.top();
                    stack.pop();

                    PycRef<PycString> varname = code->getName(operand);
                    if (varname->length() >= 2 && varname->value()[0] == '_'
                            && varname->value()[1] == '[') {
                        /* Don't show stores of list comp append objects. */
                        break;
                    }
                    if (varname->isEqual("__classcell__"))
                        break;

                    // Return private names back to their original name
                    const std::string class_prefix = std::string("_") + code->name()->strValue();
                    if (varname->startsWith(class_prefix + std::string("__")))
                        varname->setValue(varname->strValue().substr(class_prefix.size()));

                    PycRef<ASTNode> name = new ASTName(varname);

                    if ((curblock->blktype() == ASTBlock::BLK_FOR
                                || curblock->blktype() == ASTBlock::BLK_ASYNCFOR)
                            && !curblock->inited()) {
                        curblock.cast<ASTIterBlock>()->setIndex(name);
                    } else if (stack.top().type() == ASTNode::NODE_IMPORT) {
                        PycRef<ASTImport> import = stack.top().cast<ASTImport>();

                        import->add_store(new ASTStore(value, name));
                    } else if (curblock->blktype() == ASTBlock::BLK_WITH
                               && !curblock->inited()) {
                        if (value != nullptr && value.type() == ASTNode::NODE_COMPREHENSION) {
                            cleanBuild = false;
                            return false;
                        }
                        curblock.cast<ASTWithBlock>()->setExpr(value);
                        curblock.cast<ASTWithBlock>()->setVar(name);
                    } else if (value.type() == ASTNode::NODE_CHAINSTORE) {
                        append_to_chain_store(value, name, stack, curblock);
                    } else if (!tryAttachDecorators(curblock, value, name)) {
                        curblock->append(new ASTStore(value, name));

                        if (value.type() == ASTNode::NODE_INVALID)
                            break;
                    }
                }
            }
            break;
        case Pyc::STORE_ATTR_A:
            {
                inplaceStore = 0;
                if (unpack) {
                    PycRef<ASTNode> name = stack.top();
                    stack.pop();
                    PycRef<ASTNode> attr = new ASTBinary(name, new ASTName(code->getName(operand)), ASTBinary::BIN_ATTR);
                    if (unpack == unpackStar)
                        attr = new ASTUnary(attr, ASTUnary::UN_STAR);

                    PycRef<ASTNode> tup = stack.top();
                    if (tup.type() == ASTNode::NODE_TUPLE)
                        tup.cast<ASTTuple>()->add(attr);
                    else
                        fputs("Something TERRIBLE happened!\n", stderr);

                    --unpack;
                    while (unpack <= 0 && !unpackNest.empty()) {
                        PycRef<ASTNode> inner = stack.top(); stack.pop();
                        tup = stack.top();
                        if (tup.type() == ASTNode::NODE_TUPLE)
                            tup.cast<ASTTuple>()->add(inner);
                        unpack = unpackNest.back() - 1; unpackNest.pop_back();
                    }
                    if (unpack <= 0) {
                        stack.pop();
                        PycRef<ASTNode> seq = stack.top();
                        stack.pop();
                        if (seq.type() == ASTNode::NODE_CHAINSTORE) {
                            append_to_chain_store(seq, tup, stack, curblock);
                        } else {
                            curblock->append(new ASTStore(seq, tup));
                        }
                    }
                } else {
                    PycRef<ASTNode> name = stack.top();
                    stack.pop();
                    PycRef<ASTNode> value = stack.top();
                    stack.pop();
                    PycRef<ASTNode> attr = new ASTBinary(name, new ASTName(code->getName(operand)), ASTBinary::BIN_ATTR);
                    if ((curblock->blktype() == ASTBlock::BLK_FOR
                                || curblock->blktype() == ASTBlock::BLK_ASYNCFOR)
                            && !curblock->inited()) {
                        curblock.cast<ASTIterBlock>()->setIndex(attr);
                    } else if (value.type() == ASTNode::NODE_CHAINSTORE) {
                        append_to_chain_store(value, attr, stack, curblock);
                    } else {
                        curblock->append(new ASTStore(value, attr));
                    }
                }
            }
            break;
        case Pyc::STORE_SUBSCR:
            {
                inplaceStore = 0;
                if (unpack) {
                    PycRef<ASTNode> subscr = stack.top();
                    stack.pop();
                    PycRef<ASTNode> dest = stack.top();
                    stack.pop();

                    PycRef<ASTNode> save = new ASTSubscr(dest, subscr);
                    if (unpack == unpackStar)
                        save = new ASTUnary(save, ASTUnary::UN_STAR);

                    PycRef<ASTNode> tup = stack.top();
                    if (tup.type() == ASTNode::NODE_TUPLE)
                        tup.cast<ASTTuple>()->add(save);
                    else
                        fputs("Something TERRIBLE happened!\n", stderr);

                    --unpack;
                    while (unpack <= 0 && !unpackNest.empty()) {
                        PycRef<ASTNode> inner = stack.top(); stack.pop();
                        tup = stack.top();
                        if (tup.type() == ASTNode::NODE_TUPLE)
                            tup.cast<ASTTuple>()->add(inner);
                        unpack = unpackNest.back() - 1; unpackNest.pop_back();
                    }
                    if (unpack <= 0) {
                        stack.pop();
                        PycRef<ASTNode> seq = stack.top();
                        stack.pop();
                        if (seq.type() == ASTNode::NODE_CHAINSTORE) {
                            append_to_chain_store(seq, tup, stack, curblock);
                        } else {
                            curblock->append(new ASTStore(seq, tup));
                        }
                    }
                } else {
                    PycRef<ASTNode> subscr = stack.top();
                    stack.pop();
                    PycRef<ASTNode> dest = stack.top();
                    stack.pop();
                    PycRef<ASTNode> src = stack.top();
                    stack.pop();

                    // If variable annotations are enabled, we'll need to check for them here.
                    // Python handles a varaible annotation by setting:
                    // __annotations__['var-name'] = type
                    const bool found_annotated_var = (variable_annotations && dest->type() == ASTNode::Type::NODE_NAME
                                                      && dest.cast<ASTName>()->name()->isEqual("__annotations__"));

                    if (found_annotated_var) {
                        // Annotations can be done alone or as part of an assignment.
                        PycRef<ASTStore> store = (!curblock->nodes().empty()
                                && curblock->nodes().back()->type() == ASTNode::Type::NODE_STORE)
                                ? curblock->nodes().back().cast<ASTStore>() : nullptr;
                        bool sameTarget = false;
                        if (store != nullptr && store->dest() != nullptr
                                && store->dest().type() == ASTNode::NODE_NAME
                                && subscr != nullptr && subscr.type() == ASTNode::NODE_OBJECT) {
                            sameTarget = store->dest().cast<ASTName>()->name()
                                    ->isEqual(subscr.cast<ASTObject>()->object());
                        }
                        if (sameTarget) {
                            // Replace the existing NODE_STORE with a new one that includes the annotation.
                            curblock->removeLast();
                            curblock->append(new ASTStore(store->src(),
                                                          new ASTAnnotatedVar(subscr, src)));
                        } else {
                            curblock->append(new ASTAnnotatedVar(subscr, src));
                        }
                    } else {
                        if (curblock->blktype() == ASTBlock::BLK_FOR
                                && !curblock->inited()) {
                            curblock.cast<ASTIterBlock>()->setIndex(new ASTSubscr(dest, subscr));
                        } else if (dest.type() == ASTNode::NODE_MAP) {
                            dest.cast<ASTMap>()->add(subscr, src);
                        } else if (src.type() == ASTNode::NODE_CHAINSTORE) {
                            append_to_chain_store(src, new ASTSubscr(dest, subscr), stack, curblock);
                        } else {
                            curblock->append(new ASTStore(src, new ASTSubscr(dest, subscr)));
                        }
                    }
                }
            }
            break;
    }
    return true;
}

/* Open a sequence-unpacking target (`a, b, *c = ...` or a for/with target).
 * UNPACK_SEQUENCE/LIST/TUPLE set `unpack` to the element count and push an empty
 * tuple that the following stores fill; a nested unpack first saves the outer
 * count on unpackNest. Unpacking zero elements has no stores to follow, so the
 * empty `()`/`[]` target is wired up immediately (for-target, chained store, or
 * a plain store). UNPACK_EX (`a, *b, c = ...`) splits the operand into the
 * before/after counts, records the starred position in unpackStar, and pushes
 * the accumulator tuple. */
void CodeBuilder::handleUnpack(int opcode, int operand)
{
    switch (opcode) {
    case Pyc::UNPACK_LIST_A:
    case Pyc::UNPACK_TUPLE_A:
    case Pyc::UNPACK_SEQUENCE_A:
        {
            if (unpack > 0 && operand > 0)
                unpackNest.push_back(unpack);
            unpack = operand;
            unpackStar = -1;
            if (unpack > 0) {
                ASTTuple::value_t vals;
                stack.push(new ASTTuple(vals));
            } else {
                // Unpack zero values and assign it to top of stack or for loop variable.
                // E.g. [] = TOS / for [] in X
                ASTTuple::value_t vals;
                auto tup = new ASTTuple(vals);
                if (curblock->blktype() == ASTBlock::BLK_FOR
                    && !curblock->inited()) {
                    tup->setRequireParens(true);
                    curblock.cast<ASTIterBlock>()->setIndex(tup);
                } else if (stack.top().type() == ASTNode::NODE_CHAINSTORE) {
                    auto chainStore = stack.top();
                    stack.pop();
                    append_to_chain_store(chainStore, tup, stack, curblock);
                } else {
                    curblock->append(new ASTStore(stack.top(), tup));
                    stack.pop();
                }
            }
        }
        break;
    case Pyc::UNPACK_EX_A:
        {
            int countBefore = operand & 0xFF;
            int countAfter = (operand >> 8) & 0xFF;
            unpack = countBefore + 1 + countAfter;
            unpackStar = countAfter + 1;
            ASTTuple::value_t vals;
            stack.push(new ASTTuple(vals));
        }
        break;
    }
}

/* The yield opcodes, which all build an ASTReturn tagged with a yield flavour.
 *   YIELD_FROM  -> `yield from <iter>`; pops the exhausted sub-iterator and
 *       emits the yield-from of the value beneath it (a non-null send target is
 *       not yet supported).
 *   YIELD_VALUE -> inside a generator comprehension it becomes the
 *       COMP_GENERATOR result; otherwise it is a `yield`. A yield whose sent
 *       value is consumed (not immediately POP_TOP'd, Python 2.5+ PEP 342) is a
 *       yield EXPRESSION left on the stack; otherwise it is a yield statement. */
void CodeBuilder::handleYield(int opcode)
{
    switch (opcode) {
    case Pyc::YIELD_FROM:
        {
            PycRef<ASTNode> dest = stack.top();
            stack.pop();
            // TODO: Support yielding into a non-null destination
            PycRef<ASTNode> value = stack.top();
            if (value) {
                value->setProcessed();
                curblock->append(new ASTReturn(value, ASTReturn::YIELD_FROM));
            }
        }
        break;
    case Pyc::YIELD_VALUE:
    case Pyc::INSTRUMENTED_YIELD_VALUE_A:
        {
            PycRef<ASTNode> value = stack.top();
            stack.pop();
            if (curblock->blktype() == ASTBlock::BLK_FOR
                    && curblock.cast<ASTIterBlock>()->isComprehension()) {
                stack.push(new ASTComprehension(value,
                        ASTComprehension::COMP_GENERATOR));
            } else {
                bool resultUsed = false;
                /* `yield` as an expression (its sent-value result consumed) only
                   exists from Python 2.5 (PEP 342); before that a yield leaves no
                   usable value, so it is always a statement. */
                if (mod->verCompare(2, 5) >= 0) {
                    PycBuffer pk(code->code()->value(), code->code()->length());
                    pk.setPos(pos);
                    int po = -1, pa, pp = pos, steps = 0;
                    while (!pk.atEof() && steps++ < 4) {
                        bc_next(pk, mod, po, pa, pp);
                        if (po == Pyc::RESUME_A || po == Pyc::CACHE
                                || po == Pyc::INSTRUMENTED_RESUME_A) continue;
                        break;
                    }
                    resultUsed = (po != Pyc::POP_TOP && po != -1);
                }
                if (resultUsed)
                    stack.push(new ASTReturn(value, ASTReturn::YIELD_EXPR));
                else
                    curblock->append(new ASTReturn(value, ASTReturn::YIELD));
            }
        }
        break;
    }
}

/* POP_BLOCK closes the innermost open block when its runtime block ends. It is
 * ignored for blocks torn down by other opcodes (a container/finally waits for
 * END_FINALLY, a with for WITH_CLEANUP). Otherwise it restores the saved stack
 * for a suite that opened one, pops the block and appends it to its parent
 * (dropping an empty else), and then handles the follow-on structure the same
 * pop implies: opening a for/while `else:` suite, unwinding a try whose body
 * just ended, and advancing a container into its finally suite or closing it. */
void CodeBuilder::handlePopBlock()
{
    if (curblock->blktype() == ASTBlock::BLK_CONTAINER ||
            curblock->blktype() == ASTBlock::BLK_FINALLY) {
        /* These should only be popped by an END_FINALLY */
        return;
    }

    if (curblock->blktype() == ASTBlock::BLK_WITH) {
        // This should only be popped by a WITH_CLEANUP
        return;
    }

    if (curblock->nodes().size() &&
            curblock->nodes().back().type() == ASTNode::NODE_KEYWORD) {
        curblock->removeLast();
    }

    if (curblock->blktype() == ASTBlock::BLK_IF
            || curblock->blktype() == ASTBlock::BLK_ELIF
            || curblock->blktype() == ASTBlock::BLK_ELSE
            || curblock->blktype() == ASTBlock::BLK_TRY
            || curblock->blktype() == ASTBlock::BLK_EXCEPT
            || curblock->blktype() == ASTBlock::BLK_FINALLY) {
        if (!stack_hist.empty()) {
            stack = stack_hist.top();
            stack_hist.pop();
        } else {
            fprintf(stderr, "Warning: Stack history is empty, something wrong might have happened\n");
        }
    }
    PycRef<ASTBlock> tmp = curblock;
    blocks.pop();

    if (!blocks.empty())
        curblock = blocks.top();

    if (!(tmp->blktype() == ASTBlock::BLK_ELSE
            && tmp->nodes().size() == 0)) {
        curblock->append(tmp.cast<ASTNode>());
    }

    if (tmp->blktype() == ASTBlock::BLK_FOR && tmp->end() > pos) {
        stack_hist.push(stack);

        PycRef<ASTBlock> blkelse = new ASTBlock(ASTBlock::BLK_ELSE, tmp->end());
        blocks.push(blkelse);
        curblock = blocks.top();
    }

    if (curblock->blktype() == ASTBlock::BLK_TRY
            && tmp->blktype() != ASTBlock::BLK_FOR
            && tmp->blktype() != ASTBlock::BLK_ASYNCFOR
            && tmp->blktype() != ASTBlock::BLK_WHILE) {
        stack = stack_hist.top();
        stack_hist.pop();

        tmp = curblock;
        blocks.pop();
        curblock = blocks.top();

        if (!(tmp->blktype() == ASTBlock::BLK_ELSE
                && tmp->nodes().size() == 0)) {
            curblock->append(tmp.cast<ASTNode>());
        }
    }

    if (curblock->blktype() == ASTBlock::BLK_CONTAINER) {
        PycRef<ASTContainerBlock> cont = curblock.cast<ASTContainerBlock>();

        if (tmp->blktype() == ASTBlock::BLK_ELSE && !cont->hasFinally()) {

            /* Pop the container */
            blocks.pop();
            curblock = blocks.top();
            curblock->append(cont.cast<ASTNode>());

        } else if ((tmp->blktype() == ASTBlock::BLK_ELSE && cont->hasFinally())
                || (tmp->blktype() == ASTBlock::BLK_TRY && !cont->hasExcept())) {

            /* Add the finally block */
            stack_hist.push(stack);

            PycRef<ASTBlock> final = new ASTBlock(ASTBlock::BLK_FINALLY, 0, true);
            blocks.push(final);
            curblock = blocks.top();
        }
    }

    if ((curblock->blktype() == ASTBlock::BLK_FOR || curblock->blktype() == ASTBlock::BLK_ASYNCFOR)
            && curblock->end() == pos) {
        blocks.pop();
        blocks.top()->append(curblock.cast<ASTNode>());
        curblock = blocks.top();
    }
}

/* POP_TOP discards TOS, but with side meanings depending on what it discards:
 *   - a match subject being torn down is simply dropped;
 *   - a comprehension result inside a comprehension loop is kept on the stack;
 *   - the first POP_TOP of an uninitialised suite initialises it (or sets the
 *     with-manager expression) rather than emitting a statement;
 *   - an already-processed value is dropped;
 *   - otherwise the value is a bare expression statement and is appended, with a
 *     comprehension-append call turned back into an ASTComprehension.
 * Returns false (after cleanBuild=false) if a comprehension turns up as a
 * with-manager, which cannot be reconstructed. */
bool CodeBuilder::handlePopTop()
{
    PycRef<ASTNode> value = stack.top();
    stack.pop();

    if (value != nullptr) {
        std::stack<PycRef<ASTBlock> > ms = blocks;
        while (!ms.empty()) {
            if (ms.top()->blktype() == ASTBlock::BLK_MATCH) {
                if (ms.top().cast<ASTMatchBlock>()->subject() == value)
                    value = nullptr;
                break;
            }
            ms.pop();
        }
        if (value == nullptr)
            return true;
    }

    if (value != nullptr && value.type() == ASTNode::NODE_COMPREHENSION
            && curblock->blktype() == ASTBlock::BLK_FOR
            && curblock.cast<ASTIterBlock>()->isComprehension()) {
        stack.push(value);
        return true;
    }

    if (!curblock->inited()) {
        if (curblock->blktype() == ASTBlock::BLK_WITH) {
            if (value != nullptr && value.type() == ASTNode::NODE_COMPREHENSION) {
                cleanBuild = false;
                return false;
            }
            curblock.cast<ASTWithBlock>()->setExpr(value);
        } else {
            curblock->init();
        }
        return true;
    } else if (value == nullptr || value->processed()) {
        return true;
    }

    curblock->append(value);

    if (curblock->blktype() == ASTBlock::BLK_FOR
            && curblock.cast<ASTIterBlock>()->isComprehension()) {
        /* This relies on some really uncertain logic...
         * If it's a comprehension, the only POP_TOP should be
         * a call to append the iter to the list.
         */
        if (value.type() == ASTNode::NODE_CALL) {
            auto& pparams = value.cast<ASTCall>()->pparams();
            if (!pparams.empty()) {
                PycRef<ASTNode> res = pparams.front();
                stack.push(new ASTComprehension(res));
            }
        }
    }
    return true;
}

/* RERAISE re-raises the current exception. For a plain `raise` cleanup this is
 * a no-op, but when the prescan recorded a resume target for this offset
 * (finExcReraise) it also closes the reconstructed try/except: a RERAISE inside
 * the try body opens the recorded except clause, while one inside an except
 * clause closes the handler (and its container) and jumps control to the
 * recorded target. */
void CodeBuilder::handleReraise()
{
    auto fer = finExcReraise.find(curpos);
    if (fer != finExcReraise.end()) {
        int X = fer->second;
        if (curblock->blktype() == ASTBlock::BLK_TRY) {
            if (!stack_hist.empty()) {
                stack = stack_hist.top();
                stack_hist.pop();
            }
            PycRef<ASTBlock> prev = curblock;
            blocks.pop();
            curblock = blocks.top();
            curblock->append(prev.cast<ASTNode>());
            if (curblock->blktype() == ASTBlock::BLK_CONTAINER
                    && curblock.cast<ASTContainerBlock>()->hasExcept()) {
                stack_hist.push(stack);
                PycRef<ASTBlock> except = new ASTCondBlock(ASTBlock::BLK_EXCEPT, X, NULL, false);
                except->init();
                blocks.push(except);
                curblock = blocks.top();
            }
            return;
        } else if (curblock->blktype() == ASTBlock::BLK_EXCEPT) {
            if (!stack_hist.empty()) {
                stack = stack_hist.top();
                stack_hist.pop();
            }
            PycRef<ASTBlock> exc = curblock;
            blocks.pop();
            curblock = blocks.top();
            curblock->append(exc.cast<ASTNode>());
            if (curblock->blktype() == ASTBlock::BLK_CONTAINER) {
                PycRef<ASTBlock> cont = curblock;
                blocks.pop();
                curblock = blocks.top();
                curblock->append(cont.cast<ASTNode>());
            }
            source.setPos(X);
            pos = X;
            while (next_exception_entry < exception_entries.size()
                    && exception_entries[next_exception_entry].start_offset < X)
                next_exception_entry++;
            return;
        }
    }
}

/* The pre-3.11 explicit block-setup opcodes, which open a block covering the
 * range up to pos+operand:
 *   SETUP_EXCEPT   -> a try/except: reuse or open the container, save the stack,
 *       and push the BLK_TRY body (need_try cleared: it has an except clause).
 *   SETUP_FINALLY  -> a try/finally: push a bare container and set need_try.
 *   SETUP_LOOP     -> a `while` loop: push the BLK_WHILE block. */
void CodeBuilder::handleSetupBlock(int opcode, int operand)
{
    switch (opcode) {
    case Pyc::SETUP_EXCEPT_A:
        {
            if (curblock->blktype() == ASTBlock::BLK_CONTAINER) {
                curblock.cast<ASTContainerBlock>()->setExcept(pos+operand);
            } else {
                PycRef<ASTBlock> next = new ASTContainerBlock(0, pos+operand);
                blocks.push(next.cast<ASTBlock>());
            }

            /* Store the current stack for the except/finally statement(s) */
            stack_hist.push(stack);
            PycRef<ASTBlock> tryblock = new ASTBlock(ASTBlock::BLK_TRY, pos+operand, true);
            blocks.push(tryblock.cast<ASTBlock>());
            curblock = blocks.top();

            need_try = false;
        }
        break;
    case Pyc::SETUP_FINALLY_A:
        {
            PycRef<ASTBlock> next = new ASTContainerBlock(pos+operand);
            blocks.push(next.cast<ASTBlock>());
            curblock = blocks.top();

            need_try = true;
        }
        break;
    case Pyc::SETUP_LOOP_A:
        {
            PycRef<ASTBlock> next = new ASTCondBlock(ASTBlock::BLK_WHILE, pos+operand, NULL, false);
            blocks.push(next.cast<ASTBlock>());
            curblock = blocks.top();
        }
        break;
    }
}

/* Enter a `for` loop, pushing a BLK_FOR whose range covers the loop body.
 * FOR_ITER (3.x): the block end comes from the jump operand (3.8+) or the
 * enclosing SETUP_LOOP block (pre-3.8); a `<listcomp>`/`<setcomp>`/etc. code
 * object marks a comprehension. When the recorded end runs past the enclosing
 * block, the prescan loop ranges are consulted to record an early close. The
 * iterator is kept on the stack (popped pre-3.12) and a NULL placeholder is
 * pushed. FOR_LOOP is the Python 1/2 form, which also re-pushes the sequence
 * and counter the interpreter expects. */
void CodeBuilder::handleForIter(int opcode, int operand)
{
    if (opcode == Pyc::FOR_LOOP_A) {
        PycRef<ASTNode> curidx = stack.top(); // Current index
        stack.pop();
        PycRef<ASTNode> iter = stack.top(); // Iterable
        stack.pop();

        bool comprehension = false;
        PycRef<ASTBlock> top = blocks.top();
        if (top->blktype() == ASTBlock::BLK_WHILE) {
            blocks.pop();
        } else {
            comprehension = true;
        }
        PycRef<ASTIterBlock> forblk = new ASTIterBlock(ASTBlock::BLK_FOR, curpos, top->end(), iter);
        forblk->setComprehension(comprehension);
        blocks.push(forblk.cast<ASTBlock>());
        curblock = blocks.top();

        /* Python Docs say:
              "push the sequence, the incremented counter,
               and the current item onto the stack." */
        stack.push(iter);
        stack.push(curidx);
        stack.push(NULL); // We can totally hack this >_>
        return;
    }

    PycRef<ASTNode> iter = stack.top(); // Iterable
    if (mod->verCompare(3, 12) < 0) {
        // Do not pop the iterator for py 3.12+
        stack.pop();
    }
    /* Pop it? Don't pop it? */

    int end;
    bool comprehension = false;

    // before 3.8, there is a SETUP_LOOP instruction with block start and end position,
    //    the operand is usually a jump to a POP_BLOCK instruction
    // after 3.8, block extent has to be inferred implicitly; the operand is a jump to a position after the for block
    if (mod->majorVer() == 3 && mod->minorVer() >= 8) {
        end = operand;
        if (mod->verCompare(3, 10) >= 0)
            end *= sizeof(uint16_t); // // BPO-27129
        end += pos;
        const char* cn = code->name()->value();
        comprehension = strcmp(cn, "<listcomp>") == 0
                || strcmp(cn, "<setcomp>") == 0
                || strcmp(cn, "<dictcomp>") == 0
                || strcmp(cn, "<genexpr>") == 0;
    } else {
        PycRef<ASTBlock> top = blocks.top();
        end = top->end(); // block end position from SETUP_LOOP
        if (top->blktype() == ASTBlock::BLK_WHILE) {
            blocks.pop();
        } else {
            comprehension = true;
        }
    }

    if (!comprehension && !blocks.empty()
            && blocks.top()->end() > 0 && end > blocks.top()->end()) {
        int lastBE = -1;
        for (const auto& lr : loopRanges)
            if (lr.start == curpos && lr.end > lastBE) lastBE = lr.end;
        if (lastBE > curpos && lastBE < end) {
            PycBuffer bb(code->code()->value(), code->code()->length());
            bb.setPos(lastBE);
            int bo, ba, bp = lastBE;
            if (!bb.atEof()) { bc_next(bb, mod, bo, ba, bp);
                if (bp > lastBE && bp < end) forEarlyClose[curpos] = bp; }
        }
    }

    PycRef<ASTIterBlock> forblk = new ASTIterBlock(ASTBlock::BLK_FOR, curpos, end, iter);
    forblk->setComprehension(comprehension);
    blocks.push(forblk.cast<ASTBlock>());
    curblock = blocks.top();
    if (comprehension)
        compFilterFwd = false;

    stack.push(NULL);
}

/* SEND drives a generator/await sub-iterator (`yield from` / `await`). The
 * operand is the offset resumed when the sub-iterator is exhausted. When the
 * value below TOS is the awaitable/iterator being driven, this emits the
 * yield-from as a statement (its result is POP_TOP'd at the target) or as an
 * expression, then jumps past the send loop to the target. Returns false (bail)
 * for an unrecognised SEND shape. */
bool CodeBuilder::handleSend()
{
    PycRef<ASTNode> below = stack.top(2);
    int target = pos + operand * sizeof(uint16_t);
    bool validTarget = (target > pos
            && target <= (int)code->code()->length());
    if (validTarget && below != nullptr
            && below.type() == ASTNode::NODE_AWAITABLE) {
        stack.pop();
        source.setPos(target);
        pos = target;
        return true;
    }
    int targetOp = -1;
    if (validTarget) {
        PycBuffer peek(code->code()->value(), code->code()->length());
        peek.setPos(target);
        int po, pa, pp = target;
        if (!peek.atEof()) { bc_next(peek, mod, po, pa, pp); targetOp = po; }
    }
    if (validTarget && below != nullptr && targetOp == Pyc::POP_TOP) {
        stack.pop();
        PycRef<ASTNode> value = stack.top();
        value->setProcessed();
        curblock->append(new ASTReturn(value, ASTReturn::YIELD_FROM));
        source.setPos(target);
        pos = target;
        return true;
    }
    if (validTarget && below != nullptr) {
        stack.pop();
        PycRef<ASTNode> iter = stack.top();
        stack.pop();
        stack.push(new ASTReturn(iter, ASTReturn::YIELD_FROM_EXPR));
        source.setPos(target);
        pos = target;
        return true;
    }
    fprintf(stderr, "Unsupported opcode: %s (%d)\n",
            Pyc::OpcodeName(opcode), opcode);
    cleanBuild = false;
    return false;
}

/* The UNARY_* opcodes each pop one operand and push one result node.
 *   INVERT / NEGATIVE / NOT / POSITIVE  -> the ~x, -x, `not x`, +x operators
 *                                          (ASTUnary with the matching UN_*).
 *   CALL  (Python 1/2) -> call the top-of-stack callable with no arguments.
 *   CONVERT (Python 1/2) -> backtick `repr` conversion (ASTConvert).
 * Output is unchanged from the six inline cases this replaced. */
void CodeBuilder::handleUnaryOp(int opcode)
{
    PycRef<ASTNode> arg = stack.top();
    stack.pop();
    switch (opcode) {
    case Pyc::UNARY_CALL:
        stack.push(new ASTCall(arg, ASTCall::pparam_t(), ASTCall::kwparam_t()));
        break;
    case Pyc::UNARY_CONVERT:
        stack.push(new ASTConvert(arg));
        break;
    case Pyc::UNARY_INVERT:
        stack.push(new ASTUnary(arg, ASTUnary::UN_INVERT));
        break;
    case Pyc::UNARY_NEGATIVE:
        stack.push(new ASTUnary(arg, ASTUnary::UN_NEGATIVE));
        break;
    case Pyc::UNARY_NOT:
        stack.push(new ASTUnary(arg, ASTUnary::UN_NOT));
        break;
    case Pyc::UNARY_POSITIVE:
        stack.push(new ASTUnary(arg, ASTUnary::UN_POSITIVE));
        break;
    }
}

/* Arithmetic/bitwise binary operators and their in-place (`x @= y`) forms: pop
 * the right operand then the left and push an ASTBinary. Two encodings feed in:
 *   - 3.11's single BINARY_OP whose OPERAND selects the operator
 *     (ASTBinary::from_binary_op); an unknown operand is reported but still
 *     builds a node so the rest of the stream stays aligned.
 *   - the older one-opcode-per-operator forms (BINARY_ADD, INPLACE_XOR, ...)
 *     decoded from the OPCODE (ASTBinary::from_opcode); an unknown one is fatal.
 * When the operator is an in-place variant, inplaceStore is armed so the STORE
 * that consumes the result renders as an augmented assignment. */
void CodeBuilder::handleBinaryOp(int opcode, int operand)
{
    ASTBinary::BinOp op;
    if (opcode == Pyc::BINARY_OP_A) {
        op = ASTBinary::from_binary_op(operand);
        if (op == ASTBinary::BIN_INVALID)
            fprintf(stderr, "Unsupported `BINARY_OP` operand value: %d\n", operand);
    } else {
        op = ASTBinary::from_opcode(opcode);
        if (op == ASTBinary::BIN_INVALID)
            throw std::runtime_error("Unhandled opcode from ASTBinary::from_opcode");
    }
    PycRef<ASTNode> right = stack.top();
    stack.pop();
    PycRef<ASTNode> left = stack.top();
    stack.pop();
    if (op >= ASTBinary::BIN_IP_ADD && op < ASTBinary::BIN_INVALID)
        inplaceStore = 1;
    stack.push(new ASTBinary(left, right, op));
}

/* The identity (`is` / `is not`) and membership (`in` / `not in`) tests. Both
 * pop the right then the left operand and, like COMPARE_OP, either extend an
 * in-progress chained comparison, start one (when chainCmp is armed), or build
 * a plain ASTCompare. The operand's low bit selects the negated form. */
void CodeBuilder::handleIsContainsOp(int opcode, int operand)
{
    PycRef<ASTNode> right = stack.top();
    stack.pop();
    PycRef<ASTNode> left = stack.top();
    stack.pop();
    int cmpop;
    if (opcode == Pyc::CONTAINS_OP_A)
        cmpop = operand ? ASTCompare::CMP_NOT_IN : ASTCompare::CMP_IN;
    else // IS_OP_A
        cmpop = operand ? ASTCompare::CMP_IS_NOT : ASTCompare::CMP_IS;
    if (left != nullptr && left.type() == ASTNode::NODE_CHAINCOMPARE) {
        left.cast<ASTChainCompare>()->extend(cmpop, right);
        stack.push(left);
    } else if (chainCmp) {
        stack.push(new ASTChainCompare(left, right, cmpop));
    } else {
        stack.push(new ASTCompare(left, right, cmpop));
    }
    chainCmp = 0;
}

/* The collection/display builders. Each pops the operand-count elements the
 * compiler pushed (in reverse, so they are un-reversed here) and pushes one
 * container node:
 *   BUILD_LIST/SET/TUPLE  -> [..] / {..} / (..)  (BUILD_TUPLE ignores the
 *                            __build_class__ helper tuple so a class definition
 *                            renders as a class, not a tuple literal).
 *   BUILD_STRING          -> an f-string (ASTJoinedStr), same shape as a list.
 *   BUILD_MAP             -> a dict; pre-3.5 the pairs arrive later via
 *                            STORE_MAP, so only an empty ASTMap is pushed here.
 *   BUILD_CONST_KEY_MAP   -> a dict whose keys are a single constant tuple on
 *                            top with the values beneath it.
 *   BUILD_SLICE           -> a subscript slice (start:end[:step]); a None bound
 *                            becomes an omitted slice field, and the 3-arg form
 *                            nests a [a:b] slice inside a [..:step] slice. */
void CodeBuilder::handleBuildCollection(int opcode, int operand)
{
    switch (opcode) {
    case Pyc::BUILD_LIST_A:
        {
            ASTList::value_t values;
            for (int i=0; i<operand; i++) {
                values.push_front(stack.top());
                stack.pop();
            }
            stack.push(new ASTList(values));
        }
        break;
    case Pyc::BUILD_SET_A:
        {
            ASTSet::value_t values;
            for (int i=0; i<operand; i++) {
                values.push_front(stack.top());
                stack.pop();
            }
            stack.push(new ASTSet(values));
        }
        break;
    case Pyc::BUILD_MAP_A:
        if (mod->verCompare(3, 5) >= 0) {
            auto map = new ASTMap;
            std::vector<std::pair<PycRef<ASTNode>, PycRef<ASTNode> > > pairs;
            pairs.reserve(operand);
            for (int i=0; i<operand; ++i) {
                PycRef<ASTNode> value = stack.top();
                stack.pop();
                PycRef<ASTNode> key = stack.top();
                stack.pop();
                pairs.push_back(std::make_pair(key, value));
            }
            for (auto it = pairs.rbegin(); it != pairs.rend(); ++it)
                map->add(it->first, it->second);
            stack.push(map);
        } else {
            if (stack.top().type() == ASTNode::NODE_CHAINSTORE) {
                stack.pop();
            }
            stack.push(new ASTMap());
        }
        break;
    case Pyc::BUILD_CONST_KEY_MAP_A:
        // Top of stack will be a tuple of keys.
        // Values will start at TOS - 1.
        {
            PycRef<ASTNode> keys = stack.top();
            stack.pop();

            ASTConstMap::values_t values;
            values.reserve(operand);
            for (int i = 0; i < operand; ++i) {
                PycRef<ASTNode> value = stack.top();
                stack.pop();
                values.push_back(value);
            }

            stack.push(new ASTConstMap(keys, values));
        }
        break;
    case Pyc::BUILD_SLICE_A:
        {
            if (operand == 2) {
                PycRef<ASTNode> end = stack.top();
                stack.pop();
                PycRef<ASTNode> start = stack.top();
                stack.pop();

                if (start.type() == ASTNode::NODE_OBJECT
                        && start.cast<ASTObject>()->object() == Pyc_None) {
                    start = NULL;
                }

                if (end.type() == ASTNode::NODE_OBJECT
                        && end.cast<ASTObject>()->object() == Pyc_None) {
                    end = NULL;
                }

                if (start == NULL && end == NULL) {
                    stack.push(new ASTSlice(ASTSlice::SLICE0));
                } else if (start == NULL) {
                    stack.push(new ASTSlice(ASTSlice::SLICE2, start, end));
                } else if (end == NULL) {
                    stack.push(new ASTSlice(ASTSlice::SLICE1, start, end));
                } else {
                    stack.push(new ASTSlice(ASTSlice::SLICE3, start, end));
                }
            } else if (operand == 3) {
                PycRef<ASTNode> step = stack.top();
                stack.pop();
                PycRef<ASTNode> end = stack.top();
                stack.pop();
                PycRef<ASTNode> start = stack.top();
                stack.pop();

                if (start.type() == ASTNode::NODE_OBJECT
                        && start.cast<ASTObject>()->object() == Pyc_None) {
                    start = NULL;
                }

                if (end.type() == ASTNode::NODE_OBJECT
                        && end.cast<ASTObject>()->object() == Pyc_None) {
                    end = NULL;
                }

                if (step.type() == ASTNode::NODE_OBJECT
                        && step.cast<ASTObject>()->object() == Pyc_None) {
                    step = NULL;
                }

                /* We have to do this as a slice where one side is another slice */
                /* [[a:b]:c] */

                if (start == NULL && end == NULL) {
                    stack.push(new ASTSlice(ASTSlice::SLICE0));
                } else if (start == NULL) {
                    stack.push(new ASTSlice(ASTSlice::SLICE2, start, end));
                } else if (end == NULL) {
                    stack.push(new ASTSlice(ASTSlice::SLICE1, start, end));
                } else {
                    stack.push(new ASTSlice(ASTSlice::SLICE3, start, end));
                }

                PycRef<ASTNode> lhs = stack.top();
                stack.pop();

                if (step == NULL) {
                    stack.push(new ASTSlice(ASTSlice::SLICE1, lhs, step));
                } else {
                    stack.push(new ASTSlice(ASTSlice::SLICE3, lhs, step));
                }
            }
        }
        break;
    case Pyc::BUILD_STRING_A:
        {
            // Nearly identical logic to BUILD_LIST
            ASTList::value_t values;
            for (int i = 0; i < operand; i++) {
                values.push_front(stack.top());
                stack.pop();
            }
            stack.push(new ASTJoinedStr(values));
        }
        break;
    case Pyc::BUILD_TUPLE_A:
        {
            // if class is a closure code, ignore this tuple
            PycRef<ASTNode> tos = stack.top();
            if (tos && tos->type() == ASTNode::NODE_LOADBUILDCLASS) {
                break;
            }

            ASTTuple::value_t values;
            values.resize(operand);
            for (int i=0; i<operand; i++) {
                values[operand-i-1] = stack.top();
                stack.pop();
            }
            stack.push(new ASTTuple(values));
        }
        break;
    }
}

/* The simple load opcodes that just push a name/reference node (the operator
 * loads and the constant load, which need extra state, stay inline in build()).
 *   LOAD_FAST / LOAD_NAME / LOAD_GLOBAL / LOAD_DEREF / LOAD_CLASSDEREF ->
 *       push the local / name / global / closure variable by index.
 *   LOAD_FAST_LOAD_FAST (3.13) packs two local indices into one operand.
 *   LOAD_ATTR / LOAD_METHOD -> pop the object and push `obj.attr` (BIN_ATTR);
 *       LOAD_ATTR on an in-progress import is left untouched so it renders as
 *       part of the import. From 3.11/3.12 the low operand bit signals that a
 *       NULL/self placeholder is pushed first, and the real index is operand>>1.
 *   LOAD_BUILD_CLASS / LOAD_LOCALS -> push the __build_class__ helper / the
 *       frame locals marker.
 * operand is taken by value; its in-case shifts do not escape (the dispatch
 * loop re-reads operand each instruction and never uses it after the switch). */
void CodeBuilder::handleLoad(int opcode, int operand)
{
    switch (opcode) {
    case Pyc::LOAD_ATTR_A:
        {
            PycRef<ASTNode> name = stack.top();
            if (name.type() != ASTNode::NODE_IMPORT) {
                stack.pop();

                if (mod->verCompare(3, 12) >= 0) {
                    if (operand & 1) {
                        /* Changed in version 3.12:
                        If the low bit of name is set, then a NULL or self is pushed to the stack
                        before the attribute or unbound method respectively. */
                        stack.push(nullptr);
                    }
                    operand >>= 1;
                }

                stack.push(new ASTBinary(name, new ASTName(code->getName(operand)), ASTBinary::BIN_ATTR));
            }
        }
        break;
    case Pyc::LOAD_BUILD_CLASS:
        stack.push(new ASTLoadBuildClass(new PycObject()));
        break;
    case Pyc::LOAD_DEREF_A:
    case Pyc::LOAD_CLASSDEREF_A:
        stack.push(new ASTName(code->getCellVar(mod, operand)));
        break;
    case Pyc::LOAD_FAST_A:
        if (mod->verCompare(1, 3) < 0)
            stack.push(new ASTName(code->getName(operand)));
        else
            stack.push(new ASTName(code->getLocal(operand)));
        break;
    case Pyc::LOAD_FAST_LOAD_FAST_A:
        stack.push(new ASTName(code->getLocal(operand >> 4)));
        stack.push(new ASTName(code->getLocal(operand & 0xF)));
        break;
    case Pyc::LOAD_GLOBAL_A:
        if (mod->verCompare(3, 11) >= 0) {
            // Loads the global named co_names[namei>>1] onto the stack.
            if (operand & 1) {
                /* Changed in version 3.11:
                If the low bit of "NAMEI" (operand) is set,
                then a NULL is pushed to the stack before the global variable. */
                stack.push(nullptr);
            }
            operand >>= 1;
        }
        stack.push(new ASTName(code->getName(operand)));
        break;
    case Pyc::LOAD_LOCALS:
        stack.push(new ASTNode(ASTNode::NODE_LOCALS));
        break;
    case Pyc::LOAD_METHOD_A:
        {
            // Behave like LOAD_ATTR
            PycRef<ASTNode> name = stack.top();
            stack.pop();
            PycRef<ASTNode> meth = new ASTBinary(name, new ASTName(code->getName(operand)), ASTBinary::BIN_ATTR);
            if (mod->verCompare(3, 11) >= 0)
                stack.push(nullptr);
            stack.push(meth);
        }
        break;
    case Pyc::LOAD_NAME_A:
        stack.push(new ASTName(code->getName(operand)));
        break;
    }
}

/* The `del` statements: each appends an ASTDelete of the target being removed.
 *   DELETE_FAST / NAME / GLOBAL  -> del of a local / name / global (GLOBAL also
 *       records the name as global for the enclosing scope). Names of the form
 *       `_[...]` are compiler-internal list-comprehension temporaries and are
 *       not emitted.
 *   DELETE_ATTR                  -> del obj.attr.
 *   DELETE_SUBSCR                -> del obj[key].
 *   DELETE_SLICE_0..3 (Python 2) -> del of the old dedicated slice forms
 *       ([:], [lo:], [:hi], [lo:hi]) rebuilt as a subscript of an ASTSlice. */
void CodeBuilder::handleDelete(int opcode, int operand)
{
    switch (opcode) {
    case Pyc::DELETE_ATTR_A:
        {
            PycRef<ASTNode> name = stack.top();
            stack.pop();
            curblock->append(new ASTDelete(new ASTBinary(name, new ASTName(code->getName(operand)), ASTBinary::BIN_ATTR)));
        }
        break;
    case Pyc::DELETE_GLOBAL_A:
        code->markGlobal(code->getName(operand));
        /* Fall through */
    case Pyc::DELETE_NAME_A:
        {
            PycRef<PycString> varname = code->getName(operand);

            if (varname->length() >= 2 && varname->value()[0] == '_'
                    && varname->value()[1] == '[') {
                /* Don't show deletes that are a result of list comps. */
                break;
            }

            PycRef<ASTNode> name = new ASTName(varname);
            curblock->append(new ASTDelete(name));
        }
        break;
    case Pyc::DELETE_FAST_A:
        {
            PycRef<ASTNode> name;

            if (mod->verCompare(1, 3) < 0)
                name = new ASTName(code->getName(operand));
            else
                name = new ASTName(code->getLocal(operand));

            if (name.cast<ASTName>()->name()->value()[0] == '_'
                    && name.cast<ASTName>()->name()->value()[1] == '[') {
                /* Don't show deletes that are a result of list comps. */
                break;
            }

            curblock->append(new ASTDelete(name));
        }
        break;
    case Pyc::DELETE_SLICE_0:
        {
            PycRef<ASTNode> name = stack.top();
            stack.pop();

            curblock->append(new ASTDelete(new ASTSubscr(name, new ASTSlice(ASTSlice::SLICE0))));
        }
        break;
    case Pyc::DELETE_SLICE_1:
        {
            PycRef<ASTNode> upper = stack.top();
            stack.pop();
            PycRef<ASTNode> name = stack.top();
            stack.pop();

            curblock->append(new ASTDelete(new ASTSubscr(name, new ASTSlice(ASTSlice::SLICE1, upper))));
        }
        break;
    case Pyc::DELETE_SLICE_2:
        {
            PycRef<ASTNode> lower = stack.top();
            stack.pop();
            PycRef<ASTNode> name = stack.top();
            stack.pop();

            curblock->append(new ASTDelete(new ASTSubscr(name, new ASTSlice(ASTSlice::SLICE2, NULL, lower))));
        }
        break;
    case Pyc::DELETE_SLICE_3:
        {
            PycRef<ASTNode> lower = stack.top();
            stack.pop();
            PycRef<ASTNode> upper = stack.top();
            stack.pop();
            PycRef<ASTNode> name = stack.top();
            stack.pop();

            curblock->append(new ASTDelete(new ASTSubscr(name, new ASTSlice(ASTSlice::SLICE3, upper, lower))));
        }
        break;
    case Pyc::DELETE_SUBSCR:
        {
            PycRef<ASTNode> key = stack.top();
            stack.pop();
            PycRef<ASTNode> name = stack.top();
            stack.pop();

            curblock->append(new ASTDelete(new ASTSubscr(name, key)));
        }
        break;
    }
}

/* The Python 2 dedicated slice stores (`obj[:] = v`, `obj[lo:] = v`,
 * `obj[:hi] = v`, `obj[lo:hi] = v`). Each pops the bounds, then the target,
 * then the value, and appends an ASTStore of the value into a subscript of the
 * matching ASTSlice. (Modern Python routes these through STORE_SUBSCR instead,
 * which stays inline because it also carries the unpack/annotation paths.) */
void CodeBuilder::handleStoreSlice(int opcode)
{
    switch (opcode) {
    case Pyc::STORE_SLICE_0:
        {
            PycRef<ASTNode> dest = stack.top();
            stack.pop();
            PycRef<ASTNode> value = stack.top();
            stack.pop();

            curblock->append(new ASTStore(value, new ASTSubscr(dest, new ASTSlice(ASTSlice::SLICE0))));
        }
        break;
    case Pyc::STORE_SLICE_1:
        {
            PycRef<ASTNode> upper = stack.top();
            stack.pop();
            PycRef<ASTNode> dest = stack.top();
            stack.pop();
            PycRef<ASTNode> value = stack.top();
            stack.pop();

            curblock->append(new ASTStore(value, new ASTSubscr(dest, new ASTSlice(ASTSlice::SLICE1, upper))));
        }
        break;
    case Pyc::STORE_SLICE_2:
        {
            PycRef<ASTNode> lower = stack.top();
            stack.pop();
            PycRef<ASTNode> dest = stack.top();
            stack.pop();
            PycRef<ASTNode> value = stack.top();
            stack.pop();

            curblock->append(new ASTStore(value, new ASTSubscr(dest, new ASTSlice(ASTSlice::SLICE2, NULL, lower))));
        }
        break;
    case Pyc::STORE_SLICE_3:
        {
            PycRef<ASTNode> lower = stack.top();
            stack.pop();
            PycRef<ASTNode> upper = stack.top();
            stack.pop();
            PycRef<ASTNode> dest = stack.top();
            stack.pop();
            PycRef<ASTNode> value = stack.top();
            stack.pop();

            curblock->append(new ASTStore(value, new ASTSubscr(dest, new ASTSlice(ASTSlice::SLICE3, upper, lower))));
        }
        break;
    }
}

/* Subscript reads: pop the index/bounds and the container and push an
 * ASTSubscr (`obj[...]`).
 *   BINARY_SUBSCR         -> obj[key].
 *   BINARY_SLICE (3.12)   -> obj[start:end]; a None bound is dropped so the
 *                            slice renders with that field omitted.
 *   SLICE_0..3 (Python 2) -> the old dedicated forms obj[:], obj[lo:],
 *                            obj[:hi], obj[lo:hi] rebuilt as a subscript of the
 *                            matching ASTSlice. */
void CodeBuilder::handleSubscript(int opcode)
{
    switch (opcode) {
    case Pyc::BINARY_SUBSCR:
        {
            PycRef<ASTNode> subscr = stack.top();
            stack.pop();
            PycRef<ASTNode> src = stack.top();
            stack.pop();
            stack.push(new ASTSubscr(src, subscr));
        }
        break;
    case Pyc::SLICE_0:
        {
            PycRef<ASTNode> name = stack.top();
            stack.pop();

            PycRef<ASTNode> slice = new ASTSlice(ASTSlice::SLICE0);
            stack.push(new ASTSubscr(name, slice));
        }
        break;
    case Pyc::SLICE_1:
        {
            PycRef<ASTNode> lower = stack.top();
            stack.pop();
            PycRef<ASTNode> name = stack.top();
            stack.pop();

            PycRef<ASTNode> slice = new ASTSlice(ASTSlice::SLICE1, lower);
            stack.push(new ASTSubscr(name, slice));
        }
        break;
    case Pyc::SLICE_2:
        {
            PycRef<ASTNode> upper = stack.top();
            stack.pop();
            PycRef<ASTNode> name = stack.top();
            stack.pop();

            PycRef<ASTNode> slice = new ASTSlice(ASTSlice::SLICE2, NULL, upper);
            stack.push(new ASTSubscr(name, slice));
        }
        break;
    case Pyc::SLICE_3:
        {
            PycRef<ASTNode> upper = stack.top();
            stack.pop();
            PycRef<ASTNode> lower = stack.top();
            stack.pop();
            PycRef<ASTNode> name = stack.top();
            stack.pop();

            PycRef<ASTNode> slice = new ASTSlice(ASTSlice::SLICE3, lower, upper);
            stack.push(new ASTSubscr(name, slice));
        }
        break;
    case Pyc::BINARY_SLICE:
        {
            PycRef<ASTNode> end = stack.top();
            stack.pop();
            PycRef<ASTNode> start = stack.top();
            stack.pop();
            PycRef<ASTNode> dest = stack.top();
            stack.pop();

            if (start.type() == ASTNode::NODE_OBJECT
                    && start.cast<ASTObject>()->object() == Pyc_None) {
                start = NULL;
            }

            if (end.type() == ASTNode::NODE_OBJECT
                    && end.cast<ASTObject>()->object() == Pyc_None) {
                end = NULL;
            }

            PycRef<ASTNode> slice;
            if (start == NULL && end == NULL) {
                slice = new ASTSlice(ASTSlice::SLICE0);
            } else if (start == NULL) {
                slice = new ASTSlice(ASTSlice::SLICE2, start, end);
            } else if (end == NULL) {
                slice = new ASTSlice(ASTSlice::SLICE1, start, end);
            } else {
                slice = new ASTSlice(ASTSlice::SLICE3, start, end);
            }
            stack.push(new ASTSubscr(dest, slice));
        }
        break;
    }
}

/* Pure operand-stack shuffles that carry no source syntax of their own.
 *   DUP_TOP      -> duplicate TOS. On a real value this also opens an
 *                   ASTChainStore so a following `a = b = ...` renders as a
 *                   chained assignment; a NULL or an existing chain store is
 *                   duplicated as-is.
 *   DUP_TOP_TWO  -> duplicate the top two entries (used for `x[i] op= y`).
 *   DUP_TOPX     -> duplicate the top `operand` entries, order preserved.
 *   ROT_TWO/THREE/FOUR -> rotate the top 2/3/4 entries; a chain-store marker
 *                   sitting among them is dropped so the rotation lines up with
 *                   the real values. */
void CodeBuilder::handleStackManip(int opcode, int operand)
{
    switch (opcode) {
    case Pyc::DUP_TOP:
        {
            if (stack.top().type() == PycObject::TYPE_NULL) {
                stack.push(stack.top());
            } else if (stack.top().type() == ASTNode::NODE_CHAINSTORE) {
                auto chainstore = stack.top();
                stack.pop();
                stack.push(stack.top());
                stack.push(chainstore);
            } else {
                stack.push(stack.top());
                ASTNodeList::list_t targets;
                stack.push(new ASTChainStore(targets, stack.top()));
            }
        }
        break;
    case Pyc::DUP_TOP_TWO:
        {
            PycRef<ASTNode> first = stack.top();
            stack.pop();
            PycRef<ASTNode> second = stack.top();

            stack.push(first);
            stack.push(second);
            stack.push(first);
        }
        break;
    case Pyc::DUP_TOPX_A:
        {
            std::stack<PycRef<ASTNode> > first;
            std::stack<PycRef<ASTNode> > second;

            for (int i = 0; i < operand; i++) {
                PycRef<ASTNode> node = stack.top();
                stack.pop();
                first.push(node);
                second.push(node);
            }

            while (first.size()) {
                stack.push(first.top());
                first.pop();
            }

            while (second.size()) {
                stack.push(second.top());
                second.pop();
            }
        }
        break;
    case Pyc::ROT_TWO:
        {
            PycRef<ASTNode> one = stack.top();
            stack.pop();
            if (stack.top().type() == ASTNode::NODE_CHAINSTORE) {
                stack.pop();
            }
            PycRef<ASTNode> two = stack.top();
            stack.pop();

            stack.push(one);
            stack.push(two);
        }
        break;
    case Pyc::ROT_THREE:
        {
            PycRef<ASTNode> one = stack.top();
            stack.pop();
            PycRef<ASTNode> two = stack.top();
            stack.pop();
            if (stack.top().type() == ASTNode::NODE_CHAINSTORE) {
                stack.pop();
            }
            PycRef<ASTNode> three = stack.top();
            stack.pop();
            stack.push(one);
            stack.push(three);
            stack.push(two);
        }
        break;
    case Pyc::ROT_FOUR:
        {
            PycRef<ASTNode> one = stack.top();
            stack.pop();
            PycRef<ASTNode> two = stack.top();
            stack.pop();
            PycRef<ASTNode> three = stack.top();
            stack.pop();
            if (stack.top().type() == ASTNode::NODE_CHAINSTORE) {
                stack.pop();
            }
            PycRef<ASTNode> four = stack.top();
            stack.pop();
            stack.push(one);
            stack.push(four);
            stack.push(three);
            stack.push(two);
        }
        break;
    }
}

/* The container-update opcodes, which serve double duty. Inside a comprehension
 * loop (curblock is a comprehension for-block) the accumulate opcodes turn the
 * running container into the comprehension result node; everywhere else they
 * grow a literal display in place:
 *   LIST_APPEND / SET_ADD / MAP_ADD -> append/add to the list/set/dict, or
 *       start an ASTComprehension (list/set/dict). MAP_ADD carries key+value.
 *   LIST_EXTEND / SET_UPDATE        -> splice the right-hand side in: a constant
 *       tuple/frozenset is spread element-by-element, otherwise it renders as a
 *       `*rhs` unpack inside the display.
 *   DICT_UPDATE / DICT_MERGE        -> merge a mapping into the dict under
 *       construction (materialising a const-key map first if needed); a
 *       non-mapping right-hand side becomes a `**rhs` unpack (null key).
 *   LIST_TO_TUPLE -> convert a finished list display into a tuple. */
void CodeBuilder::handleCollectionUpdate(int opcode)
{
    switch (opcode) {
    case Pyc::LIST_APPEND:
    case Pyc::LIST_APPEND_A:
        {
            PycRef<ASTNode> value = stack.top();
            stack.pop();

            PycRef<ASTNode> list = stack.top();


            if (curblock->blktype() == ASTBlock::BLK_FOR
                    && curblock.cast<ASTIterBlock>()->isComprehension()) {
                stack.pop();
                stack.push(new ASTComprehension(value));
            } else if (list != nullptr && list.type() == ASTNode::NODE_LIST) {
                stack.pop();
                ASTList::value_t vals = list.cast<ASTList>()->values();
                vals.push_back(value);
                stack.push(new ASTList(vals));
            } else {
                stack.push(new ASTSubscr(list, value)); /* Total hack */
            }
        }
        break;
    case Pyc::SET_ADD_A:
        {
            PycRef<ASTNode> value = stack.top();
            stack.pop();
            PycRef<ASTNode> set = stack.top();
            if (curblock->blktype() == ASTBlock::BLK_FOR
                    && curblock.cast<ASTIterBlock>()->isComprehension()) {
                stack.pop();
                stack.push(new ASTComprehension(value, ASTComprehension::COMP_SET));
            } else if (set != nullptr && set.type() == ASTNode::NODE_SET) {
                stack.pop();
                ASTSet::value_t vals = set.cast<ASTSet>()->values();
                vals.push_back(value);
                stack.push(new ASTSet(vals));
            } else {
                stack.push(new ASTSubscr(set, value));
            }
        }
        break;
    case Pyc::MAP_ADD_A:
        {
            PycRef<ASTNode> value = stack.top();
            stack.pop();
            PycRef<ASTNode> key = stack.top();
            stack.pop();
            PycRef<ASTNode> map = stack.top();
            if (curblock->blktype() == ASTBlock::BLK_FOR
                    && curblock.cast<ASTIterBlock>()->isComprehension()) {
                stack.pop();
                stack.push(new ASTComprehension(value,
                        ASTComprehension::COMP_DICT, key));
            } else if (map.type() == ASTNode::NODE_MAP) {
                map.cast<ASTMap>()->add(key, value);
            } else {
                stack.push(new ASTSubscr(map, key));
            }
        }
        break;
    case Pyc::SET_UPDATE_A:
        {
            PycRef<ASTNode> rhs = stack.top();
            stack.pop();
            PycRef<ASTSet> lhs = stack.top().cast<ASTSet>();
            stack.pop();

            ASTSet::value_t result = lhs->values();
            if (rhs.type() == ASTNode::NODE_OBJECT
                    && rhs.cast<ASTObject>()->object()->type() == PycObject::TYPE_FROZENSET) {
                for (const auto& it : rhs.cast<ASTObject>()->object().cast<PycSet>()->values())
                    result.push_back(new ASTObject(it));
            } else {
                result.push_back(new ASTUnary(rhs, ASTUnary::UN_STAR));
            }

            stack.push(new ASTSet(result));
        }
        break;
    case Pyc::LIST_EXTEND_A:
        {
            PycRef<ASTNode> rhs = stack.top();
            stack.pop();
            PycRef<ASTList> lhs = stack.top().cast<ASTList>();
            stack.pop();

            ASTList::value_t result = lhs->values();
            if (rhs.type() == ASTNode::NODE_OBJECT
                    && (rhs.cast<ASTObject>()->object()->type() == PycObject::TYPE_TUPLE
                        || rhs.cast<ASTObject>()->object()->type() == PycObject::TYPE_SMALL_TUPLE)) {
                for (const auto& it : rhs.cast<ASTObject>()->object().cast<PycTuple>()->values())
                    result.push_back(new ASTObject(it));
            } else {
                result.push_back(new ASTUnary(rhs, ASTUnary::UN_STAR));
            }

            stack.push(new ASTList(result));
        }
        break;
    case Pyc::DICT_UPDATE_A:
    case Pyc::DICT_MERGE_A:
        {
            PycRef<ASTNode> rhs = stack.top();
            stack.pop();
            PycRef<ASTMap> target;
            if (stack.top().type() == ASTNode::NODE_MAP) {
                target = stack.top().cast<ASTMap>();
            } else if (stack.top().type() == ASTNode::NODE_CONST_MAP) {
                PycRef<ASTConstMap> cm = stack.top().cast<ASTConstMap>();
                stack.pop();
                PycTuple::value_t keys = cm->keys().cast<ASTObject>()
                        ->object().cast<PycTuple>()->values();
                ASTConstMap::values_t vals = cm->values();
                target = new ASTMap;
                for (const auto& key : keys) {
                    PycRef<ASTNode> value = vals.back();
                    vals.pop_back();
                    target->add(new ASTObject(key), value);
                }
                stack.push(target.cast<ASTNode>());
            } else {
                break;
            }
            if (opcode == Pyc::DICT_UPDATE_A
                    && rhs.type() == ASTNode::NODE_MAP) {
                for (const auto& kv : rhs.cast<ASTMap>()->values())
                    target->add(kv.first, kv.second);
            } else if (opcode == Pyc::DICT_UPDATE_A
                    && rhs.type() == ASTNode::NODE_CONST_MAP) {
                PycRef<ASTConstMap> rcm = rhs.cast<ASTConstMap>();
                PycTuple::value_t rkeys = rcm->keys().cast<ASTObject>()
                        ->object().cast<PycTuple>()->values();
                ASTConstMap::values_t rvals = rcm->values();
                for (const auto& key : rkeys) {
                    PycRef<ASTNode> value = rvals.back();
                    rvals.pop_back();
                    target->add(new ASTObject(key), value);
                }
            } else {
                target->add(nullptr, rhs);
            }
        }
        break;
    case Pyc::LIST_TO_TUPLE:
        if (!stack.empty() && stack.top() != nullptr
                && stack.top().type() == ASTNode::NODE_LIST) {
            ASTList::value_t lv = stack.top().cast<ASTList>()->values();
            stack.pop();
            ASTTuple::value_t tv(lv.begin(), lv.end());
            stack.push(new ASTTuple(tv));
        }
        break;
    }
}

/* The Python 2 `print` statement opcodes. The compiler emits one PRINT_ITEM per
 * argument and a trailing PRINT_NEWLINE; the *_TO variants take an explicit
 * stream (`print >> f, ...`). Consecutive items on the same statement are
 * merged into a single ASTPrint (matching stream, not yet terminated by a
 * newline); PRINT_NEWLINE marks that node's end of line, or emits a bare
 * `print` when none is open. */
void CodeBuilder::handlePrint(int opcode)
{
    switch (opcode) {
    case Pyc::PRINT_ITEM:
        {
            PycRef<ASTPrint> printNode;
            if (curblock->size() > 0 && curblock->nodes().back().type() == ASTNode::NODE_PRINT)
                printNode = curblock->nodes().back().try_cast<ASTPrint>();
            if (printNode && printNode->stream() == nullptr && !printNode->eol())
                printNode->add(stack.top());
            else
                curblock->append(new ASTPrint(stack.top()));
            stack.pop();
        }
        break;
    case Pyc::PRINT_ITEM_TO:
        {
            PycRef<ASTNode> stream = stack.top();
            stack.pop();

            PycRef<ASTPrint> printNode;
            if (curblock->size() > 0 && curblock->nodes().back().type() == ASTNode::NODE_PRINT)
                printNode = curblock->nodes().back().try_cast<ASTPrint>();
            if (printNode && printNode->stream() == stream && !printNode->eol())
                printNode->add(stack.top());
            else
                curblock->append(new ASTPrint(stack.top(), stream));
            stack.pop();
            if (stream)
                stream->setProcessed();
        }
        break;
    case Pyc::PRINT_NEWLINE:
        {
            PycRef<ASTPrint> printNode;
            if (curblock->size() > 0 && curblock->nodes().back().type() == ASTNode::NODE_PRINT)
                printNode = curblock->nodes().back().try_cast<ASTPrint>();
            if (printNode && printNode->stream() == nullptr && !printNode->eol())
                printNode->setEol(true);
            else
                curblock->append(new ASTPrint(nullptr));
            stack.pop();
        }
        break;
    case Pyc::PRINT_NEWLINE_TO:
        {
            PycRef<ASTNode> stream = stack.top();
            stack.pop();

            PycRef<ASTPrint> printNode;
            if (curblock->size() > 0 && curblock->nodes().back().type() == ASTNode::NODE_PRINT)
                printNode = curblock->nodes().back().try_cast<ASTPrint>();
            if (printNode && printNode->stream() == stream && !printNode->eol())
                printNode->setEol(true);
            else
                curblock->append(new ASTPrint(nullptr, stream));
            stack.pop();
            if (stream)
                stream->setProcessed();
        }
        break;
    }
}

/* The import opcodes.
 *   IMPORT_NAME  -> build an ASTImport for `import module`. In Python 1 the
 *       name is the whole statement; from Python 2 on it pops the from-list
 *       (the names in `from module import a, b`) and, from 2.5, a relative
 *       import level, and carries both on the node.
 *   IMPORT_FROM  -> push the named attribute being pulled out of the module
 *       that IMPORT_NAME left on the stack (one per imported name).
 *   IMPORT_STAR  -> `from module import *`: emit a store of the module with no
 *       target, which renders as the star import. */
void CodeBuilder::handleImport(int opcode, int operand)
{
    switch (opcode) {
    case Pyc::IMPORT_NAME_A:
        if (mod->majorVer() == 1) {
            stack.push(new ASTImport(new ASTName(code->getName(operand)), NULL));
        } else {
            PycRef<ASTNode> fromlist = stack.top();
            stack.pop();
            int level = 0;
            if (mod->verCompare(2, 5) >= 0) {
                PycRef<ASTNode> levelnode = stack.top();
                stack.pop();
                if (levelnode.type() == ASTNode::NODE_OBJECT) {
                    PycRef<PycObject> obj = levelnode.cast<ASTObject>()->object();
                    if (obj->type() == PycObject::TYPE_INT)
                        level = obj.cast<PycInt>()->value();
                }
            }
            stack.push(new ASTImport(new ASTName(code->getName(operand)), fromlist, level));
        }
        break;
    case Pyc::IMPORT_FROM_A:
        stack.push(new ASTName(code->getName(operand)));
        break;
    case Pyc::IMPORT_STAR:
        {
            PycRef<ASTNode> import = stack.top();
            stack.pop();
            curblock->append(new ASTStore(import, NULL));
        }
        break;
    }
}

/* Single-operand expression wrappers that replace TOS with a specialised node.
 *   FORMAT_VALUE  -> one `{value!conv:spec}` piece of an f-string. The operand
 *       carries the conversion flag (str/repr/ascii) and whether a format spec
 *       was pushed just below the value.
 *   GET_AWAITABLE -> the target of an `await`. A comprehension operand marks an
 *       implicit await (an async comprehension), which renders without the
 *       explicit `await` keyword. */
void CodeBuilder::handleExprWrap(int opcode, int operand)
{
    switch (opcode) {
    case Pyc::FORMAT_VALUE_A:
        {
            auto conversion_flag = static_cast<ASTFormattedValue::ConversionFlag>(operand);
            PycRef<ASTNode> format_spec = nullptr;
            if (conversion_flag & ASTFormattedValue::HAVE_FMT_SPEC) {
                format_spec = stack.top();
                stack.pop();
            }
            auto val = stack.top();
            stack.pop();
            stack.push(new ASTFormattedValue(val, conversion_flag, format_spec));
        }
        break;
    case Pyc::GET_AWAITABLE:
    case Pyc::GET_AWAITABLE_A:
        {
            PycRef<ASTNode> object = stack.top();
            stack.pop();
            bool implicitAwait = (object != nullptr
                    && object.type() == ASTNode::NODE_COMPREHENSION);
            stack.push(new ASTAwaitable(object, implicitAwait));
        }
        break;
    }
}

/* RAISE_VARARGS -> a `raise` statement. Pop the (0..3) operands the compiler
 * pushed (exception, cause) and append an ASTRaise. When the raise is the only
 * thing in an if/else branch that was opened with a saved stack (Python >= 2.6),
 * close that branch here: restore the saved stack and fold the finished block
 * into its parent, so the branch does not stay open waiting for a merge that a
 * raise never reaches. */
void CodeBuilder::handleRaiseVarargs(int operand)
{
    ASTRaise::param_t paramList;
    for (int i = 0; i < operand; i++) {
        paramList.push_front(stack.top());
        stack.pop();
    }
    curblock->append(new ASTRaise(paramList));

    if ((curblock->blktype() == ASTBlock::BLK_IF
            || curblock->blktype() == ASTBlock::BLK_ELSE)
            && stack_hist.size()
            && blocks.size() > 1
            && (mod->verCompare(2, 6) >= 0)) {
        stack = stack_hist.top();
        stack_hist.pop();

        PycRef<ASTBlock> prev = curblock;
        blocks.pop();
        curblock = blocks.top();
        curblock->append(prev.cast<ASTNode>());
    }
}

static void append_to_chain_store(const PycRef<ASTNode> &chainStore,
        PycRef<ASTNode> item, FastStack& stack, const PycRef<ASTBlock>& curblock)
{
    stack.pop();    // ignore identical source object.
    chainStore.cast<ASTChainStore>()->append(item);
    if (stack.top().type() == PycObject::TYPE_NULL) {
        curblock->append(chainStore);
    } else {
        stack.push(chainStore);
    }
}

/* Arithmetic/bitwise precedence LEVEL of a binary op (higher binds tighter).
   Boolean short-circuit ops (BIN_LOG_AND/OR) return -1 and are left to the
   caller's op-enum fallback, so their rendering is unchanged. */

static std::unordered_set<PycCode *> code_seen;


void decompyle(PycRef<PycCode> code, PycModule* mod, std::ostream& pyc_output)
{
    if (code_seen.find((PycCode *)code) != code_seen.end()) {
        fputs("WARNING: Circular reference detected\n", stderr);
        return;
    }
    code_seen.insert((PycCode *)code);

    PycRef<ASTNode> source = BuildFromCode(code, mod);
    /* Capture this code object's signature anchor NOPs before any nested code
       object is decompiled (which would overwrite the global). */
    std::set<int> sigAnchorNopOffs = g_sigAnchorNopOffs;

    PycRef<ASTNodeList> clean = source.cast<ASTNodeList>();
    if (cleanBuild) {
        // The Python compiler adds some stuff that we don't really care
        // about, and would add extra code for re-compilation anyway.
        // We strip these lines out here, and then add a "pass" statement
        // if the cleaned up code is empty
        if (clean->nodes().front().type() == ASTNode::NODE_STORE) {
            PycRef<ASTStore> store = clean->nodes().front().cast<ASTStore>();
            if (store->src().type() == ASTNode::NODE_NAME
                    && store->dest().type() == ASTNode::NODE_NAME) {
                PycRef<ASTName> src = store->src().cast<ASTName>();
                PycRef<ASTName> dest = store->dest().cast<ASTName>();
                if (src->name()->isEqual("__name__")
                        && dest->name()->isEqual("__module__")) {
                    // __module__ = __name__
                    // Automatically added by Python 2.2.1 and later
                    clean->removeFirst();
                }
            }
        }
        if (clean->nodes().front().type() == ASTNode::NODE_STORE) {
            PycRef<ASTStore> store = clean->nodes().front().cast<ASTStore>();
            if (store->src().type() == ASTNode::NODE_OBJECT
                    && store->dest().type() == ASTNode::NODE_NAME) {
                PycRef<ASTObject> src = store->src().cast<ASTObject>();
                PycRef<PycString> srcString = src->object().try_cast<PycString>();
                PycRef<ASTName> dest = store->dest().cast<ASTName>();
                if (dest->name()->isEqual("__qualname__")) {
                    // __qualname__ = '<Class Name>'
                    // Automatically added by Python 3.3 and later
                    clean->removeFirst();
                }
            }
        }

        // Class and module docstrings may only appear at the beginning of their source
        if (printClassDocstring && clean->nodes().front().type() == ASTNode::NODE_STORE) {
            PycRef<ASTStore> store = clean->nodes().front().cast<ASTStore>();
            if (store->dest().type() == ASTNode::NODE_NAME &&
                    store->dest().cast<ASTName>()->name()->isEqual("__doc__") &&
                    store->src().type() == ASTNode::NODE_OBJECT) {
                if (print_docstring(store->src().cast<ASTObject>()->object(),
                        cur_indent + (code->name()->isEqual("<module>") ? 0 : 1), mod, pyc_output))
                    clean->removeFirst();
            }
        }
        const bool isLambdaCode = code->name()->isEqual("<lambda>");
        while (!isLambdaCode && !clean->nodes().empty()
                && clean->nodes().back().type() == ASTNode::NODE_RETURN) {
            PycRef<ASTReturn> ret = clean->nodes().back().cast<ASTReturn>();
            if (ret->rettype() != ASTReturn::RETURN)
                break;
            PycRef<ASTObject> retObj = ret->value().try_cast<ASTObject>();
            if (ret->value() == NULL || ret->value().type() == ASTNode::NODE_LOCALS ||
                    (retObj && retObj->object().type() == PycObject::TYPE_NONE)) {
                /* An EXPLICIT shared trailing `return None` (several branches converge
                   on one None-return block) must be kept — dropping it recompiles to
                   separate per-branch None-returns. */
                if (keepFinalRetNone)
                    break;
                clean->removeLast();  // Always an extraneous return statement
            } else {
                break;
            }
        }
        if (!(code->flags() & PycCode::CO_OPTIMIZED)) {
            for (const auto& n : clean->nodes()) {
                if (n != nullptr && n.type() == ASTNode::NODE_BLOCK)
                    strip_module_trailing_return(n.cast<ASTBlock>(), false);
            }
        }
    }
    if (printClassDocstring)
        printClassDocstring = false;
    // This is outside the clean check so a source block will always
    // be compilable, even if decompylation failed.
    if (clean->nodes().size() == 0 && !code.isIdent(mod->code()))
        clean->append(new ASTKeyword(ASTKeyword::KW_PASS));

    bool part1clean = cleanBuild;

    if (printDocstringAndGlobals) {
        if (code->consts()->size())
            print_docstring(code->getConst(0), cur_indent + 1, mod, pyc_output);

        PycCode::globals_t globs = code->getGlobals();
        if (globs.size()) {
            start_line(cur_indent + 1, pyc_output);
            pyc_output << "global ";
            bool first = true;
            for (const auto& glob : globs) {
                if (!first)
                    pyc_output << ", ";
                pyc_output << glob->value();
                first = false;
            }
            pyc_output << "\n";
        }
        /* Emit `nonlocal` for a free variable that this nested function ASSIGNS
           (STORE_DEREF). Without the declaration, recompiling the assignment makes
           the name a fresh local (STORE_FAST) instead of rebinding the enclosing
           scope's cell -- changing every deref in the body. `nonlocal` produces no
           bytecode, so this is co_code-neutral except for restoring the free/cell
           binding. A free var that is only read needs no declaration. In 3.11 the
           free vars share the combined locals-plus array; a slot is a free var iff
           its kind byte carries CO_FAST_FREE (0x80). */
        if (mod->verCompare(3, 11) >= 0 && code->code() != nullptr
                && code->localKinds() != nullptr) {
            PycRef<PycString> kinds = code->localKinds();
            int nkinds = kinds->length();
            const char* kv = kinds->value();
            std::set<int> assignedFree;
            PycBuffer s(code->code()->value(), code->code()->length());
            int op, arg, p = 0;
            while (!s.atEof()) {
                int pv = p;
                bc_next(s, mod, op, arg, p);
                if (p <= pv) break;
                if (op == Pyc::STORE_DEREF_A && arg >= 0 && arg < nkinds
                        && (kv[arg] & 0x80))     // CO_FAST_FREE
                    assignedFree.insert(arg);
            }
            if (!assignedFree.empty()) {
                start_line(cur_indent + 1, pyc_output);
                pyc_output << "nonlocal ";
                bool first = true;
                for (int idx : assignedFree) {   // ascending; order is irrelevant
                    PycRef<PycString> nm = code->getLocal(idx);
                    if (nm == nullptr)
                        continue;
                    if (!first)
                        pyc_output << ", ";
                    pyc_output << nm->value();
                    first = false;
                }
                pyc_output << "\n";
            }
        }
        printDocstringAndGlobals = false;
    }

    /* ---- Orphaned dead code-const recovery ----
       CPython's -OO / dead-branch elimination drops the MAKE_FUNCTION (and paired
       LOAD_CONST) of a def/lambda/comprehension guarded by a compile-time-false
       branch (`if __debug__:`), but KEEPS its code object orphaned in co_consts.
       With nothing loading it, pycdc would otherwise omit it entirely, silently
       dropping recoverable source: the orphan only round-trips as a dead
       `if __debug__:` block (a recompile at -OO eliminates the block again and
       restores the orphaned const at the same code-const slot). Emit each orphan
       under `if __debug__:` at its position among the scope's code consts. */
    if (cleanBuild && clean != nullptr && code->consts() != nullptr) {
        const int nconsts = code->consts()->size();
        std::set<int> loadedIdx;
        {
            PycBuffer s(code->code()->value(), code->code()->length());
            int o, a, p = 0;
            while (!s.atEof()) {
                bc_next(s, mod, o, a, p);
                if (o == Pyc::LOAD_CONST_A)
                    loadedIdx.insert(a);
            }
        }
        std::vector<PycRef<PycCode> > orphanCode;
        std::vector<int> orphanL;   // # live code consts preceding the orphan
        int liveSeen = 0;
        for (int i = 0; i < nconsts; i++) {
            PycRef<PycObject> c = code->getConst(i);
            if (c == nullptr || c.type() != PycObject::TYPE_CODE)
                continue;
            if (loadedIdx.find(i) == loadedIdx.end()) {
                orphanCode.push_back(c.cast<PycCode>());
                orphanL.push_back(liveSeen);
            } else {
                liveSeen++;
            }
        }
        if (!orphanCode.empty()) {
            auto makeStmt = [&](PycRef<PycCode> oc) -> PycRef<ASTNode> {
                PycRef<ASTNode> codeObj = new ASTObject(oc.cast<PycObject>());
                const char* nm = oc->name() != nullptr ? oc->name()->value() : "";
                PycRef<PycString> dn = new PycString();
                if (strcmp(nm, "<lambda>") == 0) {
                    dn->setValue("__");
                    return new ASTStore(new ASTFunction(codeObj, {}, {}), new ASTName(dn));
                }
                if (strcmp(nm, "<dictcomp>") == 0 || strcmp(nm, "<setcomp>") == 0
                        || strcmp(nm, "<listcomp>") == 0 || strcmp(nm, "<genexpr>") == 0) {
                    PycRef<PycString> it = new PycString(); it->setValue("_");
                    PycRef<ASTNode> comp = InlineComprehension(oc, mod, new ASTName(it));
                    if (comp == nullptr)
                        return nullptr;
                    dn->setValue("__");
                    return new ASTStore(comp, new ASTName(dn));
                }
                dn->setValue(nm);
                return new ASTStore(new ASTFunction(codeObj, {}, {}), new ASTName(dn));
            };
            auto& lst = clean->mutableNodes();
            auto insertPos = [&](int L) -> ASTNodeList::list_t::iterator {
                if (L <= 0)
                    return lst.begin();
                int seen = 0;
                for (auto it = lst.begin(); it != lst.end(); ++it) {
                    if (stmtEmitsScopeCode(*it) && ++seen == L)
                        return ++it;
                }
                return lst.end();
            };
            // Group orphans that share the same insertion point (same L) into one block.
            size_t i = 0;
            while (i < orphanCode.size()) {
                int L = orphanL[i];
                PycRef<ASTCondBlock> ifblk;
                {
                    PycRef<PycString> dbg = new PycString(); dbg->setValue("__debug__");
                    ifblk = new ASTCondBlock(ASTBlock::BLK_IF, 0, new ASTName(dbg), false);
                }
                bool any = false;
                while (i < orphanCode.size() && orphanL[i] == L) {
                    PycRef<ASTNode> st = makeStmt(orphanCode[i]);
                    if (st != nullptr) { ifblk->append(st); any = true; }
                    i++;
                }
                if (any)
                    lst.insert(insertPos(L), ifblk.cast<ASTNode>());
            }
        }
    }

    /* Restore INTERIOR stripped statements: a docstring / bare-constant /
       debug-only line removed under -OO between two real statements survives as a
       line-anchor NOP the decompiler drops. Reproduce it by inserting a `...`
       placeholder at the matching between-statements slot. Confined to a FLAT
       straight-line body (module const tables, enum-style class bodies): no
       exception table, no compound-block nodes, and the count of bytecode
       statement terminators equal to the rendered node count, so terminators map
       1:1 onto nodes in order and the slot is exact. Runs BEFORE the leading
       placeholder pass so the node count is still pristine. */
    if (cleanBuild && clean != nullptr && mod->verCompare(3, 11) >= 0
            && code->code() != nullptr
            && code->exceptionTableEntries().empty()
            && clean->nodes().size() >= 2) {
        bool flatNodes = true;
        for (const auto& n : clean->nodes())
            if (n != nullptr && n.type() == ASTNode::NODE_BLOCK) { flatNodes = false; break; }
        if (flatNodes) {
            struct II { int op, arg, off; };
            std::vector<II> ins;
            {
                PycBuffer s(code->code()->value(), code->code()->length());
                int op, arg, p = 0;
                while (!s.atEof()) {
                    int prev = p;
                    bc_next(s, mod, op, arg, p);
                    if (p <= prev) break;
                    if (op == Pyc::CACHE) continue;
                    ins.push_back({op, arg, prev});
                }
            }
            /* Skip the code-object prologue (RESUME/cells/generator) and, for a
               class body, the __module__/__qualname__ stores. */
            size_t k = 0; bool sawRG = false;
            while (k < ins.size()) {
                int op = ins[k].op;
                if (op == Pyc::RESUME_A || op == Pyc::MAKE_CELL_A
                        || op == Pyc::COPY_FREE_VARS_A || op == Pyc::RETURN_GENERATOR) {
                    if (op == Pyc::RETURN_GENERATOR) sawRG = true;
                    k++; continue;
                }
                if (op == Pyc::POP_TOP && sawRG) { sawRG = false; k++; continue; }
                break;
            }
            auto nameIs = [&](int idx, const char* n) {
                PycRef<PycString> s = code->getName(idx);
                return s != nullptr && s->isEqual(n);
            };
            if (k + 3 < ins.size()
                    && ins[k].op == Pyc::LOAD_NAME_A && nameIs(ins[k].arg, "__name__")
                    && ins[k + 1].op == Pyc::STORE_NAME_A && nameIs(ins[k + 1].arg, "__module__")
                    && ins[k + 2].op == Pyc::LOAD_CONST_A
                    && ins[k + 3].op == Pyc::STORE_NAME_A && nameIs(ins[k + 3].arg, "__qualname__"))
                k += 4;
            auto isTerm = [](int op) {
                return op == Pyc::STORE_NAME_A || op == Pyc::STORE_GLOBAL_A
                    || op == Pyc::STORE_FAST_A || op == Pyc::STORE_DEREF_A
                    || op == Pyc::STORE_SUBSCR || op == Pyc::STORE_ATTR_A
                    || op == Pyc::POP_TOP || op == Pyc::DELETE_NAME_A
                    || op == Pyc::DELETE_GLOBAL_A || op == Pyc::DELETE_FAST_A;
            };
            /* A NOP whose statement ends in a def-creation (MAKE_FUNCTION then a
               STORE of the function's name) is a signature-continuation anchor:
               a def whose signature spans several source lines leaves one NOP per
               continuation line just before the creation sequence. When pycdc
               renders that signature on a SINGLE line it regenerates NONE of
               them, so every NOP in the run must be reproduced (the
               line-uniqueness rule below would otherwise drop the ones sharing
               the def's line). The single-line condition is verified per-def
               against the rendered AST (sigRendersInline) -- an annotation or
               default holding a list/set/dict display or comprehension makes
               pycdc wrap the signature and regenerate some anchors itself. */
            auto isDefSigNop = [&](size_t j) -> bool {
                for (size_t t = j + 1; t < ins.size(); t++) {
                    if (!isTerm(ins[t].op))
                        continue;
                    if (ins[t].op == Pyc::STORE_NAME_A || ins[t].op == Pyc::STORE_FAST_A
                            || ins[t].op == Pyc::STORE_GLOBAL_A || ins[t].op == Pyc::STORE_DEREF_A) {
                        int p = (int)t - 1;
                        while (p >= 0 && ins[p].op == Pyc::CACHE) p--;
                        return p >= 0 && ins[p].op == Pyc::MAKE_FUNCTION_A;
                    }
                    return false;
                }
                return false;
            };
            /* True when the def stored as `defNode` renders its whole signature
               on one physical line -- i.e. no annotation or default renders with
               an embedded newline, matching what the NODE_STORE signature render
               actually emits. The real render disables collection compaction
               only for a signature whose source spanned multiple lines (some
               annotation/default sits on a later line than the `def`); this
               scratch check applies the identical gate so the two agree. */
            auto sigRendersInline = [&](const PycRef<ASTNode>& defNode) -> bool {
                if (defNode == nullptr || defNode.type() != ASTNode::NODE_STORE)
                    return false;
                PycRef<ASTNode> fn = defNode.cast<ASTStore>()->src();
                if (fn == nullptr || fn.type() != ASTNode::NODE_FUNCTION)
                    return false;
                PycRef<ASTFunction> f = fn.cast<ASTFunction>();
                /* A one-per-line const-default signature (sigNopAnchors set) is
                   rendered multi-line and regenerates its anchor NOPs itself, so
                   it is NOT inline and needs no placeholder. */
                if (f->sigNopAnchors() >= 0)
                    return false;
                int defLine = defNode->srcLine();
                bool srcMulti = false;
                auto laterLine = [&](const PycRef<ASTNode>& n) {
                    if (n == nullptr)
                        return;
                    int l = n->srcLine();
                    if (l > 0 && defLine > 0 && l != defLine)
                        srcMulti = true;
                };
                for (const auto& a : f->annotations())
                    laterLine(a.second);
                for (const auto& d : f->defargs())
                    laterLine(d);
                for (const auto& d : f->kwdefargs())
                    laterLine(d);
                auto hasNL = [&](const PycRef<ASTNode>& n) -> bool {
                    if (n == nullptr)
                        return false;
                    std::ostringstream ss;
                    if (srcMulti) g_noCompact++;
                    print_src(n, mod, ss);
                    if (srcMulti) g_noCompact--;
                    return ss.str().find('\n') != std::string::npos;
                };
                for (const auto& a : f->annotations())
                    if (hasNL(a.second)) return false;
                for (const auto& d : f->defargs())
                    if (hasNL(d)) return false;
                for (const auto& d : f->kwdefargs())
                    if (hasNL(d)) return false;
                return true;
            };
            auto nodeAtSlot = [&](int slot) -> PycRef<ASTNode> {
                if (slot < 0 || slot >= (int)clean->nodes().size())
                    return nullptr;
                auto it = clean->nodes().begin();
                std::advance(it, slot);
                return *it;
            };
            /* Cheap first pass (no line lookups): find each interior NOP and the
               statement-terminator count before it. Bail early when there are no
               interior NOPs -- avoids the O(n^2) line scan below on huge
               generated data tables (tens of thousands of instructions, zero
               NOPs). */
            struct NopC { int slot, off; bool defSig; };
            std::vector<NopC> nopCand;
            int termCount = 0;
            for (size_t j = k; j < ins.size(); j++) {
                if (ins[j].op == Pyc::NOP) {
                    if (termCount >= 1 && !sigAnchorNopOffs.count(ins[j].off))
                        nopCand.push_back({termCount, ins[j].off, isDefSigNop(j)});
                } else if (isTerm(ins[j].op)) {
                    termCount++;
                }
            }
            int origSize = (int)clean->nodes().size();
            /* lineForOffset re-parses the whole line table per call, so bound the
               line-uniqueness scan to a sane body size (real stripped-statement
               bodies are small; giant data tables are not a target). */
            std::vector<std::pair<int,int> > nopAfter;  // {slot, byte offset}
            if (!nopCand.empty() && termCount == origSize && ins.size() <= 4000) {
                /* A genuine stripped statement occupies its OWN source line, so
                   its line-anchor NOP is the only thing on that line. A NOP whose
                   line ALSO carries code is a redundant anchor for that code (e.g.
                   a following def's multi-line signature / annotation) which the
                   decompiler regenerates when it renders the code -- inserting a
                   placeholder there would double it. */
                std::set<int> codeLines;
                for (size_t j = 0; j < ins.size(); j++)
                    if (ins[j].op != Pyc::NOP)
                        codeLines.insert(code->lineForOffset(ins[j].off));
                std::unordered_map<int, int> sigInlineCache;  // slot -> 0/1
                for (const auto& nc : nopCand) {
                    bool useDefSig = false;
                    if (nc.defSig) {
                        auto ci = sigInlineCache.find(nc.slot);
                        int v;
                        if (ci != sigInlineCache.end())
                            v = ci->second;
                        else {
                            v = sigRendersInline(nodeAtSlot(nc.slot)) ? 1 : 0;
                            sigInlineCache[nc.slot] = v;
                        }
                        useDefSig = (v == 1);
                    }
                    if (useDefSig
                            || codeLines.find(code->lineForOffset(nc.off)) == codeLines.end())
                        nopAfter.push_back({nc.slot, nc.off});
                }
            } else if (!nopCand.empty() && termCount != origSize && ins.size() <= 4000) {
                /* The terminator count exceeds the AST child count: a statement in
                   this flat body compiles to several terminators but ONE node (a
                   multi-name `from x import a, b`, an annotated class attribute
                   `x: T` storing into __annotations__ plus a value, a chained/tuple
                   assign). The terminator-index slot above would then be too high,
                   so map each line-free stripped NOP to its child slot by SOURCE
                   OFFSET instead: the slot is the number of children whose source
                   offset precedes the NOP, and the NOP must sit strictly between the
                   two children that bracket it. Requires ascending child offsets. */
                std::set<int> codeLines;
                for (size_t j = 0; j < ins.size(); j++)
                    if (ins[j].op != Pyc::NOP)
                        codeLines.insert(code->lineForOffset(ins[j].off));
                std::vector<int> childOff;
                bool ok = true;
                int prevOff = -1;
                for (const auto& n : clean->nodes()) {
                    int so = n ? n->srcOff() : -1;
                    if (so < 0 || so <= prevOff) { ok = false; break; }  // missing / non-ascending
                    childOff.push_back(so);
                    prevOff = so;
                }
                if (ok) {
                    for (const auto& nc : nopCand) {
                        int slot = 0;
                        for (int c : childOff)
                            if (c < nc.off) slot++;
                        if (slot < 1 || slot >= (int)childOff.size())
                            continue;   // leading / trailing -> other passes handle
                        if (!(childOff[slot - 1] < nc.off && nc.off < childOff[slot]))
                            continue;   // not strictly bracketed between two statements
                        /* A NOP whose statement is a following def's multi-line
                           signature anchor is regenerated when pycdc renders that
                           signature (one param per line), so place it ONLY if the
                           signature renders inline; otherwise require a code-free
                           line (a genuine stripped statement owns its own line). */
                        bool place;
                        if (nc.defSig)
                            place = sigRendersInline(nodeAtSlot(slot));
                        else
                            place = codeLines.find(code->lineForOffset(nc.off)) == codeLines.end();
                        if (place)
                            nopAfter.push_back({slot, nc.off});
                    }
                }
            }
            if (!nopAfter.empty()) {
                /* nopAfter is ascending (forward walk); insert from the highest
                   slot down so earlier insertions do not shift later indices. */
                for (auto rit = nopAfter.rbegin(); rit != nopAfter.rend(); ++rit) {
                    int pos = rit->first;
                    if (pos < 1 || pos >= origSize)
                        continue;
                    auto it = clean->mutableNodes().begin();
                    std::advance(it, pos);
                    PycRef<ASTNode> ph = new ASTObject(Pyc_Ellipsis);
                    ph->setSrcOff(rit->second);
                    clean->mutableNodes().insert(it, ph);
                }
            }
        }
    }

    /* General interior placeholder pass for NON-flat bodies (nested if/for/while/
       try suites, or bodies with an exception table) that the flat fast path
       above skips. Places each line-unique stripped NOP into its enclosing suite
       by byte-range; leading NOPs are left to the pass below. */
    if (cleanBuild && clean != nullptr && mod->verCompare(3, 11) >= 0
            && code->code() != nullptr && clean->nodes().size() >= 2) {
        bool hasBlock = false;
        for (const auto& n : clean->nodes())
            if (n != nullptr && n.type() == ASTNode::NODE_BLOCK) { hasBlock = true; break; }
        if (hasBlock || !code->exceptionTableEntries().empty()) {
            std::vector<int> nops = collectStrippedNops(code, mod, sigAnchorNopOffs);
            if (!nops.empty()) {
                std::set<int> remaining(nops.begin(), nops.end());
                insertGeneralPlaceholders(clean->mutableNodes(), remaining, 0);
                /* NOP-locator: place each still-unplaced stripped bare-const NOP
                   (now that try-heads are excluded, `remaining` holds genuine
                   dropped statements). When the preceding instruction ends a
                   RETURNing `if` body the NOP sits at that branch's merge -- place
                   a `...` sibling after the `if`. Otherwise place it directly after
                   the statement built by the preceding instruction, unless that
                   predecessor is another branch/exit boundary (jump/raise), where
                   the position is not a simple after-statement slot. */
                if (!remaining.empty()) {
                    struct II { int op, off; };
                    std::vector<II> bcins;
                    {
                        PycBuffer s(code->code()->value(), code->code()->length());
                        int op, arg, p = 0;
                        while (!s.atEof()) {
                            int pv = p;
                            bc_next(s, mod, op, arg, p);
                            if (p <= pv) break;
                            if (op == Pyc::CACHE) continue;
                            bcins.push_back({op, pv});
                        }
                    }
                    auto isBoundary = [](int op) {
                        return op == Pyc::RETURN_VALUE || op == Pyc::RETURN_CONST_A
                            || op == Pyc::RAISE_VARARGS_A || op == Pyc::RERAISE_A
                            || op == Pyc::JUMP_FORWARD_A || op == Pyc::JUMP_BACKWARD_A
                            || op == Pyc::JUMP_ABSOLUTE_A
                            || op == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                            || op == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                            || op == Pyc::POP_JUMP_FORWARD_IF_NONE_A
                            || op == Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A
                            || op == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A
                            || op == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A;
                    };
                    std::vector<int> todo(remaining.begin(), remaining.end());
                    for (int o : todo) {
                        int prevOff = -1, prevOp = -1;
                        for (const auto& x : bcins) {
                            if (x.off >= o) break;
                            if (x.op == Pyc::NOP) continue;
                            prevOff = x.off; prevOp = x.op;
                        }
                        if (prevOff < 0)
                            continue;
                        bool exitPrev = (prevOp == Pyc::RETURN_VALUE
                                || prevOp == Pyc::RETURN_CONST_A
                                || prevOp == Pyc::RAISE_VARARGS_A);
                        if (exitPrev) {
                            /* The NOP is at a merge whose preceding branch exited with
                               a `return` or `raise`. Try the last-leaf `if` match first;
                               if it misses, the NOP is the merge after an `if` OR a `match`
                               statement (all its cases return, so a statement after the
                               match lands at its merge = block end), keyed on the block
                               END == NOP offset. A `raise` inside an exception-table
                               protected range is part of entangled try/finally unwinding
                               whose merge anchor pycdc regenerates -- skip it entirely
                               (the `return` merge keeps its established behaviour). */
                            bool inProtected = false;
                            for (const auto& e : code->exceptionTableEntries())
                                if (prevOff >= e.start_offset && prevOff < e.end_offset) {
                                    inProtected = true; break;
                                }
                            if (prevOp == Pyc::RAISE_VARARGS_A && inProtected)
                                continue;
                            if (insertPlaceholderAfterIfEndingAt(clean->mutableNodes(),
                                                                 prevOff, o, 0)) {
                                remaining.erase(o);
                                continue;
                            }
                            if (!inProtected
                                    && insertPlaceholderAfterBlockEndingAt(
                                           clean->mutableNodes(), o, o, 0))
                                remaining.erase(o);
                            continue;
                        }
                        bool condBranch = (prevOp == Pyc::POP_JUMP_FORWARD_IF_FALSE_A);
                        if (condBranch) {
                            /* The NOP starts the fall-through `if` body -- but only
                               for a clean `if` that is not itself inside an
                               exception-table protected range (an `if` nested in a
                               `try`/`with`/loop-in-try has entangled control flow
                               whose branch-start anchor pycdc regenerates by other
                               means). */
                            bool inProtected = false;
                            for (const auto& e : code->exceptionTableEntries())
                                if (prevOff >= e.start_offset && prevOff < e.end_offset) {
                                    inProtected = true; break;
                                }
                            if (!inProtected
                                    && insertPlaceholderAtIfBodyStart(clean->mutableNodes(),
                                                                      prevOff, o, 0))
                                remaining.erase(o);
                            continue;
                        }
                        if (isBoundary(prevOp))
                            continue;
                        if (insertPlaceholderAfterOff(clean->mutableNodes(), prevOff, o, 0)) {
                            remaining.erase(o);
                            continue;
                        }
                        /* A stripped statement as the FIRST statement of a `for` body:
                           the preceding instruction is the loop-variable STORE (which
                           unpacks the FOR_ITER value), so no AST node has that offset.
                           Detect it -- walk back over the store run (and any tuple
                           UNPACK) to a FOR_ITER -- and prepend the `...` to the `for`
                           body by containment. Skip a loop var store inside a protected
                           range (entangled unwinding regenerates the anchor). */
                        bool storePrev = (prevOp == Pyc::STORE_FAST_A
                                || prevOp == Pyc::STORE_NAME_A
                                || prevOp == Pyc::STORE_GLOBAL_A
                                || prevOp == Pyc::STORE_DEREF_A);
                        if (storePrev) {
                            int pi = -1;
                            for (int k2 = 0; k2 < (int)bcins.size(); k2++)
                                if (bcins[k2].off == prevOff) { pi = k2; break; }
                            bool forVar = false;
                            for (int k2 = pi; k2 >= 0; k2--) {
                                int op2 = bcins[k2].op;
                                if (op2 == Pyc::STORE_FAST_A || op2 == Pyc::STORE_NAME_A
                                        || op2 == Pyc::STORE_GLOBAL_A || op2 == Pyc::STORE_DEREF_A
                                        || op2 == Pyc::UNPACK_SEQUENCE_A || op2 == Pyc::UNPACK_EX_A)
                                    continue;
                                forVar = (op2 == Pyc::FOR_ITER_A);
                                break;
                            }
                            bool inProtected = false;
                            for (const auto& e : code->exceptionTableEntries())
                                if (prevOff >= e.start_offset && prevOff < e.end_offset) {
                                    inProtected = true; break;
                                }
                            if (forVar && !inProtected
                                    && insertPlaceholderAtForBodyStart(clean->mutableNodes(), o, 0)) {
                                remaining.erase(o);
                                continue;
                            }
                        }
                        /* A stripped statement whose preceding instruction is a
                           POP_TOP (an expression statement whose value is discarded)
                           that ends a fall-through `if` body: the NOP is the branch
                           merge, a sibling right after the `if` (block end == NOP).
                           insertPlaceholderAfterOff can't anchor it (a POP_TOP has no
                           AST node of its own), so place it after that block. Skip
                           when the POP_TOP is inside an exception-table protected
                           range: there the `if`'s merge anchor is entangled with the
                           enclosing try/loop unwinding and pycdc regenerates it, so a
                           placeholder would double the NOP. */
                        if (prevOp == Pyc::POP_TOP) {
                            bool inProtected = false;
                            for (const auto& e : code->exceptionTableEntries())
                                if (prevOff >= e.start_offset && prevOff < e.end_offset) {
                                    inProtected = true; break;
                                }
                            if (!inProtected
                                    && insertPlaceholderAfterBlockEndingAt(
                                           clean->mutableNodes(), o, o, 0))
                                remaining.erase(o);
                        }
                    }
                }
            }
        }
    }

    /* Restore leading statements the compiler stripped to bare NOPs (see
       leadingStrippedNops): one `...` placeholder per NOP so the recompiled
       co_code reproduces the original's head-of-body NOPs byte-for-byte. */
    if (cleanBuild && clean != nullptr && !clean->nodes().empty()
            && !leadingNopAlreadyHandled(clean->nodes().front())) {
        std::vector<int> leadOffs;
        int nStripped = leadingStrippedNops(code, mod, &leadOffs, &sigAnchorNopOffs);
        /* Prepend one `...` per stripped leading statement. Each `...` is a
           bare-constant expression statement, which recompiles to a line-anchor
           NOP as long as it is NOT the final statement -- the existing body (or a
           trailing `pass` pycdc already synthesized for an empty body) keeps the
           last placeholder non-final, so N placeholders yield N NOPs.

           Injecting MORE than one placeholder is unsafe when the body's first
           statement is a compound block (`try`/`while`/`for`/... rendered as a
           NODE_BLOCK): its header carries a line-anchor NOP that pycdc
           regenerates on its own, so one of the counted leading NOPs belongs to
           it, not to a stripped statement. Restrict the multi-placeholder case to
           a simple/`pass`/def front and keep the proven single placeholder when
           the body opens with a compound block. */
        PycRef<ASTNode> front = clean->nodes().front();
        bool frontIsBlock = (front != nullptr
                && front.type() == ASTNode::NODE_BLOCK);
        /* A leading `try` (BLK_CONTAINER) or `while` (BLK_WHILE) regenerates one
           head line-anchor NOP on its own; nested ones (e.g. `try: try:`) each
           add another. Count that self-regenerated depth by descending through
           the front's leading try/while blocks so those NOPs are not
           double-counted as stripped statements. Only try/while are counted
           (if/for/with/match heads regenerate no head NOP), so this can only
           under-inject -- never over-inject onto a byte-matching function. */
        int frontAnchors = 0;
        {
            PycRef<ASTNode> n = front;
            while (n != nullptr && n.type() == ASTNode::NODE_BLOCK) {
                PycRef<ASTBlock> blk = n.cast<ASTBlock>();
                ASTBlock::BlkType bt = blk->blktype();
                if (bt == ASTBlock::BLK_CONTAINER
                        || (bt == ASTBlock::BLK_WHILE && is_infinite_while(blk))) {
                    /* Only an infinite `while True:` (like a `try`) regenerates a
                       leading head anchor NOP. A CONDITIONAL `while cond:` evaluates
                       its condition first, so its head carries a real instruction
                       and emits no leading NOP -- counting it wrongly cancels a
                       genuine stripped leading statement (e.g. a coroutine docstring
                       before a `while cond:` loop). Treat it like if/for: stop. */
                    frontAnchors++;
                    n = blk->nodes().empty() ? nullptr : blk->nodes().front();
                } else if (bt == ASTBlock::BLK_TRY) {
                    /* the try body inside a container: descend without counting */
                    n = blk->nodes().empty() ? nullptr : blk->nodes().front();
                } else {
                    break;
                }
            }
        }
        int nInject;
        if (frontAnchors > 0) {
            nInject = nStripped - frontAnchors;
            if (nInject < 0)
                nInject = 0;
        } else {
            nInject = (nStripped >= 2 && frontIsBlock) ? 0 : nStripped;
        }
        /* Insert in reverse so the placeholder for the earliest leading NOP ends
           up front-most; tag each with its NOP byte offset (the first nInject
           leading NOPs are the stripped statements) so its width can be set. */
        for (int k = nInject - 1; k >= 0; k--) {
            PycRef<ASTNode> ph = new ASTObject(Pyc_Ellipsis);
            if (k < (int)leadOffs.size())
                ph->setSrcOff(leadOffs[k]);
            clean->mutableNodes().insert(clean->mutableNodes().begin(), ph);
        }
    }

    /* A stripped module docstring can sit before a `from __future__` import -- the
       one kind of statement allowed to precede it. The `...` placeholder above is
       skipped for a `__future__` front (leadingNopAlreadyHandled) because `...` is
       illegal there, but a string-literal docstring IS legal and, compiled under
       -OO, is stripped to exactly the same head line-anchor NOP. Prepend one such
       string placeholder when the front is a `__future__` import and the head
       carries a stripped leading NOP. Only one: a second bare string before a
       future import is itself a syntax error. */
    if (cleanBuild && clean != nullptr && !clean->nodes().empty()) {
        PycRef<ASTNode> front = clean->nodes().front();
        bool futureFront = front != nullptr && front.type() == ASTNode::NODE_IMPORT
                && front.cast<ASTImport>()->name() != nullptr
                && front.cast<ASTImport>()->name().type() == ASTNode::NODE_NAME
                && front.cast<ASTImport>()->name().cast<ASTName>()->name() != nullptr
                && front.cast<ASTImport>()->name().cast<ASTName>()->name()->isEqual("__future__");
        if (futureFront) {
            std::vector<int> leadOffs;
            int nStripped = leadingStrippedNops(code, mod, &leadOffs, &sigAnchorNopOffs);
            if (nStripped >= 1) {
                PycRef<PycString> str = new PycString(PycObject::TYPE_UNICODE);
                str->setValue("");
                PycRef<ASTNode> ph = new ASTObject(str.cast<PycObject>());
                if (!leadOffs.empty())
                    ph->setSrcOff(leadOffs[0]);
                clean->mutableNodes().insert(clean->mutableNodes().begin(), ph);
            }
        }
    }

    /* Recover consts/names orphaned by a stripped `if __debug__:` block (see
       recoverOrphanTableEntries): render the block instead of a bare `...` so the
       recompile re-registers the orphaned table entries at their original indices. */
    if (cleanBuild)
        recoverOrphanTableEntries(clean, code, mod);

    /* Emit module-level `global` declarations. A `global x` at module scope
       forces STORE_GLOBAL for x; the decompiler marks such names (markGlobal)
       but only emits `global` for FUNCTION bodies, so a module's globals are
       dropped and recompile to STORE_NAME. Emit them at the top of the module
       body. Skipped when the module has a `from __future__` import (which must
       be the first statement, so a preceding `global` would be illegal). */
    if (cleanBuild && clean != nullptr && code.isIdent(mod->code())) {
        PycCode::globals_t globs = code->getGlobals();
        bool hasFuture = false;
        for (const auto& n : clean->nodes()) {
            if (n != nullptr && n.type() == ASTNode::NODE_IMPORT) {
                PycRef<ASTNode> nm = n.cast<ASTImport>()->name();
                if (nm != nullptr && nm.type() == ASTNode::NODE_NAME
                        && nm.cast<ASTName>()->name() != nullptr
                        && nm.cast<ASTName>()->name()->isEqual("__future__")) {
                    hasFuture = true;
                    break;
                }
            }
        }
        if (globs.size()) {
            /* A module `global x` must follow any `from __future__` import (which
               has to be the module's first statement) but precede x's assignment.
               When the module leads with future imports, render THOSE first and
               drop them from the body list, then emit `global` before the rest;
               otherwise emit it at the very top. `global` produces no bytecode, so
               placing it after the futures is co_code-neutral. */
            if (hasFuture) {
                auto& nodes = clean->mutableNodes();
                /* Render and drop everything that must precede the `global`: the
                   leading docstring / stripped-statement placeholder (a NODE_OBJECT)
                   and the `from __future__` import(s). Stop at the first other
                   statement (`global` may follow imports; it only has to precede the
                   assignment of a global name). */
                while (!nodes.empty() && nodes.front() != nullptr) {
                    int ft = nodes.front().type();
                    bool skip = false;
                    if (ft == ASTNode::NODE_OBJECT) {
                        skip = true;
                    } else if (ft == ASTNode::NODE_IMPORT) {
                        PycRef<ASTNode> nm = nodes.front().cast<ASTImport>()->name();
                        skip = nm != nullptr && nm.type() == ASTNode::NODE_NAME
                                && nm.cast<ASTName>()->name() != nullptr
                                && nm.cast<ASTName>()->name()->isEqual("__future__");
                    }
                    if (!skip)
                        break;
                    print_src(nodes.front(), mod, pyc_output);
                    pyc_output << "\n";
                    nodes.erase(nodes.begin());
                }
            }
            start_line(cur_indent, pyc_output);
            pyc_output << "global ";
            bool first = true;
            for (const auto& glob : globs) {
                if (!first)
                    pyc_output << ", ";
                pyc_output << glob->value();
                first = false;
            }
            pyc_output << "\n";
        }
    }

    if (cleanBuild && clean != nullptr)
        setPlaceholderWidths(clean.cast<ASTNode>(), code, true, 0);

    if (cleanBuild && clean != nullptr)
        coalesceElifChains(clean->mutableNodes(), true, 0);
    if (cleanBuild && clean != nullptr)
        markEpilogueSuppress(clean->mutableNodes(), true, false, 0);
    /* Enable line-faithful multi-line expression rendering (master switch). Each
       wrap site further gates on lfSafe(): a specific expression is wrapped only
       when it carries no nested code object, whose source line the re-wrap would
       shift (changing its line-anchor NOP -> co_code, the old Fix B wall). A flat
       expression carries no such NOP, so wrapping it at its recorded element
       lines is co_code-inert. g_funcHasNestedCode lets a fully flat function skip
       the per-expression scan. Save/restore around the recursive render. */
    bool savedLineFaithful = g_lineFaithful;
    bool savedFuncNested = g_funcHasNestedCode;
    {
        bool hasNestedCode = false;
        if (code->consts() != nullptr) {
            for (int ci = 0; ci < code->consts()->size(); ci++) {
                PycRef<PycObject> cc = code->consts()->get(ci);
                if (cc != nullptr && cc->type() == PycObject::TYPE_CODE) {
                    hasNestedCode = true;
                    break;
                }
            }
        }
        g_funcHasNestedCode = hasNestedCode;
        g_lineFaithful = true;
    }
    print_src(source, mod, pyc_output);
    g_lineFaithful = savedLineFaithful;
    g_funcHasNestedCode = savedFuncNested;

    if (!cleanBuild || !part1clean) {
        start_line(cur_indent, pyc_output);
        pyc_output << "# WARNING: Decompyle incomplete\n";
    }

    code_seen.erase((PycCode *)code);
}
