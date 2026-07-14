#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>
#include <set>
#include <vector>
#include "ASTree.h"
#include "ASTree_priv.h"
#include "bytecode.h"

/* ==========================================================================
 * ASTFaithful.cpp -- PHASE 2 of the decompiler: faithfulness passes.
 *
 * Post-build transforms on the reconstructed AST that make the emitted source
 * recompile to the SAME byte-code and source positions as the original .pyc:
 *   - recover statements the -OO compiler stripped (docstrings, asserts,
 *     `if __debug__:` blocks) as line/position-anchoring `...` placeholders;
 *   - recover constant/name table entries orphaned by that stripping;
 *   - fold terminal if-return chains into if/elif/else;
 *   - suppress the implicit fall-off `return None` epilogue.
 * These are invoked by decompyle() (in ASTree.cpp) between build and render.
 * ========================================================================== */

bool is_infinite_while(const PycRef<ASTBlock>& blk)
{
    if (blk->blktype() != ASTBlock::BLK_WHILE)
        return false;
    PycRef<ASTCondBlock> cb = blk.try_cast<ASTCondBlock>();
    if (cb == NULL || cb->negative() || cb->cond() == NULL
            || cb->cond().type() != ASTNode::NODE_OBJECT)
        return false;
    return cb->cond().cast<ASTObject>()->object().type() == PycObject::TYPE_TRUE;
}

void strip_module_trailing_return(PycRef<ASTBlock> blk, bool inInfWhile = false)
{
    ASTBlock::BlkType selfType = blk->blktype();
    bool blkIsLoop = (selfType == ASTBlock::BLK_WHILE || selfType == ASTBlock::BLK_FOR
            || selfType == ASTBlock::BLK_ASYNCFOR);
    bool effFlag = blkIsLoop ? is_infinite_while(blk) : inInfWhile;
    for (const auto& n : blk->nodes()) {
        if (n != nullptr && n.type() == ASTNode::NODE_BLOCK)
            strip_module_trailing_return(n.cast<ASTBlock>(), effFlag);
    }
    while (!blk->nodes().empty()
            && blk->nodes().back().type() == ASTNode::NODE_RETURN) {
        PycRef<ASTReturn> ret = blk->nodes().back().cast<ASTReturn>();
        if (ret->rettype() != ASTReturn::RETURN)
            break;
        PycRef<ASTObject> retObj = ret->value().try_cast<ASTObject>();
        if (ret->value() == NULL || ret->value().type() == ASTNode::NODE_LOCALS
                || (retObj && retObj->object().type() == PycObject::TYPE_NONE)) {
            if (effFlag) {
                blk->removeLast();
                blk->append(new ASTKeyword(ASTKeyword::KW_BREAK));
                break;
            }
            blk->removeLast();
        } else {
            break;
        }
    }
    if (blk->nodes().empty())
        blk->append(new ASTKeyword(ASTKeyword::KW_PASS));
}

/* True if a top-level suite statement introduces a code const of its own scope
   (a def / lambda / class / decorated def). Used to position a recovered orphan
   def among its live siblings (see the orphan-const recovery in decompyle). */
bool stmtEmitsScopeCode(const PycRef<ASTNode>& n)
{
    if (n == nullptr)
        return false;
    switch (n.type()) {
    case ASTNode::NODE_FUNCTION:
    case ASTNode::NODE_CLASS:
        return true;
    case ASTNode::NODE_STORE:
        return stmtEmitsScopeCode(n.cast<ASTStore>()->src());
    case ASTNode::NODE_CALL:
        {
            PycRef<ASTCall> c = n.cast<ASTCall>();
            if (stmtEmitsScopeCode(c->func()))
                return true;
            for (const auto& p : c->pparams())
                if (stmtEmitsScopeCode(p))
                    return true;
            return false;
        }
    default:
        return false;
    }
}

/* Detect a statement the compiler stripped from the FRONT of a code object's
   body (a docstring / bare-constant / `if __debug__` assert removed under -OO),
   which survives only as a line-anchor NOP. pycdc has no source to reconstruct
   it, so it drops the statement and the recompiled body is missing that NOP.
   Returns 1 when a single `...` placeholder should be emitted as the body's
   first statement to restore it, else 0. `...` (like any bare constant) compiles
   to exactly one NOP at every optimization level, whereas a real docstring only
   does so under -OO.

   Detection is on the ORIGINAL bytecode: skip the CACHE/RESUME prologue (and, for
   a class body, the `__module__`/`__qualname__` stores that precede any source
   statement), then require EXACTLY one NOP before the first real instruction. A
   function/module leading docstring's NOP sits right after RESUME; a class
   docstring's NOP sits right after the `__qualname__` store -- pycdc strips those
   same stores from the rendered body, so in both cases the placeholder belongs at
   the front of the cleaned statement list. */
/* The line-unique stripped-statement NOP offsets of a whole code object: a NOP
   whose source line carries NO other (non-NOP) instruction is a docstring /
   bare-constant / debug line the compiler removed under -OO, leaving only that
   line-anchor. A NOP sharing its line with real code is a redundant anchor the
   decompiler regenerates when it renders that code, so it is excluded. Returns
   empty (cheaply) when the body has no NOPs, and bails on very large bodies
   (lineForOffset is O(log n) cached, but giant generated tables have no NOPs
   anyway). */
/* ==========================================================================
 * PHASE 2 -- faithfulness passes + the decompyle() orchestrator
 * Post-build AST transforms that make the reconstructed source recompile to
 * the SAME byte-code and source positions as the original: recover statements
 * stripped under -OO as placeholders, recover orphaned constant/name table
 * entries, fold terminal if-return chains into if/elif/else, suppress the
 * implicit return-None epilogue, and reproduce multi-line layout. decompyle()
 * (at the end) ties the phases together: build -> transform -> render.
 * ========================================================================== */
std::vector<int> collectStrippedNops(PycRef<PycCode> code, PycModule* mod,
                                            const std::set<int>& exclude)
{
    std::vector<int> out;
    if (code->code() == nullptr)
        return out;
    struct II { int op, off; };
    std::vector<II> ins;
    {
        PycBuffer s(code->code()->value(), code->code()->length());
        int op, arg, p = 0;
        while (!s.atEof()) {
            int prev = p;
            bc_next(s, mod, op, arg, p);
            if (p <= prev) break;
            if (op == Pyc::CACHE) continue;
            ins.push_back({op, prev});
        }
    }
    bool anyNop = false;
    for (const auto& i : ins)
        if (i.op == Pyc::NOP) { anyNop = true; break; }
    if (!anyNop || ins.size() > 8000)
        return out;
    std::set<int> codeLines;
    for (const auto& i : ins)
        if (i.op != Pyc::NOP)
            codeLines.insert(code->lineForOffset(i.off));
    /* NOPs inside a `with`/`try` protected range are structural cleanup/exit
       line-anchors (e.g. a `return` inside `with:` triggers `__exit__` codegen)
       that the decompiler regenerates when it renders the construct -- exclude
       them; only NOPs in unprotected, straight-line regions are genuine stripped
       statements. */
    std::vector<PycExceptionTableEntry> ete = code->exceptionTableEntries();
    std::set<int> excStarts;
    for (const auto& e : ete)
        excStarts.insert(e.start_offset);
    for (size_t idx = 0; idx < ins.size(); ++idx) {
        if (ins[idx].op != Pyc::NOP)
            continue;
        int off = ins[idx].off;
        if (exclude.find(off) != exclude.end())
            continue;   // signature-continuation anchor, regenerated by one-per-line render
        if (codeLines.find(code->lineForOffset(off)) != codeLines.end())
            continue;
        bool protectedRange = false;
        for (const auto& e : ete)
            if (off >= e.start_offset && off < e.end_offset) { protectedRange = true; break; }
        if (protectedRange)
            continue;
        /* A `try:` head emits a NOP right before the protected body; pycdc
           regenerates it by rendering the `try`, so it is NOT a stripped statement.
           Walking forward from the NOP, it is a try head iff an exception-table
           protected range begins BEFORE any real (non-NOP) instruction: only the
           stacked try heads (and the protected body which may itself open with a
           NOP) lie between it and the range start. This handles nested `try: try:`
           and a `try` whose body opens with a stripped statement, and merely
           over-excludes a rare bare constant sitting immediately before a `try`
           (safe -- leaves that one anchor unreproduced, never a false NOP). */
        bool tryHead = false;
        for (size_t j = idx + 1; j < ins.size(); ++j) {
            if (excStarts.count(ins[j].off)) { tryHead = true; break; }
            if (ins[j].op != Pyc::NOP) break;
        }
        if (tryHead)
            continue;
        out.push_back(off);
    }
    return out;
}

/* Minimum source byte-offset over a node's leaf descendants (or a huge sentinel
   if it has none), giving a block an effective start without a stored one. */
static int subtreeMinOff(const PycRef<ASTNode>& n, int depth = 0)
{
    if (n == nullptr || depth > 80)
        return 0x7fffffff;
    if (n.type() != ASTNode::NODE_BLOCK) {
        int o = n->srcOff();
        return o >= 0 ? o : 0x7fffffff;
    }
    int m = 0x7fffffff;
    for (const auto& c : n.cast<ASTBlock>()->nodes()) {
        int cm = subtreeMinOff(c, depth + 1);
        if (cm < m) m = cm;
    }
    return m;
}

/* Earliest source byte-offset over an EXPRESSION's operands (not just the node's
   own srcOff, which for a compound expression is the offset of its result-producing
   op, AFTER its operands). Used to find a block's true header start (the first byte
   of its condition/iterator eval). Sets *unknown when it meets a node shape it can't
   descend, so the caller can bail rather than guess a start that is too late. */
static int exprMinOff(const PycRef<ASTNode>& n, bool* unknown, int depth = 0)
{
    if (n == nullptr)
        return 0x7fffffff;
    if (depth > 40) { *unknown = true; return 0x7fffffff; }
    int m = (n->srcOff() >= 0) ? n->srcOff() : 0x7fffffff;
    auto rec = [&](const PycRef<ASTNode>& c) {
        int cm = exprMinOff(c, unknown, depth + 1);
        if (cm < m) m = cm;
    };
    switch (n->type()) {
    case ASTNode::NODE_NAME:
    case ASTNode::NODE_OBJECT:
        break;                                  // leaf
    case ASTNode::NODE_BINARY:
    case ASTNode::NODE_COMPARE:
        rec(n.cast<ASTBinary>()->left());
        rec(n.cast<ASTBinary>()->right());
        break;
    case ASTNode::NODE_UNARY:
        rec(n.cast<ASTUnary>()->operand());
        break;
    case ASTNode::NODE_SUBSCR:
        rec(n.cast<ASTSubscr>()->name());
        rec(n.cast<ASTSubscr>()->key());
        break;
    case ASTNode::NODE_CALL:
        rec(n.cast<ASTCall>()->func());
        for (const auto& p : n.cast<ASTCall>()->pparams())
            rec(p);
        break;
    case ASTNode::NODE_TUPLE:
        for (const auto& v : n.cast<ASTTuple>()->values())
            rec(v);
        break;
    default:
        *unknown = true;                        // unrecognised: don't trust the start
        break;
    }
    return m;
}

/* Source offset of a node's LAST leaf descendant (follow last children down). */
static int lastLeafSrcOff(const PycRef<ASTNode>& n, int depth = 0)
{
    if (n == nullptr || depth > 80)
        return -1;
    if (n.type() != ASTNode::NODE_BLOCK)
        return n->srcOff();
    const auto& kids = n.cast<ASTBlock>()->nodes();
    if (kids.empty())
        return -1;
    return lastLeafSrcOff(kids.back(), depth + 1);
}

/* Insert a `...` placeholder AFTER the `if` block whose last statement is at
   prevOff (built by the branch-exit instruction preceding a dropped merge NOP): a
   statement stripped under -OO right after an `if` whose body exits leaves its NOP
   at the branch merge, as a sibling of the `if`. A plain `if` merge regenerates no
   anchor, so a `...` sibling there reproduces exactly that NOP. Restricted to
   BLK_IF (loops/try/with own their merge/exit anchors). Depth-first. */
bool insertPlaceholderAfterIfEndingAt(ASTBlock::list_t& nodes, int prevOff,
                                             int phOff, int depth = 0)
{
    if (depth > 80)
        return false;
    for (auto it = nodes.begin(); it != nodes.end(); ++it) {
        PycRef<ASTNode> n = *it;
        if (n == nullptr || n.type() != ASTNode::NODE_BLOCK)
            continue;
        if (insertPlaceholderAfterIfEndingAt(n.cast<ASTBlock>()->mutableNodes(),
                                             prevOff, phOff, depth + 1))
            return true;
        if (n.cast<ASTBlock>()->blktype() == ASTBlock::BLK_IF
                && lastLeafSrcOff(n) == prevOff) {
            PycRef<ASTNode> ph = new ASTObject(Pyc_Ellipsis);
            ph->setSrcOff(phOff);
            nodes.insert(std::next(it), ph);
            return true;
        }
    }
    return false;
}

/* Insert a `...` placeholder as a sibling right AFTER the `if` block whose byte
   range ENDS at endOff: a statement stripped under -OO right after an `if` whose
   body falls through (its last statement is an expression/discard, so the branch
   merge -- and thus the block end -- is exactly the dropped NOP offset). Unlike
   insertPlaceholderAfterIfEndingAt (which keys on a RETURNing body's last-leaf
   offset), this keys on the block END so it covers a fall-through if. A plain `if`
   merge regenerates no anchor of its own, so a `...` sibling there reproduces the
   NOP. Restricted to BLK_IF; depth-first so the deepest/innermost match wins. */
bool insertPlaceholderAfterBlockEndingAt(ASTBlock::list_t& nodes, int endOff,
                                                int phOff, int depth = 0)
{
    if (depth > 80)
        return false;
    for (auto it = nodes.begin(); it != nodes.end(); ++it) {
        PycRef<ASTNode> n = *it;
        if (n == nullptr || n.type() != ASTNode::NODE_BLOCK)
            continue;
        if (insertPlaceholderAfterBlockEndingAt(n.cast<ASTBlock>()->mutableNodes(),
                                                endOff, phOff, depth + 1))
            return true;
        ASTBlock::BlkType bt = n.cast<ASTBlock>()->blktype();
        /* A plain `if` (fall-through merge) or a `match` (its cases all exit, so a
           statement after the match sits at its merge) regenerate no trailing anchor
           of their own -- a `...` sibling at the block end reproduces the dropped NOP.
           Loops / try own their exit anchors, so they are excluded. */
        if ((bt == ASTBlock::BLK_IF || bt == ASTBlock::BLK_MATCH)
                && n.cast<ASTBlock>()->end() == endOff) {
            PycRef<ASTNode> ph = new ASTObject(Pyc_Ellipsis);
            ph->setSrcOff(phOff);
            nodes.insert(std::next(it), ph);
            return true;
        }
    }
    return false;
}

/* Insert a `...` placeholder at the START of the `if` body whose header branch
   (POP_JUMP) is at headerOff: a statement stripped under -OO as the FIRST statement
   of an `if:` suite leaves its NOP right after the branch test, before the body. A
   plain `if` header is real condition code with no anchor of its own, so prepending
   a `...` reproduces the NOP. Skip if the body already opens with a `try`/infinite
   `while` (that block regenerates the anchor). Depth-first. */
bool insertPlaceholderAtIfBodyStart(ASTBlock::list_t& nodes, int headerOff,
                                           int phOff, int depth = 0)
{
    if (depth > 80)
        return false;
    for (auto it = nodes.begin(); it != nodes.end(); ++it) {
        PycRef<ASTNode> n = *it;
        if (n == nullptr || n.type() != ASTNode::NODE_BLOCK)
            continue;
        PycRef<ASTBlock> blk = n.cast<ASTBlock>();
        if (insertPlaceholderAtIfBodyStart(blk->mutableNodes(), headerOff, phOff, depth + 1))
            return true;
        if (blk->blktype() == ASTBlock::BLK_IF && blk->srcOff() == headerOff
                && !blk->nodes().empty()) {
            PycRef<ASTNode> first = blk->nodes().front();
            if (first != nullptr && first.type() == ASTNode::NODE_BLOCK) {
                ASTBlock::BlkType bt = first.cast<ASTBlock>()->blktype();
                if (bt == ASTBlock::BLK_CONTAINER
                        || (bt == ASTBlock::BLK_WHILE
                            && is_infinite_while(first.cast<ASTBlock>())))
                    return true;
            }
            PycRef<ASTNode> ph = new ASTObject(Pyc_Ellipsis);
            ph->setSrcOff(phOff);
            blk->mutableNodes().push_front(ph);
            return true;
        }
    }
    return false;
}

/* Insert a `...` placeholder at the START of the `for` body containing the dropped
   NOP at phOff: a statement stripped under -OO as the FIRST statement of a `for:`
   suite leaves its NOP right after the loop-variable STORE(s) (which unpack the
   FOR_ITER value), before the body. The `for` header (iterator eval + FOR_ITER)
   carries real instructions and regenerates no anchor of its own, so prepending a
   `...` reproduces the NOP. Matched by containment: the deepest BLK_FOR/BLK_ASYNCFOR
   whose byte range holds phOff with phOff BEFORE its first body statement (the gap
   between the loop-variable store and the first real statement). Skip if the body
   opens with a `try`/infinite `while` (that block regenerates the anchor). */
bool insertPlaceholderAtForBodyStart(ASTBlock::list_t& nodes, int phOff, int depth = 0)
{
    if (depth > 80)
        return false;
    for (auto it = nodes.begin(); it != nodes.end(); ++it) {
        PycRef<ASTNode> n = *it;
        if (n == nullptr || n.type() != ASTNode::NODE_BLOCK)
            continue;
        PycRef<ASTBlock> blk = n.cast<ASTBlock>();
        if (insertPlaceholderAtForBodyStart(blk->mutableNodes(), phOff, depth + 1))
            return true;
        ASTBlock::BlkType bt = blk->blktype();
        if ((bt == ASTBlock::BLK_FOR || bt == ASTBlock::BLK_ASYNCFOR)
                && !blk->nodes().empty()
                && blk->srcOff() >= 0 && blk->srcOff() <= phOff && phOff < blk->end()
                && subtreeMinOff(blk->nodes().front()) > phOff) {
            PycRef<ASTNode> first = blk->nodes().front();
            if (first != nullptr && first.type() == ASTNode::NODE_BLOCK) {
                ASTBlock::BlkType fbt = first.cast<ASTBlock>()->blktype();
                if (fbt == ASTBlock::BLK_CONTAINER
                        || (fbt == ASTBlock::BLK_WHILE
                            && is_infinite_while(first.cast<ASTBlock>())))
                    return true;
            }
            PycRef<ASTNode> ph = new ASTObject(Pyc_Ellipsis);
            ph->setSrcOff(phOff);
            blk->mutableNodes().push_front(ph);
            return true;
        }
    }
    return false;
}

/* Insert a `...` placeholder (offset phOff) immediately AFTER the deepest AST node
   whose srcOff == targetOff -- the statement built by the instruction that directly
   precedes a dropped stripped NOP. Since the NOP follows that instruction in the
   bytecode, it belongs in the same suite right after that statement, however nested.
   Depth-first so the deepest match wins. */
bool insertPlaceholderAfterOff(ASTBlock::list_t& nodes, int targetOff,
                                      int phOff, int depth = 0)
{
    if (depth > 80)
        return false;
    for (auto it = nodes.begin(); it != nodes.end(); ++it) {
        PycRef<ASTNode> n = *it;
        if (n == nullptr)
            continue;
        if (n.type() == ASTNode::NODE_BLOCK
                && insertPlaceholderAfterOff(n.cast<ASTBlock>()->mutableNodes(),
                                             targetOff, phOff, depth + 1))
            return true;
        if (n->srcOff() == targetOff && n.type() != ASTNode::NODE_BLOCK) {
            /* If the statement is immediately followed by a `try` or an infinite
               `while` block, the dropped NOP is that block's head anchor, which the
               block regenerates on its own -- skip (report handled, place nothing).
               A `try:` head is normally already excluded upstream, but a `try`
               whose body opens with a stripped statement can slip through. */
            auto nx = std::next(it);
            if (nx != nodes.end() && *nx != nullptr && (*nx).type() == ASTNode::NODE_BLOCK) {
                ASTBlock::BlkType bt = (*nx).cast<ASTBlock>()->blktype();
                if (bt == ASTBlock::BLK_CONTAINER
                        || (bt == ASTBlock::BLK_WHILE
                            && is_infinite_while((*nx).cast<ASTBlock>())))
                    return true;
            }
            PycRef<ASTNode> ph = new ASTObject(Pyc_Ellipsis);
            ph->setSrcOff(phOff);
            nodes.insert(nx, ph);
            return true;
        }
    }
    return false;
}

/* Recursively insert `...` placeholders for the still-unplaced stripped NOPs in
   `remaining` into their correct suite (depth-first: inner blocks claim theirs
   first). A NOP is placed in a suite only when it falls in a gap BETWEEN two of
   that suite's children (a preceding sibling exists and a following sibling
   keeps the placeholder non-final), and NOT inside a child block's byte range.
   Leading NOPs (no preceding sibling) are left to the leading pass. */
void insertGeneralPlaceholders(ASTBlock::list_t& nodes, std::set<int>& remaining,
                                      int depth = 0)
{
    if (depth > 80)   // guard against a cyclic/shared block reference (coalescing)
        return;
    for (auto& n : nodes)
        if (n != nullptr && n.type() == ASTNode::NODE_BLOCK)
            insertGeneralPlaceholders(n.cast<ASTBlock>()->mutableNodes(), remaining, depth + 1);
    if (remaining.empty())
        return;
    /* Trailing stripped statement inside an `if` body: a statement stripped under
       -OO at the END of an `if:` suite (a tail `assert` / `if __debug__:` block)
       leaves its line-anchor NOP just before the branch's merge target -- inside
       the if's byte range, after its last real statement. It has no following
       sibling, so it is neither placed here nor by the recursion. A plain `if`
       (no else) falls through with no trailing anchor of its own, so appending a
       `...` as the suite's last statement reproduces exactly that NOP. Restricted
       to BLK_IF whose last child is a leaf (a trailing compound block owns its own
       anchors) and to NOPs strictly between that child and the block end. */
    for (auto& n : nodes) {
        if (n == nullptr || n.type() != ASTNode::NODE_BLOCK)
            continue;
        PycRef<ASTBlock> blk = n.cast<ASTBlock>();
        if (blk->blktype() != ASTBlock::BLK_IF || blk->nodes().empty())
            continue;
        int blkEnd = blk->end();
        PycRef<ASTNode> last = blk->nodes().back();
        if (last == nullptr || last.type() == ASTNode::NODE_BLOCK)
            continue;   // compound tail owns its own anchors
        int lastHi = last->srcOff();
        if (lastHi < 0 || blkEnd <= lastHi)
            continue;
        std::vector<int> here;
        for (int o : remaining)
            if (o > lastHi && o < blkEnd)
                here.push_back(o);
        for (int o : here) {
            PycRef<ASTNode> ph = new ASTObject(Pyc_Ellipsis);
            ph->setSrcOff(o);
            blk->append(ph);
            remaining.erase(o);
        }
    }
    if (remaining.empty())
        return;
    struct Child { ASTBlock::list_t::iterator it; int lo, hi; };
    std::vector<Child> ch;
    for (auto it = nodes.begin(); it != nodes.end(); ++it) {
        PycRef<ASTNode> n = *it;
        int lo, hi;
        if (n != nullptr && n.type() == ASTNode::NODE_BLOCK) {
            lo = subtreeMinOff(n);
            hi = n.cast<ASTBlock>()->end();
            if (lo == 0x7fffffff) lo = hi;
        } else {
            lo = hi = (n != nullptr ? n->srcOff() : -1);
        }
        ch.push_back({it, lo, hi});
    }
    std::vector<std::pair<ASTBlock::list_t::iterator, int> > toInsert;
    std::vector<int> placed;
    for (int o : remaining) {
        bool inside = false;
        int followIdx = -1;
        for (size_t i = 0; i < ch.size(); i++) {
            if (ch[i].lo < 0) continue;
            if (o >= ch[i].lo && o < ch[i].hi) { inside = true; break; }
            if (ch[i].lo > o) { followIdx = (int)i; break; }
        }
        if (inside || followIdx < 1)
            continue;
        /* Conservative: only fill a gap BETWEEN two plain leaf statements whose
           offsets strictly bracket the NOP (prev.srcOff < o < follow.srcOff).
           A compound-block neighbour carries its own header line-anchor that the
           decompiler regenerates (double-count), and a block-adjacent or
           edge slot is where mis-positioning / invalid placements arise. */
        Child& follow = ch[followIdx];
        Child& prev = ch[followIdx - 1];
        PycRef<ASTNode> fn = *(follow.it), pn = *(prev.it);
        bool followLeaf = fn != nullptr && fn.type() != ASTNode::NODE_BLOCK && follow.lo >= 0;
        /* The follow may also be a compound block that does NOT regenerate its own
           head line-anchor NOP -- an `if`/`for`/`with`/`async for`/`match` (a `try`
           or `while` DOES emit a head NOP, so a placeholder before it would double
           it). To place the NOP strictly BEFORE such a block, bracket it against the
           block's true HEADER start (the first byte of its condition/iterator eval),
           not subtreeMinOff (which is the block BODY start and would wrongly admit a
           NOP sitting inside a multi-line header). Bail if the header start can't be
           computed exactly. */
        int followHeader = follow.lo;
        bool followSafeBlock = false;
        if (fn != nullptr && fn.type() == ASTNode::NODE_BLOCK && follow.lo >= 0) {
            ASTBlock::BlkType bt = fn.cast<ASTBlock>()->blktype();
            if (bt == ASTBlock::BLK_IF || bt == ASTBlock::BLK_FOR
                    || bt == ASTBlock::BLK_WITH || bt == ASTBlock::BLK_ASYNCFOR) {
                bool unknown = false;
                int hs = 0x7fffffff;
                if (bt == ASTBlock::BLK_IF) {
                    hs = exprMinOff(fn.cast<ASTCondBlock>()->cond(), &unknown);
                } else if (bt == ASTBlock::BLK_FOR || bt == ASTBlock::BLK_ASYNCFOR) {
                    hs = exprMinOff(fn.cast<ASTIterBlock>()->iter(), &unknown);
                } else {
                    unknown = true;             // with: source-start not modelled here
                }
                if (!unknown && hs != 0x7fffffff && hs > 0) {
                    followHeader = hs;
                    followSafeBlock = true;
                }
            }
        }
        bool prevLeaf = pn != nullptr && pn.type() != ASTNode::NODE_BLOCK && prev.hi >= 0;
        if ((!followLeaf && !followSafeBlock) || !prevLeaf)
            continue;
        /* A NOP right before a control-flow exit (`return`/`raise`/`break`/
           `continue`) is that exit's own line-anchor / unwind-cleanup NOP, which
           the decompiler regenerates when it renders the exit in a nested block --
           not a stripped statement. */
        if (fn.type() == ASTNode::NODE_RETURN || fn.type() == ASTNode::NODE_RAISE
                || fn.type() == ASTNode::NODE_KEYWORD)
            continue;
        if (!(prev.hi < o && o < followHeader))
            continue;
        toInsert.push_back({follow.it, o});
        placed.push_back(o);
    }
    for (auto& pr : toInsert) {
        PycRef<ASTNode> ph = new ASTObject(Pyc_Ellipsis);
        ph->setSrcOff(pr.second);
        nodes.insert(pr.first, ph);
    }
    for (int o : placed)
        remaining.erase(o);
}

/* Count `...` (Ellipsis) stripped-statement placeholders anywhere in a suite. */
static int countEllipsisPlaceholders(const ASTBlock::list_t& nodes, int depth = 0)
{
    if (depth > 80)
        return 0;
    int c = 0;
    for (const auto& n : nodes) {
        if (n == nullptr)
            continue;
        if (n.type() == ASTNode::NODE_OBJECT
                && n.cast<ASTObject>()->object() == Pyc_Ellipsis)
            c++;
        else if (n.type() == ASTNode::NODE_BLOCK)
            c += countEllipsisPlaceholders(n.cast<ASTBlock>()->nodes(), depth + 1);
    }
    return c;
}

/* Source byte-offset of the first `...` placeholder (depth-first), or -1. */
static int firstEllipsisOff(const ASTBlock::list_t& nodes, int depth = 0)
{
    if (depth > 80)
        return -1;
    for (const auto& n : nodes) {
        if (n == nullptr)
            continue;
        if (n.type() == ASTNode::NODE_OBJECT
                && n.cast<ASTObject>()->object() == Pyc_Ellipsis)
            return n->srcOff();
        if (n.type() == ASTNode::NODE_BLOCK) {
            int o = firstEllipsisOff(n.cast<ASTBlock>()->nodes(), depth + 1);
            if (o != -1)
                return o;
        }
    }
    return -1;
}

/* Replace the first `...` placeholder found (depth-first) with `repl`, carrying
   over its source position. Returns true once one is replaced. */
static bool replaceFirstEllipsis(ASTBlock::list_t& nodes, PycRef<ASTNode> repl, int depth = 0)
{
    if (depth > 80)
        return false;
    for (auto it = nodes.begin(); it != nodes.end(); ++it) {
        PycRef<ASTNode> n = *it;
        if (n == nullptr)
            continue;
        if (n.type() == ASTNode::NODE_OBJECT
                && n.cast<ASTObject>()->object() == Pyc_Ellipsis) {
            repl->setSrcOff(n->srcOff());
            *it = repl;
            return true;
        }
        if (n.type() == ASTNode::NODE_BLOCK
                && replaceFirstEllipsis(n.cast<ASTBlock>()->mutableNodes(), repl, depth + 1))
            return true;
    }
    return false;
}

/* A const value simple enough to reconstruct verbatim as a literal in a fabricated
   (dead, -OO-stripped) expression -- excludes containers/code whose recompiled
   representation might not round-trip. */
static bool isSimpleOrphanConst(int t)
{
    switch (t) {
    case PycObject::TYPE_NONE:
    case PycObject::TYPE_FALSE:
    case PycObject::TYPE_TRUE:
    case PycObject::TYPE_INT:
    case PycObject::TYPE_INT64:
    case PycObject::TYPE_LONG:
    case PycObject::TYPE_FLOAT:
    case PycObject::TYPE_BINARY_FLOAT:
    case PycObject::TYPE_STRING:
    case PycObject::TYPE_INTERNED:
    case PycObject::TYPE_UNICODE:
    case PycObject::TYPE_ASCII:
    case PycObject::TYPE_ASCII_INTERNED:
    case PycObject::TYPE_SHORT_ASCII:
    case PycObject::TYPE_SHORT_ASCII_INTERNED:
        return true;
    default:
        return false;
    }
}

/* Under -OO a stripped `if __debug__:` debug-logging block is dead-code-eliminated
   AFTER codegen registered its consts/names, leaving those entries ORPHANED in the
   const/name tables (present but never referenced by any instruction). pycdc renders
   the stripped block as a bare `...` placeholder, which registers nothing, so the
   orphans vanish and every later const/name index shifts down -- the recompiled
   co_code then diverges only in its operand indices. Recover them: when the function
   has orphaned table entries and EXACTLY ONE `...` placeholder (the stripped block),
   replace it with a reconstructed `if __debug__:` block whose single statement is a
   call referencing the orphaned names (as an attribute chain) and the orphaned consts
   (as arguments), in table order. Recompiled under -OO the block is dead, so only its
   table registration survives -- re-creating the orphans at their original indices,
   with no effect on the instruction stream beyond the same head NOP the `...` gave.
   A matching function cannot have orphaned consts (pycdc could not otherwise have
   reproduced them), so this only ever affects already-diverging functions. */
void recoverOrphanTableEntries(PycRef<ASTNodeList> clean, PycRef<PycCode> code,
                                      PycModule* mod)
{
    if (clean == nullptr || code->code() == nullptr || mod->verCompare(3, 11) < 0)
        return;
    if (countEllipsisPlaceholders(clean->nodes()) != 1)
        return;
    int phOff = firstEllipsisOff(clean->nodes());
    if (phOff < 0)
        return;
    /* Referenced const/name indices, and the FIRST-reference byte offset per name
       index (its registration order proxy). */
    std::set<int> constRefs, nameRefs;
    std::unordered_map<int, int> nameFirstRef;
    {
        PycBuffer s(code->code()->value(), code->code()->length());
        int op, arg, p = 0;
        while (!s.atEof()) {
            int pv = p;
            bc_next(s, mod, op, arg, p);
            if (p <= pv) break;
            int nameIdx = -1;
            switch (op) {
            case Pyc::LOAD_CONST_A:
            case Pyc::KW_NAMES_A:
            case Pyc::RETURN_CONST_A:
                constRefs.insert(arg);
                break;
            case Pyc::LOAD_GLOBAL_A:
                nameIdx = arg >> 1;   // 3.11: name index is arg >> 1
                break;
            case Pyc::LOAD_NAME_A:
            case Pyc::STORE_NAME_A:
            case Pyc::LOAD_ATTR_A:
            case Pyc::STORE_ATTR_A:
            case Pyc::DELETE_ATTR_A:
            case Pyc::LOAD_METHOD_A:
            case Pyc::IMPORT_NAME_A:
            case Pyc::IMPORT_FROM_A:
            case Pyc::STORE_GLOBAL_A:
            case Pyc::DELETE_NAME_A:
            case Pyc::DELETE_GLOBAL_A:
                nameIdx = arg;
                break;
            default:
                break;
            }
            if (nameIdx >= 0) {
                nameRefs.insert(nameIdx);
                auto it = nameFirstRef.find(nameIdx);
                if (it == nameFirstRef.end() || pv < it->second)
                    nameFirstRef[nameIdx] = pv;
            }
        }
    }
    std::vector<PycRef<PycObject> > orphConsts;
    int nc = code->consts() != nullptr ? code->consts()->size() : 0;
    for (int i = 0; i < nc; i++) {
        if (constRefs.count(i))
            continue;
        PycRef<PycObject> c = code->getConst(i);
        if (c == nullptr)
            continue;
        if (!isSimpleOrphanConst(c.type()))
            return;   // a container/code orphan: don't fabricate, leave the function as-is
        orphConsts.push_back(c);
    }
    std::vector<PycRef<PycString> > orphNames;
    std::vector<int> orphNameIdx;
    int nn = code->names() != nullptr ? code->names()->size() : 0;
    for (int i = 0; i < nn; i++) {
        if (nameRefs.count(i))
            continue;
        PycRef<PycString> nm = code->getName(i);
        if (nm == nullptr)
            return;
        orphNames.push_back(nm);
        orphNameIdx.push_back(i);
    }
    if (orphNames.empty())
        return;   // need a name to anchor the reconstructed call; skip pure-const orphans
    /* The orphaned names must come from a SINGLE stripped block: contiguous indices
       with no used name interleaved. */
    for (size_t i = 1; i < orphNameIdx.size(); i++)
        if (orphNameIdx[i] != orphNameIdx[i - 1] + 1)
            return;
    /* POSITION check: the block registers its names starting at index
       orphNameIdx[0], so exactly that many used names must be registered BEFORE the
       placeholder. Using each used name's first-reference offset as its registration
       order, require the count of used names first-referenced before the placeholder
       to equal the orphan block's start index. This rejects the case where the sole
       placeholder is a DIFFERENT stripped statement (e.g. a leading docstring) and
       the orphans belong to a trailing block -- placing them there would shift every
       real index. */
    int usedBefore = 0;
    for (const auto& kv : nameFirstRef)
        if (kv.second < phOff)
            usedBefore++;
    if (usedBefore != orphNameIdx[0])
        return;
    /* attribute chain n0.n1.n2... registers the names in table order */
    PycRef<ASTNode> func = new ASTName(orphNames[0]);
    for (size_t i = 1; i < orphNames.size(); i++)
        func = new ASTBinary(func, new ASTName(orphNames[i]), ASTBinary::BIN_ATTR);
    /* call args = orphaned consts (their exact PycObjects), in table order */
    ASTCall::pparam_t pparams;
    for (const auto& c : orphConsts)
        pparams.push_back(new ASTObject(c));
    PycRef<ASTNode> call = new ASTCall(func, pparams, ASTCall::kwparam_t());
    PycRef<PycString> dbg = new PycString(PycObject::TYPE_STRING);
    dbg->setValue("__debug__");
    PycRef<ASTCondBlock> ifblk = new ASTCondBlock(ASTBlock::BLK_IF, 0,
            new ASTName(dbg), false);
    ifblk->append(call);
    replaceFirstEllipsis(clean->mutableNodes(), ifblk.cast<ASTNode>());
}

int leadingStrippedNops(PycRef<PycCode> code, PycModule* mod,
                               std::vector<int>* offsets = nullptr,
                               const std::set<int>* exclude = nullptr)
{
    if (mod->verCompare(3, 11) < 0 || code->code() == nullptr)
        return 0;
    struct LI { int op, arg, off; };
    std::vector<LI> ins;
    {
        PycBuffer src(code->code()->value(), code->code()->length());
        int op, arg, pos = 0;
        while (!src.atEof() && ins.size() < 24) {
            int prev = pos;
            bc_next(src, mod, op, arg, pos);
            if (pos <= prev)
                break;
            ins.push_back({op, arg, prev});
        }
    }
    auto nameIs = [&](int idx, const char* n) {
        PycRef<PycString> s = code->getName(idx);
        return s != nullptr && s->isEqual(n);
    };
    /* Skip the whole code-object prologue that precedes any source statement:
       MAKE_CELL (one per cell variable), COPY_FREE_VARS (closures), the
       RETURN_GENERATOR/POP_TOP pair (generators / coroutines / async), and
       RESUME/CACHE. None of these can begin a source statement, so skipping every
       leading occurrence is safe (the POP_TOP is skipped only as the generator
       prologue's second half). */
    size_t i = 0;
    bool sawRetGen = false;
    while (i < ins.size()) {
        int op = ins[i].op;
        if (op == Pyc::CACHE || op == Pyc::RESUME_A || op == Pyc::MAKE_CELL_A
                || op == Pyc::COPY_FREE_VARS_A || op == Pyc::RETURN_GENERATOR) {
            if (op == Pyc::RETURN_GENERATOR)
                sawRetGen = true;
            i++;
            continue;
        }
        if (op == Pyc::POP_TOP && sawRetGen) {
            sawRetGen = false;
            i++;
            continue;
        }
        break;
    }
    if (i + 3 < ins.size()
            && ins[i].op == Pyc::LOAD_NAME_A && nameIs(ins[i].arg, "__name__")
            && ins[i + 1].op == Pyc::STORE_NAME_A && nameIs(ins[i + 1].arg, "__module__")
            && ins[i + 2].op == Pyc::LOAD_CONST_A
            && ins[i + 3].op == Pyc::STORE_NAME_A && nameIs(ins[i + 3].arg, "__qualname__"))
        i += 4;

    /* A class/module body carrying annotations emits SETUP_ANNOTATIONS right
       after the (class) qualname prologue and before the first source statement,
       so a stripped leading statement (docstring / bare const) anchors its NOP
       just past it. Skip it so that NOP is counted (else e.g. an annotated
       dataclass with a stripped docstring loses its head line-anchor). */
    if (i < ins.size() && ins[i].op == Pyc::SETUP_ANNOTATIONS)
        i++;

    int count = 0;
    while (i < ins.size() && ins[i].op == Pyc::NOP) {
        /* Skip anchor NOPs a one-per-line signature reproduces itself; they are
           not leading stripped statements. */
        if (exclude == nullptr || !exclude->count(ins[i].off)) {
            if (offsets != nullptr)
                offsets->push_back(ins[i].off);
            count++;
        }
        i++;
    }
    /* `count` = the number of leading line-anchor NOPs. Each stripped simple
       statement (docstring / bare const) contributes one, but a leading `try`
       (or `while`) regenerates one head anchor NOP on its own when pycdc renders
       it. The caller subtracts those self-regenerated anchors (it has the AST and
       can count the leading try/while nesting) before injecting placeholders, so
       here we return the raw count regardless of what follows the run. */
    return count;
}

/* True if pycdc's own reconstruction already emits a leading statement that
   compiles to the head NOP (a bare constant / `...` / `pass`), so no placeholder
   is needed, OR a statement a placeholder may not legally precede (a
   `from __future__` import must be the module's first statement). */
bool leadingNopAlreadyHandled(const PycRef<ASTNode>& front)
{
    if (front == nullptr)
        return true;
    if (front.type() == ASTNode::NODE_OBJECT)
        return true;   // bare constant expression statement -> already a NOP
    /* A leading `while` (empirically `while True:`) carries its own head
       line-anchor NOP, as does a `try`; both are now accounted for at the
       injection site, which descends the leading try/while nesting and subtracts
       those self-regenerated anchors from the stripped-statement count. So a
       `while`/`try` front is NOT treated as already-handled here -- a docstring
       before it still needs its own placeholder (if/for/with/match heads
       regenerate no head NOP and were already injectable). */
    /* NOTE: a leading `pass` is NOT treated as already-handled. `pass` emits no
       bytecode, so it never carries a stripped statement's line-anchor NOP; when
       the body is a stripped docstring/bare-const FOLLOWED by a `pass` (e.g. a
       class body `"""doc"""` / `pass`), the `...` placeholder must still be
       prepended before the `pass` to reproduce that NOP. (A genuinely empty
       `pass`-only body has no leading NOP, so leadingStrippedNops returns 0 and
       this path is not reached.) */
    if (front.type() == ASTNode::NODE_IMPORT) {
        PycRef<ASTNode> nm = front.cast<ASTImport>()->name();
        if (nm != nullptr && nm.type() == ASTNode::NODE_NAME
                && nm.cast<ASTName>()->name() != nullptr
                && nm.cast<ASTName>()->name()->isEqual("__future__"))
            return true;
    }
    return false;
}

/* Column-layout engine (floor b1): give each stripped-statement `...` placeholder
   the original source-text width so it renders as a width-matched literal and
   reproduces the co_positions column span of the stripped docstring/assert.
   The placeholder was injected with setSrcOff() = the stripped NOP's byte
   offset; look up its (scol, ecol) span in this code's location table. A bare
   string statement of that width recompiles (under -OO) to the same discarded
   code as `...`, so co_code is unaffected. */
void setPlaceholderWidths(const PycRef<ASTNode>& node, PycRef<PycCode> code,
                                 bool isFirst, int depth = 0)
{
    if (node == nullptr || depth > 200)
        return;
    if (node.type() == ASTNode::NODE_NODELIST) {
        bool first = true;
        for (const auto& n : node.cast<ASTNodeList>()->nodes()) {
            setPlaceholderWidths(n, code, first, depth + 1);
            first = false;
        }
        return;
    }
    if (node.type() == ASTNode::NODE_BLOCK) {
        bool first = true;
        for (const auto& n : node.cast<ASTBlock>()->nodes()) {
            setPlaceholderWidths(n, code, first, depth + 1);
            first = false;
        }
        return;
    }
    (void)isFirst;
    if (node.type() == ASTNode::NODE_OBJECT
            && node.cast<ASTObject>()->object() == Pyc_Ellipsis
            && node->srcOff() >= 0) {
        int ln = code->lineForOffset(node->srcOff());
        int el = code->elineForOffset(node->srcOff());
        int sc = code->colForOffset(node->srcOff());
        int ec = code->ecolForOffset(node->srcOff());
        /* Anchor the placeholder on the stripped statement's own source line so
           the absolute line-placement pass positions it (an injected node
           otherwise carries a stale construction-time line). */
        if (ln >= 0)
            node->setSrcLine(ln);
        if (sc >= 0 && ec > sc) {
            node->setSrcCol(sc);
            node->setLayoutWidth(ec - sc);
        }
        /* A multi-line original (a wrapped docstring): record the span's end so
           the placeholder can close on the same line and column. */
        if (el > ln && ec >= 1) {
            if (sc >= 0)
                node->setSrcCol(sc);
            node->setLayoutEndLine(el);
            node->setLayoutEndCol(ec);
        }
    }
}

/* True for a `return None` / bare `return` statement node. */
static bool isReturnNoneNode(const PycRef<ASTNode>& n)
{
    if (n == nullptr || n.type() != ASTNode::NODE_RETURN)
        return false;
    if (n.cast<ASTReturn>()->rettype() != ASTReturn::RETURN)
        return false;
    PycRef<ASTNode> v = n.cast<ASTReturn>()->value();
    if (v == nullptr)
        return true;
    if (v.type() != ASTNode::NODE_OBJECT)
        return false;
    PycRef<PycObject> o = v.cast<ASTObject>()->object();
    return o != nullptr && o.type() == PycObject::TYPE_NONE;
}

/* Last visible (non-suppressed) statement of a block body, or null. */
static PycRef<ASTNode> lastVisibleStmt(const ASTBlock::list_t& body)
{
    PycRef<ASTNode> last;
    for (const auto& n : body)
        if (n != nullptr && !n->suppressed())
            last = n;
    return last;
}

/* A statement whose normal control flow does not fall through to the next
   sibling (so an if-chain terminated by it maps to an `else`). */
static bool isTerminalNode(const PycRef<ASTNode>& n)
{
    if (n == nullptr)
        return false;
    return n.type() == ASTNode::NODE_RETURN || n.type() == ASTNode::NODE_RAISE;
}

/* Layout-only: re-fold a terminal if-return chain into if/elif/.../else.
   pycdc reconstructs `if C1: A elif C2: B ... else: T` as SEPARATE
   `if C1: A; return None` sibling blocks ending in a terminal statement T,
   because both forms compile to byte-identical bytecode. The expanded form
   emits an extra explicit `return None` LINE per branch, pushing every later
   def down (the co_positions over-run cascade). When a maximal run of >=1
   consecutive BLK_IF, each whose body's last visible statement is an implicit
   `return None`, is immediately followed by a terminal statement (raise/return)
   that is the LAST visible node in the list, fold it back: suppress each
   branch's trailing return-None, convert the 2nd..Nth if to elif, and wrap the
   terminal in an else. co_code is unchanged (the forms are byte-identical);
   only the rendered line layout shrinks toward the original source.

   SAFETY: the fold is co_code-safe ONLY in TAIL position. A branch's implicit
   `return None` exits the FUNCTION; the elif form falls through the chain to
   whatever follows it. These coincide (both reach the function-end return None)
   only when the chain is in tail position — inside a loop the fall-through
   continues the loop instead, which is NOT equivalent. So `listTail` is
   propagated exactly as in markEpilogueSuppress and the fold runs only when
   the (terminal-ended) chain is the tail of a tail list. */
void coalesceElifChains(ASTBlock::list_t& nodes, bool listTail, int depth)
{
    if (depth > 80)
        return;
    /* Flatten to a vector of iterators for index-style back-walking. */
    std::vector<ASTBlock::list_t::iterator> it;
    for (auto i = nodes.begin(); i != nodes.end(); ++i)
        it.push_back(i);
    int N = (int)it.size();
    int lastVis = -1;
    for (int i = 0; i < N; i++)
        if (*it[i] != nullptr && !(*it[i])->suppressed())
            lastVis = i;
    /* Recurse into child block bodies first, propagating tail position. */
    for (int i = 0; i < N; i++) {
        PycRef<ASTNode> n = *it[i];
        if (n == nullptr || n.type() != ASTNode::NODE_BLOCK)
            continue;
        PycRef<ASTBlock> blk = n.cast<ASTBlock>();
        ASTBlock::BlkType bt = blk->blktype();
        /* Propagate tail ONLY through pure conditional nesting (if/elif/else).
           A try/except/finally body's fall-through carries exception-unwind
           cleanup (POP_EXCEPT etc.), so the if/elif/else vs if-return-chain
           equivalence does NOT hold there (proven: folding in an except body
           changes co_code). Loops/with never preserve tail either. */
        bool childTail = listTail && (i == lastVis)
                && (bt == ASTBlock::BLK_IF || bt == ASTBlock::BLK_ELSE
                    || bt == ASTBlock::BLK_ELIF);
        if (bt == ASTBlock::BLK_CONTAINER) {
            for (auto& c : blk->mutableNodes())
                if (c != nullptr && c.type() == ASTNode::NODE_BLOCK)
                    coalesceElifChains(c.cast<ASTBlock>()->mutableNodes(),
                                       false, depth + 1);
        } else {
            coalesceElifChains(blk->mutableNodes(), childTail, depth + 1);
        }
    }
    if (!listTail)
        return;
    int lastIdx = lastVis;
    if (lastIdx < 1 || !isTerminalNode(*it[lastIdx]))
        return;
    /* Walk back over the contiguous run of qualifying BLK_IF siblings. */
    std::vector<int> ifIdx;
    for (int j = lastIdx - 1; j >= 0; j--) {
        PycRef<ASTNode> n = *it[j];
        if (n == nullptr || n->suppressed() || n.type() != ASTNode::NODE_BLOCK)
            break;
        PycRef<ASTBlock> b = n.cast<ASTBlock>();
        if (b->blktype() != ASTBlock::BLK_IF)
            break;
        if (!isReturnNoneNode(lastVisibleStmt(b->nodes())))
            break;
        /* The trailing return-None must follow a SIMPLE (non-block) statement,
           and there must be one (body >=2 visible). If the preceding statement
           is itself a block (a nested if/for/try), the return-None is a shared
           control-flow merge target and the if-return vs if/elif/else forms
           compile to DIFFERENT bytecode (proven). Also, >=2 visible guarantees
           suppressing the return leaves a non-empty body (an empty branch
           renders `pass`, which is not fall-through-equivalent). Mirrors the
           `!prevWasBlock` guard in markEpilogueSuppress. A branch failing this
           stops the run (folding only the safe suffix closer to the terminal). */
        int vis = 0;
        PycRef<ASTNode> beforeRet, last;
        for (const auto& s : b->nodes()) {
            if (s == nullptr || s->suppressed())
                continue;
            beforeRet = last;
            last = s;
            vis++;
        }
        if (vis < 2 || beforeRet == nullptr
                || beforeRet.type() == ASTNode::NODE_BLOCK)
            break;
        ifIdx.push_back(j);
    }
    if (ifIdx.empty())
        return;
    int firstIf = ifIdx.back();
    for (int idx : ifIdx) {
        PycRef<ASTBlock> b = (*it[idx]).cast<ASTBlock>();
        PycRef<ASTNode> lastB = lastVisibleStmt(b->nodes());
        if (lastB != nullptr)
            lastB->setSuppressed(true);
        if (idx != firstIf)
            b->setBlktype(ASTBlock::BLK_ELIF);
    }
    PycRef<ASTBlock> elseb = new ASTBlock(ASTBlock::BLK_ELSE, 0);
    elseb->append(*it[lastIdx]);
    elseb->setSrcLine(-1);   // no own line-pad; the terminal inside pads to its line
    *it[lastIdx] = elseb;
}

/* Layout-only: drop the implicit trailing `return None` epilogue that pycdc
   over-renders for a terminal try/except (see setSuppressed). CPython re-creates
   the implicit epilogue when the plain try/except recompiles, so co_code is
   unchanged, but the over-rendered `else: return None` / `except …: …; return None`
   lines no longer push every later def down (the co_positions over-run cascade).
   Only the SPECIFIC shape is touched: a container try/except whose synthesized
   `else` is exactly one `return None` sharing the try body's last line (an inherited
   implicit-epilogue line, not an explicit own-line return), and whose every except
   clause ends in a `return None` that likewise inherits its preceding statement's
   line. All-or-nothing per container, so a partially-explicit shape is left intact. */
void markEpilogueSuppress(ASTBlock::list_t& nodes, bool listTail,
                                 bool inExcept, int depth = 0)
{
    if (depth > 80)
        return;
    int lastVis = -1, i = 0;
    for (const auto& n : nodes) {
        if (n != nullptr && !n->suppressed())
            lastVis = i;
        i++;
    }
    i = -1;
    bool prevWasBlock = false;
    bool prevExists = false;
    for (auto& n : nodes) {
        i++;
        /* A statement is in TAIL position (its normal exit reaches the function end,
           so an implicit `return None` there IS the fall-off epilogue) only when its
           enclosing list is tail AND it is the last visible statement in it. */
        bool isTail = listTail && (i == lastVis);
        /* A plain trailing `return None` in tail position is the implicit fall-off
           epilogue rendered explicitly: dropping it recompiles identically (the
           compiler re-adds the epilogue) and removes the over-rendered line. Require
           a plain (non-block) preceding statement: after a loop / if / try the
           control flow MERGES several predecessors on the return, and the compiler
           may duplicate vs share the epilogue -- dropping it would then change the
           block count (co_code). A simple fall-through from a leaf statement is safe. */
        if (!inExcept && isTail && n != nullptr && !n->suppressed()
                && isReturnNoneNode(n) && prevExists && !prevWasBlock) {
            n->setSuppressed(true);
            continue;
        }
        if (n != nullptr && !n->suppressed()) {
            prevExists = true;
            prevWasBlock = (n.type() == ASTNode::NODE_BLOCK);
        }
        if (n == nullptr || n.type() != ASTNode::NODE_BLOCK)
            continue;
        PycRef<ASTBlock> blk = n.cast<ASTBlock>();
        ASTBlock::BlkType bt = blk->blktype();
        if (bt == ASTBlock::BLK_CONTAINER) {
            /* Each try/else/except path independently reaches the function end when
               the container is tail. An except body carries exception-unwind cleanup,
               so a plain return-None there cannot be dropped to a fall-off (the
               container pass handles the all-fall-off case itself) -- mark inExcept. */
            for (auto& c : blk->mutableNodes()) {
                if (c == nullptr || c.type() != ASTNode::NODE_BLOCK) continue;
                bool ce = (c.cast<ASTBlock>()->blktype() == ASTBlock::BLK_EXCEPT);
                markEpilogueSuppress(c.cast<ASTBlock>()->mutableNodes(),
                                     isTail, inExcept || ce, depth + 1);
            }
        } else if (bt == ASTBlock::BLK_IF || bt == ASTBlock::BLK_ELSE
                   || bt == ASTBlock::BLK_ELIF || bt == ASTBlock::BLK_TRY
                   || bt == ASTBlock::BLK_EXCEPT) {
            markEpilogueSuppress(blk->mutableNodes(), isTail, inExcept, depth + 1);
        } else {
            markEpilogueSuppress(blk->mutableNodes(), false, inExcept, depth + 1);
        }
        if (!isTail || bt != ASTBlock::BLK_CONTAINER
                || !blk.cast<ASTContainerBlock>()->hasExcept())
            continue;
        PycRef<ASTBlock> tryb, elseb;
        std::vector<PycRef<ASTBlock> > excepts;
        bool hasFinally = false;
        for (const auto& c : blk->nodes()) {
            if (c == nullptr || c.type() != ASTNode::NODE_BLOCK)
                continue;
            ASTBlock::BlkType bt = c.cast<ASTBlock>()->blktype();
            if (bt == ASTBlock::BLK_TRY) tryb = c.cast<ASTBlock>();
            else if (bt == ASTBlock::BLK_ELSE) elseb = c.cast<ASTBlock>();
            else if (bt == ASTBlock::BLK_EXCEPT) excepts.push_back(c.cast<ASTBlock>());
            else if (bt == ASTBlock::BLK_FINALLY) hasFinally = true;
        }
        if (hasFinally || tryb == nullptr || elseb == nullptr || excepts.empty())
            continue;
        /* the try body's last statement line (the implicit epilogue inherits it) */
        PycRef<ASTNode> tryLast;
        for (auto it = tryb->nodes().rbegin(); it != tryb->nodes().rend(); ++it)
            if (*it != nullptr && !(*it)->suppressed()) { tryLast = *it; break; }
        if (tryLast == nullptr)
            continue;
        /* else must be a single `return None` sharing the try body's last line */
        PycRef<ASTNode> elseRet;
        int elseVisible = 0;
        for (const auto& c : elseb->nodes())
            if (c != nullptr && !c->suppressed()) { elseVisible++; elseRet = c; }
        if (elseVisible != 1 || !isReturnNoneNode(elseRet)
                || elseRet->srcLine() < 0 || elseRet->srcLine() != tryLast->srcLine())
            continue;
        /* every except clause must END in an inherited-line `return None` */
        bool ok = true;
        std::vector<PycRef<ASTNode> > exRets;
        for (const auto& ex : excepts) {
            PycRef<ASTNode> l1, l2;
            for (auto it = ex->nodes().rbegin(); it != ex->nodes().rend(); ++it) {
                if (*it == nullptr || (*it)->suppressed()) continue;
                if (l1 == nullptr) l1 = *it;
                else { l2 = *it; break; }
            }
            int prevLine = (l2 != nullptr) ? l2->srcLine() : ex->srcLine();
            if (l1 == nullptr || !isReturnNoneNode(l1) || l1->srcLine() < 0
                    || prevLine < 0 || l1->srcLine() != prevLine) { ok = false; break; }
            exRets.push_back(l1);
        }
        if (!ok)
            continue;
        elseb->setSuppressed(true);
        for (auto& r : exRets)
            r->setSuppressed(true);
    }
}
