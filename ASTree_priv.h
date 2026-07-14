#ifndef _PYC_ASTREE_PRIV_H
#define _PYC_ASTREE_PRIV_H

/* Internal shared declarations for the decompiler core, split across
 * ASTree.cpp (the byte-code -> AST builder and the decompyle() orchestrator)
 * and ASTRender.cpp (the AST -> source renderer / layout engine). These are
 * not part of the public API (ASTree.h); they are the file-scope state and
 * helpers the two translation units share. Each global is defined (once,
 * non-static) in whichever .cpp owns it, and documented at that definition. */

#include <string>
#include <vector>
#include <set>
#include <ostream>
#include "ASTNode.h"
#include "pyc_module.h"
#include "pyc_code.h"

/* --- owned by ASTRender.cpp, used by the builder/orchestrator --- */
extern int cur_indent;            // current output indentation depth
extern int g_noCompact;           // >0: keep collection displays multi-line
extern bool g_lineFaithful;       // reproduce original multi-line expr layout
extern bool g_funcHasNestedCode;  // code object has a nested code const

void print_block(PycRef<ASTBlock> blk, PycModule* mod, std::ostream& pyc_output);
bool print_docstring(PycRef<PycObject> obj, int indent, PycModule* mod,
                     std::ostream& pyc_output);
void print_formatted_value(PycRef<ASTFormattedValue> formatted_value,
                           PycModule* mod, std::ostream& pyc_output);
bool lfSafe(const PycRef<ASTNode>& node);
void start_line(int indent, std::ostream& pyc_output);
void end_line(std::ostream& pyc_output);

/* --- owned by ASTree.cpp, used by the renderer --- */
extern bool cleanBuild;
extern bool inLambda;
extern bool inAsyncGen;
extern bool suppressGenExprParens;
extern bool suppressLambdaParens;
extern bool printDocstringAndGlobals;
extern bool printClassDocstring;
extern std::vector<std::string> g_classNameStack;

std::string demangleName(const std::string& name);

/* --- PHASE 2: faithfulness passes (ASTFaithful.cpp), invoked by decompyle() ---
 * Declared without default arguments; the defaults live on the definitions and
 * decompyle() passes every argument explicitly. */
extern std::set<int> g_sigAnchorNopOffs;   // sig anchor NOP offsets (owned by ASTree.cpp)

bool is_infinite_while(const PycRef<ASTBlock>& blk);
void strip_module_trailing_return(PycRef<ASTBlock> blk, bool inInfWhile);
std::vector<int> collectStrippedNops(PycRef<PycCode> code, PycModule* mod,
                                     const std::set<int>& exclude);
void insertGeneralPlaceholders(ASTBlock::list_t& nodes, std::set<int>& remaining, int depth);
int leadingStrippedNops(PycRef<PycCode> code, PycModule* mod,
                        std::vector<int>* offsets, const std::set<int>* exclude);
void recoverOrphanTableEntries(PycRef<ASTNodeList> clean, PycRef<PycCode> code, PycModule* mod);
void setPlaceholderWidths(const PycRef<ASTNode>& node, PycRef<PycCode> code,
                          bool isFirst, int depth);
void coalesceElifChains(ASTBlock::list_t& nodes, bool listTail, int depth);
void markEpilogueSuppress(ASTBlock::list_t& nodes, bool listTail, bool inExcept, int depth);
bool leadingNopAlreadyHandled(const PycRef<ASTNode>& front);
bool insertPlaceholderAfterIfEndingAt(ASTBlock::list_t& nodes, int prevOff, int phOff, int depth);
bool insertPlaceholderAfterBlockEndingAt(ASTBlock::list_t& nodes, int endOff, int phOff, int depth);
bool insertPlaceholderAtIfBodyStart(ASTBlock::list_t& nodes, int headerOff, int phOff, int depth);
bool insertPlaceholderAtForBodyStart(ASTBlock::list_t& nodes, int phOff, int depth);
bool insertPlaceholderAfterOff(ASTBlock::list_t& nodes, int targetOff, int phOff, int depth);
bool stmtEmitsScopeCode(const PycRef<ASTNode>& n);

#endif
