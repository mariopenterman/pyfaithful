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

#endif
