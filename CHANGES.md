# pyfaithful — Changes relative to Decompyle++ (pycdc)

pyfaithful is a fork of [Decompyle++ / pycdc](https://github.com/zrax/pycdc).
This file summarizes the substantial changes made in the fork. The complete,
per-commit history is in the git log.

The fork's goal is **faithful reconstruction** of Python 3.11 byte-code: source
that recompiles to the same instruction bytes (`co_code`) and, as far as the
`.pyc` allows, the same per-instruction source positions (`co_positions`). Full
`co_positions` identity is inherently impossible — column offsets encode source
whitespace and literal spelling that the `.pyc` does not retain — but line
layout is reproduced where recoverable. Every change preserves the `co_code`
invariant: no change turns a byte-matching function into a mismatch.

## Python 3.11 control-flow reconstruction and correctness

Extensive work on reconstructing 3.11 control flow so that decompile → recompile
round-trips faithfully, including:

* `try` / `except` / `finally` reconstruction: nested-if finally bodies,
  finally-loop breaks in fall-through try bodies, conditional and unconditional
  returns inside `finally`, shared-merge finally coalescing, empty finally
  (`try: B finally: pass`), and resuming after an inner `except`'s return at the
  correct merge.
* `match` / `case`: OR-patterns (`case A | B`), a relocated `case _` wildcard
  inside `try`/`finally`.
* `with` / `async with` / `async for`; closing a `with`-block that returns
  before a following branch; closure-cell `as`-targets.
* Loop and branch coalescing: while-True break that doubles as an enclosing
  branch's jump-over-else, chained-comparison while loops, trailing `continue`
  emission, loops with both a jump break and a fall-through break, if/elif
  chains trailing the last clause of a nested try/except.
* Boolean and comparison faithfulness: keeping a ternary as a single
  short-circuit operand, 3+-operand OR guards (instead of de Morgan inversion),
  chained comparisons whose link is a containment test, constant-folded set
  membership rendered as a set literal.

## Statement and expression rendering

* PEP 328 relative-import leading dots, `@decorator` syntax, PEP 3132 extended
  unpacking, nested sequence-assignment targets, walrus (`:=`) and 3.11 chained
  assignment, PEP 448 unpacking generalizations, `yield` / `await` forms, class
  keyword arguments and metaclass, set/dict/generator comprehensions.
* Function parameter and return **annotations**.
* Rendering-correctness fixes where a dropped construct previously shifted
  operand indices: `lambda *args/**kwargs`, `nonlocal` declarations, a
  `from x import y` bound to a `global` target, and constant-key `**kwargs`
  calls rendered as keyword arguments rather than a dict-display spread.
* Operator-precedence parenthesization fixes: no longer over-parenthesizing the
  operand of `not`, a unary operator's attribute operand, or a power's unary
  exponent.

## Byte-identical instruction bytes (co_code)

* Faithful reconstruction of stripped statements under `-OO` (docstrings,
  asserts, `if __debug__:` blocks) as line/position-anchoring placeholders,
  including recovery of orphaned constants/names left in the code object's
  tables.
* Multi-line function-signature rendering that regenerates the per-line anchor
  NOPs the compiler emits for constant defaults spread across source lines.

## Line-faithful source layout (co_positions)

* Multi-line expression rendering: call arguments and list/set/dict/tuple
  displays are laid out on their original source lines and columns, gated so
  that only expressions with no nested code object are wrapped (preserving
  `co_code`).
* Compound statements (`a = 1; b = 2` on one source line) are rendered
  semicolon-joined; a single-statement `if`/`elif`/`else` body sharing the
  header's line is rendered inline (`if c: return x`).
* Suppression of the implicit fall-off `return None` epilogue that was
  previously over-rendered, and re-folding terminal `if`-return chains into
  `if` / `elif` / `else`.

## Tests

Added Python 3.11 test inputs and tokenized expectations for control flow,
exceptions, chained comparisons, decorators, relative imports, starred/nested
unpacking, walrus, and surrogate-escape string handling.
