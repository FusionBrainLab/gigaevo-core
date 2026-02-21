"""LLM prompt templates for the Optuna hyperparameter search-space proposal."""

_SYSTEM_PROMPT = """\
You parametrize Python code for Optuna by replacing tuneable literals with \
``_optuna_params["name"]`` references. Return ``parameters`` (ParamSpec list), \
``modifications`` (CodeModification patches), ``new_imports``, and ``reasoning``.

Every constraint below is a **runtime requirement** — violations crash all trials.

**Type selection** (wrong type = TypeError on every trial)
- ``int``: value passed to ``range()``, used as index, or must be a whole number. \
Using ``float`` here causes ``TypeError: 'float' object cannot be interpreted as an integer``.
- ``float``: continuous real (learning rate, tolerance, threshold).
- ``log_float``: log-uniform; use for values spanning orders of magnitude. \
Both ``low`` AND ``high`` must be > 0 — ``low=0`` raises ValueError.
- ``categorical``: finite set of strings, bools, or numbers. ``initial_value`` must \
exactly match one element of ``choices``.

**Bounds and initial_value**
- ``low < high`` always. ``initial_value`` must satisfy ``low <= initial_value <= high``. \
Set ``initial_value`` to the literal currently in the code.

**Name consistency** (mismatch = parameter silently ignored, wasted trials)
- Every ``ParamSpec.name`` must appear verbatim as ``_optuna_params["that_name"]`` \
in at least one ``parameterized_snippet``. Check spelling — it is case-sensitive.

**Patch geometry** (overlap = ValueError, optimization skipped entirely)
- Use line numbers from the ``N | `` prefix as-is (1-indexed, inclusive).
- No two patches may share any line. If multiple parameters fall in the same block, \
emit ONE patch covering the whole block with all references inside it.
- ``parameterized_snippet``: first line has zero leading spaces; subsequent lines \
use relative indentation (4 spaces per nesting level). Strip the ``N | `` prefix.
- Snippets must be syntactically complete — no partial expressions split across a boundary.

**new_imports** (NameError if misused)
- Only Python ``import`` statements (e.g. ``"import numpy as np"``). \
Never put ``_optuna_params`` or variable assignments here — \
``_optuna_params`` is injected by the runtime, not imported.
- Omit entirely if no new imports are needed.

**What to parametrize**
- Target constants with large impact on ``{score_key}``. Prefer 3-5 high-impact \
parameters; max 10 total.
- Skip: random seeds, file paths, print/log-only constants.
- Replace literals only — no new branches, variables, or control flow.
- Linked constants (e.g. ``uniform(-x, x)``): use ONE key, derive others via arithmetic.

**Examples**

1. Integer param (line 7): ``k = 5``
   - Correct: ``name="k", param_type="int", low=1, high=20, initial_value=5``
   - Snippet: ``k = _optuna_params['k']``
   - Wrong: ``param_type="float"`` — crashes with TypeError when ``k`` enters ``range(k)``.

2. Multi-param block (lines 12-13, both on adjacent lines):
   - ONE patch: ``start_line=12, end_line=13``
   - Snippet: ``lr = _optuna_params['lr']\\nmomentum = _optuna_params['momentum']``
   - Wrong: two separate patches on line 12 and line 13 — overlapping raises ValueError \
if the lines are contiguous in a single logical block.

3. log_float (line 9): ``alpha = 1e-4``
   - Correct: ``param_type="log_float", low=1e-6, high=1.0, initial_value=1e-4``
   - Wrong: ``low=0`` — raises ValueError (log scale requires low > 0).
"""

_USER_PROMPT_TEMPLATE = """\
Parametrize the code below for Optuna. Return:
1. ``parameters`` — name, param_type, bounds/choices, initial_value (= current literal), reason
2. ``modifications`` — non-overlapping line-range patches using the ``N | `` numbers shown

**Code:**
```python
{numbered_code}
```
{task_description_section}"""
