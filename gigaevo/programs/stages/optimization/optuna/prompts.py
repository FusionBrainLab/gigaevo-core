"""LLM prompt templates for the Optuna hyperparameter search-space proposal."""

_SYSTEM_PROMPT = """\
You parametrize Python code for Optuna hyperparameter optimization. Replace tuneable \
literals with ``_optuna_params["name"]`` references. Return a structured response with \
``parameters`` (ParamSpec list) and ``modifications`` (CodeModification patches).

The optimizer will inject different values for each trial, so every constraint below \
is a runtime requirement — violations cause all trials to fail silently.

**Hard Constraints (violations crash every trial)**
- ``initial_value`` MUST satisfy the declared bounds/choices:
  - float/int: ``low <= initial_value <= high`` (strict: ``low < high``)
  - log_float: ``0 < low <= initial_value <= high`` (BOTH bounds must be > 0; ``low=0`` raises ValueError)
  - categorical: ``initial_value`` must be one of the ``choices`` entries (exact value and type)
- Every ``ParamSpec.name`` must appear verbatim as ``_optuna_params["that_name"]`` in at \
least one ``parameterized_snippet``. A declared parameter with no matching reference \
in the code is silently ignored — optimization is wasted.
- Keys inside snippets must match ``ParamSpec.name`` exactly (case-sensitive, no typos).

**Type Selection**
- ``int``: discrete whole numbers — use whenever the value is passed to ``range()``, \
used as an index, or must be an integer (e.g. ``n_neighbors``, ``batch_size``). \
Using ``float`` for these causes ``TypeError: 'float' object cannot be interpreted as an integer``.
- ``float``: continuous real values (e.g. ``learning_rate``, ``tolerance``).
- ``log_float``: continuous on log scale — use for values spanning orders of magnitude \
(e.g. regularization strength ``1e-4`` to ``1.0``). Both ``low`` and ``high`` must be > 0.
- ``categorical``: finite set — use for algorithm choices, string flags, or boolean toggles. \
List all candidates in ``choices``; ``initial_value`` must be one of them.

**Patch Rules**
- Line numbers come from the ``N | `` prefix shown in the code. Use them as-is (1-indexed, inclusive).
- Non-overlapping: no two patches may share any line. If multiple parameters fall in the \
same block, emit ONE patch covering the whole block with ALL references inside it. \
Violating this raises ValueError and skips optimization entirely.
- ``parameterized_snippet``: first line has zero leading spaces; subsequent lines use \
relative indentation (4 spaces per nesting level). Do NOT copy the ``N | `` prefix.
- Snippets must be syntactically complete (no partial expressions split across a patch boundary).

**What to Parametrize**
- Focus on constants with large impact on ``{score_key}``.
- Do NOT parametrize: seeds (``random.seed``, ``np.random.seed``), file paths, or constants \
used only in print/log statements.
- Do NOT add new branches, variables, or control flow — only replace existing literals.
- If a constant appears in multiple semantically linked forms (e.g. ``uniform(-x, x)``, \
``linspace(0, n, n+1)``, ``k`` and ``2*k``), use a SINGLE ``_optuna_params`` key and \
derive the other occurrences via arithmetic in the snippet.

**Imports**: Add to ``new_imports`` only if the snippet introduces a symbol not already \
imported (e.g. a callable referenced via ``eval()``).

**Output length**: ``reasoning`` 2-3 sentences. Propose <= 10 parameters.

**Examples**
1) Parameter in code (line 7): ``k = 5  # number of neighbors`` \
Correct: name="k", param_type="int", low=1, high=20, initial_value=5. \
Patch: start_line=7, end_line=7, parameterized_snippet="k = _optuna_params['k']". \
Wrong: param_type="float" for a value passed to range() → runtime TypeError.
2) Multi-line block (lines 12-13): \
start_line=12, end_line=13, parameterized_snippet="lr = _optuna_params['lr']\\nmomentum = _optuna_params['momentum']". \
Two separate non-overlapping patches on lines 12 and 13 are also valid.
"""

_USER_PROMPT_TEMPLATE = """\
Parametrize the code below: (1) list **parameters** (name, type, bounds/choices, initial_value) and (2) list **modifications** (patches) using line ranges.

**Code** (with line numbers):
```python
{numbered_code}
```
{task_description_section}"""
