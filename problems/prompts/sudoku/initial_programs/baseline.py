PROMPT_TEMPLATE = """
You are a 4x4 Sudoku solver. Your task is to complete the grid by filling exactly one empty cell per step.

### SUDOKU RULES
1. Each row must contain digits 1, 2, 3, and 4 exactly once.
2. Each column must contain digits 1, 2, 3, and 4 exactly once.
3. Each 2x2 subgrid (divided by | and ----) must contain digits 1, 2, 3, and 4 exactly once.

### GRID REPRESENTATION
You must maintain this exact visual format:
3 1 | _ 2
4 2 | 3 1
----+----
2 4 | 1 3
1 3 | 2 _

- Empty cells are represented by '_'.
- Never change or remove the existing digits.
- Never change the '|' or '----+----' separators.

### SOLVING STEP CHECKLIST
Before providing the next grid, mentally verify:
- Have I filled exactly ONE '_' cell?
- Is the new digit unique in its row, column, and 2x2 subgrid?
- Is the rest of the grid identical to the previous step?

### OUTPUT FORMAT
- If the grid is not yet full: <node>GRID</node>
- If the grid is completely filled: <done>GRID</done>

Do not include any thought process, commentary, or text outside the tags. Provide ONLY the tag and the grid.
""".strip()


def entrypoint():
    return PROMPT_TEMPLATE
