from __future__ import annotations

from problems.prompts.sudoku.local_runtime.grid import Grid
from problems.prompts.sudoku.local_runtime.models import (
    Action,
    DoneAction,
    NodeAction,
    PathContext,
    ValidationResult,
)


class SudokuValidator:
    def __init__(self) -> None:
        self._valid_grids: set[str] = set()
        self._solvable_cache: dict[str, bool] = {}

    def _is_solvable_cached(self, grid: Grid) -> bool:
        key = grid.to_canonical()
        if key in self._solvable_cache:
            return self._solvable_cache[key]
        value = grid.is_solvable()
        self._solvable_cache[key] = value
        return value

    def validate(self, action: Action, context: PathContext) -> ValidationResult | None:
        if not context.nodes:
            return ValidationResult(False, "Path is empty")

        prev_grid = context.last_node.state
        if prev_grid is None:
            return ValidationResult(False, "Previous node has no state")

        if not isinstance(action, (NodeAction, DoneAction)):
            return ValidationResult(
                False, f"Unknown action type: {type(action).__name__}"
            )

        if isinstance(action, NodeAction):
            last_id = -1
            for node in reversed(context.nodes):
                if isinstance(node.action, NodeAction):
                    last_id = node.action.node_id
                    break
            expected = last_id + 1
            if action.node_id != expected:
                return ValidationResult(
                    False,
                    f"Node id should be {expected}, received {action.node_id}",
                )

        try:
            new_grid = Grid.from_string(action.text, spec=prev_grid.spec)
        except Exception as exc:
            return ValidationResult(False, f"Invalid grid format: {str(exc)[:100]}")

        is_preserved, modified_cell = prev_grid.verify_pivots_preserved(new_grid)
        if not is_preserved:
            if modified_cell is not None:
                row, col = modified_cell
                return ValidationResult(
                    False,
                    f"Pivot violation: Initial clue at ({row},{col}) was modified",
                )
            return ValidationResult(
                False, "Pivot violation: Initial clues were modified"
            )

        new_grid_hash = new_grid.to_canonical()
        if new_grid_hash not in self._valid_grids:
            if not new_grid.is_valid():
                return ValidationResult(False, "Sudoku constraints violated")
            self._valid_grids.add(new_grid_hash)

        if not prev_grid.validate_next_step(new_grid):
            return ValidationResult(
                False,
                "Invalid step: must fill exactly one empty cell with valid value",
            )

        if not self._is_solvable_cached(new_grid):
            return ValidationResult(False, "Dead-end: resulting grid is unsolvable")

        if isinstance(action, DoneAction) and not new_grid.is_complete():
            return ValidationResult(False, "Puzzle incomplete: empty cells remain")

        return ValidationResult(True, "ok")
