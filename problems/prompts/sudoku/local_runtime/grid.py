from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True, slots=True)
class SudokuSpec:
    size: int
    box_rows: int
    box_cols: int

    @property
    def cell_count(self) -> int:
        return self.size * self.size

    @property
    def box_count(self) -> int:
        return (self.size // self.box_rows) * (self.size // self.box_cols)

    def validate(self) -> None:
        if self.size not in (4, 6, 9):
            raise ValueError(f"Unsupported sudoku size: {self.size}. Supported: 4, 6, 9.")
        if self.size % self.box_rows != 0 or self.size % self.box_cols != 0:
            raise ValueError(
                f"Invalid box shape {self.box_rows}x{self.box_cols} for size {self.size}."
            )
        if self.box_rows * self.box_cols != self.size:
            raise ValueError(
                f"Box area must equal size. Got {self.box_rows}*{self.box_cols} != {self.size}."
            )

    @staticmethod
    def default_for_size(size: int) -> "SudokuSpec":
        mapping = {4: (2, 2), 6: (2, 3), 9: (3, 3)}
        if size not in mapping:
            raise ValueError(f"Unsupported size: {size}. Supported: {sorted(mapping)}")
        rows, cols = mapping[size]
        spec = SudokuSpec(size=size, box_rows=rows, box_cols=cols)
        spec.validate()
        return spec

    @staticmethod
    def from_grid_config(
        size: int, box_rows: int | None, box_cols: int | None
    ) -> "SudokuSpec":
        if box_rows is None or box_cols is None:
            return SudokuSpec.default_for_size(size)
        spec = SudokuSpec(size=size, box_rows=int(box_rows), box_cols=int(box_cols))
        spec.validate()
        return spec


def _box_index(spec: SudokuSpec, row: int, col: int) -> int:
    boxes_per_row = spec.size // spec.box_cols
    return (row // spec.box_rows) * boxes_per_row + (col // spec.box_cols)


@dataclass
class Grid:
    spec: SudokuSpec = field(default_factory=lambda: SudokuSpec.default_for_size(9))
    _data: str = field(default="", repr=False)

    _row_cache: list[int] = field(init=False, repr=False)
    _col_cache: list[int] = field(init=False, repr=False)
    _box_cache: list[int] = field(init=False, repr=False)
    _pivots: int = field(init=False, repr=False, default=0)
    _solvable_cache: Optional[bool] = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self.spec.validate()
        if not self._data:
            self._data = "0" * self.spec.cell_count
        if len(self._data) != self.spec.cell_count:
            raise ValueError(
                f"Grid data must be length {self.spec.cell_count}, got {len(self._data)}"
            )

        allowed = set("0" + "".join(str(i) for i in range(1, self.spec.size + 1)))
        invalid = {ch for ch in self._data if ch not in allowed}
        if invalid:
            raise ValueError(
                f"Grid data contains invalid symbols for size={self.spec.size}: {invalid}"
            )

        self._row_cache = [0] * self.spec.size
        self._col_cache = [0] * self.spec.size
        self._box_cache = [0] * self.spec.box_count

        for idx in range(self.spec.cell_count):
            val = int(self._data[idx])
            if val == 0:
                continue
            self._pivots |= 1 << idx
            row, col = divmod(idx, self.spec.size)
            box = _box_index(self.spec, row, col)
            bit = 1 << val
            self._row_cache[row] |= bit
            self._col_cache[col] |= bit
            self._box_cache[box] |= bit

    @classmethod
    def from_string(cls, value: str, spec: SudokuSpec | None = None) -> "Grid":
        raw = value.strip().replace("\n", "").replace(" ", "")
        if spec is None:
            size = int(len(raw) ** 0.5)
            spec = SudokuSpec.default_for_size(size)
        allowed = set("." + "0" + "".join(str(i) for i in range(1, spec.size + 1)))
        if len(raw) != spec.cell_count:
            raise ValueError(f"Expected length {spec.cell_count}, got {len(raw)}.")
        invalid = {ch for ch in raw if ch not in allowed}
        if invalid:
            raise ValueError(f"Invalid characters for size={spec.size}: {invalid}")
        return cls(spec=spec, _data=raw.replace(".", "0"))

    def to_canonical(self) -> str:
        return self._data.replace("0", ".")

    def _index(self, row: int, col: int) -> int:
        return row * self.spec.size + col

    def get(self, row: int, col: int) -> int:
        return int(self._data[self._index(row, col)])

    def get_pivots_mask(self) -> int:
        return self._pivots

    def is_valid_placement(self, row: int, col: int, val: int) -> bool:
        n = self.spec.size
        if not (0 <= row < n and 0 <= col < n):
            return False
        if not (1 <= val <= n):
            return False
        if self.get(row, col) != 0:
            return False
        bit = 1 << val
        box = _box_index(self.spec, row, col)
        return not (
            (self._row_cache[row] & bit)
            or (self._col_cache[col] & bit)
            or (self._box_cache[box] & bit)
        )

    def is_valid(self) -> bool:
        rows = [0] * self.spec.size
        cols = [0] * self.spec.size
        boxes = [0] * self.spec.box_count
        for idx in range(self.spec.cell_count):
            val = int(self._data[idx])
            if val == 0:
                continue
            row, col = divmod(idx, self.spec.size)
            box = _box_index(self.spec, row, col)
            bit = 1 << val
            if (rows[row] & bit) or (cols[col] & bit) or (boxes[box] & bit):
                return False
            rows[row] |= bit
            cols[col] |= bit
            boxes[box] |= bit
        return True

    def is_complete(self) -> bool:
        return "0" not in self._data

    def find_empty_cell(self) -> Optional[tuple[int, int]]:
        idx = self._data.find("0")
        if idx == -1:
            return None
        return divmod(idx, self.spec.size)

    def set(self, row: int, col: int, val: int) -> bool:
        n = self.spec.size
        if not (0 <= row < n and 0 <= col < n):
            return False
        if not (0 <= val <= n):
            return False

        idx = self._index(row, col)
        old_val = int(self._data[idx])
        self._solvable_cache = None
        box = _box_index(self.spec, row, col)

        if val == 0:
            if old_val != 0:
                bit_old = 1 << old_val
                self._row_cache[row] &= ~bit_old
                self._col_cache[col] &= ~bit_old
                self._box_cache[box] &= ~bit_old
            self._data = self._data[:idx] + "0" + self._data[idx + 1 :]
            return True

        if old_val == 0:
            if not self.is_valid_placement(row, col, val):
                return False
        else:
            bit_old = 1 << old_val
            self._row_cache[row] &= ~bit_old
            self._col_cache[col] &= ~bit_old
            self._box_cache[box] &= ~bit_old
            bit_new = 1 << val
            conflict = (
                (self._row_cache[row] & bit_new)
                or (self._col_cache[col] & bit_new)
                or (self._box_cache[box] & bit_new)
            )
            if conflict:
                self._row_cache[row] |= bit_old
                self._col_cache[col] |= bit_old
                self._box_cache[box] |= bit_old
                return False

        self._data = self._data[:idx] + str(val) + self._data[idx + 1 :]
        bit_new = 1 << val
        self._row_cache[row] |= bit_new
        self._col_cache[col] |= bit_new
        self._box_cache[box] |= bit_new
        return True

    def copy(self) -> "Grid":
        new_grid = Grid(spec=self.spec, _data=self._data)
        new_grid._pivots = self._pivots
        return new_grid

    def validate_next_step(self, next_grid: "Grid") -> bool:
        if not isinstance(next_grid, Grid) or next_grid.spec != self.spec:
            return False

        changes: list[tuple[int, int]] = []
        for idx in range(self.spec.cell_count):
            if self._data[idx] == next_grid._data[idx]:
                continue
            old_val = int(self._data[idx])
            new_val = int(next_grid._data[idx])
            if old_val != 0 or new_val == 0:
                return False
            changes.append((idx, new_val))

        if len(changes) != 1:
            return False

        idx, val = changes[0]
        row, col = divmod(idx, self.spec.size)
        return self.is_valid_placement(row, col, val)

    def verify_pivots_preserved(
        self, other: "Grid"
    ) -> tuple[bool, Optional[tuple[int, int]]]:
        if not isinstance(other, Grid) or other.spec != self.spec:
            return (False, None)
        for idx in range(self.spec.cell_count):
            if self._pivots & (1 << idx) and self._data[idx] != other._data[idx]:
                row, col = divmod(idx, self.spec.size)
                return (False, (row, col))
        return (True, None)

    def is_solvable(self) -> bool:
        if self._solvable_cache is not None:
            return self._solvable_cache
        if not self.is_valid():
            self._solvable_cache = False
            return False
        self._solvable_cache = self._check_solvable(self.copy())
        return self._solvable_cache

    def _check_solvable(self, grid: "Grid") -> bool:
        empty = grid.find_empty_cell()
        if empty is None:
            return True
        row, col = empty
        for val in range(1, grid.spec.size + 1):
            if grid.is_valid_placement(row, col, val):
                grid.set(row, col, val)
                if self._check_solvable(grid):
                    return True
                grid.set(row, col, 0)
        return False
