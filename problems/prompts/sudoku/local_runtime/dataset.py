from __future__ import annotations

import json
from pathlib import Path

import yaml

from problems.prompts.sudoku.local_runtime.grid import Grid, SudokuSpec
from problems.prompts.sudoku.local_runtime.models import DoneAction, Node, NodeAction


class GoldPath:
    def __init__(self, nodes: list[Node]):
        self.nodes = nodes


class SudokuAdapter:
    def __init__(self, dataset_path: str | Path) -> None:
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
        if not self.dataset_path.is_dir():
            raise ValueError(f"Dataset path must be a directory: {self.dataset_path}")

        self.puzzles_path = self.dataset_path / "puzzles.jsonl"
        self.chains_path = self.dataset_path / "chains.jsonl"
        self.index_path = self.dataset_path / "index.yaml"

        for path in (self.puzzles_path, self.chains_path, self.index_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing {path.name} in {self.dataset_path}")

        with open(self.index_path, encoding="utf-8") as handle:
            self.metadata = yaml.safe_load(handle)

        grid_meta = (self.metadata or {}).get("grid", {}) or {}
        self.spec = SudokuSpec.from_grid_config(
            size=int(grid_meta.get("size", 9)),
            box_rows=grid_meta.get("box_rows"),
            box_cols=grid_meta.get("box_cols"),
        )

    def _read_chains(self) -> list[dict]:
        chains: list[dict] = []
        with open(self.chains_path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    chains.append(json.loads(line))
        return chains

    def load_gold_paths(
        self,
        size: int | None = None,
        difficulty: str | None = None,
        start_idx: int = 0,
    ) -> list[GoldPath]:
        chains = self._read_chains()
        if difficulty is not None:
            chains = [chain for chain in chains if chain["id"].startswith(f"{difficulty}_")]
        if start_idx > 0:
            chains = chains[start_idx:]
        if size is not None:
            chains = chains[:size]
        return [self._chain_to_gold_path(chain) for chain in chains]

    def _chain_to_gold_path(self, chain_data: dict) -> GoldPath:
        initial_board = chain_data["initial_board"]
        actions = chain_data["actions"]

        root_node = Node(
            parent=None,
            action=NodeAction(0, initial_board),
            state=Grid.from_string(initial_board, spec=self.spec),
        )
        nodes = [root_node]

        for step_idx, action_data in enumerate(actions[:-1]):
            resulting_board = action_data["resulting_board"]
            node = Node(
                parent=nodes[-1],
                action=NodeAction(step_idx + 1, resulting_board),
                state=Grid.from_string(resulting_board, spec=self.spec),
            )
            nodes.append(node)

        if actions:
            final_board = actions[-1]["resulting_board"]
            nodes.append(
                Node(
                    parent=nodes[-1],
                    action=DoneAction(final_board),
                    state=Grid.from_string(final_board, spec=self.spec),
                )
            )

        return GoldPath(nodes)
