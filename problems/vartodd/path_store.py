from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path as FsPath
from typing import List, Optional, Sequence
import json
import pickle
from copy import deepcopy

import numpy as np

from mcts_dao import Path as MctsPath, Dao
from node import Node, Matrix, ActionInfo

X0_LENGTH = 200

@dataclass
class PathStore:
    root_dir: str = "data/path_backups"

    def _resolve_dir(self, name: str) -> FsPath:
        if not name:
            raise ValueError("name must be a non-empty string")
        base = FsPath(self.root_dir) / name
        return base

    def save(self, name: str, paths: Sequence[MctsPath], *, store_daos: bool = True) -> FsPath:
        base = self._resolve_dir(name)
        base.mkdir(parents=True, exist_ok=True)

        matrices_path = base / "matrices.npz"
        meta_path = base / "meta.json"
        daos_path = base / "daos.pkl"
        incoming_path = base / "incoming.pkl"

        matrices: dict[str, np.ndarray] = {}
        meta = {
            "version": 2,
            "paths": [],
        }
        incoming_payload: List[List[Optional[tuple]]] = []

        for p_idx, path in enumerate(paths):
            if path.final_node is None:
                raise ValueError(f"path at index {p_idx} has no final_node")

            nodes: List[Node] = []
            cur = path.final_node
            while cur is not None:
                nodes.append(cur)
                cur = cur.parent
            nodes.reverse()

            matrix_keys: List[str] = []
            incoming_info: List[Optional[tuple]] = []
            for s_idx, node in enumerate(nodes):
                key = f"p{p_idx}_s{s_idx}"
                matrices[key] = node.state.to_numpy()
                matrix_keys.append(key)
                if node.incoming is None:
                    incoming_info.append(None)
                else:
                    incoming_info.append((node.incoming.cand, node.incoming.global_info, node.incoming.source))
            x0s = [list(x0) for x0 in path.x0s]
            #TODO
            for x0 in x0s:
                while x0 and x0[-1] == 0:
                    x0.pop()
            meta["paths"].append(
                {
                    "matrix_keys": matrix_keys,
                    "ranks_thr": list(path.ranks_thr),
                    "x0s": x0s,
                }
            )
            incoming_payload.append(incoming_info)

        np.savez_compressed(matrices_path, **matrices)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if store_daos:
            daos_payload = [path.daos for path in paths]
            with open(daos_path, "wb") as f:
                pickle.dump(daos_payload, f)
        with open(incoming_path, "wb") as f:
            pickle.dump(incoming_payload, f)

        return base

    def load(self, name: str, *, dao_fallback: Optional[Dao] = None) -> List[MctsPath]:
        base = self._resolve_dir(name)
        matrices_path = base / "matrices.npz"
        meta_path = base / "meta.json"
        daos_path = base / "daos.pkl"
        incoming_path = base / "incoming.pkl"

        if not matrices_path.exists():
            raise FileNotFoundError(f"missing matrices file: {matrices_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"missing meta file: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        daos_payload: Optional[List[List[Dao]]] = None
        if daos_path.exists():
            with open(daos_path, "rb") as f:
                daos_payload = pickle.load(f)

        incoming_payload: Optional[List[List[Optional[tuple]]]] = None
        if incoming_path.exists():
            try:
                with open(incoming_path, "rb") as f:
                    incoming_payload = pickle.load(f)
            except Exception:
                incoming_payload = None

        out: List[MctsPath] = []
        with np.load(matrices_path) as data:
            for idx, p in enumerate(meta.get("paths", [])):
                keys = p.get("matrix_keys", [])
                if not keys:
                    raise ValueError(f"path at index {idx} has no matrix_keys")

                nodes: List[Node] = []
                prev: Optional[Node] = None
                incoming_entries: Optional[List[Optional[tuple]]] = None
                if isinstance(incoming_payload, list) and idx < len(incoming_payload):
                    per_path = incoming_payload[idx]
                    if isinstance(per_path, list) and len(per_path) == len(keys):
                        incoming_entries = per_path
                for key in keys:
                    incoming: Optional[ActionInfo] = None
                    if incoming_entries is not None:
                        entry = incoming_entries[len(nodes)]
                        if isinstance(entry, ActionInfo):
                            incoming = entry
                        elif isinstance(entry, tuple) and len(entry) == 3:
                            cand, global_info, source = entry
                            incoming = ActionInfo(cand=cand, global_info=global_info, source=source)
                    if key not in data:
                        raise KeyError(f"matrix key not found in npz: {key}")
                    mat = Matrix.from_numpy(data[key])
                    node = Node(state=mat, parent=prev, incoming=incoming, depth=0 if prev is None else prev.depth + 1)
                    nodes.append(node)
                    prev = node

                path = MctsPath(
                    final_node=nodes[-1],
                    ranks_thr=list(p.get("ranks_thr", [])),
                    daos=[],
                    x0s=[list(x0) + [0]*(X0_LENGTH - len(x0)) for x0 in p.get("x0s", [])],
                )

                if daos_payload is not None:
                    if idx >= len(daos_payload):
                        raise ValueError("daos payload length mismatch with paths")
                    path.daos = daos_payload[idx]
                elif dao_fallback is not None:
                    path.daos = [deepcopy(dao_fallback)]

                if not path.daos:
                    raise ValueError(
                        "loaded path has no dao snapshots; pass dao_fallback or save with store_daos=True"
                    )

                out.append(path)

        return out
