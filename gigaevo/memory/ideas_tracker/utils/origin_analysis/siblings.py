"""Sibling group construction — programs sharing a common parent."""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Any  # noqa: UP035


def _pick_best_parent(
    parents: list[str], programs: dict[str, dict]
) -> tuple[str, float] | None:
    best_pid = None
    best_fit = float("-inf")
    for par in parents:
        p = programs.get(par)
        if not p:
            continue
        try:
            f = float(p.get("fitness", float("nan")))
        except Exception:
            continue
        if math.isfinite(f) and f > best_fit:
            best_fit = f
            best_pid = par
    if best_pid is None:
        return None
    return best_pid, best_fit


def build_sibling_groups(
    programs: dict[str, dict],
    parents_of: dict[str, list[str]],
    mode: str,
    gen_window: int,
) -> dict[tuple, list[str]]:
    groups: dict[tuple, list[str]] = defaultdict(list)

    def bucket(gen: int) -> int:
        if gen_window <= 0:
            return gen
        return gen // (gen_window + 1)

    for pid, pars in parents_of.items():
        if not pars:
            continue
        p = programs.get(pid, {})
        gen = p.get("generation", None)
        fit = p.get("fitness", None)
        if not (isinstance(gen, int) and fit is not None):
            continue
        try:
            f = float(fit)
        except Exception:
            continue
        if not math.isfinite(f):
            continue

        if mode == "best_parent":
            best = _pick_best_parent(pars, programs)
            if best is None:
                continue
            best_pid, _ = best
            key: tuple[Any, ...] = ("best_parent", best_pid, bucket(gen))
        else:
            key = ("parent_set", tuple(sorted(pars)), bucket(gen))

        groups[key].append(pid)

    return dict(groups)


def build_sibling_groups_allgens(
    programs: dict[str, dict],
    parents_of: dict[str, list[str]],
    mode: str,
) -> dict[tuple, list[str]]:
    groups: dict[tuple, list[str]] = defaultdict(list)

    for pid, pars in parents_of.items():
        if not pars:
            continue
        p = programs.get(pid, {})
        fit = p.get("fitness", None)
        if fit is None:
            continue
        try:
            f = float(fit)
        except Exception:
            continue
        if not math.isfinite(f):
            continue

        if mode == "best_parent":
            best = _pick_best_parent(pars, programs)
            if best is None:
                continue
            best_pid, _ = best
            key_ag: tuple[Any, ...] = ("best_parent_allgens", best_pid)
        else:
            key_ag = ("parent_set_allgens", tuple(sorted(pars)))

        groups[key_ag].append(pid)

    return dict(groups)
