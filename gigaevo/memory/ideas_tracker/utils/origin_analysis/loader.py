"""JSON loading, graph construction, and root ancestry computation."""

from __future__ import annotations

from collections import defaultdict
import json


def load_ideas(path: str) -> tuple[dict[str, set[str]], dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    bank = None
    if isinstance(data, list) and data and isinstance(data[0], dict):
        if "active_bank" in data[0]:
            bank = data[0]["active_bank"]
        elif "ideas" in data[0]:
            bank = data[0]["ideas"]
    elif isinstance(data, dict) and "active_bank" in data:
        bank = data["active_bank"]

    if bank is None:
        raise ValueError("Could not find 'active_bank' list in banks JSON.")

    idea_to_origin_programs: dict[str, set[str]] = {}
    idea_desc: dict[str, str] = {}

    for idea in bank:
        if not isinstance(idea, dict):
            continue
        idea_id = idea.get("id") or idea.get("short_id") or idea.get("name")
        if not idea_id:
            continue
        origin_programs = idea.get("programs")
        if origin_programs is None:
            origin_programs = idea.get("linked_programs", [])
        origin_programs = origin_programs or []
        idea_to_origin_programs[str(idea_id)] = set(str(x) for x in origin_programs)
        idea_desc[str(idea_id)] = str(idea.get("description", "") or "")

    return idea_to_origin_programs, idea_desc


def load_programs(path: str) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    programs: dict[str, dict] = {}

    def fit_of(p: dict) -> float:
        try:
            return float(p.get("fitness", float("-inf")))
        except Exception:
            return float("-inf")

    if isinstance(data, list):
        for snap in data:
            if (
                isinstance(snap, dict)
                and "programs" in snap
                and isinstance(snap["programs"], list)
            ):
                for p in snap["programs"]:
                    if not isinstance(p, dict) or "id" not in p:
                        continue
                    pid = str(p["id"])
                    if pid not in programs or fit_of(p) > fit_of(programs[pid]):
                        programs[pid] = p
            elif isinstance(snap, dict) and "id" in snap:
                pid = str(snap["id"])
                if pid not in programs or fit_of(snap) > fit_of(programs[pid]):
                    programs[pid] = snap
    elif (
        isinstance(data, dict)
        and "programs" in data
        and isinstance(data["programs"], list)
    ):
        for p in data["programs"]:
            if not isinstance(p, dict) or "id" not in p:
                continue
            programs[str(p["id"])] = p
    else:
        raise ValueError("Unexpected programs JSON format.")

    return programs


def build_parents(programs: dict[str, dict]) -> dict[str, list[str]]:
    import json as _json

    parents_of: dict[str, list[str]] = {}
    for pid, p in programs.items():
        parents = p.get("parents", []) or []
        if isinstance(parents, str):
            try:
                parents = _json.loads(parents)
            except (ValueError, TypeError):
                parents = []
        parents = [str(x) for x in parents if str(x) in programs]
        parents_of[str(pid)] = parents
    return parents_of


def build_children(parents_of: dict[str, list[str]]) -> dict[str, list[str]]:
    children_of: dict[str, list[str]] = defaultdict(list)
    for child, pars in parents_of.items():
        for par in pars:
            children_of[par].append(child)
    return dict(children_of)


def invert_idea_to_programs(
    idea_to_programs: dict[str, set[str]],
) -> dict[str, set[str]]:
    prog_to_ideas: dict[str, set[str]] = defaultdict(set)
    for idea, pids in idea_to_programs.items():
        for pid in pids:
            prog_to_ideas[str(pid)].add(idea)
    return dict(prog_to_ideas)


def compute_roots_memoized(parents_of: dict[str, list[str]]) -> dict[str, set[str]]:
    memo: dict[str, set[str]] = {}

    def roots(pid: str) -> set[str]:
        if pid in memo:
            return memo[pid]
        pars = parents_of.get(pid, [])
        if not pars:
            memo[pid] = {pid}
            return memo[pid]
        out: set[str] = set()
        for par in pars:
            out |= roots(par)
        memo[pid] = out
        return out

    for pid in parents_of.keys():
        roots(pid)
    return memo
