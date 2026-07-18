#!/usr/bin/env python3
"""Offline memory-card injection→use funnel for heilbron memory runs.

Read-only. Answers "is an injected card even read, and does using it help?" by
joining each child's frozen ``memory_injected_idea_ids`` (the GAM/extra-channel
slate) with the mutator-declared ``card_ids_used``. A card counts as *used* for a
child when it is in ``base_selected ∩ card_ids_used`` — the same credit rule the
system uses for ``gain_events`` (compute_contextual_gains).

Pre-registration + decision thresholds: docs/audits/card_use_offline_prereg_2026-06-25.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

ARMS = ("BD1", "BD2", "AP1", "AP2")


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                return  # tolerate a partial trailing write on a live run


def normalize_program(raw: dict) -> dict:
    """Project a disk-storage program JSON onto the fields the funnel needs."""
    md = raw.get("metadata") or {}
    mo = md.get("mutation_output") or {}
    metrics = raw.get("metrics") or {}
    fitness = None
    if metrics.get("is_valid"):
        fitness = metrics.get("fitness")
    base_metrics = md.get("memory_base_metrics") or {}
    base_fitness = base_metrics.get("fitness") if base_metrics.get("is_valid") else None
    return {
        "id": raw.get("id"),
        "injected": list(md.get("memory_injected_idea_ids") or []),
        "base_selected": list(md.get("memory_base_selected_idea_ids") or []),
        "card_ids_used": list(mo.get("card_ids_used") or []),
        "fitness": fitness,
        "base_fitness": base_fitness,
        "iteration": raw.get("iteration", 0),
    }


def cited_cards(child: dict) -> set[str]:
    """Cards credited as used: selected for the base parent AND declared used."""
    return set(child["base_selected"]) & set(child["card_ids_used"])


def funnel(children: list[dict]) -> dict:
    """Injected→used funnel over the extra (GAM) channel.

    Instance-level counts an (child, card) pair; child-level counts children.
    ``unread`` = injected card instances never credited as used.
    """
    n_children = len(children)
    children_injected = children_used_any = 0
    injected_instances = used_instances = 0
    for c in children:
        inj = set(c["injected"])
        used = cited_cards(c) & inj
        if inj:
            children_injected += 1
        if used:
            children_used_any += 1
        injected_instances += len(inj)
        used_instances += len(used)
    unread_instances = injected_instances - used_instances
    return {
        "n_children": n_children,
        "children_injected": children_injected,
        "children_used_any": children_used_any,
        "injected_instances": injected_instances,
        "used_instances": used_instances,
        "unread_instances": unread_instances,
        "unread_frac": (unread_instances / injected_instances)
        if injected_instances
        else 0.0,
    }


def _gain(child: dict, higher_is_better: bool):
    if child["fitness"] is None or child["base_fitness"] is None:
        return None
    delta = child["fitness"] - child["base_fitness"]
    return delta if higher_is_better else -delta


def use_conditional_gain(
    children: list[dict], *, higher_is_better: bool = True
) -> dict:
    """Median child gain split by whether ≥1 injected card was used vs not."""
    used_gains, unread_gains = [], []
    for c in children:
        if not c["injected"]:
            continue
        g = _gain(c, higher_is_better)
        if g is None:
            continue
        if cited_cards(c) & set(c["injected"]):
            used_gains.append(g)
        else:
            unread_gains.append(g)
    return {
        "n_used": len(used_gains),
        "n_unread": len(unread_gains),
        "median_gain_used": median(used_gains) if used_gains else None,
        "median_gain_unread": median(unread_gains) if unread_gains else None,
    }


def basin_share(
    children: list[dict], *, threshold: float = 0.007, higher_is_better: bool = True
) -> dict:
    """Share of positive child gains below the basin-tweak ceiling."""
    positives = []
    for c in children:
        if not c["injected"]:
            continue
        g = _gain(c, higher_is_better)
        if g is not None and g > 0:
            positives.append(g)
    below = sum(1 for g in positives if g < threshold)
    return {
        "n_positive": len(positives),
        "basin_share": (below / len(positives)) if positives else 0.0,
    }


def load_children(arm_root: Path) -> list[dict]:
    prog_dir = arm_root / "storage" / "heilbron" / "programs"
    children = []
    if not prog_dir.exists():
        return children
    for f in sorted(prog_dir.glob("*.json")):
        try:
            raw = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        children.append(normalize_program(raw))
    return children


def analyze_arm(name: str, root: Path) -> dict:
    children = load_children(root / name)
    return {
        "name": name,
        "n": len(children),
        "funnel": funnel(children),
        "gain": use_conditional_gain(children),
        "basin": basin_share(children),
    }


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def build_report(results: list[dict]) -> str:
    lines = [
        "# Memory-Card Injection→Use Funnel (offline)",
        "",
        "> Read-only. Extra (GAM) channel: a card is *used* for a child when it is in",
        "> `base_selected ∩ card_ids_used`. Pre-reg + thresholds:",
        "> `docs/audits/card_use_offline_prereg_2026-06-25.md`.",
        "",
        "## Check 1 — injected→use funnel",
        "",
        "| metric | " + " | ".join(r["name"] for r in results) + " |",
        "|---" * (len(results) + 1) + "|",
    ]
    funnel_keys = [
        "children_injected",
        "children_used_any",
        "injected_instances",
        "used_instances",
        "unread_instances",
        "unread_frac",
    ]
    for k in funnel_keys:
        lines.append(
            f"| {k} | " + " | ".join(_fmt(r["funnel"][k]) for r in results) + " |"
        )
    lines += ["", "## Check 2 — use-conditional gain", ""]
    lines.append("| metric | " + " | ".join(r["name"] for r in results) + " |")
    lines.append("|---" * (len(results) + 1) + "|")
    for k in ["n_used", "n_unread", "median_gain_used", "median_gain_unread"]:
        lines.append(
            f"| {k} | " + " | ".join(_fmt(r["gain"][k]) for r in results) + " |"
        )
    lines += ["", "## Check 3 — basin profile (gain < 0.007)", ""]
    lines.append("| metric | " + " | ".join(r["name"] for r in results) + " |")
    lines.append("|---" * (len(results) + 1) + "|")
    for k in ["n_positive", "basin_share"]:
        lines.append(
            f"| {k} | " + " | ".join(_fmt(r["basin"][k]) for r in results) + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_root", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    results = [analyze_arm(a, args.run_root) for a in arms]
    for r in results:
        f = r["funnel"]
        print(
            f"{r['name']}: n={r['n']} injected={f['children_injected']} "
            f"used_any={f['children_used_any']} unread_frac={f['unread_frac']:.2f} "
            f"gain_used={_fmt(r['gain']['median_gain_used'])} "
            f"gain_unread={_fmt(r['gain']['median_gain_unread'])}"
        )
    report = build_report(results)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        print(f"[report] wrote {args.out}")


if __name__ == "__main__":
    main()
