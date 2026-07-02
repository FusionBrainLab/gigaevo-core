"""Diagnostic study of the memory-card selection bandit.

Read-only. Treats each memory card as a Beta-Bernoulli bandit arm and tests the
selection trace for known bandit failure modes (RQ1-RQ6 in
plans/curried-zooming-teapot.md). Emits the machine tables to
docs/audits/bandit_health_data.md; the hand-authored narrative lives in
docs/audits/bandit_health_report.md.

Usage:
    python tools/analyze_bandit_health.py RUN_ROOT [--arms BD1,BD2,AP1,AP2]
                                          [--out docs/audits] [--no-figs]

Each arm directory under RUN_ROOT must contain memory/memory_events.jsonl,
memory/write_ledger.jsonl, memory/cards.json and
metrics/program_metrics:*.jsonl.

The runs may be live: the last line of any jsonl can be a partial write, so every
record is parsed defensively and bad lines are skipped.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

from gigaevo.memory.cards import CardStatsBlock, ContextualGain
from gigaevo.memory.read.reputation import block_from_events

PRIOR_MAGNITUDE = 0.1  # EVThompsonAuctioneer cold-card bid magnitude
BASELINE_MEAN = 0.5  # Beta(3,3) abstain-arm mean
COLD_MAG_TOL = 1e-9


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # partial trailing write on a live run


def _load_bank(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    cards = payload.get("cards") if isinstance(payload, Mapping) else None
    if not isinstance(cards, Mapping):
        return {}
    return {cid: rec for cid, rec in cards.items() if isinstance(rec, Mapping)}


@dataclass
class Arm:
    name: str
    root: Path
    selections: list = field(default_factory=list)  # MEMORY_READ_SELECTION rows
    restamps: list = field(default_factory=list)  # MEMORY_GAIN_RESTAMP rows
    researches: list = field(default_factory=list)  # MEMORY_RESEARCH rows
    research_steps: list = field(default_factory=list)  # MEMORY_RESEARCH_STEP rows
    saves: int = 0  # MEMORY_STORE_WRITE op=save
    merges: int = 0  # MEMORY_STORE_WRITE op=merge
    ledger: list = field(default_factory=list)
    bank: dict = field(default_factory=dict)  # card_id -> card record
    frontier: list = field(default_factory=list)  # cumulative-max-able (s, v)

    @property
    def mem(self) -> Path:
        return self.root / "memory"


def load_arm(name: str, root: Path) -> Arm:
    arm = Arm(name=name, root=root / name)
    for e in _iter_jsonl(arm.mem / "memory_events.jsonl"):
        ev = e.get("event")
        if ev == "MEMORY_READ_SELECTION":
            e["_parent"] = (e.get("parent_ids") or [None])[0]
            arm.selections.append(e)
        elif ev == "MEMORY_GAIN_RESTAMP":
            arm.restamps.append(e)
        elif ev == "MEMORY_RESEARCH":
            arm.researches.append(e)
        elif ev == "MEMORY_RESEARCH_STEP":
            arm.research_steps.append(e)
        elif ev == "MEMORY_STORE_WRITE":
            if e.get("op") == "save":
                arm.saves += 1
            elif e.get("op") == "merge":
                arm.merges += 1
    arm.ledger = list(_iter_jsonl(arm.mem / "write_ledger.jsonl"))
    arm.bank = _load_bank(arm.mem / "cards.json")
    fr = list(
        _iter_jsonl(
            arm.root / "metrics" / "program_metrics:valid_frontier_fitness.jsonl"
        )
    )
    arm.frontier = [(row.get("s"), row.get("v")) for row in fr if "v" in row]
    return arm


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #


def is_program(cid: str, bank: Mapping[str, Mapping] | None = None) -> bool:
    if bank is not None:
        rec = bank.get(cid)
        if rec is not None:
            return rec.get("kind") == "program"
    return isinstance(cid, str) and cid.startswith("program-")


def is_cold_mag(mag) -> bool:
    """Card rode the optimistic prior magnitude (no learned magnitude)."""
    return mag is not None and abs(mag - PRIOR_MAGNITUDE) < COLD_MAG_TOL


def is_prior_post(a, b) -> bool:
    return a == 1.0 and b == 1.0


def injected_selections(arm: Arm):
    """Selections that actually injected a card (non-empty budgeted_ids)."""
    return [s for s in arm.selections if s.get("budgeted_ids")]


def winner_rows(sel: dict):
    bset = set(sel.get("budgeted_ids") or [])
    return [r for r in (sel.get("slate") or []) if r.get("card_id") in bset]


def latest_posterior(arm: Arm):
    """Latest (a,b) seen on the slate per card_id (any slate row, not just winners)."""
    out = {}
    for s in arm.selections:
        for r in s.get("slate") or []:
            out[r["card_id"]] = (r["posterior_a"], r["posterior_b"])
    return out


def _card_block(rec: Mapping[str, Any] | None) -> CardStatsBlock | None:
    """Live reputation block recomputed from a card's persisted ``gain_events``
    with the same ``block_from_events`` math the auction reads. ``None`` for a
    missing card or one with no events."""
    if not rec:
        return None
    raw = rec.get("gain_events")
    if not isinstance(raw, list):
        return None
    events = [ContextualGain.model_validate(e) for e in raw if isinstance(e, Mapping)]
    return block_from_events(events)


def gini(counts) -> float:
    xs = sorted(float(c) for c in counts)
    n = len(xs)
    if n == 0 or sum(xs) == 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return (2 * cum) / (n * sum(xs)) - (n + 1) / n


def hhi(counts) -> float:
    tot = sum(counts)
    if tot == 0:
        return 0.0
    return sum((c / tot) ** 2 for c in counts)


def deciles(seq, k=10):
    n = len(seq)
    if n == 0:
        return []
    size = math.ceil(n / k)
    return [seq[i : i + size] for i in range(0, n, size)]


# --------------------------------------------------------------------------- #
# RQ1 — optimistic-prior calibration
# --------------------------------------------------------------------------- #


def rq1(arm: Arm) -> dict:
    inj = injected_selections(arm)
    cold_mag = prior_post = program = total = 0
    displaced = 0
    win_mags = []
    cold_flags = []  # per injection (order preserved) -> rode prior magnitude
    for s in inj:
        rows = winner_rows(s)
        for r in rows:
            total += 1
            win_mags.append(r["magnitude"])
            cm = is_cold_mag(r["magnitude"])
            cold_flags.append(cm)
            if cm:
                cold_mag += 1
            if is_prior_post(r["posterior_a"], r["posterior_b"]):
                prior_post += 1
            if is_program(r["card_id"], arm.bank):
                program += 1
        # counterfactual displacement: a cold winner while an evidence-bearing
        # (learned-magnitude, positive-bid) card sat on the same slate
        if rows and all(is_cold_mag(r["magnitude"]) for r in rows):
            for r in s["slate"]:
                if (
                    r["card_id"] not in set(s["budgeted_ids"])
                    and not is_cold_mag(r["magnitude"])
                    and (r.get("bid") or 0) > 0
                ):
                    displaced += 1
                    break
    first_half = cold_flags[: len(cold_flags) // 2]
    second_half = cold_flags[len(cold_flags) // 2 :]
    fh = sum(first_half) / len(first_half) if first_half else 0.0
    sh = sum(second_half) / len(second_half) if second_half else 0.0
    cold_frac = cold_mag / total if total else 0.0
    # healthy if cold fraction tapers in the back half; pathological if high+flat
    pathological = cold_frac > 0.5 and sh >= fh - 0.1
    return {
        "n_injections": total,
        "cold_magnitude_frac": cold_frac,
        "prior_posterior_frac": prior_post / total if total else 0.0,
        "program_win_frac": program / total if total else 0.0,
        "cold_frac_first_half": fh,
        "cold_frac_second_half": sh,
        "tapers": sh < fh - 0.1,
        "displacement_events": displaced,
        "win_magnitudes": win_mags,
        "cold_flags": cold_flags,
        "verdict": "PATHOLOGICAL"
        if pathological
        else ("HEALTHY" if cold_frac < 0.3 else "WATCH"),
    }


# --------------------------------------------------------------------------- #
# RQ2 — starvation / monopolization
# --------------------------------------------------------------------------- #


def rq2(arm: Arm) -> dict:
    counts = Counter()
    for s in arm.selections:
        for cid in s.get("budgeted_ids") or []:
            counts[cid] += 1
    total = sum(counts.values())
    post = latest_posterior(arm)
    bad_arms = []
    if counts:
        q75 = sorted(counts.values())[int(0.75 * (len(counts) - 1))]
        for cid, n in counts.items():
            a, b = post.get(cid, (1.0, 1.0))
            # prefer the live block recomputed from the card's gain_events
            block = _card_block(arm.bank.get(cid))
            if block is not None and block.posterior_a is not None:
                a, b = block.posterior_a, block.posterior_b
            mean = a / (a + b)
            if n >= q75 and mean <= BASELINE_MEAN:
                bad_arms.append((cid, n, round(mean, 3)))
    cvals = list(counts.values())
    max_share = (max(cvals) / total) if total else 0.0
    bank = len(arm.bank)
    pathological = max_share >= 0.15 and bool(bad_arms)
    return {
        "n_injections": total,
        "distinct_budgeted": len(counts),
        "bank_size": bank,
        "gini": gini(cvals),
        "hhi": hhi(cvals),
        "max_single_share": max_share,
        "top5": counts.most_common(5),
        "bad_arm_monopoly": sorted(bad_arms, key=lambda x: -x[1]),
        "counts": dict(counts),
        "verdict": "PATHOLOGICAL"
        if pathological
        else ("WATCH" if max_share >= 0.15 else "HEALTHY"),
    }


# --------------------------------------------------------------------------- #
# RQ3 — posterior credibility
# --------------------------------------------------------------------------- #


def rq3(arm: Arm) -> dict:
    post = latest_posterior(arm)
    ever = set()
    for s in arm.selections:
        ever.update(s.get("budgeted_ids") or [])
    off_prior = 0
    for cid in ever:
        a, b = post.get(cid, (1.0, 1.0))
        if not is_prior_post(a, b):
            off_prior += 1
    off_prior_frac = off_prior / len(ever) if ever else 0.0
    credited_series = [r.get("credited_card_count", 0) for r in arm.restamps]
    # confidence snapshot from the bank: blocks recomputed per card
    with_block = confident = 0
    for rec in arm.bank.values():
        block = _card_block(rec)
        if block is None:
            continue
        with_block += 1
        if block.efficacy_confident:
            confident += 1
    # reconciliation: the global block recomputed from each card's gain_events
    # vs the (a,b) stamped on its latest slate row. Informational only (does not
    # feed the verdict). Expect a HIGH match under a global reputation, but a LOW
    # match under bd_proximity is BY DESIGN, not drift: BDProximityReputation
    # stamps a cell-subset posterior on the slate while _card_block recomputes
    # over all gain_events, so the two diverge by construction on BD arms.
    dir_ok = dir_total = 0
    for cid, rec in arm.bank.items():
        block = _card_block(rec)
        slate = post.get(cid)
        if block is None or block.posterior_a is None or slate is None:
            continue
        dir_total += 1
        if (
            abs(block.posterior_a - slate[0]) < 1e-6
            and abs(block.posterior_b - slate[1]) < 1e-6
        ):
            dir_ok += 1
    # pathological if credit is too sparse to overrule the prior
    confident_share = (confident / with_block) if with_block else 0.0
    pathological = confident_share <= 0.34 or off_prior_frac < 0.5
    return {
        "ever_injected": len(ever),
        "off_prior_frac": off_prior_frac,
        "credited_series": credited_series,
        "cards_with_block": with_block,
        "confident_cards": confident,
        "last_confident_share": confident_share,
        "direction_consistent": f"{dir_ok}/{dir_total}",
        "verdict": "PATHOLOGICAL"
        if pathological
        else ("HEALTHY" if confident_share > 0.5 and off_prior_frac > 0.7 else "WATCH"),
    }


# --------------------------------------------------------------------------- #
# RQ4 — non-stationarity / rotting arms (decision-order proxy)
# --------------------------------------------------------------------------- #


def rq4(arm: Arm) -> dict:
    inj = injected_selections(arm)
    n = len(inj)
    # cards that first won in the first quarter of injections ("early arms")
    early_cut = n // 4
    early_arms = set()
    for s in inj[:early_cut]:
        early_arms.update(s.get("budgeted_ids") or [])
    # share of late-half injections still going to early arms
    late = inj[n // 2 :]
    late_to_early = 0
    late_total = 0
    for s in late:
        for cid in s.get("budgeted_ids") or []:
            late_total += 1
            if cid in early_arms:
                late_to_early += 1
    persistence = late_to_early / late_total if late_total else 0.0
    # frontier regime shift over the run
    best = 0.0
    fr_curve = []
    for _s, v in arm.frontier:
        if isinstance(v, (int, float)) and v > best:
            best = v
        fr_curve.append(best)
    fr_rise = (fr_curve[-1] - fr_curve[0]) if fr_curve else 0.0
    # rotting risk: frontier climbed a lot AND early arms still dominate late, with
    # no posterior decay mechanism in the system
    pathological = persistence >= 0.5 and fr_rise > 0
    return {
        "early_arms": len(early_arms),
        "late_to_early_persistence": persistence,
        "frontier_rise": fr_rise,
        "frontier_final": fr_curve[-1] if fr_curve else None,
        "verdict": "WATCH (latent)"
        if pathological
        else ("HEALTHY" if persistence < 0.3 else "WATCH"),
        "_note": "decision-order proxy; no parent-id->fitness link in trace (see report)",
    }


# --------------------------------------------------------------------------- #
# RQ5 — idea generation over time
# --------------------------------------------------------------------------- #


def rq5(arm: Arm) -> dict:
    outcomes = Counter(r.get("outcome") for r in arm.ledger)
    prog_in = sum(
        1 for r in arm.ledger if is_program(r.get("incoming_id", ""), arm.bank)
    )
    added_ids = {r.get("final_id") for r in arm.ledger if r.get("outcome") == "added"}
    admitted = sum(outcomes.get(k, 0) for k in ("added", "updated", "merged"))
    refused = sum(outcomes.get(k, 0) for k in ("discarded", "rejected_harm"))
    admission_rate = admitted / (admitted + refused) if (admitted + refused) else 0.0
    # research novelty: unique vs total hits over each decision's step trail
    steps_by_decision = defaultdict(list)
    for st in arm.research_steps:
        steps_by_decision[st.get("decision_id")].append(st)
    research_novelty = []
    for steps in steps_by_decision.values():
        hits = [h for st in steps for h in (st.get("hit_ids") or [])]
        if hits:
            research_novelty.append(len(set(hits)) / len(hits))
    ever_budgeted = set()
    for s in arm.selections:
        ever_budgeted.update(s.get("budgeted_ids") or [])
    added = outcomes.get("added", 0)
    recycled = sum(
        outcomes.get(k, 0) for k in ("merged", "discarded", "updated", "evicted")
    )
    # pathological if inflow is dominated by recycling rather than genuinely new arms
    pathological = added > 0 and recycled > added
    return {
        "ledger_outcomes": dict(outcomes),
        "distinct_added": len(added_ids),
        "program_inflow": prog_in,
        "added_vs_recycled": (added, recycled),
        "admission_rate": round(admission_rate, 3),
        "research_novelty_mean": round(sum(research_novelty) / len(research_novelty), 3)
        if research_novelty
        else None,
        "coverage": {
            "ever_budgeted": len(ever_budgeted),
            "bank_size": len(arm.bank),
            "total_added": len(added_ids),
        },
        "verdict": "WATCH"
        if pathological
        else ("HEALTHY" if added >= recycled else "WATCH"),
    }


# --------------------------------------------------------------------------- #
# RQ6 — auction integrity / abstention
# --------------------------------------------------------------------------- #


def rq6(arm: Arm) -> dict:
    abstain = Counter()
    n_with_candidates = 0
    gate_pass = gate_total = 0
    floor_fire = 0
    gate_flip = floor_flip = 0
    integrity_ok = True
    for s in arm.selections:
        cc = len(s.get("candidate_ids") or [])
        injected = bool(s.get("budgeted_ids"))
        if cc > 0:
            n_with_candidates += 1
            if not injected:
                abstain[s.get("empty_reason") or "gated_out"] += 1
        # integrity: budgeted subset of winners, at most max_cards
        winners = set(s.get("auction_winner_ids") or [])
        bud = set(s.get("budgeted_ids") or [])
        if not bud <= winners or len(bud) > (s.get("max_cards", 1) or 1):
            integrity_ok = False
        for r in s.get("slate") or []:
            gate_total += 1
            gate = r["theta"] > r["baseline_theta"]
            if gate:
                gate_pass += 1
            bid = r.get("bid") or 0
            if not r.get("selected"):
                if not gate:
                    gate_flip += 1
                elif bid <= 0:
                    floor_flip += 1
            if bid < 0:
                floor_fire += 1
    abstain_rate = (
        (sum(abstain.values()) / n_with_candidates) if n_with_candidates else 0.0
    )
    gate_pass_rate = gate_pass / gate_total if gate_total else 0.0
    # pathological if the abstain arm essentially never fires and the floor is inert
    pathological = abstain_rate < 0.02 and floor_fire == 0
    return {
        "selections_with_candidates": n_with_candidates,
        "abstain_rate": abstain_rate,
        "abstain_reasons": dict(abstain),
        "gate_pass_rate": gate_pass_rate,
        "gate_flips": gate_flip,
        "floor_flips": floor_flip,
        "floor_negative_mag_rows": floor_fire,
        "integrity_ok": integrity_ok,
        "verdict": "WATCH" if pathological else "HEALTHY",
    }


RQS = [
    ("RQ1 optimistic-prior", rq1),
    ("RQ2 monopolization", rq2),
    ("RQ3 posterior credibility", rq3),
    ("RQ4 rotting arms", rq4),
    ("RQ5 idea generation", rq5),
    ("RQ6 auction integrity", rq6),
]


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #


def make_figures(arms, results, figdir: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[figs] matplotlib unavailable ({exc}); skipping plots")
        return
    figdir.mkdir(parents=True, exist_ok=True)
    names = [a.name for a in arms]

    # RQ1: cold-winner fraction first vs second half
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(names))
    fh = [results[n]["RQ1 optimistic-prior"]["cold_frac_first_half"] for n in names]
    sh = [results[n]["RQ1 optimistic-prior"]["cold_frac_second_half"] for n in names]
    ax.bar([i - 0.2 for i in x], fh, 0.4, label="first half")
    ax.bar([i + 0.2 for i in x], sh, 0.4, label="second half")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylabel("cold-magnitude winner fraction")
    ax.set_title("RQ1: optimistic-prior wins (taper = healthy)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "rq1_cold_winner_taper.png", dpi=110)
    plt.close(fig)

    # RQ2: injection concentration (sorted counts per arm)
    fig, ax = plt.subplots(figsize=(7, 4))
    for n in names:
        cvals = sorted(
            results[n]["RQ2 monopolization"]["counts"].values(), reverse=True
        )
        ax.plot(range(1, len(cvals) + 1), cvals, marker="o", label=n)
    ax.set_xlabel("card rank")
    ax.set_ylabel("injection count")
    ax.set_title("RQ2: injection concentration across arms")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "rq2_concentration.png", dpi=110)
    plt.close(fig)

    # RQ3: credited cards per gain-restamp sweep
    fig, ax = plt.subplots(figsize=(7, 4))
    for n in names:
        cs = results[n]["RQ3 posterior credibility"]["credited_series"]
        if cs:
            ax.plot(range(1, len(cs) + 1), cs, marker="s", label=n)
    ax.set_xlabel("gain-restamp sweep")
    ax.set_ylabel("credited_card_count")
    ax.set_title("RQ3: cards credited with gain events per sweep")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "rq3_credited.png", dpi=110)
    plt.close(fig)

    # RQ1 magnitude histogram (all arms pooled)
    fig, ax = plt.subplots(figsize=(7, 4))
    allmags = []
    for n in names:
        allmags += [
            m for m in results[n]["RQ1 optimistic-prior"]["win_magnitudes"] if m
        ]
    if allmags:
        ax.hist(allmags, bins=40)
        ax.axvline(PRIOR_MAGNITUDE, color="red", ls="--", label="prior_magnitude=0.1")
        ax.set_xlabel("winning-card magnitude")
        ax.set_ylabel("count")
        ax.set_title("RQ1: winning magnitudes (spike at 0.1 = cold-prior wins)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figdir / "rq1_magnitude_hist.png", dpi=110)
    plt.close(fig)
    print(f"[figs] wrote plots to {figdir}")


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def build_report(arms, results) -> str:
    names = [a.name for a in arms]
    L = []
    L.append("# Memory-Card Selection Bandit — Health Report\n")
    L.append(
        "> Generated by `tools/analyze_bandit_health.py` over a **live-run snapshot**. "
        "Each card is a Beta-Bernoulli bandit arm; the auction is Thompson sampling with an "
        "EV-weighted bid and a Beta(3,3) abstain arm. Findings map to the bandit failure modes "
        "in `plans/curried-zooming-teapot.md`.\n"
    )
    L.append("## Caveats (read first)\n")
    L.append(
        "- **Variance floor:** with few seeds per arm, all cross-arm contrasts are "
        "**directional / hypothesis-generating only**; compare each A/B gap against the "
        "within-arm seed spread before reading anything into it.\n"
        "- **Load-bearing evidence is within-run structural** (RQ1 cold fraction, RQ2 concentration, "
        "RQ3 stuck-at-prior, RQ6 abstain/floor): large-N over decisions inside a single run, "
        "independent of the seed floor.\n"
        "- This study **diagnoses only** — no fix is proposed here.\n"
    )

    # executive summary
    L.append("## Executive summary (user's three questions)\n")
    L.append("| Question | RQs | Verdict (per arm) |")
    L.append("|---|---|---|")

    def verds(rqkeys):
        cells = []
        for n in names:
            vs = "/".join(results[n][k]["verdict"].split()[0] for k in rqkeys)
            cells.append(f"{n}:{vs}")
        return " · ".join(cells)

    L.append(
        f"| Suspicious selection? | RQ1,RQ2,RQ6 | "
        f"{verds(['RQ1 optimistic-prior', 'RQ2 monopolization', 'RQ6 auction integrity'])} |"
    )
    L.append(
        f"| Reputation updates as expected? | RQ3 | {verds(['RQ3 posterior credibility'])} |"
    )
    L.append(f"| New ideas over time? | RQ5 | {verds(['RQ5 idea generation'])} |")
    L.append(f"| Latent: rotting arms | RQ4 | {verds(['RQ4 rotting arms'])} |")
    L.append("")

    # per-RQ tables
    for title, _fn in RQS:
        L.append(f"## {title}\n")
        keys = sorted(
            {k for n in names for k in results[n][title] if not k.startswith("_")}
        )
        # drop bulky raw series from the table
        drop = {
            "win_magnitudes",
            "cold_flags",
            "counts",
            "credited_series",
        }
        keys = [k for k in keys if k not in drop]
        L.append("| metric | " + " | ".join(names) + " |")
        L.append("|" + "---|" * (len(names) + 1))
        for k in keys:
            row = [k]
            for n in names:
                row.append(fmt(results[n][title].get(k, "")))
            L.append("| " + " | ".join(row) + " |")
        note = next(
            (
                results[n][title].get("_note")
                for n in names
                if results[n][title].get("_note")
            ),
            None,
        )
        if note:
            L.append(f"\n_Note: {note}_")
        L.append("")

    return "\n".join(L)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root")
    ap.add_argument("--arms", default="BD1,BD2,AP1,AP2")
    ap.add_argument("--out", default="docs/audits")
    ap.add_argument("--no-figs", action="store_true")
    args = ap.parse_args()

    root = Path(args.run_root)
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    arms = []
    for nm in arm_names:
        if not (root / nm).exists():
            print(f"[skip] {nm}: {root / nm} not found")
            continue
        arms.append(load_arm(nm, root))

    results = {}
    for arm in arms:
        results[arm.name] = {title: fn(arm) for title, fn in RQS}
        r1 = results[arm.name]["RQ1 optimistic-prior"]
        r2 = results[arm.name]["RQ2 monopolization"]
        print(
            f"{arm.name}: injections={r1['n_injections']} "
            f"cold_mag={r1['cold_magnitude_frac']:.0%} "
            f"program_win={r1['program_win_frac']:.0%} "
            f"max_share={r2['max_single_share']:.0%} "
            f"distinct={r2['distinct_budgeted']}/{r2['bank_size']}"
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = build_report(arms, results)
    (out / "bandit_health_data.md").write_text(report)
    print(f"[report] wrote {out / 'bandit_health_data.md'}")
    if not args.no_figs:
        make_figures(arms, results, out / "figs")


if __name__ == "__main__":
    main()
