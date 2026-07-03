"""Structural tests for the summarizer problem package (no network)."""

from __future__ import annotations

import json
from pathlib import Path

from mmar_carl.chain import ReasoningChain

from gigaevo.chains.dag_changes import AllowedDagChanges
from problems.chains.summarizer.shared_config import (
    load_dataset,
    outer_context_builder,
)
from problems.chains.summarizer.validate import INVALID_METRICS, validate

PROBLEM_DIR = Path("problems/chains/summarizer")


def test_seed_genomes_parse_and_carry_platform_extras():
    seeds = sorted(PROBLEM_DIR.glob("initial_programs/*.json"))
    assert len(seeds) == 4
    for seed in seeds:
        doc = json.loads(seed.read_text())
        chain = ReasoningChain.from_dict(doc, use_typed_steps=True)
        assert chain.steps
        assert doc["task_description"]
        assert doc["steps"][-1]["is_output_step"] is True


def test_seeds_are_valid_diff_parents():
    changes = AllowedDagChanges()
    parents = {
        "A": (PROBLEM_DIR / "initial_programs" / "chain_2step.json").read_text(),
        "B": (PROBLEM_DIR / "initial_programs" / "chain_3step.json").read_text(),
    }
    schema = changes.build_schema(parents)
    assert schema.json_schema["title"] == "chain_dag_diff"
    assert changes.render_parents(parents)


def test_dataset_rows_have_required_keys():
    rows = load_dataset()
    assert len(rows) == 8
    for row in rows:
        assert row["input"] and row["task"] and row["expected"]
        context = outer_context_builder(row)
        assert row["input"] in context and row["task"] in context


def test_validate_rejects_broken_genome_without_network():
    metrics, artifact = validate({"steps": "garbage"})
    assert metrics == INVALID_METRICS
    assert artifact["error"].startswith("carl_validation_error")


def test_validate_client_copies_share_call_log():
    from problems.chains.client import CallLog
    from problems.chains.summarizer.validate import _LogAggregatingClient

    client = _LogAggregatingClient(model="m")
    clone = client.copy()
    clone._call_logs.append(
        CallLog(prompt_tokens=3, completion_tokens=7, cost=0.0, cost_utilization=0.0)
    )
    assert sum(log.completion_tokens for log in client.call_logs) == 7
