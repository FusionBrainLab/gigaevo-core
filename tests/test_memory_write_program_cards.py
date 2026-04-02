import json

from gigaevo.memory.memory_write_example import load_memory_cards
from gigaevo.memory.shared_memory.card_conversion import normalize_memory_card
from gigaevo.memory.shared_memory.memory import AmemGamMemory


def _write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_load_memory_cards_adds_top_program_cards(tmp_path):
    banks_path = tmp_path / "banks.json"
    best_ideas_path = tmp_path / "best_ideas.json"
    programs_path = tmp_path / "programs.json"

    _write_json(
        banks_path,
        [
            {
                "timestamp": "2026-03-23 00:00:00",
                "active_bank": [
                    {
                        "id": "idea-1",
                        "description": "Use simulated annealing for local refinement.",
                        "task_description": "Solve the task.",
                        "task_description_summary": "Solve the task efficiently.",
                        "programs": ["prog-1", "prog-9"],
                    },
                    {
                        "id": "idea-2",
                        "description": "Add a boundary-aware repair step.",
                        "task_description": "Solve the task.",
                        "task_description_summary": "Solve the task efficiently.",
                        "programs": ["prog-9"],
                    },
                ],
                "inactive_bank": [],
            }
        ],
    )
    _write_json(
        best_ideas_path,
        [
            {
                "timestamp": "2026-03-23 00:00:00",
                "best_ideas": [
                    {"idea_id": "idea-1", "description": "Use simulated annealing."}
                ],
            }
        ],
    )
    _write_json(
        programs_path,
        [
            {
                "timestamp": "2026-03-23 00:00:00",
                "programs": [
                    {
                        "id": f"prog-{idx}",
                        "fitness": float(idx),
                        "generation": idx,
                        "strategy": "hybrid",
                        "task_description": "Solve the task.",
                        "task_description_summary": "Solve the task efficiently.",
                        "code": f"def run_code():\n    return {idx}\n",
                    }
                    for idx in range(10)
                ],
            }
        ],
    )

    cards = load_memory_cards(
        banks_path,
        best_ideas_path=best_ideas_path,
        programs_path=programs_path,
        best_programs_percent=5.0,
    )

    idea_cards = [card for card in cards if card.category != "program"]
    program_cards = [card for card in cards if card.category == "program"]

    assert len(idea_cards) == 1
    assert len(program_cards) == 1

    program_card = program_cards[0]
    assert program_card.program_id == "prog-9"
    assert program_card.fitness == 9.0
    assert program_card.task_description_summary == "Solve the task efficiently."
    assert "def run_code()" in program_card.code
    assert program_card.connected_ideas == [
        {
            "idea_id": "idea-1",
            "description": "Use simulated annealing for local refinement.",
        },
        {
            "idea_id": "idea-2",
            "description": "Add a boundary-aware repair step.",
        },
    ]
    assert set(program_card.keys()) == {
        "id",
        "category",
        "program_id",
        "task_description",
        "task_description_summary",
        "description",
        "fitness",
        "code",
        "connected_ideas",
    }


def test_program_cards_bypass_idea_dedup(tmp_path):
    memory = AmemGamMemory(
        checkpoint_path=str(tmp_path / "memory"),
        use_api=False,
        enable_llm_synthesis=False,
        enable_memory_evolution=False,
        enable_llm_card_enrichment=False,
        sync_on_init=False,
        card_update_dedup_config={"enabled": True},
    )
    memory.save_card(
        {
            "id": "idea-1",
            "category": "general",
            "description": "Repair invalid candidates before scoring.",
            "task_description": "Solve task.",
            "task_description_summary": "Solve task.",
        }
    )
    memory.llm_service = object()

    def _unexpected_call(*args, **kwargs):
        raise AssertionError("Program cards should not use idea-card dedup.")

    memory._score_retrieved_candidates = _unexpected_call  # type: ignore[method-assign]

    card_id = memory.save_card(
        {
            "id": "program-prog-1",
            "category": "program",
            "program_id": "prog-1",
            "fitness": 12.5,
            "task_description": "Solve task.",
            "task_description_summary": "Solve task.",
            "description": "Top evolved program for Solve task. (fitness=12.5).",
            "code": "def run_code():\n    return 1\n",
            "connected_ideas": [
                {
                    "idea_id": "idea-1",
                    "description": "Repair invalid candidates before scoring.",
                }
            ],
        }
    )

    stored = memory.get_card(card_id)
    assert card_id == "program-prog-1"
    assert stored is not None
    assert stored.program_id == "prog-1"
    assert stored.fitness == 12.5
    assert stored.connected_ideas == [
        {
            "idea_id": "idea-1",
            "description": "Repair invalid candidates before scoring.",
        }
    ]


def test_normalize_program_card_is_minimal_shape():
    card = normalize_memory_card(
        {
            "id": "program-prog-1",
            "category": "program",
            "program_id": "prog-1",
            "task_description": "Solve task.",
            "task_description_summary": "Solve task.",
            "description": "Top evolved program.",
            "fitness": 12.5,
            "code": "def run_code():\n    return 1\n",
            "connected_ideas": [{"idea_id": "idea-1"}],
            "links": ["idea-1"],
            "strategy": "unused",
        }
    )

    assert card == {
        "id": "program-prog-1",
        "category": "program",
        "program_id": "prog-1",
        "task_description": "Solve task.",
        "task_description_summary": "Solve task.",
        "description": "Top evolved program.",
        "fitness": 12.5,
        "code": "def run_code():\n    return 1\n",
        "connected_ideas": [{"idea_id": "idea-1"}],
    }
