from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import os
from pathlib import Path

from shared_memory.memory import AmemGamMemory
from shared_memory.memory_config import ApiConfig, GamConfig, MemoryConfig


def main() -> None:
    checkpoint_dir = (
        Path(__file__).resolve().parent / "memory_usage_store" / "api_usage"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    memory_api_url = os.getenv("MEMORY_API_URL", "http://localhost:8000")
    namespace = os.getenv("MEMORY_NAMESPACE", "demo")
    use_api = os.getenv("MEMORY_USE_API", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    gam_pipeline_mode = os.getenv("MEMORY_GAM_PIPELINE_MODE", "default")

    api_config = (
        ApiConfig(base_url=memory_api_url, namespace=namespace) if use_api else None
    )
    mem_config = MemoryConfig(
        checkpoint_path=checkpoint_dir,
        api=api_config,
        gam=GamConfig(pipeline_mode=gam_pipeline_mode),
    )
    memory = AmemGamMemory(config=mem_config)

    print("\n==============================")
    print("API Memory Demo: Save + Search")
    print("==============================\n")

    memories = [
        "Convert broad goals into explicit deliverables with acceptance criteria.",
        "When requirements are vague, propose a default concrete plan with optional refinements.",
        "For status updates, use changed/next/blocked to keep communication actionable.",
    ]

    memory_ids: list[str] = []
    for text in memories:
        memory_ids.append(memory.save(text, category="project-management"))

    print("Saved cards:")
    for memory_id in memory_ids:
        card = memory.get_card(memory_id) or {}
        print(f"- {memory_id}: {card.get('description', '')}")

    question = "How should I turn a vague onboarding goal into measurable implementation steps?"
    print(f"\nQuestion: {question}\n")
    print(memory.search(question, memory_state="Onboarding initiative"))


if __name__ == "__main__":
    main()
