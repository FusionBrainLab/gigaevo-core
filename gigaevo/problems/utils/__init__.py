"""
Problem-type-specific utilities.

This package contains shared utilities for different problem types.
Each problem type has its own subdirectory with working code that can be
imported by problems of that type.

Structure:
    utils/
    ├── base/              # Base problem type utilities (minimal)
    ├── prompt_evolution/  # Prompt evolution utilities (e.g., LLM clients)
    └── rl/                # Reinforcement learning utilities

Usage:
    from gigaevo.problems.utils.prompt_evolution import llm_client
"""
