# -*- coding: utf-8 -*-
"""
Prompts Module

This module contains all prompt templates and prompt-related tools for the GAM (General-Agentic-Memory) framework.
Prompts are used to guide the behavior and responses of different agents (e.g., MemoryAgent, ResearchAgent) for various tasks, such as memory management and research reasoning.

Available Prompts:
- memory_prompts: Templates for memory management and updating.
- research_prompts: Templates for research, reasoning, and scientific inquiry.
"""
from .memory_prompts import MemoryAgent_PROMPT
from .research_prompts import (
    Planning_PROMPT,
    Integrate_PROMPT,
    InfoCheck_PROMPT,
    GenerateRequests_PROMPT,
    ExperimentalDecision_PROMPT,
)

__all__ = [
    "MemoryAgent_PROMPT",
    "Planning_PROMPT",
    "Integrate_PROMPT",
    "InfoCheck_PROMPT",
    "GenerateRequests_PROMPT",
    "ExperimentalDecision_PROMPT",
]
