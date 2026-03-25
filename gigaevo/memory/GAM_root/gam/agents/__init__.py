# -*- coding: utf-8 -*-
"""
Agent Module

This module contains all agent classes for the GAM (General-Agentic-Memory) framework.
Agents are responsible for information processing, reasoning, memory storage, and retrieval.

Available Agents:
- ResearchAgent: Handles research and reasoning tasks.
- MemoryAgent: Handles memory management, storage, and retrieval.
"""

from __future__ import annotations

from .memory_agent import MemoryAgent
from .research_agent import ResearchAgent

__all__ = [
    "ResearchAgent",
    "MemoryAgent",
]
