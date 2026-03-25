# -*- coding: utf-8 -*-
"""
Configuration Module

This module contains all configuration classes for the GAM framework.
Configurations use dataclasses for type safety and easy serialization.

Available Configurations:
- GeneratorConfigs: OpenAI, VLLM generator settings
- RetrieverConfigs: Dense, BM25, Index retriever settings
"""

from __future__ import annotations

from .generator import OpenAIGeneratorConfig, VLLMGeneratorConfig
from .retriever import DenseRetrieverConfig, IndexRetrieverConfig, BM25RetrieverConfig

__all__ = [
    # Generator configurations
    "OpenAIGeneratorConfig",
    "VLLMGeneratorConfig",
    
    # Retriever configurations  
    "DenseRetrieverConfig",
    "IndexRetrieverConfig",
    "BM25RetrieverConfig",
]
