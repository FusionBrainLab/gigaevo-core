from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Union, Optional, Dict

@dataclass
class OpenAIGeneratorConfig:
    """OpenAI生成器配置"""
    model_name: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None
    n: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 300
    thread_count: int | None = None
    system_prompt: str | None = None
    timeout: float = 60.0
    use_schema: bool = False


@dataclass
class VLLMGeneratorConfig:
    """
    vLLM 生成器（本地 OpenAI 兼容端点 /v1/chat/completions）
    注意：这是“客户端调用”所需的字段；与“启动 vLLM 服务器”的参数不同。
    """
    model_name: str = "Qwen2.5-7B-Instruct"   
    api_key: Optional[str] = "empty"          
    base_url: str = "http://localhost:8000/v1"
    n: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 300
    thread_count: Optional[int] = None
    system_prompt: Optional[str] = None
    timeout: float = 60.0
    use_schema: bool = False