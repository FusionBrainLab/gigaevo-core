from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from omegaconf import OmegaConf

from gigaevo.memory.runtime_config import resolve_settings_path

# Load repo-root .env. override=False: .env sets defaults, not overrides.
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)


@dataclass
class _HuggingFaceSchema:
    etag_timeout: str = "60"
    download_timeout: str = "300"


@dataclass
class _ModelsSchema:
    openrouter_model_name: str = "openai/gpt-4.1-mini"
    amem_embedding_model_name: str = "all-MiniLM-L6-v2"
    gam_dense_retriever_model_name: str = "BAAI/bge-m3"
    openai_base_url: str = ""
    llm_base_url: str = ""


def _normalize_env(value: str | None) -> str | None:
    """Strip quotes and whitespace; return None for blank values."""
    if not value:
        return None
    stripped = value.strip().strip('"').strip("'")
    return stripped or None


def _merge_section(cfg: Any, dotted_key: str, schema_cls: type) -> dict[str, Any]:
    raw = OmegaConf.select(cfg, dotted_key) or {}
    merged = OmegaConf.merge(OmegaConf.structured(schema_cls), raw)
    return OmegaConf.to_container(merged, resolve=True)  # type: ignore[return-value]


def _configure_huggingface_env(hf: dict[str, Any]) -> None:
    if not os.getenv("HF_HUB_ETAG_TIMEOUT"):
        os.environ["HF_HUB_ETAG_TIMEOUT"] = str(hf["etag_timeout"])
    if not os.getenv("HF_HUB_DOWNLOAD_TIMEOUT"):
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(hf["download_timeout"])


def _load() -> tuple[dict[str, Any], dict[str, Any]]:
    settings_path = resolve_settings_path()
    cfg = (
        OmegaConf.load(settings_path)
        if settings_path.exists()
        else OmegaConf.create({})
    )
    hf = _merge_section(cfg, "downloads.huggingface", _HuggingFaceSchema)
    _configure_huggingface_env(hf)
    models: dict[str, Any] = _merge_section(cfg, "models", _ModelsSchema)
    reasoning_raw = OmegaConf.select(cfg, "reasoning")
    reasoning: dict[str, Any] = (
        OmegaConf.to_container(reasoning_raw, resolve=True)  # type: ignore[assignment]
        if reasoning_raw is not None
        else {}
    )
    return models, reasoning if isinstance(reasoning, dict) else {}


_models, _reasoning = _load()

OPENAI_API_KEY = _normalize_env(
    os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
)
OPENROUTER_API_KEY = _normalize_env(
    os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
)
OPENROUTER_MODEL_NAME: str = os.getenv("OPENROUTER_MODEL_NAME") or str(
    _models["openrouter_model_name"]
)
LLM_BASE_URL = _normalize_env(
    os.getenv("OPENAI_BASE_URL")
    or os.getenv("LLM_BASE_URL")
    or os.getenv("BASE_URL")
    or str(_models["openai_base_url"] or "")
    or str(_models["llm_base_url"] or "")
)
OPENROUTER_REASONING: dict[str, object] = _reasoning
AMEM_EMBEDDING_MODEL_NAME: str = str(_models["amem_embedding_model_name"])
GAM_DENSE_RETRIEVER_MODEL_NAME: str = str(_models["gam_dense_retriever_model_name"])
