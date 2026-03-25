import os
from pathlib import Path

from dotenv import load_dotenv

try:
    from .runtime_config import deep_get, load_settings, to_str
except ImportError:  # pragma: no cover - script-style import fallback
    from runtime_config import deep_get, load_settings, to_str

# Always load env from repository root, regardless of process cwd.
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)


def _normalize_env(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().strip('"').strip("'")


_SETTINGS = load_settings()

OPENAI_API_KEY = _normalize_env(
    os.getenv("OPENAI_API_KEY")
    or os.getenv("OPENROUTER_API_KEY")
    or to_str(deep_get(_SETTINGS, "models.openai_api_key"), default=None)
    or to_str(deep_get(_SETTINGS, "models.openrouter_api_key"), default=None)
)
OPENROUTER_API_KEY = _normalize_env(
    os.getenv("OPENROUTER_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or to_str(deep_get(_SETTINGS, "models.openrouter_api_key"), default=None)
    or to_str(deep_get(_SETTINGS, "models.openai_api_key"), default=None)
)

OPENROUTER_MODEL_NAME = os.getenv(
    "OPENROUTER_MODEL_NAME",
    to_str(deep_get(_SETTINGS, "models.openrouter_model_name"), default="openai/gpt-4.1-mini"),
)
LLM_BASE_URL = _normalize_env(
    os.getenv("OPENAI_BASE_URL")
    or os.getenv("LLM_BASE_URL")
    or os.getenv("BASE_URL")
    or to_str(deep_get(_SETTINGS, "models.openai_base_url"), default=None)
    or to_str(deep_get(_SETTINGS, "models.llm_base_url"), default=None)
)

_OPENROUTER_REASONING = deep_get(_SETTINGS, "reasoning", default={})
if not isinstance(_OPENROUTER_REASONING, dict):
    _OPENROUTER_REASONING = {}
OPENROUTER_REASONING: dict[str, object] = _OPENROUTER_REASONING

AMEM_EMBEDDING_MODEL_NAME = to_str(
    deep_get(_SETTINGS, "models.amem_embedding_model_name"),
    default="all-MiniLM-L6-v2",
)
GAM_DENSE_RETRIEVER_MODEL_NAME = to_str(
    deep_get(_SETTINGS, "models.gam_dense_retriever_model_name"),
    default="BAAI/bge-m3",
)
