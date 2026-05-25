"""
AMemGenerator

Adapter generator that wraps A-mem's LLMService for GAM.
"""

from __future__ import annotations

import json
from typing import Any

from gigaevo.memory._vendor.GAM_root.gam.generator.base import AbsGenerator


class AMemGenerator(AbsGenerator):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.llm_service = config.get("llm_service")
        if self.llm_service is None:
            raise ValueError(
                "AMemGenerator requires an LLMService instance in config['llm_service']"
            )

    @staticmethod
    def _coerce_to_parameters_schema(schema: dict[str, Any]) -> dict[str, Any]:
        """Unwrap legacy ``{"type":"json_schema","json_schema":{...}}`` payloads.

        Tools-API ``parameters`` wants the bare JSON Schema dict, not the
        OpenAI response_format wrapper.
        """
        if schema.get("type") == "json_schema":
            body = schema.get("json_schema", {}).get("schema")
            if isinstance(body, dict):
                return body
        return schema

    def _messages_to_prompt(self, messages: list[dict[str, str]]) -> str:
        parts: list[str] = []
        for msg in messages:
            role = (msg.get("role") or "user").upper()
            content = msg.get("content") or ""
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def generate_single(
        self,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        schema: dict[str, Any] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if (prompt is None) and (not messages):
            raise ValueError("Either prompt or messages is required.")
        if (prompt is not None) and messages:
            raise ValueError("Pass either prompt or messages, not both.")
        if extra_params:
            raise ValueError("AMemGenerator does not support extra_params.")

        if messages is not None:
            prompt = self._messages_to_prompt(messages)
        assert prompt is not None

        wire_schema = (
            self._coerce_to_parameters_schema(schema) if schema is not None else None
        )
        text, response, _, _ = self.llm_service.generate(prompt, schema=wire_schema)

        out: dict[str, Any] = {"text": text or "", "json": None, "response": response}
        if schema is not None and out["text"]:
            out["json"] = json.loads(out["text"])
        return out

    def generate_batch(
        self,
        prompts: list[str] | None = None,
        messages_list: list[list[dict[str, str]]] | None = None,
        schema: dict[str, Any] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if (prompts is None) and (not messages_list):
            raise ValueError("Either prompts or messages_list is required.")
        if (prompts is not None) and messages_list:
            raise ValueError("Pass either prompts or messages_list, not both.")
        if extra_params:
            raise ValueError("AMemGenerator does not support extra_params.")

        if prompts is not None:
            if isinstance(prompts, str):
                prompts = [prompts]
            messages_list = [[{"role": "user", "content": p}] for p in prompts]

        results = []
        assert messages_list is not None
        for msgs in messages_list:
            results.append(self.generate_single(messages=msgs, schema=schema))
        return results
