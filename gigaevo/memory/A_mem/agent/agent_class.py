import json
import threading
from typing import Any


class LLMService:
    def __init__(
        self,
        service: str,
        model_name: str,
        api_key: str | None = None,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        streaming: bool = False,
        thinking: bool = False,
        openrouter_referer: str | None = None,
        openrouter_title: str | None = None,
        gigachat_scope: str = "GIGACHAT_API_CORP",
        gigachat_verify_ssl: bool = False,
        system_prompt: str = "",
    ):
        """
        service: 'openai', 'ollama', 'hf', 'openrouter', 'openrouter_openai' or 'gigachat'
        """
        self.service = service.lower()
        self.model_name = model_name
        self.streaming = streaming
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt

        # Defaults so attributes exist regardless of branch
        self.client = None
        self.tokenizer = None
        self.model = None
        self.openrouter_key = None
        self.openrouter_referer = openrouter_referer
        self.openrouter_title = openrouter_title
        self.giga_client = None

        if self.service == "openai":
            if not api_key:
                raise ValueError("You must pass an API key for OpenAI")
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError(
                    "Install the 'openai' package to use service='openai'"
                ) from e
            self.client = OpenAI(api_key=api_key)

        elif self.service in ("openrouter_openai",):
            if not api_key:
                raise ValueError("You must pass an API key for OpenRouter")
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError(
                    "Install the 'openai' package to use service='openrouter_openai'"
                ) from e
            # OpenAI client pointed at OpenRouter
            self.client = OpenAI(
                api_key=api_key, base_url="https://openrouter.ai/api/v1"
            )

        elif self.service == "ollama":
            # no heavy imports here; done in generate()
            pass

        elif self.service.startswith("hf"):
            # Lazy import Transformers + Torch only for HF
            try:
                import torch  # noqa: F401 (device_map uses it)
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as e:
                raise ImportError(
                    "Install 'transformers' and 'torch' to use service='hf*'"
                ) from e
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype="auto", device_map="auto"
            )

        elif self.service == "openrouter":
            if not api_key:
                raise ValueError("You must pass an API key for OpenRouter")
            self.openrouter_key = api_key

        elif self.service == "gigachat":
            if not api_key:
                raise ValueError("You must pass GigaChat credentials")
            try:
                from langchain_gigachat.chat_models import GigaChat
            except ImportError as e:
                raise ImportError(
                    "Install 'langchain-gigachat' to use service='gigachat'"
                ) from e
            # Create client; messages will be constructed in generate()
            # Temperature support varies by version; try to pass it, else fall back.
            try:
                self.giga_client = GigaChat(
                    model=self.model_name,
                    credentials=api_key,
                    scope=gigachat_scope,
                    verify_ssl_certs=gigachat_verify_ssl,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,  # may not be supported in older versions
                    use_api_for_tokens=True,
                    timeout=400,
                )
            except TypeError:
                self.giga_client = GigaChat(
                    model=self.model_name,
                    credentials=api_key,
                    scope=gigachat_scope,
                    verify_ssl_certs=gigachat_verify_ssl,
                    max_tokens=self.max_tokens,
                    use_api_for_tokens=True,
                    timeout=400,
                )

        else:
            raise ValueError(f"Unknown service {service!r}")

    def generate(self, data: str) -> tuple[str, Any, int | None, float | None]:
        """
        Returns:
            final_text   – model’s end-user answer
            raw_response – full response incl. reasoning / metadata
            token_count  – number of tokens used (provider-specific)
            cost         – provider-specific cost (if available)
        """
        # ---------- OpenAI ----------
        if self.service == "openai":
            args = {"model": self.model_name, "input": data}
            if self.temperature is not None:
                args["temperature"] = self.temperature
            if self.model_name.startswith("o"):
                # NOTE: keep both keys if your SDK supports them together.
                # If not, merge as needed.
                args["reasoning"] = {"effort": self.reasoning_effort, "summary": "auto"}
            resp = self.client.responses.create(**args)
            resp_dict = resp.model_dump()
            final_text = resp.output_text
            usage = getattr(resp, "usage", None) or {}
            token_count = getattr(usage, "total_tokens", None) or usage.get(
                "total_tokens"
            )
            return final_text, resp_dict, token_count, None

        # ---------- OpenRouter via OpenAI SDK ----------
        elif self.service in ("openrouter_openai",):
            extra_headers = {}
            if self.openrouter_referer:
                extra_headers["HTTP-Referer"] = self.openrouter_referer
            if self.openrouter_title:
                extra_headers["X-Title"] = self.openrouter_title

            extra_body = {"usage": {"include": True}}
            if self.reasoning_effort:
                extra_body["reasoning"] = {"effort": self.reasoning_effort}

            kwargs = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": data}],
                "extra_body": extra_body,
            }
            if extra_headers:
                kwargs["extra_headers"] = extra_headers
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature

            resp = self.client.chat.completions.create(**kwargs)
            resp_dict = resp.model_dump()
            final_text = resp.choices[0].message.content if resp.choices else ""
            usage = resp_dict.get("usage", {}) or {}
            token_count = usage.get("total_tokens")
            cost = usage.get("cost")
            return final_text, resp_dict, token_count, cost

        # ---------- Hugging Face (Transformers) ----------
        elif self.service.startswith("hf"):
            try:
                from transformers import TextIteratorStreamer
            except ImportError as e:
                raise ImportError("Install 'transformers' to use service='hf*'") from e

            tok, model = self.tokenizer, self.model
            messages = [{"role": "user", "content": data}]
            text = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.thinking,
            )
            inputs = tok([text], return_tensors="pt").to(model.device)

            gen_common = {}
            if self.temperature is not None:
                gen_common["temperature"] = self.temperature

            if self.streaming:
                streamer = TextIteratorStreamer(
                    tok, skip_prompt=True, skip_special_tokens=True
                )
                gen_kwargs = dict(
                    **inputs, streamer=streamer, max_new_tokens=32768, **gen_common
                )
                thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
                thread.start()

                output_text = ""
                token_count = 0
                for token in streamer:
                    print(token, end="", flush=True)
                    output_text += token
                    token_count += 1
                thread.join()
                return output_text, output_text, token_count, None
            else:
                generated = model.generate(**inputs, max_new_tokens=32768, **gen_common)
                output_ids = generated[0][len(inputs.input_ids[0]) :].tolist()
                token_count = len(output_ids)
                raw_text = tok.decode(output_ids, skip_special_tokens=True)
                # Try to split off any reasoning tokens if your model uses them
                try:
                    split_idx = len(output_ids) - output_ids[::-1].index(151668)
                except ValueError:
                    split_idx = 0
                final_text = tok.decode(
                    output_ids[split_idx:], skip_special_tokens=True
                ).strip()
                return final_text, raw_text, token_count, None

        # ---------- OpenRouter (HTTP) ----------
        elif self.service == "openrouter":
            try:
                import requests
            except ImportError as e:
                raise ImportError(
                    "Install 'requests' to use service='openrouter'"
                ) from e

            headers = {
                "Authorization": f"Bearer {self.openrouter_key}",
                "Content-Type": "application/json",
            }
            if self.openrouter_referer:
                headers["HTTP-Referer"] = self.openrouter_referer
            if self.openrouter_title:
                headers["X-Title"] = self.openrouter_title

            if (
                self.model_name.startswith(("google", "anthropic"))
                and self.max_tokens is not None
            ):
                payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": data}],
                    "usage": {"include": True},
                    "reasoning": {"max_tokens": self.max_tokens},
                }
                if self.temperature is not None:
                    payload["temperature"] = self.temperature
            else:
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": "Set reasoning effort to high"},
                        {"role": "user", "content": data},
                    ],
                    "usage": {"include": True},
                    "reasoning": {
                        "effort": self.reasoning_effort,
                        "exclude": False,
                        "enabled": True,
                    },
                }
                if self.temperature is not None:
                    payload["temperature"] = self.temperature
                if self.max_tokens is not None:
                    payload["max_tokens"] = self.max_tokens

            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload),
            )
            result = response.json()
            final_text = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {}) or {}
            return final_text, result, usage.get("completion_tokens"), usage.get("cost")

        # ---------- GigaChat ----------
        elif self.service == "gigachat":
            try:
                from langchain_core.messages import HumanMessage, SystemMessage
            except ImportError as e:
                raise ImportError(
                    "Install 'langchain-core' (comes with LangChain) to use service='gigachat'"
                ) from e

            msgs = [SystemMessage(self.system_prompt or ""), HumanMessage(content=data)]

            # Some versions may expose temperature at call-time via .bind; try it safely.
            try:
                if self.temperature is not None:
                    res = self.giga_client.bind(temperature=self.temperature)(msgs)
                else:
                    res = self.giga_client(msgs)
            except Exception:
                # Fallback to a plain call if bind is unsupported
                res = self.giga_client(msgs)

            res_dict = {
                "content": getattr(res, "content", None),
                "additional_kwargs": getattr(res, "additional_kwargs", None),
                "response_metadata": getattr(res, "response_metadata", None),
                "usage_metadata": getattr(res, "usage_metadata", None),
                "id": getattr(res, "id", None),
            }

            usage = res_dict.get("usage_metadata") or {}
            if not usage:
                rm = res_dict.get("response_metadata") or {}
                token_usage = rm.get("token_usage") or {}
                usage = {
                    "input_tokens": token_usage.get("prompt_tokens"),
                    "output_tokens": token_usage.get("completion_tokens"),
                    "total_tokens": token_usage.get("total_tokens"),
                    "input_token_details": {
                        "cache_read": token_usage.get("precached_prompt_tokens")
                    },
                    "model_name": rm.get("model_name"),
                    "finish_reason": rm.get("finish_reason"),
                    "x_headers": rm.get("x_headers"),
                }

            output_tokens = (
                usage.get("output_tokens")
                or usage.get("completion_tokens")
                or (usage.get("token_usage", {}) or {}).get("completion_tokens")
            )
            final_text = res_dict["content"] or ""
            return final_text, res_dict, output_tokens, None

        else:
            raise RuntimeError("Unsupported service")
