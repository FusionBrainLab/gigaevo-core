from __future__ import annotations

from dataclasses import dataclass
import gc

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from problems.prompts.sudoku.local_runtime.models import PathContext


@dataclass(slots=True)
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0


class LocalVLLMSolver:
    _STOP_TOKENS = ["</node>", "</done>", "</backtrack>"]

    def __init__(
        self,
        *,
        model_name: str,
        generation_config: GenerationConfig,
        system_prompt: str,
        user_prompt: str,
        gpu_memory_utilization: float = 0.7,
        max_model_len: int = 2048,
        bf16: bool = True,
    ) -> None:
        self.model_name = model_name
        self.generation_config = generation_config
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.bf16 = bf16
        self._vllm: LLM | None = None
        self._tokenizer = None

    @staticmethod
    def _cuda_mem_gib() -> tuple[float, float]:
        if not torch.cuda.is_available():
            return 0.0, 0.0
        free_b, total_b = torch.cuda.mem_get_info()
        gib = 1024**3
        return free_b / gib, total_b / gib

    def _effective_vllm_utilization(self, desired: float) -> float:
        if not torch.cuda.is_available():
            return float(desired)
        free_gib, total_gib = self._cuda_mem_gib()
        if total_gib <= 0:
            return float(desired)
        cap = max(0.01, (free_gib - 1.0) / total_gib)
        return min(float(desired), cap)

    def _ensure_vllm(self) -> LLM:
        if self._vllm is None:
            dtype = "bfloat16" if self.bf16 else "float16"
            util = self._effective_vllm_utilization(self.gpu_memory_utilization)
            self._vllm = LLM(
                model=self.model_name,
                gpu_memory_utilization=util,
                max_model_len=self.max_model_len,
                trust_remote_code=True,
                tensor_parallel_size=1,
                disable_log_stats=True,
                dtype=dtype,
                seed=0,
                enforce_eager=util < float(self.gpu_memory_utilization),
            )
        return self._vllm

    def _ensure_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
                padding_side="left",
                trust_remote_code=True,
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    def format_prompt(self, context: PathContext) -> str:
        tokenizer = self._ensure_tokenizer()
        ctx_text = (
            "\n".join(str(node.action) for node in context.nodes) if context else ""
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt.format(ctx=ctx_text)},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def inference(self, *, context: PathContext, max_actions: int = 1) -> list[str]:
        vllm = self._ensure_vllm()
        prompt = self.format_prompt(context)
        sampling_params = SamplingParams(
            n=max_actions,
            temperature=self.generation_config.temperature,
            top_p=self.generation_config.top_p,
            max_tokens=self.generation_config.max_new_tokens,
            repetition_penalty=self.generation_config.repetition_penalty,
            stop=self._STOP_TOKENS,
            include_stop_str_in_output=True,
        )
        outputs = vllm.generate([prompt], sampling_params)
        return [output.text for output in outputs[0].outputs]

    def close(self) -> None:
        if self._vllm is not None:
            llm = self._vllm
            self._vllm = None
            del llm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
