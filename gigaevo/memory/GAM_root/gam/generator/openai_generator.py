import time
import json
import os

from openai import OpenAI
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from GAM_root.gam.generator.base import AbsGenerator
from GAM_root.gam.config import OpenAIGeneratorConfig
from typing import Any, Dict, List, Optional


class OpenAIGenerator(AbsGenerator):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get("model_name", "gpt-4o-mini")
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url")
        self.n = config.get("n", 1)
        self.temperature = config.get("temperature", 0.0)
        self.top_p = config.get("top_p", 1.0)
        self.max_tokens = config.get("max_tokens", 300)
        self.thread_count = config.get("thread_count")
        self.system_prompt = config.get("system_prompt")
        self.timeout = config.get("timeout", 60.0)
        self.use_schema = config.get("use_schema", False)

        if self.api_key is not None:
            os.environ["OPENAI_API_KEY"] = self.api_key
        if self.base_url is not None:
            os.environ["OPENAI_BASE_URL"] = self.base_url


    def generate_single(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        schema: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        极简 Chat 调用（OpenAI SDK）
        - 二选一：prompt 文本 或 messages 列表
        - 若传 schema：使用 response_format.json_schema 进行结构化输出
        返回：
          {"text": str, "json": dict|None, "response": dict}
        """
        if (prompt is None) and (not messages):
            raise ValueError("Either prompt or messages is required.")
        if (prompt is not None) and messages:
            raise ValueError("Pass either prompt or messages, not both.")

        # 构造 messages
        if messages is None:
            messages = [{"role": "user", "content": prompt}]  # type: ignore[arg-type]
        if self.system_prompt and not any(m.get("role") == "system" for m in messages):
            messages = [{"role": "system", "content": self.system_prompt}] + messages

        # 构造 response_format
        response_format = None
        if schema is not None and self.use_schema:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "auto_schema",
                    "schema": schema,
                    "strict": True
                }
            }

        client = OpenAI(api_key=self.api_key, base_url=self.base_url.rstrip("/") if self.base_url else None)
        cclient = client.with_options(timeout=self.timeout) if hasattr(client, "with_options") else client

        params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if response_format is not None:
            params["response_format"] = response_format
        if extra_params:
            params.update(extra_params)

        times = 0
        while True:
            try:
                resp = cclient.chat.completions.create(**params)
                break
            except Exception as e:
                print(str(e), 'times:', times)
                times += 1
                if times > 3:  # 最多重试3次
                    raise e
                time.sleep(5)

        try:
            text = resp.choices[0].message.content or ""
        except Exception:
            text = ""
        
        text = text.split('</think>')[-1]

        out: Dict[str, Any] = {"text": text, "json": None, "response": resp.model_dump()}

        if schema is not None:
            try:
                out["json"] = json.loads(text[text.find('{'): text.rfind('}') + 1])
            except Exception:
                out["json"] = None
        return out

    def generate_batch(
        self,
        prompts: Optional[List[str]] = None,
        messages_list: Optional[List[List[Dict[str, str]]]] = None,
        schema: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        批量生成响应
        返回格式: [{"text": str, "json": dict|None, "response": dict}, ...]
        """
        if (prompts is None) and (not messages_list):
            raise ValueError("Either prompts or messages_list is required.")
        if (prompts is not None) and messages_list:
            raise ValueError("Pass either prompts or messages_list, not both.")

        if prompts is not None:
            if isinstance(prompts, str):
                prompts = [prompts]
            # 转换为 messages_list 格式
            messages_list = [[{"role": "user", "content": prompt}] for prompt in prompts]

        if self.thread_count is None:
            thread_count = cpu_count()
        else:
            thread_count = self.thread_count

        # 创建部分应用的函数
        def generate_single_wrapper(messages):
            return self.generate_single(
                messages=messages,
                schema=schema,
                extra_params=extra_params
            )

        results = []
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            # Map the fixed function to the messages_list
            results = list(tqdm(executor.map(generate_single_wrapper, messages_list), total=len(messages_list)))

        return results
    
    @classmethod
    def from_config(cls, config: OpenAIGeneratorConfig) -> "OpenAIGenerator":
        """从配置类创建 OpenAIGenerator 实例"""
        return cls(config.__dict__)
