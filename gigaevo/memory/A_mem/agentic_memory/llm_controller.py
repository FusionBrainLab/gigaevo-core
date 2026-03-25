from typing import Optional, Literal, Any
import os
import json
from abc import ABC, abstractmethod
from litellm import completion
from A_mem.agent.agent_class import LLMService

class BaseLLMController(ABC):
    @abstractmethod
    def get_completion(
        self,
        prompt: str,
        response_format: Optional[dict] = None,
        temperature: float = 0.7,
    ) -> str:
        """Get completion from LLM"""
        pass

class OpenAIController(BaseLLMController):
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        try:
            from openai import OpenAI
            self.model = model
            if api_key is None:
                api_key = os.getenv('OPENAI_API_KEY')
            if api_key is None:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("OpenAI package not found. Install it with: pip install openai")
    
    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt}
            ],
            response_format=response_format,
            temperature=temperature,
            max_tokens=1000
        )
        return response.choices[0].message.content

class OllamaController(BaseLLMController):
    def __init__(self, model: str = "llama2"):
        from ollama import chat
        self.model = model
    
    def _generate_empty_value(self, schema_type: str, schema_items: dict = None) -> Any:
        if schema_type == "array":
            return []
        elif schema_type == "string":
            return ""
        elif schema_type == "object":
            return {}
        elif schema_type == "number":
            return 0
        elif schema_type == "boolean":
            return False
        return None

    def _generate_empty_response(self, response_format: dict) -> dict:
        if "json_schema" not in response_format:
            return {}
            
        schema = response_format["json_schema"]["schema"]
        result = {}
        
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                result[prop_name] = self._generate_empty_value(prop_schema["type"], 
                                                            prop_schema.get("items"))
        
        return result

    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        # Allow exceptions (like ConnectionError) to bubble up for better debugging
        response = completion(
            model="ollama_chat/{}".format(self.model),
            messages=[
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt}
            ],
            response_format=response_format,
        )
        return response.choices[0].message.content

class LLMServiceController(BaseLLMController):
    """Wraps an existing LLMService instance so it can be used by the memory system."""

    def __init__(self, service: LLMService):
        self.service = service

    def _format_prompt(self, prompt: str, response_format: Optional[dict]) -> str:
        """Inject schema instructions for services that only accept a plain string prompt."""
        if not response_format:
            return prompt

        schema_instruction = "You must respond with a JSON object."
        if response_format.get("type") == "json_schema":
            schema = response_format.get("json_schema", {}).get("schema")
            schema_json = json.dumps(schema, indent=2)
            schema_name = response_format.get("json_schema", {}).get("name")
            strict = response_format.get("json_schema", {}).get("strict")

            extra = []
            if schema_name:
                extra.append(f"Schema name: {schema_name}")
            if strict:
                extra.append("Only include fields defined in the schema.")
            extra.append("Here is the JSON schema your response must follow:")
            extra.append(schema_json)
            extra.append("Return only JSON without commentary.")
            schema_instruction = "\n".join(extra)
        else:
            schema_instruction = (
                "You must respond with JSON that matches this specification:\n"
                f"{json.dumps(response_format, indent=2)}"
            )

        return f"{schema_instruction}\n\nUser prompt:\n{prompt}"

    def get_completion(
        self,
        prompt: str,
        response_format: Optional[dict] = None,
        temperature: float = 0.7,
    ) -> str:
        formatted_prompt = self._format_prompt(prompt, response_format)

        if hasattr(self.service, "temperature"):
            try:
                self.service.temperature = temperature
            except Exception:
                pass

        final_text, _, _, _ = self.service.generate(formatted_prompt)
        return final_text

class LLMController:
    """LLM-based controller for memory metadata generation"""

    def __init__(
        self,
        backend: Literal["openai", "ollama", "custom"] = "openai",
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        llm_service: Optional[LLMService] = None,
    ):
        if llm_service is not None:
            self.llm = LLMServiceController(llm_service)
        elif backend == "openai":
            self.llm = OpenAIController(model, api_key)
        elif backend == "ollama":
            self.llm = OllamaController(model)
        elif backend == "custom":
            raise ValueError("Provide an llm_service instance when using backend='custom'")
        else:
            raise ValueError("Backend must be one of: 'openai', 'ollama', 'custom'")

    def get_completion(
        self,
        prompt: str,
        response_format: dict = None,
        temperature: float = 0.7,
    ) -> str:
        return self.llm.get_completion(prompt, response_format, temperature)
