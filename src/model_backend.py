"""Compliant local model backend using Ollama's HTTP API and stdlib only."""

import json
import os
from typing import Any
from urllib import error, request

from src.schema import FunctionSchema


class OllamaModel:
    """Call a locally running Ollama model without ML Python dependencies."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model or os.environ.get("CALL_ME_MAYBE_MODEL", "qwen3:0.6b")
        self.base_url = (
            base_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")
        ).rstrip("/")

    def generate_call(
        self, prompt: str, functions: list[FunctionSchema]
    ) -> dict[str, Any]:
        """Generate one JSON function call using Ollama's JSON schema mode."""
        function_text = json.dumps(
            [function.model_dump() for function in functions], indent=2
        )
        names = [function.name for function in functions]
        output_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": names},
                "parameters": {"type": "object"},
            },
            "required": ["name", "parameters"],
            "additionalProperties": False,
        }
        instruction = (
            "Choose the function that matches the request and extract every "
            "argument using its declared type. Return only JSON matching the "
            f"provided schema. Functions:\n{function_text}\n\nRequest: {prompt}"
        )
        payload = {
            "model": self.model,
            "stream": False,
            "format": output_schema,
            "options": {"temperature": 0},
            "messages": [{"role": "user", "content": instruction}],
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=300) as response:
                envelope = json.load(response)
        except error.URLError as exc:
            raise RuntimeError(
                "Cannot reach Ollama. Start Ollama and pull the model with "
                f"'ollama pull {self.model}'."
            ) from exc

        content = envelope.get("message", {}).get("content")
        if not isinstance(content, str):
            raise ValueError("Ollama returned no JSON message content")
        result = json.loads(content)
        if not isinstance(result, dict):
            raise ValueError("Model response must be a JSON object")
        return result