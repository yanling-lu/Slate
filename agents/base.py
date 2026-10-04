from __future__ import annotations
import json, hashlib
from openai import OpenAI
from config import settings


class LLMAgent:
    """Thin wrapper. API key from settings; prompts injected at construction."""

    def __init__(self, model: str, system_prompt: str, temperature: float = 0.0):
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self._client = OpenAI(api_key=settings.openai_api_key())

    @property
    def prompt_hash(self) -> str:
        return hashlib.sha256(self.system_prompt.encode()).hexdigest()[:12]

    def call(self, user_payload: dict, json_mode: bool = True) -> dict:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"} if json_mode else None,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json.dumps(user_payload, default=str)},
            ],
        )
        return json.loads(resp.choices[0].message.content)