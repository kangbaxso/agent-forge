"""OpenAI-compatible chat-completions provider.

Talks to any endpoint that implements POST /chat/completions with the OpenAI
schema: OpenAI, OpenRouter, Groq, Together, Mistral, local llama.cpp server,
Ollama (/v1), LM Studio, etc. No SDK dependency — just `requests`.
"""
from __future__ import annotations

import json
import os
import typing as t

import requests


class OpenAICompatible:
    """Thin client for the /chat/completions endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int = 60,
    ):
        self.model = model
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
    ) -> dict:
        """Send one chat-completions request, return the first message dict."""
        payload: dict[str, t.Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]
