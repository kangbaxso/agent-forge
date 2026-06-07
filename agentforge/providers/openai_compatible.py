"""OpenAI-compatible chat-completions provider.

Talks to any endpoint that implements POST /chat/completions with the OpenAI
schema: OpenAI, OpenRouter, Groq, Together, Mistral, local llama.cpp server,
Ollama (/v1), LM Studio, etc. No SDK dependency — just `requests`.

Features:
  - automatic retry with exponential backoff on 429 / 5xx / network errors
  - streaming via Server-Sent Events (chat_stream)
"""
from __future__ import annotations

import json
import os
import time
import typing as t

import requests


class ProviderError(RuntimeError):
    """Raised when the provider fails after exhausting retries."""


class OpenAICompatible:
    """Thin client for the /chat/completions endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
        backoff: float = 1.0,
    ):
        self.model = model
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, messages, tools, temperature, stream) -> dict:
        payload: dict[str, t.Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if stream:
            payload["stream"] = True
        return payload

    def _request(self, payload: dict, stream: bool) -> requests.Response:
        """POST with retry on 429 / 5xx / network errors (exponential backoff)."""
        url = f"{self.base_url}/chat/completions"
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    url,
                    headers=self._headers(),
                    data=json.dumps(payload),
                    timeout=self.timeout,
                    stream=stream,
                )
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_err = ProviderError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    raise last_err
                resp.raise_for_status()
                return resp
            except (requests.RequestException, ProviderError) as e:
                last_err = e
                if attempt >= self.max_retries:
                    break
                # honor Retry-After if present, else exponential backoff
                delay = self.backoff * (2 ** attempt)
                time.sleep(delay)
        raise ProviderError(f"request failed after {self.max_retries + 1} attempts: {last_err}")

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
    ) -> dict:
        """Send one chat-completions request, return the first message dict."""
        resp = self._request(self._payload(messages, tools, temperature, stream=False), stream=False)
        data = resp.json()
        return data["choices"][0]["message"]

    def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
    ) -> t.Iterator[str]:
        """Stream a plain-text completion token-by-token (SSE).

        Tool calls are not streamed — use `chat()` when tools are involved.
        Yields content deltas as they arrive.
        """
        resp = self._request(self._payload(messages, None, temperature, stream=True), stream=True)
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            chunk = line[len("data:"):].strip()
            if chunk == "[DONE]":
                break
            try:
                delta = json.loads(chunk)["choices"][0]["delta"]
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            piece = delta.get("content")
            if piece:
                yield piece
