"""The Agent: a minimal reason-act loop over an OpenAI-compatible model.

The loop:
  1. Send conversation + tool schemas to the model.
  2. If the model requests tool calls, execute them locally and feed results back.
  3. Repeat until the model returns a plain text answer or max_steps is hit.

Sync (`run`, `stream`) and async (`arun`) entry points are provided.
"""
from __future__ import annotations

import asyncio
import typing as t

from .providers.openai_compatible import OpenAICompatible
from .tools import Tool


class Agent:
    """A tool-using agent backed by any OpenAI-compatible chat model."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        tools: list[Tool] | None = None,
        system: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_steps: int = 8,
        temperature: float = 0.7,
        verbose: bool = False,
        max_retries: int = 3,
    ):
        self.provider = OpenAICompatible(
            model=model, base_url=base_url, api_key=api_key, max_retries=max_retries
        )
        self.tools: dict[str, Tool] = {tl.name: tl for tl in (tools or [])}
        self.system = system
        self.max_steps = max_steps
        self.temperature = temperature
        self.verbose = verbose

    def _log(self, *a):
        if self.verbose:
            print("[agentforge]", *a)

    def _seed_history(self, prompt: str, messages: list[dict] | None) -> list[dict]:
        history: list[dict] = list(messages or [])
        if self.system and not any(m.get("role") == "system" for m in history):
            history.insert(0, {"role": "system", "content": self.system})
        history.append({"role": "user", "content": prompt})
        return history

    def _schemas(self):
        return [tl.to_schema() for tl in self.tools.values()] or None

    def _run_tool_calls(self, tool_calls: list[dict]) -> list[dict]:
        """Execute every requested tool call, return the tool-result messages."""
        results = []
        for tc in tool_calls:
            fn = tc["function"]["name"]
            raw_args = tc["function"].get("arguments", "{}")
            self._log(f"calling {fn}({raw_args})")
            tool = self.tools.get(fn)
            if tool is None:
                output = f"Error: unknown tool '{fn}'"
            else:
                try:
                    output = tool.call(raw_args)
                except Exception as e:  # surface tool errors back to the model
                    output = f"Error executing {fn}: {e}"
            results.append({
                "role": "tool",
                "tool_call_id": tc.get("id", fn),
                "content": output,
            })
        return results

    # --- sync ---------------------------------------------------------------

    def run(self, prompt: str, messages: list[dict] | None = None) -> str:
        """Run the agent on a prompt and return the final text answer."""
        history = self._seed_history(prompt, messages)
        schemas = self._schemas()
        for _ in range(self.max_steps):
            msg = self.provider.chat(history, tools=schemas, temperature=self.temperature)
            history.append(msg)
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                return msg.get("content") or ""
            history.extend(self._run_tool_calls(tool_calls))
        return "[agentforge] max_steps reached without a final answer"

    def stream(self, prompt: str, messages: list[dict] | None = None) -> t.Iterator[str]:
        """Stream the final answer token-by-token.

        Any tool-calling rounds run first (non-streamed); once the model is ready
        to answer in plain text, that answer is streamed. Yields content deltas.
        """
        history = self._seed_history(prompt, messages)
        schemas = self._schemas()
        for _ in range(self.max_steps):
            msg = self.provider.chat(history, tools=schemas, temperature=self.temperature)
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                # final turn: re-issue the same history as a stream so the
                # answer arrives token-by-token instead of all at once
                yield from self.provider.chat_stream(history, temperature=self.temperature)
                return
            history.append(msg)
            history.extend(self._run_tool_calls(tool_calls))
        yield "[agentforge] max_steps reached without a final answer"

    # --- async --------------------------------------------------------------

    async def arun(self, prompt: str, messages: list[dict] | None = None) -> str:
        """Async wrapper around `run` (executes the blocking loop in a thread)."""
        return await asyncio.to_thread(self.run, prompt, messages)
