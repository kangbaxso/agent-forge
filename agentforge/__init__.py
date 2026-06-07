"""AgentForge — a tiny, provider-agnostic framework for building tool-using AI agents.

Build an agent in 5 lines:

    from agentforge import Agent, tool

    @tool
    def add(a: int, b: int) -> int:
        "Add two numbers."
        return a + b

    agent = Agent(model="gpt-4o-mini", tools=[add])
    print(agent.run("What is 21 + 21?"))

Works with any OpenAI-compatible endpoint (OpenAI, OpenRouter, Groq, Together,
local llama.cpp / Ollama) — just point `base_url` at it.
"""
from .agent import Agent
from .tools import tool, Tool
from .providers.openai_compatible import OpenAICompatible, ProviderError

__version__ = "0.2.0"
__all__ = ["Agent", "tool", "Tool", "OpenAICompatible", "ProviderError"]
