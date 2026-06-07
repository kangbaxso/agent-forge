"""Live example — a weather+math agent.

Set your endpoint first (any OpenAI-compatible provider):

    export OPENAI_API_KEY=sk-...                 # your key
    export OPENAI_BASE_URL=https://api.openai.com/v1   # or OpenRouter/Groq/local

Then:  python examples/quickstart.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentforge import Agent, tool


@tool
def get_weather(city: str) -> dict:
    """Get the current weather for a city (demo: returns canned data)."""
    fake = {"tokyo": "18C, clear", "jakarta": "31C, humid", "london": "11C, rain"}
    return {"city": city, "weather": fake.get(city.lower(), "unknown")}


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY (and OPENAI_BASE_URL for non-OpenAI providers) first.")
        sys.exit(1)

    agent = Agent(
        model=os.getenv("AGENTFORGE_MODEL", "gpt-4o-mini"),
        tools=[get_weather, add],
        system="You are a concise assistant. Use tools when helpful.",
        verbose=True,
    )
    print(agent.run("What's the weather in Tokyo, and what is 19 plus 23?"))
