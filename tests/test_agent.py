"""Offline tests — mock the provider so no API key / network is needed.

Verifies: tool schema generation, the tool-calling loop, and plain answers.
Run: python -m pytest tests/ -q   (or: python tests/test_agent.py)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentforge import Agent, tool


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


def test_tool_schema():
    schema = multiply.to_schema()
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "multiply"
    assert fn["description"] == "Multiply two integers."
    assert fn["parameters"]["properties"]["a"]["type"] == "integer"
    assert set(fn["parameters"]["required"]) == {"a", "b"}


def test_tool_call_executes():
    assert multiply.call('{"a": 6, "b": 7}') == "42"


class _MockProvider:
    """Replays a scripted sequence of model responses."""

    def __init__(self, script):
        self.script = list(script)

    def chat(self, messages, tools=None, temperature=0.7):
        return self.script.pop(0)


def test_agent_runs_tool_then_answers():
    agent = Agent(tools=[multiply])
    agent.provider = _MockProvider([
        {  # 1st turn: model asks to call the tool
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "c1",
                "function": {"name": "multiply", "arguments": '{"a": 6, "b": 7}'},
            }],
        },
        {"role": "assistant", "content": "The answer is 42."},  # 2nd turn: final
    ])
    out = agent.run("What is 6 times 7?")
    assert out == "The answer is 42."


def test_agent_plain_answer_no_tools():
    agent = Agent()
    agent.provider = _MockProvider([{"role": "assistant", "content": "Hello!"}])
    assert agent.run("hi") == "Hello!"


def test_unknown_tool_is_handled():
    agent = Agent(tools=[multiply])
    agent.provider = _MockProvider([
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "function": {"name": "ghost", "arguments": "{}"}}],
        },
        {"role": "assistant", "content": "done"},
    ])
    assert agent.run("call ghost") == "done"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} tests passed")
