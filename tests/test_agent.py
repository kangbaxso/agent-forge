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


def test_async_run():
    import asyncio

    agent = Agent()
    agent.provider = _MockProvider([{"role": "assistant", "content": "async hi"}])
    out = asyncio.run(agent.arun("hi"))
    assert out == "async hi"


def test_stream_yields_tokens():
    class _StreamProvider:
        def chat(self, messages, tools=None, temperature=0.7):
            return {"role": "assistant", "content": None,
                    "tool_calls": None}  # no tool calls -> go straight to stream
        def chat_stream(self, messages, temperature=0.7):
            for piece in ["Hel", "lo ", "world"]:
                yield piece

    agent = Agent()
    agent.provider = _StreamProvider()
    out = "".join(agent.stream("say hello"))
    assert out == "Hello world"


def test_provider_retry_then_success(monkeypatch=None):
    """Provider retries on 500 then succeeds on the 2nd attempt."""
    import agentforge.providers.openai_compatible as oc

    calls = {"n": 0}

    class _Resp:
        def __init__(self, status, payload=None):
            self.status_code = status
            self.text = "boom" if status >= 500 else ""
            self._payload = payload or {}
        def json(self):
            return self._payload
        def raise_for_status(self):
            pass

    def fake_post(url, headers=None, data=None, timeout=None, stream=False):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Resp(500)
        return _Resp(200, {"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    orig_post = oc.requests.post
    orig_sleep = oc.time.sleep
    oc.requests.post = fake_post
    oc.time.sleep = lambda *_: None  # no real delay in tests
    try:
        prov = oc.OpenAICompatible(model="x", api_key="k", max_retries=2, backoff=0.01)
        msg = prov.chat([{"role": "user", "content": "hi"}])
        assert msg["content"] == "ok"
        assert calls["n"] == 2  # failed once, succeeded on retry
    finally:
        oc.requests.post = orig_post
        oc.time.sleep = orig_sleep


def test_provider_gives_up_after_retries():
    import agentforge.providers.openai_compatible as oc

    class _Resp:
        status_code = 503
        text = "down"
        def raise_for_status(self): pass
        def json(self): return {}

    orig_post = oc.requests.post
    orig_sleep = oc.time.sleep
    oc.requests.post = lambda *a, **k: _Resp()
    oc.time.sleep = lambda *_: None
    try:
        prov = oc.OpenAICompatible(model="x", api_key="k", max_retries=2, backoff=0.01)
        raised = False
        try:
            prov.chat([{"role": "user", "content": "hi"}])
        except oc.ProviderError:
            raised = True
        assert raised
    finally:
        oc.requests.post = orig_post
        oc.time.sleep = orig_sleep


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} tests passed")
