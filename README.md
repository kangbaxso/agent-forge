# AgentForge

A tiny, provider-agnostic framework for building tool-using AI agents in Python.

No SDK lock-in, no heavy dependencies (just `requests`), no magic. One small
`Agent` class, a `@tool` decorator, and a clean reason-act loop you can actually
read in one sitting. Works with **any OpenAI-compatible endpoint**: OpenAI,
OpenRouter, Groq, Together, Mistral, or a local `llama.cpp` / Ollama server.

```python
from agentforge import Agent, tool

@tool
def add(a: int, b: int) -> int:
    "Add two numbers."
    return a + b

agent = Agent(model="gpt-4o-mini", tools=[add])
print(agent.run("What is 21 + 21?"))
# -> "21 + 21 equals 42."
```

## Why

Most agent frameworks are huge. If all you want is "call my Python functions
from an LLM and loop until done," you shouldn't need 40 dependencies and a
graph DSL. AgentForge is ~200 lines of readable code that does exactly that.

- **Provider-agnostic** — point `base_url` at any OpenAI-compatible API.
- **Zero ceremony** — a function + a docstring becomes a tool. Types are
  inferred from annotations into JSON schema automatically.
- **Transparent loop** — the whole reason-act cycle lives in one method you can
  read and patch.
- **Resilient** — automatic retry with exponential backoff on 429 / 5xx /
  network errors.
- **Streaming + async** — stream the final answer token-by-token, or `await`
  the agent from async code.
- **Tested offline** — the test suite mocks the provider, so it runs with no
  API key and no network.

## Install

```bash
pip install agentforge-mini
```

Or from source:

```bash
git clone https://github.com/kangbaxso/agent-forge
cd agent-forge
pip install -e .
```

## Usage

### 1. Define tools

Any function with a docstring becomes a tool. Parameter types come from
annotations:

```python
from agentforge import tool

@tool
def get_weather(city: str) -> dict:
    "Get the current weather for a city."
    ...
```

### 2. Build an agent

```python
from agentforge import Agent

agent = Agent(
    model="gpt-4o-mini",
    tools=[get_weather],
    system="You are a concise assistant.",
    verbose=True,        # log each tool call
    max_steps=8,         # safety cap on the loop
)
```

### 3. Run it

```python
answer = agent.run("What's the weather in Tokyo?")
```

### Stream the answer

```python
for token in agent.stream("Explain async IO in one paragraph."):
    print(token, end="", flush=True)
```

Tool-calling rounds run first (non-streamed); once the model is ready to answer
in plain text, that answer streams token-by-token.

### Use it from async code

```python
answer = await agent.arun("What's 21 + 21?")
```

### Resilience

The provider retries automatically on `429` and `5xx` responses and on network
errors, with exponential backoff. Tune it:

```python
agent = Agent(model="gpt-4o-mini", max_retries=5)
```

After exhausting retries it raises `agentforge.ProviderError`.

## Point it at any provider

Set two environment variables — that's the whole config:

```bash
# OpenAI (default)
export OPENAI_API_KEY=sk-...

# OpenRouter
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export OPENAI_API_KEY=sk-or-...

# Groq
export OPENAI_BASE_URL=https://api.groq.com/openai/v1
export OPENAI_API_KEY=gsk_...

# Local llama.cpp / Ollama
export OPENAI_BASE_URL=http://localhost:8080/v1
export OPENAI_API_KEY=not-needed
```

…or pass `base_url=` / `api_key=` directly to `Agent(...)`.

## Run the example

```bash
export OPENAI_API_KEY=sk-...
python examples/quickstart.py
```

## Run the tests

No API key needed — the provider is mocked:

```bash
python tests/test_agent.py
# or
python -m pytest tests/ -q
```

## How the loop works

1. Send the conversation + your tool schemas to the model.
2. If the model asks to call tools, AgentForge runs them locally and feeds the
   results back as `role: tool` messages.
3. Repeat until the model returns a plain answer or `max_steps` is reached.

That's it. The entire loop is in `agentforge/agent.py`.

## License

MIT — see [LICENSE](LICENSE).
