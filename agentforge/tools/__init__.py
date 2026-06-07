"""Tool abstraction: turn a plain Python function into an LLM-callable tool.

A `Tool` introspects a function's signature + docstring and produces the JSON
schema that OpenAI-style function calling expects. The `@tool` decorator is the
ergonomic entry point.
"""
from __future__ import annotations

import inspect
import json
import typing as t
from dataclasses import dataclass

_PY_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_type(annotation: t.Any) -> str:
    """Map a Python annotation to a JSON-schema type string."""
    if annotation is inspect.Parameter.empty:
        return "string"
    origin = t.get_origin(annotation)
    if origin in (list, tuple):
        return "array"
    if origin is dict:
        return "object"
    return _PY_TO_JSON.get(annotation, "string")


@dataclass
class Tool:
    """A callable wrapped with the metadata an LLM needs to invoke it."""

    func: t.Callable
    name: str
    description: str
    parameters: dict

    @classmethod
    def from_function(cls, func: t.Callable) -> "Tool":
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or func.__name__
        props: dict = {}
        required: list[str] = []
        for pname, param in sig.parameters.items():
            if pname in ("self", "cls"):
                continue
            props[pname] = {"type": _json_type(param.annotation)}
            if param.default is inspect.Parameter.empty:
                required.append(pname)
        return cls(
            func=func,
            name=func.__name__,
            description=doc.strip().split("\n")[0],
            parameters={
                "type": "object",
                "properties": props,
                "required": required,
            },
        )

    def to_schema(self) -> dict:
        """Render the OpenAI tools[] schema entry."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def call(self, arguments: str | dict) -> str:
        """Invoke the underlying function with JSON (or dict) arguments."""
        args = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
        result = self.func(**args)
        return result if isinstance(result, str) else json.dumps(result, default=str)


def tool(func: t.Callable) -> Tool:
    """Decorator: turn a function into a Tool. Use the first docstring line as its description."""
    return Tool.from_function(func)
