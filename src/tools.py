"""Tool definitions and executor.

Three tools are registered with the model:

- get_current_datetime : returns the current local date/time.
- calculator           : evaluates a math expression via a safe AST walker (no eval()).
- web_search           : top DuckDuckGo results (uses the `duckduckgo-search` lib).

A fourth tool, `retrieve_from_notes`, is registered dynamically by main.py when
a RAG index is loaded — see rag.py.

The executor (`run_tool`) is provider-agnostic: it takes a tool name plus its
JSON-decoded arguments and returns a string result, which main.py packages into
a `tool` role message for the next API call.
"""

from __future__ import annotations

import ast
import math
import operator as op
from datetime import datetime
from typing import Any, Callable

# -------- OpenAI-style tool schemas --------
# Kept as plain dicts so they serialise cleanly to JSON for the API.

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Return the current local date and time as an ISO-8601 string. Useful when the student asks for today's date, the time, or how far away a deadline is.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression like '2 ** 10 + 5' or 'sqrt(2) * pi'. Supports +, -, *, /, //, %, **, parentheses, and the functions: sqrt, log, log10, exp, sin, cos, tan, asin, acos, atan, floor, ceil, abs, round. Constants: pi, e.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate.",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web via DuckDuckGo and return the top results. Use this for current events, recent news, or facts the student wants verified.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (1-10).",
                        "default": 5,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]


# -------- implementations --------

def get_current_datetime() -> str:
    now = datetime.now().astimezone()
    return now.isoformat(timespec="seconds")


# Safe calculator via AST.
# We explicitly allowlist operator nodes and function names. Anything else
# (attribute access, imports, names, subscripts, etc.) raises.

_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}

_UNARY_OPS: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}

_FUNCS: dict[str, Callable[..., float]] = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "floor": math.floor,
    "ceil": math.ceil,
    "abs": abs,
    "round": round,
}

_CONSTS: dict[str, float] = {"pi": math.pi, "e": math.e}


def _safe_eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        fn = _BIN_OPS.get(type(node.op))
        if fn is None:
            raise ValueError(f"operator {type(node.op).__name__} not allowed")
        return fn(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        fn = _UNARY_OPS.get(type(node.op))
        if fn is None:
            raise ValueError(f"unary operator {type(node.op).__name__} not allowed")
        return fn(_safe_eval(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise ValueError(f"name {node.id!r} is not allowed")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise ValueError("only whitelisted function calls are allowed")
        if node.keywords:
            raise ValueError("keyword arguments are not allowed")
        args = [_safe_eval(a) for a in node.args]
        return _FUNCS[node.func.id](*args)
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def calculator(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
    except ZeroDivisionError:
        return "error: division by zero"
    except (ValueError, SyntaxError, TypeError) as e:
        return f"error: {e}"
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


def web_search(query: str, max_results: int = 5) -> str:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return "error: duckduckgo-search is not installed. Run `pip install duckduckgo-search`."
    max_results = max(1, min(int(max_results or 5), 10))
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:  # the library raises a grab-bag of errors
        return f"error: web search failed ({e})"
    if not results:
        return "no results"
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title") or "(no title)"
        href = r.get("href") or r.get("url") or ""
        body = (r.get("body") or "").strip().replace("\n", " ")
        if len(body) > 240:
            body = body[:237] + "..."
        lines.append(f"{i}. {title}\n   {href}\n   {body}")
    return "\n".join(lines)


# -------- executor --------

def run_tool(name: str, arguments: dict[str, Any]) -> str:
    """Dispatch a tool call from the model. Returns a string for the tool message."""
    if name == "get_current_datetime":
        return get_current_datetime()
    if name == "calculator":
        return calculator(arguments.get("expression", ""))
    if name == "web_search":
        return web_search(
            arguments.get("query", ""),
            int(arguments.get("max_results", 5) or 5),
        )
    return f"error: unknown tool {name!r}"
