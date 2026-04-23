"""LLM API wrapper — native Anthropic + OpenAI-compatible.

StudyMate keeps one message format internally (OpenAI-style, because tool calls
came from there first) and this module translates to/from the Anthropic
Messages API when the provider is `anthropic`. Callers always get back an
object with `.content` (str) and `.tool_calls` (list | None), regardless of
which provider answered.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable

import anthropic
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)


class LLMError(Exception):
    """Raised with a human-readable message when an API call fails."""


# ---------- response shim (what main.py consumes) ----------

@dataclass
class _FunctionShim:
    name: str
    arguments: str  # JSON string — matches the OpenAI SDK shape


@dataclass
class _ToolCallShim:
    id: str
    function: _FunctionShim
    type: str = "function"


@dataclass
class _MessageShim:
    content: str | None
    tool_calls: list[_ToolCallShim] | None


# ---------- config ----------

@dataclass
class LLMConfig:
    provider: str  # "anthropic" | "openai" | "groq" | ...
    api_key: str
    model: str
    base_url: str | None

    @classmethod
    def from_env(cls) -> "LLMConfig":
        api_key = os.environ.get("STUDYMATE_API_KEY", "").strip()
        if not api_key:
            raise LLMError(
                "STUDYMATE_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        provider = os.environ.get("STUDYMATE_PROVIDER", "anthropic").strip().lower()
        base_url = os.environ.get("STUDYMATE_BASE_URL", "").strip() or None
        if provider == "anthropic":
            model = os.environ.get("STUDYMATE_MODEL", "claude-opus-4-7").strip()
        else:
            model = os.environ.get("STUDYMATE_MODEL", "llama-3.3-70b-versatile").strip()
            if base_url is None and provider == "groq":
                base_url = "https://api.groq.com/openai/v1"
        return cls(provider=provider, api_key=api_key, model=model, base_url=base_url)


# ---------- OpenAI → Anthropic conversion helpers ----------

def _messages_openai_to_anthropic(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Convert OpenAI-style messages into (system_string, anthropic_messages).

    - system messages are pulled out and concatenated (Anthropic takes system as a top-level field)
    - tool_calls on assistant turns become `tool_use` content blocks
    - consecutive `role: tool` messages collapse into a single user message of `tool_result` blocks
    """
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    i = 0
    n = len(messages)
    while i < n:
        msg = messages[i]
        role = msg.get("role")

        if role == "system":
            if msg.get("content"):
                system_parts.append(str(msg["content"]))
            i += 1

        elif role == "user":
            out.append({"role": "user", "content": str(msg.get("content") or "")})
            i += 1

        elif role == "assistant":
            blocks: list[dict[str, Any]] = []
            if msg.get("content"):
                blocks.append({"type": "text", "text": str(msg["content"])})
            for tc in msg.get("tool_calls") or []:
                args_raw = tc.get("function", {}).get("arguments") or "{}"
                try:
                    tool_input = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    if not isinstance(tool_input, dict):
                        tool_input = {}
                except json.JSONDecodeError:
                    tool_input = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": tc.get("function", {}).get("name", ""),
                        "input": tool_input,
                    }
                )
            # Anthropic requires non-empty content; a single space keeps edge-case empty turns valid.
            out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": " "}]})
            i += 1

        elif role == "tool":
            # Batch consecutive tool results into one user-role message.
            tool_blocks: list[dict[str, Any]] = []
            while i < n and messages[i].get("role") == "tool":
                m = messages[i]
                tool_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": str(m.get("tool_call_id", "")),
                        "content": str(m.get("content", "")),
                    }
                )
                i += 1
            out.append({"role": "user", "content": tool_blocks})

        else:
            # Unknown role — skip rather than crash.
            i += 1

    return "\n\n".join(system_parts), out


def _tools_openai_to_anthropic(tools: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for t in tools:
        fn = t.get("function", {})
        result.append(
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return result


def _anthropic_response_to_shim(resp: Any) -> _MessageShim:
    text_parts: list[str] = []
    tool_calls: list[_ToolCallShim] = []
    for block in resp.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_parts.append(block.text)
        elif btype == "tool_use":
            tool_calls.append(
                _ToolCallShim(
                    id=block.id,
                    function=_FunctionShim(name=block.name, arguments=json.dumps(block.input)),
                )
            )
        # Ignore thinking / server_tool_use blocks — StudyMate doesn't need them.
    content = "".join(text_parts).strip() or None
    return _MessageShim(content=content, tool_calls=tool_calls or None)


# ---------- error translation ----------

def _anthropic_llm_error(e: anthropic.APIError, model: str) -> LLMError:
    msg = getattr(e, "message", str(e))
    if isinstance(e, anthropic.AuthenticationError):
        return LLMError(f"Authentication failed — check STUDYMATE_API_KEY. ({msg})")
    if isinstance(e, anthropic.PermissionDeniedError):
        return LLMError(
            "Your API key does not have permission for this model. "
            "Check the model name or upgrade your workspace."
        )
    if isinstance(e, anthropic.NotFoundError):
        return LLMError(f"Model {model!r} not found. Check STUDYMATE_MODEL in .env.")
    if isinstance(e, anthropic.RateLimitError):
        return LLMError("Rate limit hit. Wait a moment and try again, or switch to claude-haiku-4-5.")
    if isinstance(e, anthropic.APITimeoutError):
        return LLMError("The API took too long to respond. Try again.")
    if isinstance(e, anthropic.APIConnectionError):
        return LLMError("Could not reach the API. Check your internet connection.")
    if isinstance(e, anthropic.BadRequestError):
        return LLMError(f"Bad request sent to the model: {msg}")
    status = getattr(e, "status_code", None)
    return LLMError(f"API error ({status}): {msg}" if status else f"API error: {msg}")


def _openai_llm_error(e: APIError) -> LLMError:
    msg = getattr(e, "message", str(e))
    if isinstance(e, AuthenticationError):
        return LLMError(f"Authentication failed — check STUDYMATE_API_KEY. ({msg})")
    if isinstance(e, RateLimitError):
        return LLMError("Rate limit hit. Wait a moment and try again, or switch to a smaller model.")
    if isinstance(e, APITimeoutError):
        return LLMError("The API took too long to respond. Try again.")
    if isinstance(e, APIConnectionError):
        return LLMError("Could not reach the API. Check your internet connection.")
    if isinstance(e, BadRequestError):
        return LLMError(f"Bad request sent to the model: {msg}")
    return LLMError(f"API error: {msg}")


# ---------- client ----------

class LLMClient:
    """Provider-agnostic chat client. Returns a `_MessageShim` regardless of provider."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_env()
        if self.config.provider == "anthropic":
            self._anthropic = anthropic.Anthropic(
                api_key=self.config.api_key, timeout=60.0, max_retries=2
            )
            self._openai = None
        else:
            self._anthropic = None
            self._openai = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=60.0,
                max_retries=2,
            )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Iterable[dict[str, Any]] | None = None,
        temperature: float = 0.4,
        max_tokens: int = 4096,
    ) -> Any:
        if self.config.provider == "anthropic":
            return self._chat_anthropic(messages, tools, max_tokens)
        return self._chat_openai(messages, tools, temperature)

    def _chat_anthropic(
        self,
        messages: list[dict[str, Any]],
        tools: Iterable[dict[str, Any]] | None,
        max_tokens: int,
    ) -> _MessageShim:
        system, converted = _messages_openai_to_anthropic(messages)
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "messages": converted,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = _tools_openai_to_anthropic(tools)
        # Note: temperature / top_p / top_k are NOT passed — Opus 4.7 rejects them (400).
        try:
            resp = self._anthropic.messages.create(**kwargs)
        except anthropic.APIError as e:
            raise _anthropic_llm_error(e, self.config.model) from e
        return _anthropic_response_to_shim(resp)

    def _chat_openai(
        self,
        messages: list[dict[str, Any]],
        tools: Iterable[dict[str, Any]] | None,
        temperature: float,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = list(tools)
            kwargs["tool_choice"] = "auto"
        try:
            resp = self._openai.chat.completions.create(**kwargs)
        except APIError as e:
            raise _openai_llm_error(e) from e
        if not resp.choices:
            raise LLMError("Model returned no choices.")
        return resp.choices[0].message
