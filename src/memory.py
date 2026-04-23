"""Conversation memory: short-term (in-process list) and long-term (JSON file).

Design notes:
- We persist only `role`, `content`, and (when present) `tool_calls` / `tool_call_id` / `name`,
  so restored history replays cleanly through the OpenAI-compatible chat API.
- Tool messages are kept so the model can see prior tool outputs across sessions.
- The system prompt and few-shot examples are NOT stored here — those are prepended fresh
  each session from `prompts.py` so we can iterate on them without breaking old memory files.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

MEMORY_FILE = Path(os.environ.get("STUDYMATE_MEMORY_FILE", "memory.json"))

# Keys we keep when serialising a message. `tool_calls` is a list of dicts,
# `tool_call_id` + `name` appear on tool-result messages.
_KEEP_KEYS = ("role", "content", "tool_calls", "tool_call_id", "name")


def _serialise(msg: dict[str, Any]) -> dict[str, Any]:
    return {k: msg[k] for k in _KEEP_KEYS if k in msg and msg[k] is not None}


class Memory:
    """In-process history plus load/save to a JSON file."""

    def __init__(self, path: Path = MEMORY_FILE) -> None:
        self.path = Path(path)
        self.history: list[dict[str, Any]] = []
        self.created_at: str = datetime.now().isoformat(timespec="seconds")

    # ---- short-term (in-session) ----

    def append(self, message: dict[str, Any]) -> None:
        self.history.append(_serialise(message))

    def messages(self) -> list[dict[str, Any]]:
        """Return a copy of the history for sending to the model."""
        return [dict(m) for m in self.history]

    def summary_text(self, limit: int = 8) -> str:
        """Human-readable summary of the last `limit` user/assistant turns."""
        relevant = [
            m for m in self.history
            if m["role"] in ("user", "assistant") and m.get("content")
        ]
        if not relevant:
            return "No prior conversation recorded."
        lines = []
        for m in relevant[-limit:]:
            who = "You" if m["role"] == "user" else "StudyMate"
            content = str(m["content"]).strip().replace("\n", " ")
            if len(content) > 240:
                content = content[:237] + "..."
            lines.append(f"- {who}: {content}")
        return "\n".join(lines)

    # ---- long-term (persistent) ----

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # Corrupt file — start fresh rather than crash the REPL.
            self.history = []
            return
        self.history = [_serialise(m) for m in data.get("history", []) if isinstance(m, dict)]
        self.created_at = data.get("created_at", self.created_at)

    def save(self) -> None:
        payload = {
            "created_at": self.created_at,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "history": self.history,
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def clear(self) -> None:
        """Wipe both in-memory history and the persisted file."""
        self.history = []
        self.created_at = datetime.now().isoformat(timespec="seconds")
        if self.path.exists():
            try:
                self.path.unlink()
            except OSError:
                pass
