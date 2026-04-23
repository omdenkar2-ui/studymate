"""MCP-facing tool implementations.

These functions are thin wrappers that can be called directly (by
`mcp_server.py`) or composed into tests. They reuse the same LLM client,
memory, and RAG modules the CLI uses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .llm import LLMClient, LLMError
from .memory import MEMORY_FILE, Memory
from .prompts import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT
from .rag import (
    RETRIEVE_TOOL_SCHEMA,
    Chunk,
    RagIndex,
    _embed,
    _get_model,
    build_index,
    run_retrieve_tool,
)
from .tools import TOOL_SCHEMAS, run_tool

_MAX_TOOL_ROUNDS = 6
_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---- lazy singletons, so the MCP server only pays the cost if a tool is called ----

_client: LLMClient | None = None
_rag: RagIndex | None = None
_rag_path: Path | None = None


def _get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def _get_rag(docs: str | None) -> RagIndex | None:
    """Build or return the cached RAG index. Returns None if no docs are given
    and none have been loaded yet."""
    global _rag, _rag_path
    if docs:
        p = Path(docs)
        if _rag is None or _rag_path != p:
            _rag = build_index(p, verbose=False)
            _rag_path = p
    return _rag


def _parse_args(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


# ---- public tool functions ----

def ask_studymate(question: str, docs: str | None = None) -> str:
    """Ask StudyMate a question and get the final answer as a string.

    Runs a single turn (with tool calls) against the model but does NOT
    persist memory — MCP callers manage their own session state.
    """
    if not question or not question.strip():
        return "error: question is empty"
    client = _get_client()
    rag = _get_rag(docs)

    tools = list(TOOL_SCHEMAS)
    if rag is not None:
        tools.append(RETRIEVE_TOOL_SCHEMA)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOT_EXAMPLES,
        {"role": "user", "content": question.strip()},
    ]

    for _ in range(_MAX_TOOL_ROUNDS):
        try:
            reply = client.chat(messages=messages, tools=tools)
        except LLMError as e:
            return f"error: {e}"

        tool_calls = getattr(reply, "tool_calls", None) or []
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": reply.content or ""}
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ]
        messages.append(assistant_msg)

        if not tool_calls:
            return reply.content or ""

        for tc in tool_calls:
            name = tc.function.name
            args = _parse_args(tc.function.arguments)
            try:
                if name == "retrieve_from_notes":
                    result = run_retrieve_tool(rag, args) if rag is not None else "error: no notes loaded"
                else:
                    result = run_tool(name, args)
            except Exception as e:
                result = f"error: tool {name!r} raised {type(e).__name__}: {e}"
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "name": name, "content": result}
            )
    return "(stopped: exceeded maximum tool-call rounds)"


def add_to_knowledge_base(text: str, source: str, cache_dir: str = "rag_index") -> str:
    """Append a new chunk of text to the RAG store and persist it."""
    global _rag, _rag_path
    if not text or not text.strip():
        return "error: text is empty"
    if not source or not source.strip():
        return "error: source is empty"

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    meta_file = cache / "index.json"
    vec_file = cache / "vectors.npy"

    # Start from the in-memory index, or the on-disk cache, or empty.
    if _rag is not None:
        chunks: list[Chunk] = list(_rag.chunks)
        vectors: np.ndarray | None = _rag.embeddings
        model_name = _rag.model_name
    elif meta_file.exists() and vec_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        chunks = [Chunk(**c) for c in meta["chunks"]]
        vectors = np.load(vec_file)
        model_name = meta.get("model_name", _DEFAULT_MODEL)
    else:
        chunks, vectors, model_name = [], None, _DEFAULT_MODEL

    next_id = sum(1 for c in chunks if c.source == source)
    new_chunk = Chunk(source=source, chunk_id=next_id, text=text.strip())
    new_vec = _embed(_get_model(model_name), [new_chunk.text])

    vectors = new_vec if vectors is None or len(vectors) == 0 else np.vstack([vectors, new_vec])
    chunks.append(new_chunk)

    meta_file.write_text(
        json.dumps(
            {
                "manifest_hash": "manual",  # manual additions bypass manifest tracking
                "model_name": model_name,
                "chunks": [{"source": c.source, "chunk_id": c.chunk_id, "text": c.text} for c in chunks],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    np.save(vec_file, vectors)

    _rag = RagIndex(chunks=chunks, embeddings=vectors, model_name=model_name, manifest_hash="manual")
    _rag_path = None
    return f"added chunk {new_chunk.citation} ({len(chunks)} total)"


def get_session_summary(memory_file: str | None = None) -> str:
    """Return a short summary of the current StudyMate session memory."""
    mem = Memory(Path(memory_file) if memory_file else MEMORY_FILE)
    mem.load()
    total_user = sum(1 for m in mem.history if m.get("role") == "user")
    total_assistant = sum(1 for m in mem.history if m.get("role") == "assistant")
    header = (
        f"Session started: {mem.created_at}\n"
        f"Turns — user: {total_user}, assistant: {total_assistant}\n"
        f"Recent exchanges:"
    )
    return f"{header}\n{mem.summary_text(limit=6)}"
