"""StudyMate — a CLI AI study assistant.

Usage:
    python main.py                      # plain chat
    python main.py --docs notes/        # chat with RAG over your notes folder
    python main.py --docs lecture.pdf   # chat with RAG over a single file

Commands inside the REPL:
    /clear   wipe memory and start fresh
    /help    list commands
    /summary print a summary of the current session
    quit, exit, :q   leave
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.llm import LLMClient, LLMError
from src.memory import Memory, MEMORY_FILE
from src.prompts import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT
from src.rag import RETRIEVE_TOOL_SCHEMA, RagIndex, build_index, run_retrieve_tool
from src.tools import TOOL_SCHEMAS, run_tool

MAX_TOOL_ROUNDS = 6
EXIT_WORDS = {"quit", "exit", ":q", "q"}

HELP_TEXT = """Commands:
  /clear    — wipe memory and start fresh
  /summary  — show a summary of this session
  /help     — show this help
  quit      — exit StudyMate"""


def build_system_messages(rag_loaded: bool) -> list[dict[str, Any]]:
    system = SYSTEM_PROMPT
    if rag_loaded:
        system += (
            "\n\nThe student has loaded study notes. Prefer the `retrieve_from_notes` tool "
            "for any question that sounds like it could be answered from their notes, and "
            "cite the returned passages inline as [source: filename#chunk_id]."
        )
    return [{"role": "system", "content": system}, *FEW_SHOT_EXAMPLES]


def dispatch_tool(name: str, arguments: dict[str, Any], rag: RagIndex | None) -> str:
    if name == "retrieve_from_notes":
        if rag is None:
            return "error: no notes are loaded — rerun with --docs to enable this tool"
        return run_retrieve_tool(rag, arguments)
    return run_tool(name, arguments)


def serialise_assistant_message(msg: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
    if getattr(msg, "tool_calls", None):
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    return out


def _parse_args(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def run_turn(
    client: LLMClient,
    memory: Memory,
    user_input: str,
    tools: list[dict[str, Any]],
    rag: RagIndex | None,
) -> str:
    memory.append({"role": "user", "content": user_input})

    for _ in range(MAX_TOOL_ROUNDS):
        messages = build_system_messages(rag is not None) + memory.messages()
        reply = client.chat(messages=messages, tools=tools)
        memory.append(serialise_assistant_message(reply))

        tool_calls = getattr(reply, "tool_calls", None) or []
        if not tool_calls:
            return reply.content or ""

        for tc in tool_calls:
            name = tc.function.name
            raw_args = tc.function.arguments or "{}"
            arguments = _parse_args(raw_args)
            try:
                result = dispatch_tool(name, arguments, rag)
            except Exception as e:  # never let a tool crash the REPL
                result = f"error: tool {name!r} raised {type(e).__name__}: {e}"
            print(f"  [tool] {name}({_short(raw_args)}) → {_short(result, 120)}")
            memory.append(
                {"role": "tool", "tool_call_id": tc.id, "name": name, "content": result}
            )
    return "(stopped: exceeded maximum tool-call rounds)"


def _short(value: Any, n: int = 80) -> str:
    s = str(value).replace("\n", " ")
    return s if len(s) <= n else s[: n - 3] + "..."


def load_rag(docs: str | None) -> RagIndex | None:
    if not docs:
        return None
    try:
        return build_index(Path(docs))
    except Exception as e:
        print(f"[rag] failed to load notes: {e}")
        return None


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="studymate", description="Personal AI study assistant.")
    parser.add_argument("--docs", help="Path to a .txt/.md/.pdf file or a folder of them.")
    parser.add_argument("--memory-file", default=str(MEMORY_FILE), help="Path to memory JSON file.")
    args = parser.parse_args(argv)

    try:
        client = LLMClient()
    except LLMError as e:
        print(f"StudyMate: {e}")
        return 1

    memory = Memory(Path(args.memory_file))
    memory.load()

    rag = load_rag(args.docs)
    tools = list(TOOL_SCHEMAS)
    if rag is not None:
        tools.append(RETRIEVE_TOOL_SCHEMA)

    print("StudyMate ready. Type /help for commands, 'quit' to exit.")
    if memory.history:
        print(f"(Loaded {len(memory.history)} messages from previous session.)")
    if rag is not None:
        print(f"(Knowledge base: {len(rag)} chunks from {args.docs})")

    try:
        while True:
            try:
                user_input = input("\nyou> ").strip()
            except EOFError:
                print()
                break

            if not user_input:
                continue
            if user_input.lower() in EXIT_WORDS:
                break
            if user_input == "/help":
                print(HELP_TEXT)
                continue
            if user_input == "/clear":
                memory.clear()
                print("Memory cleared.")
                continue
            if user_input == "/summary":
                print(memory.summary_text())
                continue

            try:
                reply = run_turn(client, memory, user_input, tools, rag)
            except LLMError as e:
                print(f"StudyMate: {e}")
                continue
            except KeyboardInterrupt:
                print("\n(interrupted)")
                continue

            print(f"\nstudymate> {reply}")
            # Persist after every turn so a crash doesn't lose the conversation.
            try:
                memory.save()
            except OSError as e:
                print(f"(warning: could not save memory: {e})")
    finally:
        try:
            memory.save()
        except OSError:
            pass
    print("Bye!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
