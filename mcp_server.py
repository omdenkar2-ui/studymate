"""StudyMate MCP server.

Exposes three tools over the Model Context Protocol so any MCP client
(Claude Desktop, Cursor, etc.) can talk to StudyMate.

Run with:
    python mcp_server.py

The server speaks MCP over stdio, which is what Claude Desktop expects.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from src.mcp_tools import add_to_knowledge_base, ask_studymate, get_session_summary

load_dotenv()

server: Server = Server("studymate")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="ask_studymate",
            description=(
                "Ask StudyMate a study question and return the answer. "
                "Optionally pass `docs` (a path to a file or folder of notes) "
                "to enable RAG retrieval for this query."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question to ask StudyMate."},
                    "docs": {
                        "type": "string",
                        "description": "Optional path to a .txt/.md/.pdf file or folder of notes.",
                    },
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="add_to_knowledge_base",
            description=(
                "Add a new chunk of text to StudyMate's RAG store so future "
                "queries can retrieve it. Returns the chunk's citation."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The note text to store."},
                    "source": {
                        "type": "string",
                        "description": "A label for the source (e.g. 'lecture-3' or 'textbook-ch5').",
                    },
                },
                "required": ["text", "source"],
            },
        ),
        Tool(
            name="get_session_summary",
            description=(
                "Return a short summary of the most recent StudyMate chat "
                "session (topic counts + the last few exchanges)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_file": {
                        "type": "string",
                        "description": "Optional path to a memory.json file (defaults to ./memory.json).",
                    }
                },
            },
        ),
    ]


_DISPATCH = {
    "ask_studymate": lambda args: ask_studymate(args.get("question", ""), args.get("docs")),
    "add_to_knowledge_base": lambda args: add_to_knowledge_base(
        args.get("text", ""), args.get("source", "")
    ),
    "get_session_summary": lambda args: get_session_summary(args.get("memory_file")),
}


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    handler = _DISPATCH.get(name)
    try:
        result = (
            await asyncio.to_thread(handler, arguments)
            if handler
            else f"error: unknown tool {name!r}"
        )
    except Exception as e:
        result = f"error: {type(e).__name__}: {e}"
    return [TextContent(type="text", text=str(result))]


async def _run() -> None:
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    if not os.environ.get("STUDYMATE_API_KEY"):
        print(
            "[studymate-mcp] warning: STUDYMATE_API_KEY is not set — "
            "ask_studymate calls will fail until you set it."
        )
    asyncio.run(_run())


if __name__ == "__main__":
    main()
