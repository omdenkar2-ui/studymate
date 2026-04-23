# StudyMate

A command-line AI study assistant. Asks and answers questions, remembers your
past sessions, reasons over your own notes (RAG), can use tools (calculator,
web search, datetime), and can be exposed as an MCP server to other clients.

Built for the TETR AI-Engineering individual assignment.

## Features

- **Milestone 1 — LLM API**: OpenAI-compatible chat loop with graceful errors (rate limit, auth, timeout, connection).
- **Milestone 2 — Prompt engineering**: tutor role + chain-of-thought + few-shot examples. See [PROMPTS.md](PROMPTS.md).
- **Milestone 3 — Memory**: in-session history plus cross-session persistence to `memory.json`; `/clear` wipes it.
- **Milestone 4 — Tool calling**: `get_current_datetime`, `calculator` (safe AST parser, **no `eval`**), `web_search` (DuckDuckGo).
- **Milestone 5 — RAG**: ingest `.txt` / `.md` / `.pdf` files, chunk, embed with Sentence-Transformers, retrieve top-k, cite inline.
- **Milestone 6 — MCP server**: exposes `ask_studymate`, `add_to_knowledge_base`, `get_session_summary` over stdio.

## Project layout

```
studymate/
├── main.py              # Entry point / chat REPL
├── mcp_server.py        # Milestone 6 MCP server
├── memory.json          # Auto-generated (example committed)
├── PROMPTS.md           # Milestone 2 writeup
├── README.md
├── requirements.txt
├── .env.example
└── src/
    ├── llm.py           # LLM API wrapper (OpenAI-compatible)
    ├── memory.py        # Memory load/save
    ├── prompts.py       # System prompt + few-shot
    ├── tools.py         # Tool schemas + safe calculator + web search
    ├── rag.py           # Chunking, embedding, retrieval
    └── mcp_tools.py     # MCP tool implementations
```

## Setup

```bash
# 1. clone and enter
git clone <your-fork-url> studymate
cd studymate

# 2. create a virtualenv
python -m venv .venv
# on Windows
.venv\Scripts\activate
# on macOS / Linux
source .venv/bin/activate

# 3. install deps
pip install -r requirements.txt

# 4. add your API key
cp .env.example .env
# then edit .env and set STUDYMATE_API_KEY
```

### Getting an API key

StudyMate supports two backends out of the box:

- **Anthropic** (default) — native SDK; create a key at
  <https://console.anthropic.com/settings/keys>. Default model is
  `claude-opus-4-7`; switch to `claude-sonnet-4-6` or `claude-haiku-4-5` in
  `.env` for cheaper runs.
- **Groq / OpenAI / any OpenAI-compatible provider** — flip
  `STUDYMATE_PROVIDER` in `.env` and set `STUDYMATE_BASE_URL`. Groq has a free
  tier at <https://console.groq.com/keys>.

## Running

Plain chat:

```bash
python main.py
```

Chat with RAG over a folder of notes:

```bash
python main.py --docs notes/
```

Or a single PDF / text file:

```bash
python main.py --docs lecture-3.pdf
```

### REPL commands

| Command | Action |
| --- | --- |
| `/help` | List commands |
| `/summary` | Show a summary of the current session |
| `/clear` | Wipe memory (both in-process and `memory.json`) |
| `quit` / `exit` | Leave StudyMate |

## MCP server

StudyMate can also be used as an MCP tool server.

```bash
python mcp_server.py
```

The server speaks MCP over stdio and exposes three tools:

- `ask_studymate(question, docs?)` — ask a question, optionally with RAG over a docs path.
- `add_to_knowledge_base(text, source)` — append a chunk to the RAG store.
- `get_session_summary(memory_file?)` — summarise the latest chat session.

### Connecting Claude Desktop

Add this to your Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "studymate": {
      "command": "python",
      "args": ["C:/absolute/path/to/studymate/mcp_server.py"],
      "env": {
        "STUDYMATE_API_KEY": "sk-ant-your-key-here",
        "STUDYMATE_PROVIDER": "anthropic",
        "STUDYMATE_MODEL": "claude-opus-4-7"
      }
    }
  }
}
```

Restart Claude Desktop; you should see `studymate` listed under MCP tools.

### Connecting other MCP clients

Any client that speaks MCP over stdio can launch `python mcp_server.py` and
communicate with it. See the [MCP docs](https://modelcontextprotocol.io) for
the full protocol.

## Example session

```
$ python main.py --docs notes/
[rag] loaded cached index (12 chunks)
StudyMate ready. Type /help for commands, 'quit' to exit.
(Knowledge base: 12 chunks from notes/)

you> what did we cover about gradient descent in the last lecture?
  [tool] retrieve_from_notes({"query": "gradient descent"}) → [source: lecture-4.txt#2]...

studymate> From your notes, gradient descent updates parameters in the
direction of steepest descent of the loss... [source: lecture-4.txt#2]

you> what is 2**16?
  [tool] calculator({"expression": "2 ** 16"}) → 65536

studymate> 2^16 is 65536.

you> quit
Bye!
```

## Safety notes

- The calculator uses an AST walker that only permits arithmetic operators,
  math-module functions, and the constants `pi` and `e`. It will reject any
  attribute access, name lookup, or imports. **No `eval()` is ever called on
  user input.**
- API keys are read from environment variables only — never hardcoded.
  `.env` is gitignored.
