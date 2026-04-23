"""RAG: document ingestion, chunking, embedding, and retrieval.

Design:
- Chunks are ~400 tokens (measured as ~4 chars per token) with 50-token overlap.
- Embeddings come from sentence-transformers (`all-MiniLM-L6-v2`) — free, local,
  384-dim. Fast enough for a student's notes folder.
- The index is a numpy float32 array plus a list of chunk metadata. Cosine
  similarity is computed by dot product on L2-normalised vectors.
- The index is cached in `rag_index/` so we don't re-embed on every restart.
  If the set of source files changes (tracked by a manifest hash), we re-ingest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_CHUNK_CHARS = 1600  # ~400 tokens at ~4 chars/token
_CHUNK_OVERLAP = 200  # ~50 tokens


@dataclass
class Chunk:
    """One retrievable unit. `source` is the filename, `chunk_id` is its index in that file."""
    source: str
    chunk_id: int
    text: str

    @property
    def citation(self) -> str:
        return f"{self.source}#{self.chunk_id}"


@dataclass
class RagIndex:
    chunks: list[Chunk] = field(default_factory=list)
    embeddings: np.ndarray | None = None  # shape (N, D), L2-normalised
    model_name: str = _DEFAULT_MODEL
    manifest_hash: str = ""

    def __len__(self) -> int:
        return len(self.chunks)


# ---------- loading source files ----------

def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(parts)


def _load_doc(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    return _read_txt(path)


def _collect_files(docs_path: Path) -> list[Path]:
    if docs_path.is_file():
        return [docs_path]
    files: list[Path] = []
    for ext in ("*.txt", "*.md", "*.pdf"):
        files.extend(sorted(docs_path.rglob(ext)))
    return files


def _manifest_hash(files: list[Path]) -> str:
    h = hashlib.sha256()
    for f in files:
        try:
            stat = f.stat()
        except OSError:
            continue
        h.update(str(f).encode("utf-8"))
        h.update(str(stat.st_size).encode("utf-8"))
        h.update(str(int(stat.st_mtime)).encode("utf-8"))
    return h.hexdigest()


# ---------- chunking ----------

def chunk_text(text: str, source: str, chunk_chars: int = _CHUNK_CHARS, overlap: int = _CHUNK_OVERLAP) -> list[Chunk]:
    """Simple sliding-window chunker on character count.

    We try to break on a nearby newline/paragraph boundary so chunks don't
    start mid-sentence, but we don't require it — for dense PDFs without
    paragraph breaks, a hard cut is fine.
    """
    text = text.strip()
    if not text:
        return []
    chunks: list[Chunk] = []
    start = 0
    chunk_id = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_chars, n)
        if end < n:
            # prefer to cut at the last double-newline or period near the end
            window = text[start:end]
            for delim in ("\n\n", ". ", "\n", " "):
                idx = window.rfind(delim)
                if idx > chunk_chars // 2:
                    end = start + idx + len(delim)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(source=source, chunk_id=chunk_id, text=piece))
            chunk_id += 1
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


# ---------- embedding + index ----------

_model_cache: dict[str, Any] = {}


def _get_model(model_name: str):
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer

        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def _embed(model, texts: list[str]) -> np.ndarray:
    vecs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype("float32")


def _chunk_to_dict(c: Chunk) -> dict[str, Any]:
    return {"source": c.source, "chunk_id": c.chunk_id, "text": c.text}


def build_index(
    docs_path: Path,
    *,
    cache_dir: Path = Path("rag_index"),
    model_name: str = _DEFAULT_MODEL,
    verbose: bool = True,
) -> RagIndex:
    """Ingest files under `docs_path`, chunk, embed, and cache to `cache_dir`."""
    docs_path = Path(docs_path)
    if not docs_path.exists():
        raise FileNotFoundError(f"docs path does not exist: {docs_path}")
    files = _collect_files(docs_path)
    if not files:
        raise ValueError(f"no .txt, .md, or .pdf files found under {docs_path}")

    manifest = _manifest_hash(files)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_file = cache_dir / "index.json"
    vec_file = cache_dir / "vectors.npy"

    if meta_file.exists() and vec_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta.get("manifest_hash") == manifest and meta.get("model_name") == model_name:
            if verbose:
                print(f"[rag] loaded cached index ({len(meta['chunks'])} chunks)")
            return RagIndex(
                chunks=[Chunk(**c) for c in meta["chunks"]],
                embeddings=np.load(vec_file),
                model_name=model_name,
                manifest_hash=manifest,
            )

    if verbose:
        print(f"[rag] ingesting {len(files)} file(s) from {docs_path} ...")
    all_chunks: list[Chunk] = []
    for f in files:
        try:
            text = _load_doc(f)
        except Exception as e:
            print(f"[rag] skipped {f.name}: {e}")
            continue
        chunks = chunk_text(text, source=f.name)
        all_chunks.extend(chunks)
        if verbose:
            print(f"[rag]   {f.name}: {len(chunks)} chunks")
    if not all_chunks:
        raise ValueError("ingested files produced no chunks")

    if verbose:
        print(f"[rag] embedding {len(all_chunks)} chunks with {model_name} ...")
    embeddings = _embed(_get_model(model_name), [c.text for c in all_chunks])

    meta_file.write_text(
        json.dumps(
            {
                "manifest_hash": manifest,
                "model_name": model_name,
                "chunks": [_chunk_to_dict(c) for c in all_chunks],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    np.save(vec_file, embeddings)
    if verbose:
        print(f"[rag] cached index to {cache_dir}/")
    return RagIndex(
        chunks=all_chunks,
        embeddings=embeddings,
        model_name=model_name,
        manifest_hash=manifest,
    )


def retrieve(index: RagIndex, query: str, top_k: int = 4) -> list[tuple[Chunk, float]]:
    """Return up to top_k chunks ranked by cosine similarity."""
    if not index.chunks or index.embeddings is None:
        return []
    model = _get_model(index.model_name)
    qvec = _embed(model, [query])[0]
    sims = index.embeddings @ qvec  # both normalised, so dot == cosine
    top_idx = np.argsort(-sims)[: min(top_k, len(index.chunks))]
    return [(index.chunks[i], float(sims[i])) for i in top_idx]


def format_context(hits: Iterable[tuple[Chunk, float]]) -> str:
    """Render retrieved chunks as a prompt-ready context block."""
    return "\n\n".join(
        f"[source: {chunk.citation}] (score={score:.3f})\n{chunk.text}"
        for chunk, score in hits
    )


# ---------- tool integration ----------

RETRIEVE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "retrieve_from_notes",
        "description": "Search the student's loaded study notes for passages relevant to a question. Returns the top passages with citations. Use this whenever the student asks about something from their notes, a topic from class, or a document they uploaded.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A natural-language search query."},
                "top_k": {"type": "integer", "default": 4, "description": "How many chunks to return (1-8)."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def run_retrieve_tool(index: RagIndex, arguments: dict[str, Any]) -> str:
    query = arguments.get("query", "").strip()
    if not query:
        return "error: query is empty"
    top_k = int(arguments.get("top_k", 4) or 4)
    top_k = max(1, min(top_k, 8))
    hits = retrieve(index, query, top_k=top_k)
    if not hits:
        return "no relevant passages found in the loaded notes"
    return format_context(hits)
