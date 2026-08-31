"""Thin wrapper around a local ChromaDB store.

Persisted to chroma_db/ at the project root (gitignored — holds client
conversation content and cached filing text, same PII reasoning as
clients.db). Uses Chroma's bundled default embedding function: a small
local ONNX MiniLM model, downloaded once on first use, no API key.

Append-only by construction, not just by convention:
- Every write is an upsert with a caller-supplied, deterministic id —
  never a random UUID — so re-adding the same chunk overwrites itself
  instead of duplicating.
- `has_chunks` lets a caller check for existing content *before* doing any
  fetch/embed work, so already-ingested content is never reprocessed.
- Nothing here ever deletes a collection or clears rows outside the ids
  a given call is explicitly given.
"""

import logging
from pathlib import Path

import chromadb

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent.parent / "chroma_db"

_client = None


def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(DB_PATH))
    return _client


def get_collection(name: str):
    """Idempotent: reattaches to the existing on-disk collection if one
    exists. Never deletes or recreates it."""
    return get_client().get_or_create_collection(name)


def add_chunks(
    collection_name: str,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
) -> None:
    """Upsert chunks by their deterministic ids. Safe to call repeatedly
    with the same ids — overwrites those rows in place, never duplicates,
    never touches any other row in the collection."""
    if not ids:
        return
    collection = get_collection(collection_name)
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    logger.info(
        "vector_store: upserted %d chunk(s) into %s", len(ids), collection_name
    )


def has_chunks(collection_name: str, where: dict) -> bool:
    """Existence check — used to skip re-fetching/re-embedding entirely,
    not just to rely on upsert dedup."""
    collection = get_collection(collection_name)
    result = collection.get(where=where, limit=1)
    return len(result.get("ids", [])) > 0


def query(
    collection_name: str,
    query_text: str,
    n_results: int = 5,
    where: dict | None = None,
) -> list[dict]:
    """Top-k {document, metadata, distance} matches, optionally filtered."""
    collection = get_collection(collection_name)
    if collection.count() == 0:
        return []
    n_results = min(n_results, collection.count())
    result = collection.query(
        query_texts=[query_text], n_results=n_results, where=where
    )
    matches = []
    documents = result.get("documents") or [[]]
    metadatas = result.get("metadatas") or [[]]
    distances = result.get("distances") or [[]]
    for doc, meta, dist in zip(documents[0], metadatas[0], distances[0]):
        matches.append({"document": doc, "metadata": meta, "distance": dist})
    return matches
