"""
vector_store.py — wraps ChromaDB for storing and retrieving code chunks.

Deliberately kept as a thin abstraction (not calling chromadb directly
elsewhere in the codebase) so that swapping to Qdrant or Pinecone later
is a localized change to this one file, not a search-and-replace across
every agent that uses RAG.

Collections are cached per repo identity (repo_id + commit hash). If you
call get_or_build_collection() twice for the same repo at the same
commit, the second call skips re-chunking and re-embedding entirely —
this matters because embedding calls count against your free Gemini
quota just like generation calls do, and re-embedding an unchanged repo
on every single bug report would burn through that quota fast.
"""

import os
import hashlib
import chromadb
from rag.chunker import chunk_repository
from rag.embeddings import embed_batch, embed_text

CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")

_chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

EMBED_BATCH_SIZE = 20  # keep batches small and predictable for free-tier limits


def _collection_name(repo_id: str, commit_hash: str) -> str:
    """
    ChromaDB collection names have character restrictions, so we hash
    the repo_id + commit_hash into a safe fixed-length name rather than
    using the raw repo URL (which contains slashes/colons).
    """
    raw = f"{repo_id}:{commit_hash}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"repo_{digest}"


def collection_exists(repo_id: str, commit_hash: str) -> bool:
    name = _collection_name(repo_id, commit_hash)
    existing = [c.name for c in _chroma_client.list_collections()]
    return name in existing


def get_or_build_collection(repo_path: str, repo_id: str, commit_hash: str):
    """
    Returns a ChromaDB collection for this exact repo+commit, building
    it (chunk -> embed -> store) only if it doesn't already exist.

    Args:
        repo_path: local filesystem path to the cloned repo.
        repo_id: a stable identifier for the repo, e.g. its URL.
        commit_hash: the commit the repo is currently checked out at.
            Pass a real commit hash where possible — using "latest" or
            similar defeats the cache, since it'll never match a future
            run even if the repo hasn't changed.

    Returns:
        A chromadb Collection object, ready to .query() against.
    """
    name = _collection_name(repo_id, commit_hash)

    if collection_exists(repo_id, commit_hash):
        print(f"[vector_store] cache hit — reusing collection {name}")
        return _chroma_client.get_collection(name)

    print(f"[vector_store] cache miss — building collection {name}")
    collection = _chroma_client.create_collection(name)

    chunks = chunk_repository(repo_path)
    if not chunks:
        print("[vector_store] warning: no indexable files found in repo")
        return collection

    for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[batch_start: batch_start + EMBED_BATCH_SIZE]
        texts = [c["text"] for c in batch]
        vectors = embed_batch(texts)

        collection.add(
            ids=[f"{c['filename']}::{c['chunk_index']}::{batch_start + i}" for i, c in enumerate(batch)],
            embeddings=vectors,
            documents=texts,
            metadatas=[{"filename": c["filename"], "chunk_index": c["chunk_index"]} for c in batch],
        )

    print(f"[vector_store] indexed {len(chunks)} chunks")
    return collection


def query_collection(collection, query_text: str, n_results: int = 10) -> list[dict]:
    """
    Embeds query_text and retrieves the n_results most similar chunks.

    Returns:
        List of dicts: {"text": str, "filename": str, "chunk_index": int,
        "distance": float (lower = more similar)}
    """
    query_vector = embed_text(query_text)

    results = collection.query(query_embeddings=[query_vector], n_results=n_results)

    output = []
    for i in range(len(results["documents"][0])):
        output.append({
            "text": results["documents"][0][i],
            "filename": results["metadatas"][0][i]["filename"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
            "distance": results["distances"][0][i],
        })
    return output


if __name__ == "__main__":
    # Quick manual test — run with: python -m rag.vector_store
    import subprocess
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd="sample-repo",
        capture_output=True, text=True
    ).stdout.strip()

    collection = get_or_build_collection(
        repo_path="sample-repo",
        repo_id="local-sample-repo",
        commit_hash=commit,
    )

    print("\n--- Querying for: 'password is None causes a crash' ---")
    results = query_collection(collection, "password is None causes a crash", n_results=3)
    for r in results:
        print(f"\n[{r['filename']} chunk {r['chunk_index']}] distance={r['distance']:.4f}")
        print(r["text"][:200])
