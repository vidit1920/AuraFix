"""
chunker.py — splits source files into overlapping text chunks suitable
for embedding.

Chunk boundaries are biased toward Python's structure (class/def
boundaries first, then blank lines, then plain newlines) so that
chunks tend to preserve whole functions rather than cutting a function
in half mid-body. This directly affects retrieval quality downstream:
a chunk that contains half of one function and half of another is a
worse embedding target than a chunk containing one complete function.
"""

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Order matters: split at definition-like boundaries first (these cover
# Python, Java, C/C++ and C#), then blank lines, then any newline, then
# whitespace. Keeping whole functions intact improves retrieval quality.
CODE_SEPARATORS = [
    "\nclass ", "\ndef ", "\nfunction ", "\npublic ", "\nprivate ",
    "\nprotected ", "\nstatic ", "\nvoid ", "\n\n", "\n", " ",
]

# Dirs we never want to descend into — noise without searchable signal, and
# wasted embedding quota. Includes common build output for the JVM/.NET/C
# toolchains so we don't index generated artifacts.
SKIP_DIRS = {".git", "__pycache__", "venv", ".venv", "node_modules", ".pytest_cache",
             "target", "build", "bin", "obj", "dist"}

# Source extensions we index — mirrors the languages the navigator
# understands (analyzers/call_graph.py): Python, Java, C, C++, C#. We
# whitelist rather than blacklist so config/meta files (.gitignore, .lock,
# .md) can't leak into the index as if they were code.
INDEXABLE_EXTENSIONS = {".py", ".java", ".c", ".h", ".cpp", ".cc", ".cxx",
                        ".hpp", ".hh", ".hxx", ".cs"}


def _should_index(file_path: str) -> bool:
    parts = file_path.split(os.sep)
    if any(part in SKIP_DIRS for part in parts):
        return False
    _, ext = os.path.splitext(file_path)
    return ext in INDEXABLE_EXTENSIONS


def chunk_repository(repo_path: str) -> list[dict]:
    """
    Walks every file in repo_path, splits each into chunks, and returns
    a flat list of chunk records ready for embedding.

    Returns:
        List of dicts: {"text": str, "filename": str (relative path),
        "chunk_index": int}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CODE_SEPARATORS,
    )

    all_chunks = []

    for root, dirs, files in os.walk(repo_path):
        # Prune skip-dirs in place so os.walk doesn't descend into them
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for filename in files:
            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, repo_path)
            # Normalize to forward slashes regardless of OS — keeps
            # filenames consistent with analyzers/call_graph.py and
            # with git's own diff header convention. See call_graph.py
            # for the full rationale.
            rel_path = rel_path.replace(os.sep, "/")

            if not _should_index(rel_path):
                continue

            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except (UnicodeDecodeError, PermissionError, OSError):
                # Binary file or unreadable — skip silently, this is
                # expected for some files and shouldn't halt indexing.
                continue

            if not content.strip():
                continue

            file_chunks = splitter.split_text(content)
            for i, chunk_text in enumerate(file_chunks):
                all_chunks.append({
                    "text": chunk_text,
                    "filename": rel_path,
                    "chunk_index": i,
                })

    return all_chunks


if __name__ == "__main__":
    # Quick manual test — run with: python -m rag.chunker
    chunks = chunk_repository("sample-repo")
    print(f"Indexed {len(chunks)} chunks from sample-repo\n")
    for c in chunks[:3]:
        print(f"--- {c['filename']} [chunk {c['chunk_index']}] ---")
        print(c["text"][:200])
        print()
