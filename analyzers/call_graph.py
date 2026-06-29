"""
call_graph.py — static analysis that extracts the real code structure.

This module has NO LLM dependency. It extracts ground-truth structural
facts (function definitions, what each function calls, what each module
imports) directly from source code.

This is what makes the Code Navigation Agent a "secret weapon" rather
than just more RAG: semantic search (rag/vector_store.py) finds code
that *sounds* related to a bug description. This module finds code
that is *actually* in the execution path, regardless of how it's
worded or commented.

Languages: Python uses the built-in `ast` module (highest fidelity).
Java, C, C++, and C# are parsed with tree-sitter (analyzers/treesitter_nav.py),
which gives one front-end for many grammars without a C toolchain. Both
paths return the same per-file shape, so everything downstream treats every
language identically. Import-following in trace_call_path() is Python-specific
(dotted module paths); for other languages the trace covers the seeded files'
structure rather than resolving imports across files.
"""

import ast
import os

from analyzers.treesitter_nav import EXT_LANG, extract_structure

# Source files we know how to analyze: Python via stdlib `ast`, the rest via
# tree-sitter. Anything else is skipped during the repo walk.
SUPPORTED_EXTS = {".py"} | set(EXT_LANG.keys())


def _extract_python(source: str, file_path: str) -> dict | None:
    """The original Python path: parse with `ast` and pull imports +
    functions + the calls inside each. Returns None on a syntax error so
    the caller skips the file rather than crashing the whole pass."""
    try:
        tree = ast.parse(source, filename=file_path)
    except (SyntaxError, ValueError):
        return None

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.extend(f"{module}.{alias.name}" for alias in node.names)

    functions = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            calls = []
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call):
                    # Direct call: foo()
                    if isinstance(inner.func, ast.Name):
                        calls.append(inner.func.id)
                    # Method/attribute call: obj.foo() — record the
                    # attribute name, since that's usually the
                    # meaningful part for tracing (e.g. ".strip()")
                    elif isinstance(inner.func, ast.Attribute):
                        calls.append(inner.func.attr)
            functions[node.name] = {"line": node.lineno, "calls": calls}

    return {"imports": imports, "functions": functions}


def extract_file_structure(file_path: str) -> dict:
    """
    Parses a single source file and extracts its functions, what each
    function calls, and what it imports — dispatching to the Python `ast`
    path or the tree-sitter path by file extension.

    Returns:
        {
            "file": str,
            "imports": [str, ...],
            "functions": {func_name: {"line": int, "calls": [str, ...]}, ...},
        }
        or None if the file can't be read or parsed (callers skip it rather
        than crash the whole navigation pass on one bad file).
    """
    ext = os.path.splitext(file_path)[1].lower()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
    except (UnicodeDecodeError, OSError):
        return None

    if ext == ".py":
        result = _extract_python(source, file_path)
    elif ext in EXT_LANG:
        result = extract_structure(source, EXT_LANG[ext])
    else:
        return None

    if result is None:
        return None

    return {
        "file": file_path,
        "imports": result["imports"],
        "functions": result["functions"],
    }


def build_repo_structure(repo_path: str) -> dict:
    """
    Walks every supported source file in repo_path and extracts its structure.

    Returns:
        Dict mapping relative file path -> extract_file_structure() output
        (files that failed to parse are omitted, not included as None).
    """
    structures = {}
    skip_dirs = {".git", "__pycache__", "venv", ".venv", "node_modules", ".pytest_cache",
                 "target", "build", "bin", "obj", "dist"}

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for filename in files:
            if os.path.splitext(filename)[1].lower() not in SUPPORTED_EXTS:
                continue
            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, repo_path)
            # Normalize to forward slashes regardless of OS. Git diffs
            # always use forward slashes in their headers (a git
            # convention, not an OS thing) — if we let Windows-style
            # backslashes leak into file paths here, every downstream
            # agent (Root Cause, Patch Generator) inherits them, and
            # the Patch Generator's diff headers end up with backslash
            # paths that `git apply` rejects outright.
            rel_path = rel_path.replace(os.sep, "/")
            structure = extract_file_structure(full_path)
            if structure is not None:
                structures[rel_path] = structure

    return structures


def _module_to_possible_files(module_name: str) -> list:
    """
    Converts an import like 'auth.auth_service' into candidate relative
    file paths like 'auth/auth_service.py'. Returns multiple candidates
    since we don't know the repo root from the import string alone.
    """
    parts = module_name.split(".")
    candidates = []
    # Try matching from each suffix length, since 'from auth.auth_service
    # import authenticate' gives us 'auth.auth_service.authenticate' —
    # we need to try trimming the trailing piece (the imported name).
    for cut in range(len(parts), 0, -1):
        # Use "/" directly rather than os.path.join, so this always
        # matches the forward-slash-normalized keys in repo_structures
        # regardless of OS.
        path = "/".join(parts[:cut]) + ".py"
        candidates.append(path)
    return candidates


def trace_call_path(repo_structures: dict, seed_files: list, max_depth: int = 3) -> dict:
    """
    Starting from seed_files (e.g. the Issue Agent's possible_files
    guesses, or files RAG retrieved), traces outward through actual
    function calls and imports to build a real execution path.

    This is a breadth-first traversal: at each step, look at what
    functions in the current file call, resolve those calls to other
    files via imports, and continue outward up to max_depth hops.

    Args:
        repo_structures: output of build_repo_structure().
        seed_files: relative paths to start tracing from. Files that
            don't exist in repo_structures are skipped (a seed file
            might be a guess from the Issue Agent that doesn't
            actually exist — that's expected and not an error).
        max_depth: how many hops outward to follow. Keeps the trace
            bounded so a deeply interconnected repo doesn't explode
            into "everything calls everything."

    Returns:
        {
            "execution_path": [
                {"file": str, "function": str, "line": int, "calls_into": [str, ...]},
                ...
            ],
            "files_visited": [str, ...],
        }
    """
    visited_files = set()
    execution_path = []
    queue = [(f, 0) for f in seed_files if f in repo_structures]

    while queue:
        current_file, depth = queue.pop(0)

        if current_file in visited_files or depth > max_depth:
            continue
        visited_files.add(current_file)

        structure = repo_structures[current_file]

        for func_name, func_info in structure["functions"].items():
            execution_path.append({
                "file": current_file,
                "function": func_name,
                "line": func_info["line"],
                "calls_into": func_info["calls"],
            })

        # Resolve this file's imports to other files in the repo, and
        # queue them up for the next hop outward.
        for imported in structure["imports"]:
            for candidate in _module_to_possible_files(imported):
                if candidate in repo_structures and candidate not in visited_files:
                    queue.append((candidate, depth + 1))

    return {
        "execution_path": execution_path,
        "files_visited": sorted(visited_files),
    }


if __name__ == "__main__":
    # Quick manual test — run with: python -m analyzers.call_graph
    structures = build_repo_structure("sample-repo")

    print("=== Repo structure ===")
    for filename, structure in structures.items():
        print(f"\n{filename}")
        print(f"  imports: {structure['imports']}")
        for func_name, info in structure["functions"].items():
            print(f"  def {func_name}() [line {info['line']}] calls: {info['calls']}")

    print("\n=== Tracing from seed: login.py ===")
    trace = trace_call_path(structures, seed_files=["login.py"], max_depth=3)
    print(f"\nFiles visited: {trace['files_visited']}")
    print("\nExecution path:")
    for step in trace["execution_path"]:
        print(f"  {step['file']} :: {step['function']}() [line {step['line']}] -> calls {step['calls_into']}")
