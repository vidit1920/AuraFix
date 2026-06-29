"""
treesitter_nav.py — multi-language structural extraction via tree-sitter.

This is the non-Python counterpart to the AST extractor in call_graph.py.
Python keeps using the built-in `ast` module (highest fidelity); every other
supported language is parsed here with tree-sitter, which gives one parser
front-end for many grammars without a C toolchain (grammars ship as wheels
via tree-sitter-language-pack).

We extract the same shape the Python extractor produces — per file: the
functions/methods defined, the line each starts on, and the names it calls —
so the Code Navigation Agent and Root Cause Agent treat every language
identically.

NOTE on the binding: tree-sitter-language-pack's compiled nodes expose their
accessors as *methods* (`node.kind()`, `node.child_count()`, `node.child(i)`,
`node.start_byte()`) and the parser takes `str`, not `bytes`. That's
non-standard versus py-tree-sitter, so all access goes through the small
helpers below rather than the usual `.type` / `.children` attributes.

Scope is deliberately per-file (functions + the calls inside them), not full
cross-file call resolution — same depth as the Python path for non-imported
calls. Good enough to give the Root Cause Agent real structure in any of the
supported languages.
"""

from __future__ import annotations

# Extension -> tree-sitter language name.
EXT_LANG = {
    ".java": "java",
    ".c": "c",
    ".h": "c",          # ambiguous; treat bare .h as C (C++ headers use .hpp/.hh/.hxx)
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
    ".hpp": "cpp", ".hh": "cpp", ".hxx": "cpp",
    ".cs": "csharp",
}

# Per-language node kinds for "a function/method definition" and "a call".
_DEF_KINDS = {
    "java": {"method_declaration", "constructor_declaration"},
    "c": {"function_definition"},
    "cpp": {"function_definition"},
    "csharp": {"method_declaration", "constructor_declaration", "local_function_statement"},
}
_CALL_KINDS = {
    "java": {"method_invocation"},
    "c": {"call_expression"},
    "cpp": {"call_expression"},
    "csharp": {"invocation_expression"},
}
_IDENT_KINDS = {"identifier", "field_identifier", "qualified_identifier", "type_identifier"}

_parsers: dict = {}


def _get_parser(lang: str):
    """Lazily load + cache a parser. Imported here (not at module top) so the
    rest of the project still works if tree-sitter isn't installed and only
    Python repos are used."""
    if lang not in _parsers:
        from tree_sitter_language_pack import get_parser
        _parsers[lang] = get_parser(lang)
    return _parsers[lang]


# --- node helpers (this binding exposes everything as methods) ---
def _kids(n):
    return [n.child(i) for i in range(n.child_count())]


def _walk(n):
    yield n
    for c in _kids(n):
        yield from _walk(c)


def _text(n, src_bytes: bytes) -> str:
    return src_bytes[n.start_byte():n.end_byte()].decode("utf-8", "replace")


def _line(n, src_bytes: bytes) -> int:
    return src_bytes[:n.start_byte()].count(b"\n") + 1


def _def_name(node, src_bytes: bytes) -> str:
    """The function/method name. Tries the 'name' field first (Java/C#),
    then walks the declarator chain (C/C++), then falls back to the first
    identifier descendant."""
    nm = node.child_by_field_name("name")
    if nm is not None:
        return _text(nm, src_bytes)
    decl = node.child_by_field_name("declarator")
    seen = 0
    while decl is not None and seen < 6:
        if decl.kind() in _IDENT_KINDS:
            return _text(decl, src_bytes)
        decl = decl.child_by_field_name("declarator")
        seen += 1
    for d in _walk(node):
        if d.kind() in _IDENT_KINDS:
            return _text(d, src_bytes)
    return "<anonymous>"


def _call_name(node, src_bytes: bytes) -> str:
    """The called name. For a member call like obj.method() we want the
    *last* identifier (the method), not the receiver — so we return the last
    identifier found in the callee expression."""
    fn = node.child_by_field_name("function") or node.child_by_field_name("name") or node
    last = None
    for d in _walk(fn):
        if d.kind() in _IDENT_KINDS:
            last = d
    return _text(last, src_bytes) if last is not None else "<call>"


def extract_structure(source: str, lang: str) -> dict:
    """
    Parses `source` (already-read file text) in `lang` and returns:
        {"imports": [], "functions": {name: {"line": int, "calls": [str, ...]}}}

    `imports` is left empty for tree-sitter languages: cross-file resolution
    is Python-specific (see call_graph.trace_call_path), so non-Python traces
    cover the seeded files' structure rather than following imports. Returns
    None if parsing blows up, so callers can skip the file like the Python
    path does.
    """
    def_kinds = _DEF_KINDS.get(lang)
    call_kinds = _CALL_KINDS.get(lang, set())
    if not def_kinds:
        return None

    try:
        parser = _get_parser(lang)
        src_bytes = source.encode("utf-8")
        root = parser.parse(source).root_node()
    except Exception:
        return None

    functions: dict = {}
    for node in _walk(root):
        if node.kind() in def_kinds:
            calls = [_call_name(c, src_bytes) for c in _walk(node) if c.kind() in call_kinds]
            name = _def_name(node, src_bytes)
            # If the same name appears twice (overloads), keep the first; the
            # exact line matters less than having the symbol present.
            functions.setdefault(name, {"line": _line(node, src_bytes), "calls": calls})

    return {"imports": [], "functions": functions}


if __name__ == "__main__":
    # Quick manual test — run with: python -m analyzers.treesitter_nav
    samples = {
        "java": "class A { void foo() { bar(); this.baz(); } }",
        "c": "int foo(int x){ bar(); return 0; }",
        "cpp": "int foo(int x){ bar(); obj.baz(); return 0; }",
        "csharp": "class A { void Foo() { Bar(); this.Baz(); } }",
    }
    for lang, code in samples.items():
        print(f"{lang:7}: {extract_structure(code, lang)}")
