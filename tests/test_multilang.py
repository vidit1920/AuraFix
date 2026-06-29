"""
test_multilang.py — deterministic, offline tests for multi-language support.

Covers the three language-aware layers without needing any non-Python
toolchain installed:
  - navigation: tree-sitter extracts functions + calls for Java/C/C++/C#
  - indexing:   the RAG chunker now indexes those source types
  - validation: detect_runner picks the right build tool per project marker

The actual *running* of non-Python test suites needs that language's
toolchain (mvn/gradle/dotnet/cmake) and is verified on a real repo — here we
only assert the detection + parsing logic.
"""

import os
import tempfile

from analyzers.call_graph import build_repo_structure, SUPPORTED_EXTS
from analyzers.treesitter_nav import extract_structure
from rag.chunker import chunk_repository
from validators.test_validator import detect_runner


def _mkrepo(files: dict) -> str:
    d = tempfile.mkdtemp(prefix="aurafix_ml_")
    for rel, content in files.items():
        path = os.path.join(d, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(rel) else None
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    return d


# --------------------------------------------------------------------------
# Navigation (tree-sitter)
# --------------------------------------------------------------------------
def test_treesitter_extracts_java():
    s = extract_structure("class A { int add(int a){ return helper(a); } int helper(int x){ return x; } }", "java")
    assert set(s["functions"]) == {"add", "helper"}
    assert "helper" in s["functions"]["add"]["calls"]


def test_treesitter_extracts_c_and_cpp():
    c = extract_structure("int main(){ compute(); return 0; } int compute(){ return 1; }", "c")
    assert set(c["functions"]) == {"main", "compute"}
    assert "compute" in c["functions"]["main"]["calls"]

    cpp = extract_structure("int run(){ obj.step(); return 0; }", "cpp")
    assert "step" in cpp["functions"]["run"]["calls"]  # member call -> method name


def test_treesitter_extracts_csharp():
    s = extract_structure("class P { void Run(){ Step(); } void Step(){} }", "csharp")
    assert set(s["functions"]) == {"Run", "Step"}
    assert "Step" in s["functions"]["Run"]["calls"]


def test_build_repo_structure_is_multilanguage():
    repo = _mkrepo({
        "A.java": "class A { void f(){ g(); } void g(){} }",
        "m.c": "int main(){ return 0; }",
        "P.cs": "class P { void R(){} }",
        "x.py": "def foo():\n    bar()\n",
        "notes.md": "# ignored",  # non-source -> skipped
    })
    s = build_repo_structure(repo)
    assert "A.java" in s and "m.c" in s and "P.cs" in s and "x.py" in s
    assert "notes.md" not in s
    assert "f" in s["A.java"]["functions"]


# --------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------
def test_chunker_indexes_target_languages():
    assert {".java", ".c", ".cpp", ".cs"} <= SUPPORTED_EXTS
    repo = _mkrepo({
        "Calc.java": "public class Calc { int add(int a,int b){ return a+b; } }",
        "util.cpp": "int square(int x){ return x*x; }",
        "readme.txt": "not indexed",
    })
    chunks = chunk_repository(repo)
    files = {c["filename"] for c in chunks}
    assert "Calc.java" in files
    assert "util.cpp" in files
    assert "readme.txt" not in files


# --------------------------------------------------------------------------
# Validation runner detection
# --------------------------------------------------------------------------
def test_detect_runner_python_default():
    repo = _mkrepo({"main.py": "print(1)"})
    assert detect_runner(repo)["language"] == "Python"


def test_detect_runner_java_maven_and_gradle():
    # Detection is by dominant language, so a realistic repo has source files
    # of that language alongside the build marker.
    assert detect_runner(_mkrepo({"pom.xml": "<project/>", "App.java": "class App {}"}))["language"] == "Java"
    assert detect_runner(_mkrepo({"build.gradle": "plugins {}", "App.java": "class App {}"}))["language"] == "Java"


def test_detect_runner_csharp_and_c():
    assert detect_runner(_mkrepo({"App.csproj": "<Project/>", "App.cs": "class App {}"}))["language"] == "C#"
    assert detect_runner(_mkrepo({"CMakeLists.txt": "project(x)", "m.c": "int main(){return 0;}"}))["language"] == "C/C++"
    assert detect_runner(_mkrepo({"Makefile": "test:\n\techo ok", "m.c": "int main(){return 0;}"}))["language"] == "C/C++"


def test_detect_runner_python_with_makefile_stays_python():
    # Regression: a Python repo that ships a Makefile must NOT be seen as C
    # (the bug found by testing against psf/requests).
    repo = _mkrepo({"app.py": "print(1)", "util.py": "x=1", "Makefile": "test:\n\tpytest"})
    assert detect_runner(repo)["language"] == "Python"
