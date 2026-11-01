"""Static safety checks for model-generated Python before HumanEval execution."""

from __future__ import annotations

import ast
from dataclasses import dataclass

MAX_PROGRAM_CHARS = 65_536
MAX_AST_NODES = 8_000

# Root modules allowed in ``import x`` / ``from x import y`` (HumanEval subset).
ALLOWED_IMPORT_ROOTS = frozenset({"typing"})

# Bare names that must not appear as call targets or loads.
FORBIDDEN_NAMES = frozenset(
    {
        "__builtins__",
        "__import__",
        "__loader__",
        "__spec__",
        "eval",
        "exec",
        "compile",
        "open",
        "breakpoint",
        "input",
        "globals",
        "locals",
        "vars",
        "dir",
        "help",
        "memoryview",
        "bytearray",
        "bytes",
    }
)

# Attribute calls blocked on any receiver (e.g. os.system, pathlib.Path.write_text).
FORBIDDEN_ATTR_CALLS = frozenset(
    {
        "system",
        "popen",
        "spawn",
        "fork",
        "execv",
        "execve",
        "remove",
        "unlink",
        "rmdir",
        "removedirs",
        "rename",
        "replace",
        "chmod",
        "chown",
        "kill",
        "killpg",
        "write",
        "write_text",
        "write_bytes",
        "read_text",
        "read_bytes",
        "open",
        "connect",
        "send",
        "sendall",
        "urlopen",
        "request",
        "load",
        "loads",
        "dump",
        "dumps",
        "getattr",
        "setattr",
        "delattr",
        "subclasses",
        "mro",
        "__import__",
    }
)

FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "socket",
        "pathlib",
        "shutil",
        "pickle",
        "ctypes",
        "multiprocessing",
        "threading",
        "asyncio",
        "importlib",
        "builtins",
        "runpy",
        "pty",
        "fcntl",
        "signal",
        "tempfile",
        "glob",
        "io",
        "code",
        "codeop",
        "marshal",
        "shelve",
        "sqlite3",
        "http",
        "urllib",
        "ftplib",
        "smtplib",
        "webbrowser",
        "tkinter",
        "pip",
        "site",
        "ssl",
        "resource",
    }
)


class CodeSafetyError(ValueError):
    """Raised when generated code fails static safety validation."""


@dataclass(frozen=True)
class CodeSafetyReport:
    """Outcome of static validation."""

    ok: bool
    reason: str | None = None


class _SafetyVisitor(ast.NodeVisitor):
    """Reject constructs that are unnecessary for HumanEval and high-risk locally."""

    def __init__(self) -> None:
        self._node_count = 0

    def generic_visit(self, node: ast.AST) -> None:
        self._node_count += 1
        if self._node_count > MAX_AST_NODES:
            raise CodeSafetyError("program exceeds AST node limit")
        super().generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in FORBIDDEN_IMPORT_ROOTS:
                raise CodeSafetyError(f"import blocked: {alias.name}")
            if root not in ALLOWED_IMPORT_ROOTS:
                raise CodeSafetyError(f"import not allowed: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None:
            raise CodeSafetyError("relative import not allowed")
        root = node.module.split(".")[0]
        if root in FORBIDDEN_IMPORT_ROOTS:
            raise CodeSafetyError(f"import blocked: {node.module}")
        if root not in ALLOWED_IMPORT_ROOTS:
            raise CodeSafetyError(f"import not allowed: {node.module}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FORBIDDEN_NAMES:
            raise CodeSafetyError(f"forbidden name: {node.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            node.attr.startswith("__")
            and node.attr.endswith("__")
            and node.attr
            not in {
                "__init__",
                "__name__",
            }
        ):
            raise CodeSafetyError(f"dunder attribute blocked: {node.attr}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        target = node.func
        if isinstance(target, ast.Name) and target.id in FORBIDDEN_NAMES:
            raise CodeSafetyError(f"forbidden call: {target.id}")
        if isinstance(target, ast.Attribute) and target.attr in FORBIDDEN_ATTR_CALLS:
            raise CodeSafetyError(f"forbidden method call: {target.attr}")
        self.generic_visit(node)


def validate_program_safety(source: str) -> CodeSafetyReport:
    """Parse and statically validate a HumanEval program before subprocess execution.

    Args:
        source: Full Python source (solution + tests + check call).

    Returns:
        CodeSafetyReport with ``ok=True`` when safe to run in the isolated runner.

    Raises:
        CodeSafetyError: When validation fails (also reported via ``ok=False`` in callers
            that catch and convert).
    """
    if len(source) > MAX_PROGRAM_CHARS:
        return CodeSafetyReport(ok=False, reason="program exceeds size limit")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return CodeSafetyReport(ok=False, reason=f"syntax error: {exc.msg}")

    try:
        _SafetyVisitor().visit(tree)
    except CodeSafetyError as exc:
        return CodeSafetyReport(ok=False, reason=str(exc))

    return CodeSafetyReport(ok=True)
