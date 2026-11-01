"""Isolated HumanEval program runner (stdin program, exit 0 on passing check())."""

from __future__ import annotations

import sys
from typing import Any

from model_eval_core.coding.safety import CodeSafetyError, validate_program_safety


def _restricted_import(
    name: str,
    globals_: dict[str, Any] | None = None,
    locals_: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    """Allow only ``typing`` imports for HumanEval annotation stubs."""
    if level != 0:
        raise ImportError("relative import not allowed")
    root = name.split(".")[0]
    if root != "typing":
        raise ImportError(f"import not allowed: {name}")
    import typing  # noqa: PLC0415

    return typing


def _safe_builtins() -> dict[str, Any]:
    """Minimal builtins sufficient for HumanEval tasks and assert-based tests."""
    allowed = {
        "__import__": _restricted_import,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "pow": pow,
        "range": range,
        "reversed": reversed,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "map": map,
        "filter": filter,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "type": type,
        "True": True,
        "False": False,
        "None": None,
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "IndexError": IndexError,
        "KeyError": KeyError,
        "AssertionError": AssertionError,
    }
    return {"__builtins__": allowed}


def _run_program(source: str) -> None:
    """Execute validated program in a restricted global namespace."""
    report = validate_program_safety(source)
    if not report.ok:
        raise CodeSafetyError(report.reason or "safety validation failed")

    namespace: dict[str, Any] = _safe_builtins()
    exec(compile(source, "<humaneval>", "exec"), namespace, namespace)  # noqa: S102


def main() -> None:
    """Read program from stdin, run HumanEval check(), exit 0 on success."""
    source = sys.stdin.read()
    try:
        _run_program(source)
    except CodeSafetyError as exc:
        print(f"safety: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except AssertionError as exc:
        print(f"assert: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except Exception as exc:  # noqa: BLE001 — surface failures to parent process
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    raise SystemExit(0)


if __name__ == "__main__":
    main()
