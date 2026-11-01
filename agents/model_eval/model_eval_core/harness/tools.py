"""Deterministic tools for fair ReAct eval (same tool surface for every model)."""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from langchain_core.tools import tool

_SAFE_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval_expr(expression: str) -> float:
    """Evaluate a numeric expression with a restricted AST subset.

    Args:
        expression: Arithmetic expression string.

    Returns:
        Numeric result.

    Raises:
        ValueError: When the expression uses unsupported syntax.
    """
    node = ast.parse(expression.strip(), mode="eval")

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _SAFE_OPS:
            return float(_SAFE_OPS[type(n.op)](_eval(n.operand)))
        if isinstance(n, ast.BinOp) and type(n.op) in _SAFE_OPS:
            return float(_SAFE_OPS[type(n.op)](_eval(n.left), _eval(n.right)))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            name = n.func.id
            if name in {"sqrt", "sin", "cos", "log", "exp"}:
                args = [_eval(a) for a in n.args]
                fn = getattr(math, name)
                return float(fn(*args))
        msg = f"Unsupported expression: {expression}"
        raise ValueError(msg)

    return _eval(node)


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression (numbers, + - * / **, sqrt, sin, cos, log, exp)."""
    try:
        return str(_safe_eval_expr(expression))
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError) as exc:
        return f"error: {exc}"


@tool
def lookup_reference(key: str) -> str:
    """Look up a short fact from the eval knowledge base (keys like capital_france)."""
    kb = {
        "capital_france": "Paris",
        "speed_of_light_m_s": "299792458",
        "year_moon_landing": "1969",
        "author_1984": "George Orwell",
    }
    normalized = key.strip().lower().replace(" ", "_")
    return kb.get(normalized, "unknown_key")


@tool
def get_weather(city: str) -> str:
    """Return deterministic mock weather for supported eval cities."""
    table = {
        "paris": "12C, overcast, wind 8km/h",
        "tokyo": "22C, clear, wind 5km/h",
        "sydney": "18C, rain, wind 14km/h",
        "chicago": "4C, snow, wind 20km/h",
    }
    key = city.strip().lower()
    return table.get(key, "unknown_city")


@tool
def run_python_snippet(code: str) -> str:
    """Execute a tiny Python snippet for coding probes (no imports, print only)."""
    if "import" in code or "__" in code:
        return "error: imports and dunder access are blocked"
    safe_globals = {"__builtins__": {"print": print, "range": range, "len": len}}
    safe_locals: dict[str, Any] = {}
    try:
        exec(code, safe_globals, safe_locals)  # noqa: S102 — bounded eval sandbox
    except Exception as exc:  # noqa: BLE001 — surface tool errors to the agent
        return f"error: {exc}"
    if "result" in safe_locals:
        return str(safe_locals["result"])
    return "ok"


def default_eval_tools() -> list[Any]:
    """Return the standard tool list used for every model in this harness."""
    return [calculator, lookup_reference, get_weather, run_python_snippet]
