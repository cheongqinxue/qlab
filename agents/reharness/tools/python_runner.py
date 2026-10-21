"""Run Python in a subprocess inside the agent container."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from langchain_core.tools import tool

_MAX_OUTPUT_CHARS = 32_000
_DEFAULT_TIMEOUT = 30


@tool
def run_python(code: str) -> str:
    """Execute Python 3 code in a subprocess and return stdout/stderr.

    Args:
        code: Python source to run.

    Returns:
        Combined execution output and exit code.
    """
    timeout = int(os.environ.get("PYTHON_RUN_TIMEOUT_SECONDS", str(_DEFAULT_TIMEOUT)))
    with tempfile.TemporaryDirectory(prefix="reharness_") as tmp:
        script = Path(tmp) / "main.py"
        script.write_text(code, encoding="utf-8")
        try:
            completed = subprocess.run(
                ["python", str(script)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return f"exit_code=-1\nstderr:\nExecution timed out after {timeout}s"

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if len(stdout) > _MAX_OUTPUT_CHARS:
        stdout = stdout[:_MAX_OUTPUT_CHARS] + "\n... (truncated)"
    if len(stderr) > _MAX_OUTPUT_CHARS:
        stderr = stderr[:_MAX_OUTPUT_CHARS] + "\n... (truncated)"
    parts = [f"exit_code={completed.returncode}"]
    if stdout:
        parts.append(f"stdout:\n{stdout}")
    if stderr:
        parts.append(f"stderr:\n{stderr}")
    return "\n".join(parts)
