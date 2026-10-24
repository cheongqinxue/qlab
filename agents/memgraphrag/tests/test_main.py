"""Tests for ``main`` entrypoint helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestRun:
    """Uvicorn launcher."""

    def test_run_invokes_uvicorn(self) -> None:
        with patch("uvicorn.run") as mock_run:
            import main

            main.run()
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == "main:app"
