"""Streamlit tester for the ReHarness agent."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
import streamlit as st

_REHARNESS_ROOT = Path(__file__).resolve().parent.parent
_AGENTS_ROOT = _REHARNESS_ROOT.parent
for _path in (_AGENTS_ROOT, _REHARNESS_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from agents_utils.env import load_agents_env
from reharness_utils.message_content import extract_visible_assistant_text

load_agents_env()

AGENT_URL = os.environ.get("REHARNESS_AGENT_URL", "http://localhost:8080").rstrip("/")


def _stream_ndjson(client: httpx.Client, method: str, url: str, **kwargs: Any) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with client.stream(method, url, **kwargs) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            events.append(json.loads(line))
    return events


def _reset_session_chat() -> None:
    st.session_state.messages = []
    st.session_state.task_id = None
    st.session_state.phase = "idle"


def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(page_title="ReHarness tester", layout="centered")
    st.title("ReHarness agent tester")
    st.caption(f"Agent: `{AGENT_URL}`")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "task_id" not in st.session_state:
        st.session_state.task_id = None
    if "phase" not in st.session_state:
        st.session_state.phase = "idle"

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Task or feedback…")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        batch_events: list[dict[str, Any]] = []
        try:
            with httpx.Client(timeout=600.0) as client:
                if st.session_state.phase in ("idle", "complete"):
                    st.session_state.phase = "running"
                    batch_events = _stream_ndjson(
                        client,
                        "POST",
                        f"{AGENT_URL}/tasks/stream",
                        json={"task": prompt},
                    )
                elif st.session_state.phase == "awaiting_feedback" and st.session_state.task_id:
                    st.session_state.phase = "running"
                    batch_events = _stream_ndjson(
                        client,
                        "POST",
                        f"{AGENT_URL}/tasks/{st.session_state.task_id}/feedback/stream",
                        json={"feedback": prompt},
                    )
                else:
                    st.error("Start a new task from the sidebar.")
                    st.stop()
        except httpx.HTTPError as exc:
            st.error(f"Agent request failed: {exc}")
            st.stop()

        for event in batch_events:
            if event.get("type") == "task":
                st.session_state.task_id = event.get("task_id")
            if event.get("type") == "interrupt":
                st.session_state.phase = "awaiting_feedback"
            if event.get("type") == "complete":
                st.session_state.phase = "complete"

        assistant_text = ""
        for event in reversed(batch_events):
            if event.get("type") == "llm_message" and event.get("text"):
                assistant_text = str(event["text"])
                break
            if event.get("type") == "interrupt":
                payload = event.get("payload") or {}
                if isinstance(payload, dict) and payload.get("prompt"):
                    assistant_text = extract_visible_assistant_text(payload["prompt"])
                break

        if assistant_text:
            st.session_state.messages.append({"role": "assistant", "content": assistant_text})
            with st.chat_message("assistant"):
                st.markdown(assistant_text)

        st.rerun()

    with st.sidebar:
        st.markdown("### Session")
        if st.button("New task"):
            _reset_session_chat()
            st.rerun()

        st.markdown("### Reflection bank")
        st.caption("Clears all Qdrant reflections for a fresh experiment.")
        confirm = st.checkbox("Confirm wipe reflection bank")
        if st.button("Reset Qdrant bank", disabled=not confirm):
            try:
                with httpx.Client(timeout=60.0) as client:
                    response = client.post(f"{AGENT_URL}/memory/reset")
                    response.raise_for_status()
                _reset_session_chat()
                st.success("Reflection bank cleared.")
                st.rerun()
            except httpx.HTTPError as exc:
                st.error(f"Reset failed: {exc}")


if __name__ == "__main__":
    main()
