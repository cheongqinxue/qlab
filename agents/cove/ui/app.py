"""Streamlit tester for the CoVe agent (four-step trace)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import streamlit as st

_COVE_ROOT = Path(__file__).resolve().parent.parent
_AGENTS_ROOT = _COVE_ROOT.parent
for _path in (_AGENTS_ROOT, _COVE_ROOT):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from agents_utils.env import load_agents_env

load_agents_env()

AGENT_URL = os.environ.get("COVE_AGENT_URL", "http://localhost:8083").rstrip("/")


def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(page_title="CoVe tester", layout="wide")
    st.title("Chain-of-Verification tester")
    st.caption(f"Agent `{AGENT_URL}`")

    query = st.text_area(
        "Query",
        value="When did the Mexican–American War start and end?",
        height=100,
    )
    use_toy = st.checkbox("Use toy fixture (query_id=mexican_american_war)", value=True)

    if st.button("Run CoVe", type="primary"):
        body: dict[str, str] = {}
        if use_toy:
            body["query_id"] = "mexican_american_war"
            if query.strip():
                body["query"] = query.strip()
        elif query.strip():
            body["query"] = query.strip()
        else:
            st.error("Enter a query or enable the toy fixture.")
            return

        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(f"{AGENT_URL}/verify", json=body)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            st.error(f"Request failed: {exc}")
            return

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("1. Baseline")
            st.write(data.get("baseline_response", ""))
            st.subheader("2. Verification questions")
            for q in data.get("verification_questions", []):
                st.markdown(f"- {q}")
        with col2:
            st.subheader("3. Isolated answers")
            for pair in data.get("verification_pairs", []):
                st.markdown(f"**Q:** {pair.get('question', '')}")
                st.caption(pair.get("answer", ""))
            st.subheader("4. Final verified")
            st.success(data.get("final_response", ""))


if __name__ == "__main__":
    main()
