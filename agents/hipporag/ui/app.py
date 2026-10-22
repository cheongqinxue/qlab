"""Streamlit tester for the HippoRAG agent."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
import streamlit as st

_HIPPORAG_ROOT = Path(__file__).resolve().parent.parent
_AGENTS_ROOT = _HIPPORAG_ROOT.parent
for _path in (_AGENTS_ROOT, _HIPPORAG_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from agents_utils.env import load_agents_env

load_agents_env()

AGENT_URL = os.environ.get("HIPPORAG_AGENT_URL", "http://localhost:8081").rstrip("/")
NEO4J_BROWSER = os.environ.get(
    "HIPPORAG_NEO4J_BROWSER_URL",
    "http://localhost:7474",
).rstrip("/")

TOY_QUESTIONS = [
    "Which Stanford professor works on the neuroscience of Alzheimer's?",
    "Which Stanford professor works on Alzheimer's?",
]


def _get_json(client: httpx.Client, path: str) -> dict[str, Any]:
    response = client.get(f"{AGENT_URL}{path}")
    response.raise_for_status()
    return response.json()


def _post_json(client: httpx.Client, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    response = client.post(f"{AGENT_URL}{path}", json=body or {})
    response.raise_for_status()
    return response.json()


def _render_health(client: httpx.Client) -> None:
    try:
        health = _get_json(client, "/health")
    except httpx.HTTPError as exc:
        st.error(f"Agent unreachable: {exc}")
        return
    cols = st.columns(5)
    cols[0].metric("Passages", health.get("passages", "—"))
    cols[1].metric("Entities", health.get("entities", "—"))
    cols[2].metric("Triples", health.get("triples", "—"))
    cols[3].metric("REL edges", health.get("rel_edges", "—"))
    cols[4].metric("Synonym", health.get("synonym_edges", "—"))


def _render_hits(hits: list[dict[str, Any]]) -> None:
    if not hits:
        st.info("No passages returned.")
        return
    for rank, hit in enumerate(hits, start=1):
        score = hit.get("score", 0.0)
        pid = hit.get("passage_id", "?")
        text = hit.get("text", "")
        st.markdown(f"**{rank}. `{pid}`** — score `{score:.4f}`")
        st.caption(text)


def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(page_title="HippoRAG tester", layout="wide")
    st.title("HippoRAG tester")
    st.caption(f"Agent `{AGENT_URL}` · Neo4j Browser [{NEO4J_BROWSER}]({NEO4J_BROWSER})")

    if "top_k" not in st.session_state:
        st.session_state.top_k = 5

    with st.sidebar:
        st.markdown("### Index")
        st.caption("Toy corpus uses fixture OpenIE (no LLM cost).")
        if st.button("Bootstrap toy corpus"):
            try:
                with httpx.Client(timeout=120.0) as client:
                    result = _post_json(client, "/corpus/bootstrap-toy")
                st.success(
                    f"Indexed {result.get('indexed_passages', 0)} passages, "
                    f"{result.get('entities', 0)} entities."
                )
            except httpx.HTTPError as exc:
                st.error(f"Bootstrap failed: {exc}")

        if st.button("Re-run index (pending passages)"):
            try:
                with httpx.Client(timeout=120.0) as client:
                    result = _post_json(client, "/index")
                st.success(f"Indexed {result.get('indexed_passages', 0)} passage(s).")
            except httpx.HTTPError as exc:
                st.error(f"Index failed: {exc}")

        confirm_reset = st.checkbox("Confirm graph reset")
        if st.button("Reset graph", disabled=not confirm_reset):
            try:
                with httpx.Client(timeout=60.0) as client:
                    _post_json(client, "/reset")
                st.success("Graph cleared (passage bodies remain staged).")
            except httpx.HTTPError as exc:
                st.error(f"Reset failed: {exc}")

        st.markdown("### Retrieval")
        st.session_state.top_k = st.slider("top_k", min_value=1, max_value=20, value=st.session_state.top_k)

        st.markdown("### Add passage")
        with st.form("add_passage"):
            new_id = st.text_input("Passage id", placeholder="p_custom")
            new_text = st.text_area("Text", height=100)
            submitted = st.form_submit_button("Stage passage")
            if submitted and new_id.strip() and new_text.strip():
                try:
                    with httpx.Client(timeout=60.0) as client:
                        _post_json(
                            client,
                            "/corpus/passages",
                            {"id": new_id.strip(), "text": new_text.strip()},
                        )
                    st.success(f"Staged `{new_id.strip()}`. Run **Re-run index** to extract.")
                except httpx.HTTPError as exc:
                    st.error(f"Staging failed: {exc}")

    with httpx.Client(timeout=120.0) as client:
        st.markdown("### Graph stats")
        _render_health(client)

    tab_retrieve, tab_ask = st.tabs(["Retrieve (PPR)", "Ask (RAG)"])

    with tab_retrieve:
        example = st.selectbox("Example question", TOY_QUESTIONS, index=0)
        query = st.text_area("Query", value=example, height=80)
        if st.button("Retrieve", type="primary"):
            try:
                with httpx.Client(timeout=120.0) as client:
                    result = _post_json(
                        client,
                        "/retrieve",
                        {"query": query, "top_k": st.session_state.top_k},
                    )
                st.markdown("**Query entities**")
                st.code(json.dumps(result.get("query_entities", []), indent=2))
                st.markdown("**Seed entities (graph nodes)**")
                st.code(json.dumps(result.get("seed_entities", []), indent=2))
                st.markdown("**Ranked passages**")
                _render_hits(list(result.get("hits", [])))
            except httpx.HTTPError as exc:
                st.error(f"Retrieve failed: {exc}")

    with tab_ask:
        ask_query = st.text_area("Question", value=TOY_QUESTIONS[0], height=80, key="ask_query")
        if st.button("Ask", type="primary"):
            try:
                with httpx.Client(timeout=300.0) as client:
                    result = _post_json(
                        client,
                        "/ask",
                        {"query": ask_query, "top_k": st.session_state.top_k},
                    )
                st.markdown("**Answer**")
                st.markdown(result.get("answer", ""))
                st.markdown("**Supporting passages**")
                _render_hits(list(result.get("hits", [])))
            except httpx.HTTPError as exc:
                st.error(f"Ask failed: {exc}")


if __name__ == "__main__":
    main()
