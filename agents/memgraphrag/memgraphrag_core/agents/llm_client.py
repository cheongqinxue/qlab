"""Shared LLM client for MemGraphRAG agents (re-export)."""

from agents_utils.llm.openai_chat import create_chat_model, default_chat_model_name

__all__ = ["create_chat_model", "default_chat_model_name"]
