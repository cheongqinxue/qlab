"""Token throughput aggregation from JSONL."""

from __future__ import annotations

from model_eval_core.metrics import aggregate_suite_tokens, tokens_from_messages


class TestTokenStats:
    """Suite-level tok/s estimates."""

    def test_tokens_from_messages_sums_ai_usage(self) -> None:
        messages = [
            {"type": "human", "content": "hi"},
            {"type": "ai", "content": "a", "usage": {"completion_tokens": 10, "prompt_tokens": 5}},
            {"type": "ai", "content": "b", "usage": {"completion_tokens": 20, "prompt_tokens": 3}},
        ]
        comp, prompt = tokens_from_messages(messages)
        assert comp == 30
        assert prompt == 8

    def test_completion_tok_per_s_uses_wall_s(self) -> None:
        rows = [
            {
                "messages": [
                    {"type": "ai", "usage": {"completion_tokens": 100, "prompt_tokens": 10}},
                ],
                "latency_s": 5.0,
            },
        ]
        stats = aggregate_suite_tokens(rows, wall_s=10.0)
        assert stats["completion_tokens"] == 100
        assert stats["completion_tok_per_s"] == 10.0
        assert stats["denominator_source"] == "wall_s"
