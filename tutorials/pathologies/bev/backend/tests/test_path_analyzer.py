"""Tests for BEV metric path analysis."""

import json

from bev.catalog.path_analyzer import analyze_path, infer_dtype


class TestInferDtype:
    def test_float(self) -> None:
        assert infer_dtype(1.5) == "float"

    def test_int(self) -> None:
        assert infer_dtype(3) == "int"

    def test_null(self) -> None:
        assert infer_dtype(None) == "null"


class TestAnalyzePath:
    def test_layer_token(self) -> None:
        payload = json.loads(analyze_path("train/grad_norm/layer.12"))
        assert payload["parts"] == ["train", "grad_norm", "layer.12"]
        layer_token = payload["tokens"][-1]
        assert layer_token["kind"] == "layer"
        assert layer_token["index"] == 12

    def test_head_token(self) -> None:
        payload = json.loads(analyze_path("train/attention_max_weight/module/head.1"))
        head_token = payload["tokens"][-1]
        assert head_token["kind"] == "head"
        assert head_token["index"] == 1
