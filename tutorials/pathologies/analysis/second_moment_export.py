"""Export Adam ``exp_avg_sq`` heatmaps from Trainer checkpoints."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForMaskedLM, Trainer, TrainingArguments

_MAX_HEATMAP_SIDE = 400


def discover_checkpoints(checkpoints_dir: Path) -> list[tuple[int, Path]]:
    """Return sorted ``(step, path)`` for ``checkpoint-step-*`` directories."""
    found: list[tuple[int, Path]] = []
    for child in checkpoints_dir.iterdir():
        if not child.is_dir():
            continue
        match = re.fullmatch(r"checkpoint-step-(\d+)", child.name)
        if match:
            found.append((int(match.group(1)), child))
    return sorted(found, key=lambda item: item[0])


def downsample_2d(matrix: torch.Tensor, max_side: int) -> torch.Tensor:
    """Average-pool a 2D tensor until both sides are at most ``max_side``."""
    if matrix.dim() != 2:
        msg = f"Expected 2D tensor, got shape {tuple(matrix.shape)}"
        raise ValueError(msg)
    out = matrix
    while out.shape[0] > max_side or out.shape[1] > max_side:
        h, w = out.shape
        h_even = h - (h % 2)
        w_even = w - (w % 2)
        out = out[:h_even, :w_even]
        out = out.reshape(h_even // 2, 2, w_even // 2, 2).mean(dim=(1, 3))
    return out


def moment_heatmap(exp_avg_sq: torch.Tensor, max_side: int) -> dict[str, Any]:
    """Build a JSON-serializable sqrt(second moment) grid for Plotly."""
    values = exp_avg_sq.detach().float().sqrt()
    if values.dim() == 1:
        values = values.unsqueeze(0)
    elif values.dim() > 2:
        values = values.reshape(values.shape[0], -1)
    values = downsample_2d(values, max_side)
    z = values.cpu().tolist()
    return {
        "rows": len(z),
        "cols": len(z[0]) if z else 0,
        "z": z,
    }


def build_param_tree(param_names: list[str]) -> list[dict[str, Any]]:
    """Turn dotted parameter names into a nested navigation tree."""

    def insert(root: dict[str, Any], parts: list[str], full_name: str) -> None:
        if not parts:
            return
        head, *tail = parts
        if head not in root:
            root[head] = {"label": head, "children": {}, "params": []}
        if tail:
            insert(root[head]["children"], tail, full_name)
        else:
            root[head]["params"].append(full_name)

    roots: dict[str, Any] = {}
    for name in param_names:
        insert(roots, name.split("."), name)

    def to_list(node_map: dict[str, Any]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for key in sorted(node_map):
            node = node_map[key]
            children = to_list(node["children"])
            for param_name in sorted(node["params"]):
                short = param_name.split(".")[-1]
                children.append(
                    {
                        "id": param_name,
                        "label": short,
                        "param": param_name,
                        "leaf": True,
                    }
                )
            items.append(
                {
                    "id": key,
                    "label": node["label"],
                    "leaf": False,
                    "children": children,
                }
            )
        return items

    return to_list(roots)


def load_moments(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    """Load ``exp_avg_sq`` per parameter name from a Trainer checkpoint."""
    model = AutoModelForMaskedLM.from_pretrained(str(checkpoint_dir))
    args = TrainingArguments(output_dir=str(checkpoint_dir.parent), save_only_model=False)
    trainer = Trainer(model=model, args=args)
    trainer.create_optimizer()
    optimizer_state = torch.load(
        checkpoint_dir / "optimizer.pt",
        map_location="cpu",
        weights_only=True,
    )
    trainer.optimizer.load_state_dict(optimizer_state)
    moments: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        state = trainer.optimizer.state[param]
        if "exp_avg_sq" in state:
            moments[name] = state["exp_avg_sq"].detach().cpu()
        else:
            moments[name] = torch.zeros_like(param, dtype=torch.float32)
    return moments


def export_viewer_data(checkpoints_dir: Path, output_dir: Path) -> Path:
    """Write ``data.json`` for the second-moment viewer.

    Args:
        checkpoints_dir: Directory containing ``checkpoint-step-*`` folders.
        output_dir: Directory for ``data.json`` (``index.html`` should live here too).

    Returns:
        Path to the written ``data.json``.
    """
    steps_and_paths = discover_checkpoints(checkpoints_dir)
    if not steps_and_paths:
        msg = f"No checkpoint-step-* directories under {checkpoints_dir}"
        raise FileNotFoundError(msg)

    checkpoint_meta = [
        {"id": f"step-{step}", "step": step, "label": f"step {step}"}
        for step, _path in steps_and_paths
    ]

    all_tensors: dict[str, dict[str, Any]] = {}
    param_names: list[str] | None = None

    for step, path in steps_and_paths:
        moments = load_moments(path)
        if param_names is None:
            param_names = sorted(moments.keys())
        ckpt_id = f"step-{step}"
        all_tensors[ckpt_id] = {
            name: moment_heatmap(moments[name], _MAX_HEATMAP_SIDE) for name in param_names
        }

    payload = {
        "checkpoints": checkpoint_meta,
        "tree": build_param_tree(param_names or []),
        "tensors": all_tensors,
        "value_label": "sqrt(exp_avg_sq)",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "data.json"
    data_path.write_text(json.dumps(payload), encoding="utf-8")

    template_index = Path(__file__).resolve().parents[1] / "second_moment_viewer" / "index.html"
    if template_index.is_file():
        shutil.copy2(template_index, output_dir / "index.html")

    return data_path
