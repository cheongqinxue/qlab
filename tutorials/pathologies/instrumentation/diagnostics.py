"""Training diagnostic scalar functions."""

import re
from collections import defaultdict

import torch


def l2_norm(tensor: torch.Tensor) -> float:
    """Compute the L2 norm of a tensor."""
    return tensor.float().norm().item()


def global_grad_norm(parameters: list[torch.Tensor]) -> float:
    """Compute the L2 norm over all parameter gradients."""
    squared = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        squared += grad.square().sum().item()
    return squared**0.5


def max_param_grad_norm(model: torch.nn.Module) -> tuple[float, str]:
    """Return the largest per-parameter gradient norm and its name."""
    max_norm = 0.0
    max_name = ""
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        norm = param.grad.detach().float().norm().item()
        if norm > max_norm:
            max_norm = norm
            max_name = name
    return max_norm, max_name


def mean_update_to_weight_ratio(
    before: dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> float:
    """Mean ||ΔW|| / ||W|| over parameters present in ``before``."""
    ratios: list[float] = []
    for name, param in model.named_parameters():
        if name not in before:
            continue
        prior = before[name]
        weight_norm = prior.float().norm().item()
        if weight_norm == 0.0:
            continue
        delta_norm = (param.detach().float() - prior.float()).norm().item()
        ratios.append(delta_norm / weight_norm)
    if not ratios:
        return 0.0
    return sum(ratios) / len(ratios)


def group_mean_update_to_weight_ratio(
    before: dict[str, torch.Tensor],
    model: torch.nn.Module,
    predicate,
) -> float:
    """Mean ||ΔW|| / ||W|| over parameters whose names satisfy ``predicate``."""
    ratios: list[float] = []
    for name, param in model.named_parameters():
        if name not in before or not predicate(name):
            continue
        prior = before[name]
        weight_norm = prior.float().norm().item()
        if weight_norm == 0.0:
            continue
        delta_norm = (param.detach().float() - prior.float()).norm().item()
        ratios.append(delta_norm / weight_norm)
    if not ratios:
        return 0.0
    return sum(ratios) / len(ratios)


def tensor_rms(tensor: torch.Tensor) -> float:
    """Root-mean-square over all elements."""
    values = tensor.float()
    return values.square().mean().sqrt().item()


def logit_entropy(logits: torch.Tensor) -> float:
    """Mean per-example entropy of the softmax distribution."""
    probs = torch.softmax(logits.float(), dim=-1)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
    return entropy.mean().item()


def attention_entropy(attention: torch.Tensor) -> float:
    """Mean row-wise entropy of an attention weight tensor."""
    probs = attention.float()
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
    return entropy.mean().item()


def attention_max_weight(attention: torch.Tensor) -> float:
    """Mean peak attention mass per query row (1.0 = fully collapsed)."""
    return attention.float().amax(dim=-1).mean().item()


def attention_per_head_metrics(attention: torch.Tensor) -> dict[str, float]:
    """Per-head peak attention and an imbalance ratio (max/min per-head peak).

    Args:
        attention: Attention weights ``(batch, heads, seq, seq)`` or lower rank.

    Returns:
        Mapping with ``_imbalance`` plus ``head.{index}`` peak weights. Empty if not 4D.
    """
    probs = attention.float()
    if probs.ndim != 4:
        return {}
    per_head_peak = probs.amax(dim=-1).mean(dim=(0, 2))
    metrics: dict[str, float] = {
        f"head.{index}": peak.item() for index, peak in enumerate(per_head_peak)
    }
    min_peak = per_head_peak.min().clamp_min(1e-8)
    metrics["_imbalance"] = (per_head_peak.max() / min_peak).item()
    return metrics


def _optimizer_adam_hyperparams(optimizer: torch.optim.Optimizer) -> tuple[float, float, float, float]:
    """Return ``(beta1, beta2, eps, lr)`` from the first Adam(W) param group."""
    group = optimizer.param_groups[0]
    betas = group.get("betas", (0.9, 0.999))
    return float(betas[0]), float(betas[1]), float(group.get("eps", 1e-8)), float(group["lr"])


def _bias_corrected_adam_moments(
    state: dict,
    beta1: float,
    beta2: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return bias-corrected ``(m_hat, v_hat)`` tensors for one parameter."""
    if "exp_avg" not in state or "exp_avg_sq" not in state:
        return None
    step = int(state["step"].item())
    if step < 1:
        return None
    bc1 = 1.0 - beta1**step
    bc2 = 1.0 - beta2**step
    m_hat = state["exp_avg"].float() / bc1
    v_hat = state["exp_avg_sq"].float() / bc2
    return m_hat, v_hat


def adam_pre_step_spike_metrics(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """SPAM-style gradient spike score using gradients **before** the optimizer step.

    Computes ``max(g^2 / (v_hat + eps))`` over trainable parameters with Adam state.

    Args:
        model: Model with ``param.grad`` populated.
        optimizer: Adam or AdamW instance.

    Returns:
        Keys ``adam_max_g2_over_v`` (empty if no grads).
    """
    beta1, beta2, eps, _lr = _optimizer_adam_hyperparams(optimizer)
    max_g2_over_v = 0.0
    found = False
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is None or not param.requires_grad:
                continue
            state = optimizer.state.get(param)
            moments = (
                _bias_corrected_adam_moments(state, beta1, beta2)
                if state is not None
                else None
            )
            if moments is None:
                if state is not None and "exp_avg_sq" in state:
                    v_hat = state["exp_avg_sq"].float()
                else:
                    v_hat = torch.zeros_like(param, dtype=torch.float32)
            else:
                _, v_hat = moments
            grad = param.grad.detach().float()
            if grad.shape != v_hat.shape:
                continue
            spike = (grad.square() / (v_hat + eps)).max().item()
            max_g2_over_v = max(max_g2_over_v, spike)
            found = True
    if not found:
        return {}
    return {"adam_max_g2_over_v": max_g2_over_v}


def adam_post_step_health_metrics(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    low_moment_eps_multiplier: float = 10.0,
    low_moment_median_ratio: float = 0.01,
    r_threshold: float = 1_000.0,
) -> dict[str, float]:
    """Adam fragility and effective-step severity after an optimizer step.

    Logs:

    - ``adam_low_moment_frac``: share of elements with
      ``sqrt(v_hat) < max(eps * mult, median_ratio * tensor_median)``.
    - ``adam_max_r``, ``adam_p99_r``: max / 99th percentile of
      ``|m_hat| / (sqrt(v_hat) + eps)`` (proportional to per-coord step size / lr).
    - ``adam_frac_r_above_threshold``: fraction of elements with ``r > r_threshold``.

    Args:
        model: Trained model.
        optimizer: Optimizer after ``step()``.
        low_moment_eps_multiplier: Count ``sqrt(v)`` below ``eps *`` this as fragile.
        low_moment_median_ratio: Also count below this fraction of per-tensor median.
        r_threshold: Cutoff for ``adam_frac_r_above_threshold``.

    Returns:
        Metric dict (empty if optimizer state is uninitialized).
    """
    beta1, beta2, eps, _lr = _optimizer_adam_hyperparams(optimizer)
    r_chunks: list[torch.Tensor] = []
    low_count = 0
    total_elems = 0

    with torch.no_grad():
        for param in model.parameters():
            if not param.requires_grad:
                continue
            state = optimizer.state.get(param)
            if state is None:
                continue
            moments = _bias_corrected_adam_moments(state, beta1, beta2)
            if moments is None:
                continue
            m_hat, v_hat = moments
            sqrt_v = v_hat.sqrt()
            denom = sqrt_v + eps
            r = m_hat.abs() / denom
            r_chunks.append(r.flatten())

            median = sqrt_v.median().clamp_min(eps)
            threshold = max(eps * low_moment_eps_multiplier, median * low_moment_median_ratio)
            low_count += int((sqrt_v < threshold).sum().item())
            total_elems += int(sqrt_v.numel())

    if not r_chunks or total_elems == 0:
        return {}

    r_all = torch.cat(r_chunks)
    return {
        "adam_low_moment_frac": low_count / total_elems,
        "adam_max_r": r_all.max().item(),
        "adam_p99_r": torch.quantile(r_all, 0.99).item(),
        "adam_frac_r_above_threshold": (r_all > r_threshold).float().mean().item(),
    }


def adam_second_moment_by_layer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Per-layer RMS of Adam ``exp_avg_sq`` (second moment estimate).

    Args:
        model: Trained model whose parameters may carry optimizer state.
        optimizer: Optimizer instance (typically AdamW).

    Returns:
        Keys like ``adam_second_moment/layer.N`` or group keys for BiLSTM stacks.
    """
    layer_groups: dict[int, list[float]] = defaultdict(list)
    for name, param in model.named_parameters():
        state = optimizer.state.get(param)
        if state is None or "exp_avg_sq" not in state:
            continue
        moment_rms = state["exp_avg_sq"].float().sqrt().norm().item()
        layer_index = _decoder_layer_index_from_param_name(name)
        if layer_index is not None:
            layer_groups[layer_index].append(moment_rms)

    return {
        f"adam_second_moment/layer.{index}": sum(values) / len(values)
        for index, values in sorted(layer_groups.items())
    }


def _decoder_layer_index_from_param_name(name: str) -> int | None:
    """Map a parameter name to a decoder block index (BERT, Llama, or BiLSTM)."""
    bert_match = re.search(r"bert\.encoder\.layer\.(\d+)\.", name)
    if bert_match:
        return int(bert_match.group(1))
    llama_match = re.search(r"(?:^model\.)?layers\.(\d+)\.", name)
    if llama_match:
        return int(llama_match.group(1))
    lstm_match = re.search(r"lstm\.weight_(?:ih|hh)_l(\d+)", name)
    if lstm_match:
        return int(lstm_match.group(1))
    return None


def update_ranks(
    before: dict[str, torch.Tensor],
    model: torch.nn.Module,
    parameter_patterns: list[str],
) -> dict[str, float]:
    """Effective rank (participation ratio) of weight updates ``ΔW``.

    Args:
        before: Parameter tensors captured before ``optimizer.step()``.
        model: Model after the optimizer step.
        parameter_patterns: fnmatch / ``re:`` patterns (see :func:`name_matches`).

    Returns:
        ``update_rank/{param_name}`` entries for matched parameters.
    """
    from pathologies.instrumentation.patterns import name_matches

    ranks: dict[str, float] = {}
    for name, param in model.named_parameters():
        if name not in before or not name_matches(name, parameter_patterns):
            continue
        delta = param.detach().float() - before[name].float()
        if delta.abs().max().item() < 1e-12:
            ranks[f"update_rank/{name}"] = 0.0
            continue
        ranks[f"update_rank/{name}"] = participation_ratio(delta)
    return ranks


def depth_grad_norms(model: torch.nn.Module) -> dict[str, float]:
    """Per-depth gradient norms for BERT, Llama-style blocks, or stacked BiLSTM weights."""
    if any(name.startswith("bert.encoder.layer.") for name, _ in model.named_parameters()):
        return _bert_encoder_layer_grad_norms(model)
    if any(
        name.startswith("model.layers.") or re.match(r"layers\.\d+\.", name)
        for name, _ in model.named_parameters()
    ):
        return _llama_decoder_layer_grad_norms(model)
    if any("lstm.weight_ih_l0" in name for name, _ in model.named_parameters()):
        return _bilstm_stack_grad_norms(model)
    return {}


def depth_grad_early_to_late_ratio(depth_norms: dict[str, float]) -> float | None:
    """Ratio of shallowest to deepest ``grad_norm/layer.*`` entry (vanishing → ≪ 1)."""
    layer_indices = sorted(
        int(key.rsplit(".", maxsplit=1)[-1])
        for key in depth_norms
        if key.startswith("grad_norm/layer.")
    )
    if len(layer_indices) < 2:
        return None
    early = depth_norms[f"grad_norm/layer.{layer_indices[0]}"]
    late = depth_norms[f"grad_norm/layer.{layer_indices[-1]}"]
    if late == 0.0:
        return None
    return early / late


def _bert_encoder_layer_grad_norms(model: torch.nn.Module) -> dict[str, float]:
    groups: dict[int, list[torch.nn.Parameter]] = defaultdict(list)
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        match = re.search(r"bert\.encoder\.layer\.(\d+)\.", name)
        if match:
            groups[int(match.group(1))].append(param)
    return {
        f"grad_norm/layer.{index}": global_grad_norm(parameters)
        for index, parameters in sorted(groups.items())
    }


def _llama_decoder_layer_grad_norms(model: torch.nn.Module) -> dict[str, float]:
    groups: dict[int, list[torch.nn.Parameter]] = defaultdict(list)
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        match = re.search(r"(?:^model\.)?layers\.(\d+)\.", name)
        if match:
            groups[int(match.group(1))].append(param)
    metrics = {
        f"grad_norm/layer.{index}": global_grad_norm(parameters)
        for index, parameters in sorted(groups.items())
    }
    if any(name.startswith("model.embed_tokens") for name, _ in model.named_parameters()):
        metrics["grad_norm/embedding"] = group_grad_norm(
            model, lambda name: name.startswith("model.embed_tokens")
        )
    metrics["grad_norm/head"] = group_grad_norm(
        model, lambda name: name.startswith("lm_head")
    )
    return metrics


def _bilstm_stack_grad_norms(model: torch.nn.Module) -> dict[str, float]:
    groups: dict[int, list[torch.nn.Parameter]] = defaultdict(list)
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        match = re.search(r"lstm\.weight_(?:ih|hh)_l(\d+)", name)
        if match:
            groups[int(match.group(1))].append(param)
    metrics = {
        f"grad_norm/layer.{index}": global_grad_norm(parameters)
        for index, parameters in sorted(groups.items())
    }
    if any(name.startswith("embedding.") for name, _ in model.named_parameters()):
        metrics["grad_norm/embedding"] = group_grad_norm(
            model, lambda name: name.startswith("embedding.")
        )
    metrics["grad_norm/classifier"] = group_grad_norm(
        model, lambda name: name.startswith("classifier.")
    )
    return metrics


def group_grad_norm(model: torch.nn.Module, predicate) -> float:
    """L2 norm over gradients whose parameter names satisfy ``predicate(name)``."""
    parameters = [
        param
        for name, param in model.named_parameters()
        if param.grad is not None and predicate(name)
    ]
    return global_grad_norm(parameters)


def embedding_anisotropy(weight: torch.Tensor, sample_size: int = 256) -> float:
    """Mean pairwise cosine similarity among a sample of embedding rows."""
    matrix = weight.float().detach()
    if matrix.shape[0] <= 1:
        return 0.0
    count = min(sample_size, matrix.shape[0])
    rows = matrix[:count]
    normalized = rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-8)
    similarity = normalized @ normalized.T
    mask = ~torch.eye(count, dtype=torch.bool, device=similarity.device)
    return similarity[mask].mean().item()


def participation_ratio(weight: torch.Tensor) -> float:
    """Effective rank via singular-value participation ratio."""
    matrix = weight.float()
    if matrix.abs().max().item() < 1e-12:
        return 0.0
    if matrix.ndim > 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    if matrix.ndim < 2:
        matrix = matrix.unsqueeze(0)
    singular_values = torch.linalg.svdvals(matrix)
    if singular_values.numel() == 0:
        return 0.0
    denominator = singular_values.square().sum().item()
    if denominator == 0.0:
        return 0.0
    return (singular_values.sum().square() / singular_values.square().sum()).item()
