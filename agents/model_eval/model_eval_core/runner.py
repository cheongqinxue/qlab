"""Run one curated case: ReAct+tools for tool_use, plain LLM otherwise."""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from agents_utils.observability.langfuse_tracing import (
    build_graph_invoke_config,
    flush_langfuse,
    langfuse_task_session,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from openai import BadRequestError

from model_eval_core.benchmarks import CuratedPrompt
from model_eval_core.execution_mode import uses_react_tools
from model_eval_core.gpu_metrics import (
    CaseGpuMetrics,
    eval_gpu_index,
    ollama_api_base,
    query_gpu_vram_mib,
    query_ollama_loaded_vram_bytes,
    track_vram_peak,
)
from model_eval_core.ollama_tool_errors import (
    ToolBindingSkippedError,
    is_ollama_tool_binding_unsupported,
    probe_tool_binding_unsupported,
)
from model_eval_core.probes import ProbeResult, probe_case
from model_eval_core.react_agent import build_react_agent
from model_eval_core.scoring import case_passed, emit_langfuse_scores

logger = logging.getLogger(__name__)


@dataclass
class CaseRunResult:
    """Result of a single eval case."""

    case_id: str
    dimension: str
    model: str
    passed: bool
    probe: ProbeResult
    final_answer: str
    latency_s: float = 0.0
    trace_id: str | None = None
    messages_dump: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


def _serialize_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Serialize LangChain messages for storing in results JSONL."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        entry: dict[str, Any] = {
            "type": msg.type,
            "content": msg.content,
        }
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                entry["tool_calls"] = msg.tool_calls
            if msg.additional_kwargs:
                entry["additional_kwargs"] = msg.additional_kwargs
            if getattr(msg, "response_metadata", None):
                entry["usage"] = msg.response_metadata.get("token_usage")
        elif isinstance(msg, ToolMessage):
            entry["name"] = msg.name
            entry["tool_call_id"] = msg.tool_call_id
        out.append(entry)
    return out


def build_ollama_chat_model(
    model: str,
    *,
    ollama_options: dict[str, Any] | None = None,
    base_url: str | None = None,
) -> ChatOpenAI:
    """Create a ChatOpenAI client pointed at Ollama with publisher options.

    Args:
        model: Ollama model tag.
        ollama_options: Options dict from ``ollama.yaml`` (temperature, top_p, num_ctx, num_predict, etc.).
        base_url: Optional OpenAI-compatible base URL (defaults to local Ollama).

    Returns:
        Configured ChatOpenAI instance.
    """
    opts = dict(ollama_options or {})
    url = base_url or os.environ.get("OLLAMA_OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
    api_key = os.environ.get("OLLAMA_API_KEY", "ollama")

    temperature = opts.pop("temperature", 0.1)
    top_p = opts.pop("top_p", None)
    max_tokens = opts.pop("num_predict", 32768)
    presence_penalty = opts.pop("presence_penalty", None)
    stop = opts.pop("stop", None)

    extra_body: dict[str, Any] = {"options": opts} if opts else {}
    kwargs: dict[str, Any] = {
        "model": model,
        "base_url": url,
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": 300.0,
        "max_retries": 1,
    }
    if top_p is not None:
        kwargs["top_p"] = top_p
    if presence_penalty is not None:
        kwargs["presence_penalty"] = presence_penalty
    if stop:
        kwargs["stop"] = stop
    if extra_body:
        kwargs["extra_body"] = extra_body

    return ChatOpenAI(**kwargs)


async def run_case(
    case: CuratedPrompt,
    *,
    model: str | None = None,
    ollama_options: dict[str, Any] | None = None,
    session_id: str | None = None,
    use_ollama: bool = True,
    model_results_dir: str | None = None,
    skip_react_tool_binding: bool = False,
) -> CaseRunResult:
    """Execute one curated prompt with the standard ReAct agent.

    Args:
        case: Curated eval case.
        model: Model tag or API model name.
        ollama_options: Optional Ollama generation/options dictionary.
        session_id: Langfuse session id (e.g. pytest or batch run id).
        use_ollama: When True, connect directly to local Ollama OpenAI endpoint.
        model_results_dir: Folder name under eval/results/ for human qualitative scoring.
        skip_react_tool_binding: When True, skip ReAct invoke (same tag as prior tool_binding error).

    Returns:
        CaseRunResult with probe diagnostics and full message trace.
    """
    run_id = uuid.uuid4().hex
    model_name = model or os.environ.get("OPENAI_MODEL", "qwen3.5:4b")
    if use_ollama:
        llm = build_ollama_chat_model(model_name, ollama_options=ollama_options)
    else:
        from agents_utils.llm.openai_chat import create_chat_model

        llm = create_chat_model(model=model_name)

    config = build_graph_invoke_config(run_id)
    trace_id: str | None = None
    handler = config.get("callbacks", [None])[0] if config.get("callbacks") else None
    react_mode = uses_react_tools(case.dimension)
    invoke_mode = "react" if react_mode else "direct"

    session = session_id or f"model-eval-{run_id[:8]}"
    t0 = time.perf_counter()
    gpu_index = eval_gpu_index()
    gpu_before = query_gpu_vram_mib(gpu_index)
    gpu_metrics = CaseGpuMetrics(
        gpu_index=gpu_index,
        used_mib_before=gpu_before.used_mib if gpu_before else None,
        total_mib=gpu_before.total_mib if gpu_before else None,
    )
    tool_binding_exc: BaseException | None = None
    messages: list[BaseMessage] = []
    async with track_vram_peak(gpu_index) as vram_peak:
        with langfuse_task_session(
            task_id=run_id,
            task_preview=case.prompt,
            session_id=session,
            extra_metadata={
                "case_id": case.id,
                "dimension": case.dimension.value,
                "model": model_name,
                "invoke_mode": invoke_mode,
            },
        ):
            if react_mode and skip_react_tool_binding:
                messages = [HumanMessage(content=case.prompt)]
                tool_binding_exc = ToolBindingSkippedError()
            elif react_mode:
                graph = build_react_agent(llm)
                react_config = {**config, "recursion_limit": 16}
                state_input = {"messages": [HumanMessage(content=case.prompt)]}
                try:
                    result = await graph.ainvoke(state_input, config=react_config)
                except BadRequestError as exc:
                    # Ollama returns 400 when the model cannot accept tool/grammar binding; continue the suite.
                    if not is_ollama_tool_binding_unsupported(exc):
                        raise
                    logger.warning(
                        "Ollama tool binding unsupported for model=%s case=%s",
                        model_name,
                        case.id,
                    )
                    tool_binding_exc = exc
                    messages = [HumanMessage(content=case.prompt)]
                else:
                    messages = result.get("messages", [])
            else:
                # No system prompt or tool schemas — user prompt only.
                ai = await llm.ainvoke([HumanMessage(content=case.prompt)], config=config)
                messages = [HumanMessage(content=case.prompt), ai]
    peak_used = vram_peak[0]
    if peak_used == 0 and gpu_before is not None:
        peak_used = gpu_before.used_mib
    gpu_metrics.used_mib_peak = peak_used if peak_used > 0 else None
    gpu_after = query_gpu_vram_mib(gpu_index)
    if gpu_after is not None:
        gpu_metrics.used_mib_after = gpu_after.used_mib
        gpu_metrics.total_mib = gpu_after.total_mib
    if use_ollama:
        gpu_metrics.ollama_loaded_vram_bytes = query_ollama_loaded_vram_bytes(ollama_api_base())
    latency_s = round(time.perf_counter() - t0, 3)

    if handler is not None and hasattr(handler, "last_trace_id"):
        trace_id = getattr(handler, "last_trace_id", None)

    if tool_binding_exc is not None:
        probe = probe_tool_binding_unsupported(
            tool_binding_exc,
            skipped_without_invoke=skip_react_tool_binding,
        )
    else:
        probe = probe_case(
            dimension=case.dimension,
            prompt=case.prompt,
            messages=messages,
            metadata=case.metadata,
            case_id=case.id,
        )
    passed = case_passed(
        probe,
        dimension=case.dimension,
        model_dir=model_results_dir,
        case_id=case.id,
    )
    final = ""
    for message in reversed(messages):
        if (
            isinstance(message, AIMessage)
            and isinstance(message.content, str)
            and message.content.strip()
        ):
            final = message.content
            break
    if not final and tool_binding_exc is not None:
        final = "[N/A] ollama tool binding unsupported"
    emit_langfuse_scores(
        trace_id=trace_id,
        probe=probe,
        passed=passed,
        dimension=case.dimension.value,
        model=model_name,
        case_id=case.id,
    )
    flush_langfuse()

    return CaseRunResult(
        case_id=case.id,
        dimension=case.dimension.value,
        model=model_name,
        passed=passed,
        probe=probe,
        final_answer=final,
        latency_s=latency_s,
        trace_id=trace_id,
        messages_dump=_serialize_messages(messages),
        raw={
            "failure_modes": probe.failure_modes,
            "signals": probe.signals,
            "invoke_mode": invoke_mode,
            "runtime": {"wall_s": latency_s},
            "gpu": gpu_metrics.as_dict(),
        },
    )
