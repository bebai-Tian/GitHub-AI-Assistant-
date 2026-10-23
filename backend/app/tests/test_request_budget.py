"""Behavioral checks for the complete-request gate (no provider/network calls)."""

import copy
import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.core.config import settings
from app.services.agents.chat_agent import ChatAgent
from app.services.context_compression import estimate_tokens
from app.services.request_budget import measure_request, request_budget


class RecordingModel:
    def __init__(self, *, overflow: bool = False) -> None:
        self.calls: list[list[Any]] = []
        self.output_limits: list[int] = []
        self.overflow = overflow

    def bind_tools(self, tools: list[Any]) -> "RecordingModel":
        return self

    def bind(self, *, max_tokens: int) -> "RecordingModel":
        self.output_limits.append(max_tokens)
        return self

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.calls.append(messages)
        if self.overflow:
            raise RuntimeError("maximum context length exceeded")
        return AIMessage(content="checked answer")


@pytest.fixture(autouse=True)
def small_window(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "context_model_window_tokens", 8192)
    monkeypatch.setattr(settings, "context_dynamic_model_window_enabled", True)
    monkeypatch.setattr(settings, "context_max_input_tokens", 24000)
    monkeypatch.setattr(settings, "context_reserved_response_tokens", 1024)
    monkeypatch.setattr(settings, "context_request_safety_tokens", 512)
    monkeypatch.setattr(settings, "context_request_safety_ratio", 0.03)


def state(**changes: Any) -> dict[str, Any]:
    return {
        "message": "Continue investigating the timeout; do not change the API.",
        "context": {}, "tools": {}, "tool_name_map": {},
        "observations": [], "agent_events": [], "tool_calls": [],
        "step": 0, "completion_status": "running", **changes,
    }


async def unused_tool(_input: dict[str, Any], _context: dict[str, Any]) -> dict[str, Any]:
    raise AssertionError("This test must not execute a tool")


def test_budget_never_exceeds_a_small_model_window() -> None:
    budget = request_budget()
    assert budget.window_tokens == 8192  # Legacy section budgets may still be 24k.
    assert budget.input_limit_tokens == 6656
    assert budget.input_limit_tokens + budget.reserved_response_tokens + budget.safety_margin_tokens == 8192


def test_explicit_input_cap_and_safety_margin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "context_dynamic_model_window_enabled", False)
    monkeypatch.setattr(settings, "context_max_input_tokens", 6000)
    assert request_budget().input_limit_tokens == 4464
    monkeypatch.setattr(settings, "context_request_safety_ratio", 0.1)
    assert request_budget().input_limit_tokens == 4376
    monkeypatch.setattr(settings, "context_reserved_response_tokens", 10000)
    assert request_budget().input_limit_tokens == 0


def test_measurement_includes_serialization_and_real_tool_definitions() -> None:
    agent = ChatAgent(model=RecordingModel())
    tools = agent._build_langchain_tools({"workspace.read_file": unused_tool}, {})
    schemas = [convert_to_openai_tool(tool) for tool in tools.values()]
    system, user = "Rules", 'History: "quoted"\nFile C:\\project\\app.py'
    result = measure_request(system, user, schemas)
    payload = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "tools": schemas}
    assert result["estimated_prompt_tokens"] == estimate_tokens(json.dumps(payload, ensure_ascii=False))
    assert result["request_envelope_tokens"] > 0
    assert result["tool_schema_tokens"] > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["user", "context", "history", "skill"])
async def test_oversized_required_material_is_blocked_before_any_call(source: str) -> None:
    model = RecordingModel()
    agent = ChatAgent(model=model)
    material = "must preserve this requirement " * 5000
    current = state()
    if source == "user":
        current["message"] = material
    elif source == "context":
        current["context"] = {"server_context": material}
    elif source == "history":
        current["context"] = {"server_messages": [{"role": "user", "content": material}]}
    else:
        current["context"] = {"active_skills": [{"skill_name": "large", "instructions": material}]}
    original = copy.deepcopy(current)
    result = await agent._graph_model_decision(current, model)
    assert model.calls == []
    assert result["completion_status"] == "blocked"
    assert result["stop_reason"] == "context_budget_exceeded"
    assert "停止发送" in result["final_answer"]
    assert current["context"] == original["context"]
    assert current["message"] == original["message"]
    assert all(not row["sent_to_model"] for row in result["prompt_diagnostics"])
    assert result["prompt_diagnostics"][-1]["budget_action"] == "blocked"


@pytest.mark.asyncio
async def test_large_tool_schema_cannot_bypass_gate() -> None:
    model = RecordingModel()
    agent = ChatAgent(model=model)
    tools = agent._build_langchain_tools({"workspace.read_file": unused_tool}, {})
    next(iter(tools.values())).description = "large tool contract " * 6000
    result = await agent._graph_model_decision(state(tools=tools), model)
    assert model.calls == []
    assert result["prompt_diagnostics"][-1]["tool_schema_tokens"] > request_budget().input_limit_tokens


@pytest.mark.asyncio
async def test_large_event_index_is_reduced_without_mutating_audit_or_constraints() -> None:
    model = RecordingModel()
    agent = ChatAgent(model=model)
    events = [{"type": "tool_result", "tool_name": "read", "reason": "old diagnostic " * 10000}]
    history = [{"role": "user", "content": "Never change the public API response format."}]
    current = state(agent_events=events, context={"server_messages": history, "server_context": "Keep the user constraints."})
    audit_copy = copy.deepcopy(events)
    result = await agent._graph_model_decision(current, model)
    assert len(model.calls) == 1
    assert events == audit_copy
    assert current["context"]["server_messages"] == history
    prompt = model.calls[0][1].content
    assert history[0]["content"] in prompt
    assert current["message"] in prompt
    assert "omitted_tool_events" in prompt
    assert result["prompt_diagnostics"][0]["budget_action"] == "reduce_tools"
    assert result["prompt_diagnostics"][-1]["headroom_ok"] is True
    assert result["prompt_diagnostics"][-1]["sent_to_model"] is True
    assert model.output_limits == [1024]


@pytest.mark.asyncio
async def test_unbound_final_answer_does_not_charge_unused_schemas() -> None:
    model = RecordingModel()
    agent = ChatAgent(model=model)
    tools = agent._build_langchain_tools({"workspace.read_file": unused_tool}, {})
    next(iter(tools.values())).description = "not sent in this path " * 6000
    current = state(tools=tools)
    assert await agent._final_from_observations(current) == "checked answer"
    assert len(model.calls) == 1
    assert current["prompt_diagnostics"][-1]["tool_schema_tokens"] == 0
    assert model.output_limits == [1024]


@pytest.mark.asyncio
async def test_recent_recall_gate_returns_blocked_status_instead_of_success() -> None:
    model = RecordingModel()
    final = await ChatAgent(model=model).run(
        {"message": "我刚刚说了什么？"},
        {"tools": {}, "server_messages": [
            {"role": "user", "content": "preserve my exact words " * 6000},
            {"role": "assistant", "content": "Acknowledged."},
        ], "compression_stats": {"history_coverage": {"complete_and_exact": True, "exact_suffix_messages": 2}}},
    )
    assert model.calls == []
    assert final["completion"]["status"] == "blocked"
    assert final["route"] == "context_budget_exceeded"


@pytest.mark.asyncio
async def test_final_answer_cannot_silently_fallback_when_required_context_is_oversized() -> None:
    model = RecordingModel()
    current = state(context={"server_context": "Required context " * 10000})
    answer = await ChatAgent(model=model)._final_from_observations(current)
    assert "停止发送" in answer
    assert current["completion_status"] == "blocked"
    assert model.calls == []


@pytest.mark.asyncio
async def test_normal_water_mark_never_invokes_compaction() -> None:
    """水位停在 normal：不做任何压缩，请求照发。"""
    calls: list[str] = []

    async def compact(context, message, measure, limit, *, stage: str = "auto_compact"):
        calls.append(stage)
        return {"attempted": True, "compaction_applied": False, "headroom_ok": False, "reason": "stub"}

    model = RecordingModel()
    current = state(context={"_context_compactor": compact, "_context_base": "base", "_history_tools": []})
    await ChatAgent(model=model)._invoke_guarded_model(
        current, model, phase="test", system_prompt="Rules",
        prompt_builder=lambda aggressive: "short prompt",
        include_tools=False,
    )
    assert calls == []
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_water_mark_escalates_the_full_compaction_ladder(monkeypatch: pytest.MonkeyPatch) -> None:
    """水位到 auto_compact：第二级按 auto_compact → aggressive → manual_path 逐档升级。"""
    monkeypatch.setattr(settings, "context_warning_buffer_ratio", 0.999)
    monkeypatch.setattr(settings, "context_auto_compact_buffer_ratio", 0.999)
    stages: list[str] = []

    async def compact(context, message, measure, limit, *, stage: str = "auto_compact"):
        stages.append(stage)
        return {"attempted": True, "compaction_applied": False, "headroom_ok": False, "reason": "stub"}

    model = RecordingModel()
    current = state(context={"_context_compactor": compact, "_context_base": "base",
                              "_context_required": "base", "_history_tools": []})
    await ChatAgent(model=model)._invoke_guarded_model(
        current, model, phase="test", system_prompt="Rules",
        prompt_builder=lambda aggressive: "context material " * 500,
        include_tools=False,
    )
    assert stages == ["auto_compact", "aggressive", "manual_path"]
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_provider_overflow_is_not_retried() -> None:
    """供应商报上下文超限时不再重试——异常直接抛出。"""
    model = RecordingModel(overflow=True)
    agent = ChatAgent(model=model)
    current = state()
    with pytest.raises(RuntimeError, match="maximum context"):
        await agent._invoke_guarded_model(
            current, model, phase="test", system_prompt="Rules",
            prompt_builder=lambda aggressive: "Reduced" if aggressive else "Normal",
            include_tools=False,
        )
    assert len(model.calls) == 1
