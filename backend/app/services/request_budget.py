"""Budget the complete ChatAgent request, independently of legacy section quotas."""

import json
from dataclasses import dataclass
from typing import Any

from app.core.config import settings
from app.services.context_compression import (
    estimate_tokens,
    model_context_window_tokens,
    model_output_window_tokens,
)


@dataclass(frozen=True)
class RequestBudget:
    window_tokens: int
    reserved_response_tokens: int
    safety_margin_tokens: int
    input_limit_tokens: int


def request_budget() -> RequestBudget:
    window = model_context_window_tokens()
    if not settings.context_dynamic_model_window_enabled:
        window = min(window, settings.context_max_input_tokens)
    reserved = min(model_output_window_tokens(), max(512, settings.context_reserved_response_tokens))
    margin = max(settings.context_request_safety_tokens, int(window * settings.context_request_safety_ratio))
    return RequestBudget(window, reserved, margin, max(0, window - reserved - margin))


def measure_request(system_prompt: str, user_prompt: str, tools: list[dict[str, Any]]) -> dict[str, int]:
    """Estimate the serialized messages AND the tool definitions actually bound.

    This is deliberately labelled an estimate: provider tokenization and hidden
    message framing can differ. The separate safety margin accounts for that.
    """
    payload: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if tools:
        payload["tools"] = tools
    system_tokens = estimate_tokens(system_prompt)
    user_tokens = estimate_tokens(user_prompt)
    schema_tokens = estimate_tokens(json.dumps(tools, ensure_ascii=False)) if tools else 0
    total = estimate_tokens(json.dumps(payload, ensure_ascii=False))
    return {
        "estimated_prompt_tokens": total,
        "system_tokens": system_tokens,
        "user_context_tokens": user_tokens,
        "tool_schema_tokens": schema_tokens,
        "request_envelope_tokens": max(0, total - system_tokens - user_tokens - schema_tokens),
    }


class RequestBudgetExceeded(RuntimeError):
    def __init__(self, estimated_tokens: int, input_limit_tokens: int) -> None:
        self.estimated_tokens = estimated_tokens
        self.input_limit_tokens = input_limit_tokens
        super().__init__(
            f"本次完整请求估算为 {estimated_tokens:,} tokens，超过可用输入预算 "
            f"{input_limit_tokens:,} tokens。整理可缩减材料后仍无法容纳，因此已停止发送。"
            "请缩小本次输入、减少附带资料，或把任务拆成几步。"
        )
