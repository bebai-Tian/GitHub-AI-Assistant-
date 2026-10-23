"""Small, tool-specific contracts. Repository/session scope is server-owned."""

from functools import lru_cache
from typing import Any

from pydantic import ConfigDict, Field, create_model


FIELDS: dict[str, tuple[Any, Any]] = {
    "message": (str | None, Field(default=None, description="任务或用户请求")),
    "query": (str | None, Field(default=None, description="检索问题")),
    "keyword": (str | None, Field(default=None, description="原文关键词")),
    "limit": (int | None, Field(default=None, ge=1, le=200, description="结果数量")),
    "path": (str | None, Field(default=None, description="工作区相对路径")),
    "pattern": (str | None, Field(default=None, description="搜索文本或文件名")),
    "start_line": (int | None, Field(default=None, ge=1)),
    "line_count": (int | None, Field(default=None, ge=1)),
    "message_id": (str | None, Field(default=None, description="精确消息 UUID")),
    "before_message_id": (str | None, Field(default=None)),
    "after_message_id": (str | None, Field(default=None)),
    "session_id": (str | None, Field(default=None, description="封存片段 UUID")),
    "offset": (int | None, Field(default=None, ge=0, description="原文字符偏移")),
    "max_chars": (int | None, Field(default=None, ge=200, le=20000)),
    "include_transcript": (bool | None, Field(default=None)),
}

TOOL_FIELDS = {
    "workspace.list_files": ("path", "pattern", "limit"),
    "workspace.read_file": ("path", "start_line", "line_count"),
    "workspace.search_code": ("query", "pattern", "path", "limit"),
    "devflow_search_evidence": ("query", "limit"),
    "devflow_get_thread_context": ("keyword", "limit", "message_id", "before_message_id", "after_message_id", "offset", "max_chars"),
    "devflow_read_session_events": ("session_id", "limit", "offset", "max_chars", "include_transcript"),
    "rag.search_similar_documents": ("query", "limit"),
    "repo_health.generate_summary": (),
    "issue_agent.analyze_issue": ("message",),
    "pr_review_agent.analyze_pr": ("message",),
    "ci_debug_agent.analyze_workflow_run": ("message",),
    "report_agent.generate_weekly_report": ("message",),
    "safety_agent.classify_action_risk": ("message",),
    "workflow.run_engineering_review": ("message",),
}


@lru_cache(maxsize=64)
def tool_args_schema(name: str):
    fields = TOOL_FIELDS.get(name, ("message", "query", "limit"))
    return create_model(
        name.replace(".", "_") + "Args", __config__=ConfigDict(extra="forbid"),
        **{key: FIELDS[key] for key in fields},
    )
