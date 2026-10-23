"""Source-linked working state: exact constraints survive prompt compaction.

Only explicit user corrections supersede a prior assertion automatically.
Ambiguous corrections remain visible as conflicts requiring verification.
"""

import copy
import hashlib
import re
from datetime import datetime, timezone
from typing import Any


def update_working_state(previous: dict | None, message: str, message_id: str) -> dict:
    if previous and message_id in previous.get("processed_message_ids", []):
        return previous
    state = copy.deepcopy(previous or {})
    entries = state.setdefault("entries", [])
    conflicts = state.setdefault("conflicts", [])
    now = datetime.now(timezone.utc).isoformat()

    def add(kind: str, content: str) -> None:
        content = content.strip(" 。；;\n")
        if not content or any(e["kind"] == kind and e["content"] == content and e["status"] == "active" for e in entries):
            return
        entries.append({"id": hashlib.sha256(f"{message_id}:{kind}:{content}".encode()).hexdigest()[:24],
                        "kind": kind, "content": content, "status": "active",
                        "source_message_id": message_id, "updated_at": now})

    if not any(entry["kind"] == "goal" for entry in entries):
        # A labelled excerpt is an orientation aid, not a replacement for the
        # full request or an extracted constraint.
        add("goal", "初始请求摘要线索：" + message[:500] + (" [原文更长，需回查]" if len(message) > 500 else ""))

    correction_found = False
    patterns = [r"(?:更正|纠正|修正)[：:]\s*(.+?)\s*(?:->|→|改为|替换为)\s*(.+)",
                r"把[“\"]?(.+?)[”\"]?改为[“\"]?(.+?)[”\"]?$",
                r"不是(.+?)[，,]?而是(.+)"]
    for clause in re.split(r"[\n；;。]", message):
        clause = clause.strip()
        if not clause:
            continue
        match = next((m for pattern in patterns if (m := re.search(pattern, clause))), None)
        if match:
            correction_found = True
            old, new = (v.strip(" ‘’“”\"") for v in match.groups())
            matches = [e for e in entries if e["status"] == "active" and e["content"] == old]
            if matches:
                for entry in matches:
                    entry.update(status="superseded", superseded_by_message_id=message_id, updated_at=now)
                    add(entry["kind"], new)
                for entry in list(entries):
                    if entry["kind"] == "goal" and entry["status"] == "active" and entry["content"].startswith("初始请求摘要线索：") and old in entry["content"]:
                        entry["status"] = "superseded"
                        add("goal", entry["content"].replace(old, new))
            else:
                conflicts.append({"source_message_id": message_id, "content": clause,
                                  "reason": "无法唯一匹配旧结论，保留原文核验", "status": "open"})
                add("correction", clause)
            continue
        if re.search(r"(?:更正|纠正|之前.*(?:错|不对)|不是.*问题)", clause):
            correction_found = True
            conflicts.append({"source_message_id": message_id, "content": clause,
                              "reason": "纠正对象不明确，不能自动覆盖旧事实", "status": "open"})
            add("correction", clause)
        if re.match(r"(?:目标|当前目标)[：:]", clause):
            for entry in entries:
                if entry["kind"] == "goal" and entry["status"] == "active":
                    entry.update(status="superseded", superseded_by_message_id=message_id)
            add("goal", re.split(r"[：:]", clause, maxsplit=1)[1])
        elif re.search(r"必须|不能|不要|不允许|禁止|务必|仅允许|只允许|保持|不得|must|never|do not", clause, re.I):
            add("constraint", clause)
        elif re.match(r"(?:决定|决策|已确认)[：:]", clause):
            add("decision", re.split(r"[：:]", clause, maxsplit=1)[1])
        elif re.match(r"(?:事实|结论)[：:]", clause):
            add("fact", re.split(r"[：:]", clause, maxsplit=1)[1])
        elif re.match(r"(?:待办|下一步|任务)[：:]", clause):
            add("task", re.split(r"[：:]", clause, maxsplit=1)[1])
        elif re.match(r"(?:已完成|已排除|撤销)[：:]", clause):
            verb, target = re.split(r"[：:]", clause, maxsplit=1)
            matches = [e for e in entries if e["status"] == "active" and e["content"] == target.strip()]
            for entry in matches:
                entry.update(status="completed" if verb == "已完成" else "invalidated", updated_at=now,
                             superseded_by_message_id=message_id)
            add("resolution", clause)
    state["processed_message_ids"] = [*state.get("processed_message_ids", []), message_id]
    state["revision"] = int(state.get("revision") or 0) + 1
    state["last_message_id"] = message_id
    state["updated_at"] = now
    if correction_found:
        state["last_correction_message_id"] = message_id
    return state


def render_working_state(state: dict | None) -> str:
    if not state:
        return ""
    lines = ["当前有效工作状态（以用户原文和纠正为准；摘要中的旧推断不得覆盖这些要求）："]
    for entry in state.get("entries", []):
        if entry.get("status") == "active":
            lines.append(f"- [{entry['kind']}] {entry['content']} (message_id={entry['source_message_id']})")
    for conflict in state.get("conflicts", []):
        if conflict.get("status") == "open":
            lines.append(f"- [需核验冲突] {conflict['content']} (message_id={conflict['source_message_id']})")
    retired = [e for e in state.get("entries", []) if e.get("status") in {"superseded", "invalidated", "completed"}]
    for entry in retired:
        lines.append(f"- [已失效/完成，不作当前事实] {entry['content']}")
    return "\n".join(lines) if len(lines) > 1 else ""


def reconcile_memory(memory: dict[str, Any], working: dict) -> dict[str, Any]:
    """Remove explicitly retired assertions from legacy string snapshots."""
    result = dict(memory)
    retired = {e["content"] for e in working.get("entries", []) if e.get("status") in {"superseded", "invalidated", "completed"}}
    for key in ("facts", "decisions", "tasks", "open_questions", "user_preferences", "repo_context"):
        result[key] = [v for v in result.get(key, []) if not any(old == v or old in v for old in retired)]
    if any(old in str(result.get("summary") or "") for old in retired):
        result["summary"] = "旧摘要含已失效结论。请以当前工作状态为准，需要历史依据时回查原文。"
    return result
