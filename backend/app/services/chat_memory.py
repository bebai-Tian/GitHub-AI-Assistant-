import re
import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import and_, or_, func
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import (
    AgentRun,
    ChatMessage,
    ChatSession,
    ChatSession as ChatSessionModel,
    Conversation,
    ConversationMemory,
    Document,
    EvidenceItem,
    Issue,
    PullRequest,
    Repository,
    ThreadMemory,
    WorkflowRun,
)
from app.services.context_compression import (
    COMPACT_BOUNDARY_SUBTYPE,
    COMPACT_SUMMARY_SYSTEM_PROMPT,
    adjust_keep_start_to_preserve_api_invariants,
    calculate_context_pressure,
    clip_to_token_budget,
    dumps_compact,
    ensure_compaction_summary_preserves_key_facts,
    estimate_tokens,
    extract_relevant_snippet,
    fallback_memory_update,
    fallback_compaction_summary,
    format_compaction_summary,
    is_compact_boundary_message,
    messages_after_compact_boundary,
    memory_document_text,
    merge_memory,
    microcompact_messages,
    render_structured_memory,
    structured_memory_from_model,
)
from app.services.llm.client import LLMClient
from app.services.memory_hub import capture_memory_candidates, record_recall_event
from app.services.rag.indexing import CI_LOG_CHUNK_CHARS, sanitize_ci_log
from app.services.rag.retrieval import search_similar_documents
from app.services.rag.source_policy import RAG_SOURCE_TYPES
from app.services.rag.vector_store import MilvusUnavailableError
from app.services.working_memory import update_working_state, render_working_state, reconcile_memory
from app.services.request_budget import request_budget
from app.services.memory_lifecycle import memory_is_active


def _repo_filter(model: Any, repo_id: uuid.UUID | None) -> Any:
    return model.repo_id.is_(None) if repo_id is None else model.repo_id == repo_id


def _conversation_filter(model: Any, conversation_id: uuid.UUID | None) -> Any:
    return model.conversation_id.is_(None) if conversation_id is None else model.conversation_id == conversation_id


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_\-\u4e00-\u9fff]+", text.lower())


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: max(0, limit - 3)].rstrip() + "..."


class MessageStore:
    def __init__(self, db: Session) -> None:
        self.db = db

    def append(
        self,
        *,
        repo_id: uuid.UUID | None,
        conversation_id: uuid.UUID | None,
        role: str,
        content: str,
        route: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> ChatMessage:
        if role == "user" and repo_id and conversation_id:
            self.db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.repo_id == repo_id).with_for_update().one()
        message = ChatMessage(
            repo_id=repo_id,
            conversation_id=conversation_id,
            role=role,
            content=content,
            route=route,
            tool_calls=tool_calls or [],
            meta=meta or {},
            created_at=datetime.now(timezone.utc),
        )
        self.db.add(message)
        self.db.flush()
        if role == "user" and repo_id and conversation_id:
            memory = self.db.query(ConversationMemory).filter(ConversationMemory.conversation_id == conversation_id).first()
            if memory is None:
                memory = ConversationMemory(repo_id=repo_id, conversation_id=conversation_id, summary="")
                self.db.add(memory)
                self.db.flush()
            if not memory.working_state:
                working = {}
                for prior in self.db.query(ChatMessage).filter(
                    ChatMessage.repo_id == repo_id, ChatMessage.conversation_id == conversation_id, ChatMessage.role == "user"
                ).order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc()).yield_per(200):
                    working = update_working_state(working, prior.content, str(prior.id))
                memory.working_state = working
            memory.working_state = update_working_state(memory.working_state, content, str(message.id))
            revised = reconcile_memory({key: getattr(memory, key) for key in
                                       ("summary", "facts", "decisions", "tasks", "open_questions", "user_preferences", "repo_context")},
                                      memory.working_state)
            for key, value in revised.items():
                setattr(memory, key, value)
        return message

    def recent(self, repo_id: uuid.UUID | None, conversation_id: uuid.UUID | None, limit: int | None = 20) -> list[ChatMessage]:
        """Return the active context projection, not the complete transcript.

        New compact boundaries keep references to the original recent messages
        instead of copying them.  Older databases can still contain
        ``preserved_after_compact`` copies, so this method understands both
        layouts during the migration period.
        """
        boundary = self.last_compact_boundary(repo_id, conversation_id)
        if boundary is None:
            return self.timeline(repo_id, conversation_id, limit=limit, include_system=True)

        boundary_meta = boundary.meta or {}
        preserved_ids = [
            value
            for value in boundary_meta.get("preserved_message_ids") or []
            if value
        ]
        preserved_rows: list[ChatMessage] = []
        if preserved_ids:
            parsed_ids: list[uuid.UUID] = []
            for value in preserved_ids:
                try:
                    parsed_ids.append(uuid.UUID(str(value)))
                except ValueError:
                    continue
            if parsed_ids:
                by_id = {
                    row.id: row
                    for row in self.db.query(ChatMessage)
                    .filter(
                        _repo_filter(ChatMessage, repo_id),
                        _conversation_filter(ChatMessage, conversation_id),
                        ChatMessage.id.in_(parsed_ids),
                    )
                    .all()
                }
                preserved_rows = [by_id[item] for item in parsed_ids if item in by_id]

        rows_after_boundary = (
            self.db.query(ChatMessage)
            .filter(
                _repo_filter(ChatMessage, repo_id),
                _conversation_filter(ChatMessage, conversation_id),
                ChatMessage.created_at >= boundary.created_at,
            )
            .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
            .yield_per(200)
        )
        projected = [boundary, *preserved_rows]
        projected.extend(
            row
            for row in rows_after_boundary
            if row.id != boundary.id
            and str(row.id) not in preserved_ids
            and not (
                preserved_ids
                and (row.meta or {}).get("subtype") == "preserved_after_compact"
            )
        )
        return projected if limit is None else projected[-max(1, limit) :]

    def timeline(
        self,
        repo_id: uuid.UUID | None,
        conversation_id: uuid.UUID | None,
        *,
        limit: int | None = 500,
        include_system: bool = False,
        before_message_id: uuid.UUID | None = None,
        after_message_id: uuid.UUID | None = None,
        keyword: str = "",
    ) -> list[ChatMessage]:
        """Return the durable transcript while hiding compatibility copies.

        Compact boundaries change the model's active projection; they do not
        erase the transcript.  This method is the authoritative read path for
        UI history and exact-memory tools.
        """
        query = self.db.query(ChatMessage).filter(
            _repo_filter(ChatMessage, repo_id),
            _conversation_filter(ChatMessage, conversation_id),
        )
        if not include_system:
            query = query.filter(ChatMessage.role != "system")
        query = query.filter(func.coalesce(ChatMessage.meta["subtype"].as_string(), "") != "preserved_after_compact")
        if keyword:
            terms = _tokens(keyword)
            if terms:
                query = query.filter(or_(*(func.lower(ChatMessage.content).contains(term, autoescape=True) for term in terms)))

        if before_message_id:
            before = (
                self.db.query(ChatMessage)
                .filter(
                    _repo_filter(ChatMessage, repo_id),
                    _conversation_filter(ChatMessage, conversation_id),
                    ChatMessage.id == before_message_id,
                )
                .one_or_none()
            )
            if before and before.created_at:
                query = query.filter(
                    or_(
                        ChatMessage.created_at < before.created_at,
                        and_(
                            ChatMessage.created_at == before.created_at,
                            ChatMessage.id < before.id,
                        ),
                    )
                )
        if after_message_id:
            after = (
                self.db.query(ChatMessage)
                .filter(
                    _repo_filter(ChatMessage, repo_id),
                    _conversation_filter(ChatMessage, conversation_id),
                    ChatMessage.id == after_message_id,
                )
                .one_or_none()
            )
            if after and after.created_at:
                query = query.filter(
                    or_(
                        ChatMessage.created_at > after.created_at,
                        and_(
                            ChatMessage.created_at == after.created_at,
                            ChatMessage.id > after.id,
                        ),
                    )
                )

        if limit is None:
            return list(query.order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc()).yield_per(200))
        if after_message_id and not before_message_id:
            return query.order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc()).limit(max(1, limit)).all()
        return list(reversed(query.order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc()).limit(max(1, limit)).all()))

    def active_for_context(
        self,
        repo_id: uuid.UUID | None,
        conversation_id: uuid.UUID | None,
        limit: int | None = None,
    ) -> list[ChatMessage]:
        return self.recent(repo_id, conversation_id, limit=limit)

    def all_for_context(self, repo_id: uuid.UUID | None, conversation_id: uuid.UUID | None, limit: int = 200) -> list[ChatMessage]:
        rows = (
            self.db.query(ChatMessage)
            .filter(_repo_filter(ChatMessage, repo_id), _conversation_filter(ChatMessage, conversation_id))
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
            .all()
        )
        return list(reversed(rows))

    def last_compact_boundary(self, repo_id: uuid.UUID | None, conversation_id: uuid.UUID | None) -> ChatMessage | None:
        return (
            self.db.query(ChatMessage)
            .filter(_repo_filter(ChatMessage, repo_id), _conversation_filter(ChatMessage, conversation_id), ChatMessage.role == "system",
                    ChatMessage.meta["subtype"].as_string() == COMPACT_BOUNDARY_SUBTYPE)
            .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
            .first()
        )


class EvidenceStore:
    def __init__(self, db: Session) -> None:
        self.db = db

    def rebuild_repo(self, repo: Repository | None, conversation: Conversation | None = None) -> None:
        repo_id = repo.id if repo else None
        conversation_id = conversation.id if conversation else None
        self._seen: set[tuple] = set()
        # Source signatures avoid rebuilding unchanged repository bodies on
        # every turn; conversation memory/session records are upserted separately.
        signature = []
        if repo:
            for model in (Issue, PullRequest, WorkflowRun, Document):
                stamp = getattr(model, "updated_at", model.created_at)
                count, newest = self.db.query(func.count(model.id), func.max(stamp)).filter(model.repo_id == repo.id).one()
                signature.append([model.__tablename__, count, str(newest)])
            signature.append([repo.full_name, repo.description, repo.default_branch])
        fingerprint = hashlib.sha256(dumps_compact(signature).encode()).hexdigest()
        marker = self.db.query(EvidenceItem).filter(_repo_filter(EvidenceItem, repo_id), EvidenceItem.conversation_id.is_(None),
                                                   EvidenceItem.source_type == "index_state", EvidenceItem.source_id == str(repo_id)).first()
        project_changed = marker is None or (marker.meta or {}).get("fingerprint") != fingerprint
        existing_query = self.db.query(EvidenceItem).filter(_repo_filter(EvidenceItem, repo_id))
        if project_changed:
            existing_query = existing_query.filter(or_(EvidenceItem.conversation_id == conversation_id, EvidenceItem.conversation_id.is_(None)))
        else:
            existing_query = existing_query.filter(_conversation_filter(EvidenceItem, conversation_id), EvidenceItem.source_type.in_(["memory", "session", "agent_run"]))
        self._existing = {(row.conversation_id, row.source_type, row.source_id): row for row in existing_query.all()}
        if repo and project_changed:
            self._add(
                repo_id,
                None,
                "repository",
                str(repo.id),
                repo.full_name,
                "\n".join(filter(None, [repo.description or "", f"默认分支：{repo.default_branch or '未知'}"])),
                {"anchor": f"repo:{repo.full_name}"},
            )
            for issue in self.db.query(Issue).filter(Issue.repo_id == repo.id).all():
                self._add(
                    repo.id,
                    None,
                    "issue",
                    str(issue.id),
                    f"Issue #{issue.number}: {issue.title}",
                    "\n".join(filter(None, [issue.title, issue.body or "", f"state: {issue.state}", f"labels: {', '.join(issue.labels or [])}"])),
                    {"number": issue.number, "state": issue.state},
                )
            for pr in self.db.query(PullRequest).filter(PullRequest.repo_id == repo.id).all():
                self._add(
                    repo.id,
                    None,
                    "pull_request",
                    str(pr.id),
                    f"PR #{pr.number}: {pr.title}",
                    "\n".join(filter(None, [pr.title, pr.body or "", f"state: {pr.state}", f"base: {pr.base_branch}", f"head: {pr.head_branch}"])),
                    {"number": pr.number, "state": pr.state},
                )
            for run in self.db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo.id).all():
                if run.conclusion != "failure" or not (run.logs_text or "").strip():
                    continue
                sanitized_logs = sanitize_ci_log(run.logs_text or "")
                safe_logs = sanitized_logs[:CI_LOG_CHUNK_CHARS]
                self._add(
                    repo.id,
                    None,
                    "workflow_run",
                    str(run.id),
                    f"CI: {run.name}",
                    "\n".join(filter(None, [run.name, run.status, run.conclusion or "", safe_logs])),
                    {
                        "status": run.status,
                        "conclusion": run.conclusion,
                        "logs_truncated": len(sanitized_logs) > CI_LOG_CHUNK_CHARS,
                        "secrets_redacted": True,
                    },
                )
            for doc in (
                self.db.query(Document)
                .filter(Document.repo_id == repo.id, Document.source_type.in_(RAG_SOURCE_TYPES))
                .all()
            ):
                if not memory_is_active(doc.meta):
                    continue
                self._add(repo.id, None, doc.source_type, str(doc.source_id), doc.title, doc.content, doc.meta or {})
            self._add(repo_id, None, "index_state", str(repo_id), "Evidence index", "source signature", {"fingerprint": fingerprint})

        memories = (
            self.db.query(ConversationMemory).filter(ConversationMemory.conversation_id == conversation_id).all()
            if conversation_id
            else self.db.query(ThreadMemory).filter(_repo_filter(ThreadMemory, repo_id)).all()
        )
        for memory in memories:
            self._add(
                repo_id,
                conversation_id,
                "memory",
                str(memory.id),
                "Thread memory",
                memory_document_text(memory),
                {"sessions_incorporated": memory.sessions_incorporated},
            )
        session_query = self.db.query(ChatSession).filter(_repo_filter(ChatSession, repo_id))
        run_query = self.db.query(AgentRun).filter(_repo_filter(AgentRun, repo_id))
        if conversation_id:
            session_query = session_query.filter(ChatSession.conversation_id == conversation_id)
            run_query = run_query.filter(AgentRun.conversation_id == conversation_id)
        else:
            session_query = session_query.filter(ChatSession.conversation_id.is_(None))
            run_query = run_query.filter(AgentRun.conversation_id.is_(None))
        for session in session_query.order_by(ChatSession.created_at.desc()).limit(30):
            self._add(repo_id, conversation_id, "session", str(session.id), f"Session digest: {session.route}", session.digest, {"route": session.route})
        for run in run_query.order_by(AgentRun.created_at.desc()).limit(30):
            self._add(repo_id, conversation_id, "agent_run", str(run.id), f"Agent run: {run.route}", f"{run.user_message}\n{run.final_answer}", {"route": run.route})
        for key, row in self._existing.items():
            refresh_scope = (key[0] == conversation_id and key[1] in {"memory", "session", "agent_run"}) or (project_changed and key[0] is None)
            if refresh_scope and key not in self._seen:
                self.db.delete(row)
        self.db.flush()

    def search(self, repo_id: uuid.UUID | None, conversation_id: uuid.UUID | None, query: str, limit: int = 5) -> list[dict[str, Any]]:
        query_terms = _tokens(query)
        if not query_terms:
            return []
        results: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        prefer_conversation = any(token in query.lower() for token in ["之前", "刚才", "上次", "原话", "会话", "记得", "previous", "earlier", "remember", "session"])
        if repo_id and not prefer_conversation:
            try:
                rag_results = search_similar_documents(
                    self.db,
                    repo_id,
                    query,
                    source_type=None,
                    limit=max(limit, 8),
                )
            except MilvusUnavailableError:
                rag_results = []
            for item in rag_results:
                key = (str(item.get("source_type") or ""), str(item.get("source_id") or ""))
                if key in seen:
                    continue
                seen.add(key)
                results.append(
                    {
                        "title": item.get("title") or "",
                        "source_type": item.get("source_type") or "",
                        "source_id": item.get("source_id") or "",
                        "snippet": item.get("snippet") or "",
                        "score": item.get("score") or 0,
                        "metadata": item.get("metadata") or {},
                        "retrieval": item.get("retrieval") or {},
                    }
                )
                if len(results) >= limit:
                    return results
        candidates = (
            self.db.query(EvidenceItem)
            .filter(
                _repo_filter(EvidenceItem, repo_id),
                or_(EvidenceItem.conversation_id == conversation_id, EvidenceItem.conversation_id.is_(None)),
                EvidenceItem.source_type != "index_state",
            )
            .order_by(EvidenceItem.created_at.desc())
            .limit(500)
            .all()
        )
        ranked: list[tuple[int, EvidenceItem]] = []
        for item in candidates:
            if not memory_is_active(item.meta):
                continue
            if item.source_type == "memory_note":
                docs = self.db.query(Document).filter(Document.repo_id == repo_id, Document.source_type == "memory_note", Document.source_id == uuid.UUID(item.source_id)).all()
                if not any(memory_is_active(doc.meta) for doc in docs):
                    continue
            key = (item.source_type, item.source_id)
            if key in seen:
                continue
            haystack = f"{item.title}\n{item.content}".lower()
            score = sum(haystack.count(term) for term in query_terms)
            if score:
                ranked.append((score + (1000 if prefer_conversation and item.conversation_id == conversation_id else 0), item))
        ranked.sort(key=lambda pair: (pair[0], pair[1].created_at.replace(tzinfo=timezone.utc).timestamp()), reverse=True)
        for score, item in ranked:
            key = (item.source_type, item.source_id)
            if key in seen:
                continue
            seen.add(key)
            results.append(
                {
                    "title": item.title,
                    "source_type": item.source_type,
                    "source_id": item.source_id,
                    "snippet": extract_relevant_snippet(query, item.content, max_tokens=90),
                    "score": score,
                    "metadata": item.meta or {},
                    "retrieval": {"sources": ["evidence_item_keyword"]},
                }
            )
            if len(results) >= limit:
                break
        return results

    def _add(self, repo_id: uuid.UUID | None, conversation_id: uuid.UUID | None, source_type: str, source_id: str, title: str, content: str, meta: dict[str, Any]) -> None:
        if not content.strip():
            return
        key = (conversation_id, source_type, source_id)
        if hasattr(self, "_seen"):
            self._seen.add(key)
        row = getattr(self, "_existing", {}).get(key)
        if row is None:
            row = EvidenceItem(repo_id=repo_id, conversation_id=conversation_id, source_type=source_type, source_id=source_id, title=title, content=content, meta=meta)
            self.db.add(row)
            if hasattr(self, "_existing"):
                self._existing[key] = row
        elif row.title != title or row.content != content or row.meta != meta:
            row.title, row.content, row.meta = title, content, meta
            row.created_at = datetime.now(timezone.utc)


@dataclass
class AssembledContext:
    system_context: str
    history_messages: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    compression_stats: dict[str, Any] = field(default_factory=dict)
    base_context: str = ""
    tool_messages: list[dict[str, Any]] = field(default_factory=list)
    active_message_ids: list[str] = field(default_factory=list)
    required_context: str = ""
    memory_context: str = ""

    def runtime_fields(self) -> dict[str, Any]:
        return {
            "server_context": self.system_context, "server_messages": self.history_messages,
            "evidence": self.evidence, "compression_stats": self.compression_stats,
            "_context_base": self.base_context, "_history_tools": self.tool_messages,
            "_active_message_ids": self.active_message_ids,
            "_context_required": self.required_context, "_context_memory": self.memory_context,
        }


def compose_server_context(base: str, evidence: list[dict], tool_messages: list[dict]) -> str:
    sections = [base] if base else []
    if evidence:
        sections.append("[Related Evidence]\n" + "\n".join(
            f"- [{item.get('source_type')}] {item.get('title')}: {item.get('snippet') or ''}"
            for item in evidence) + "\n[/Related Evidence]")
    if tool_messages:
        sections.append("[Historical Tool Results]\n" + dumps_compact(tool_messages) + "\n[/Historical Tool Results]")
    return "\n\n".join(sections)


class ContextAssembler:
    def __init__(self, db: Session) -> None:
        self.db = db

    def assemble(self, repo: Repository | None, conversation: Conversation | None, current_message: str, limit: int | None = None) -> AssembledContext:
        # No message-count or per-message truncation: fetch the complete active
        # projection. The complete provider request is the only capacity gate.
        repo_id = repo.id if repo else None
        conversation_id = conversation.id if conversation else None
        store = MessageStore(self.db)
        boundary = store.last_compact_boundary(repo_id, conversation_id)
        rows = store.active_for_context(repo_id, conversation_id, limit=None)
        memory = (
            self.db.query(ConversationMemory).filter(ConversationMemory.conversation_id == conversation_id).first()
            if conversation_id else self.db.query(ThreadMemory).filter(_repo_filter(ThreadMemory, repo_id)).first()
        )
        if memory is None and repo_id and conversation_id:
            memory = ConversationMemory(repo_id=repo_id, conversation_id=conversation_id, summary="")
            self.db.add(memory)
            self.db.flush()
        if memory is not None:
            working = dict(memory.working_state or {})
            processed = set(working.get("processed_message_ids") or [])
            for row in rows:
                if row.role == "user" and str(row.id) not in processed:
                    working = update_working_state(working, row.content, str(row.id))
                    processed.add(str(row.id))
            memory.working_state = working
        history = [{"role": row.role, "content": row.content, "id": str(row.id)}
                   for row in rows if row.role in {"user", "assistant"}]
        tool_messages = [
            {"id": str(row.id), "role": row.role, "content": row.content, "route": row.route,
             "meta": {**(row.meta or {}), "recovery": {"tool": "devflow_get_thread_context", "message_id": str(row.id)}}}
            for row in rows if row.role in {"tool", "tool_result"}
        ]
        evidence = EvidenceStore(self.db).search(repo_id, conversation_id, current_message, limit=8)
        sections: list[str] = []
        if repo:
            sections.append(f"[Current Repository]\n{repo.full_name}\n[/Current Repository]")
        if conversation:
            sections.append(f"[Current Conversation]\n{conversation.title}\n[/Current Conversation]")
        if memory is not None:
            protected = render_working_state(memory.working_state)
            if protected:
                sections.append(f"[Current Working State]\n{protected}\n[/Current Working State]")
        required_context = "\n\n".join(sections)
        memory_context = ""
        if boundary:
            sections.append(f"[Compaction Boundary]\n{boundary.content}\n"
                            "历史摘要可能含旧推断，以当前工作状态和用户纠正为准。"
                            "完整原文通过 devflow_get_thread_context 回查。\n[/Compaction Boundary]")
        if memory is not None and memory_document_text(memory).strip():
            memory_context = f"[Thread Memory]\n{render_structured_memory(memory)}\n[/Thread Memory]"
            sections.append(memory_context)
        base = "\n\n".join(sections)
        system_context = compose_server_context(base, evidence, tool_messages)
        current_in_history = bool(history and history[-1]["role"] == "user" and history[-1]["content"] == current_message)
        pending = 0 if current_in_history else estimate_tokens(current_message)
        history_tokens = estimate_tokens(dumps_compact(history))
        total = estimate_tokens(system_context) + history_tokens + pending
        coverage = {
            "source_messages": len(history), "prompt_messages": len(history), "omitted_messages": 0,
            "truncated_messages": 0, "exact_suffix_messages": len(history),
            "source_limit_reached": False, "has_compaction_boundary": boundary is not None,
            "microcompacted_messages": 0, "complete_and_exact": boundary is None,
        }
        stats = {
            "mode": "shared_request_budget", "pressure": calculate_context_pressure(total).as_dict(),
            "history": {"input_messages": len(history), "kept_messages": len(history), "estimated_tokens": history_tokens,
                        "omitted_messages": 0, "truncated_messages": 0, "exact_suffix_messages": len(history)},
            "history_coverage": coverage, "microcompact": None,
            "compact_boundary": {"id": str(boundary.id) if boundary else None},
            "assembled_prompt_estimated_tokens": total, "system_context_estimated_tokens": estimate_tokens(system_context),
            "history_estimated_tokens": history_tokens, "pending_user_estimated_tokens": pending,
            "prompt_breakdown": {"system_memory_evidence": estimate_tokens(system_context),
                                "recent_history": history_tokens, "pending_user": pending,
                                "reserved_response": request_budget().reserved_response_tokens},
        }
        return AssembledContext(system_context, history, evidence, stats, base, tool_messages,
                                [str(row.id) for row in rows if row.role in {"user", "assistant", "tool", "tool_result"}],
                                required_context, memory_context)


class ProgressiveContextManager:
    def __init__(self, db: Session, llm: LLMClient | None = None) -> None:
        self.db = db
        self.llm = llm or LLMClient()

    async def ensure_headroom(self, repo: Repository | None, conversation: Conversation | None, current_message: str) -> dict[str, Any]:
        # Capacity is assessed only after skills, tools and fixed instructions
        # have been assembled by ChatAgent. Raw database size is not a trigger.
        return {"attempted": False, "compaction_applied": False, "reason": "deferred_to_complete_request"}

    def request_callback(self, repo: Repository, conversation: Conversation):
        async def compact(context: dict, current_message: str, measure, input_limit: int, *, stage: str = "auto_compact") -> dict:
            result = await self.compact_request(repo, conversation, current_message, context, measure, input_limit, stage=stage)
            if not context.get("evaluation_mode"):
                self.db.commit()
            else:
                self.db.flush()
            return result
        return compact

    async def compact_request(
        self, repo, conversation, current_message, context, measure, input_limit,
        *, stage: str = "auto_compact",
    ) -> dict:
        """Run the second-level compaction for ONE water-mark stage.

        The caller derives ``stage`` from the context water mark and only this
        stage's plan is executed. The first level (microcompaction) belongs to
        the caller, so ``stage == "warning"`` is a no-op here.
        """
        if not settings.context_auto_compact_enabled:
            return {"attempted": False, "compaction_applied": False, "headroom_ok": False, "reason": "disabled"}
        if stage == "warning":
            return {"attempted": False, "compaction_applied": False, "headroom_ok": False,
                    "reason": "microcompact_stage"}
        rows = MessageStore(self.db).active_for_context(repo.id, conversation.id)
        source_snapshot = [str(row.id) for row in rows if row.role in {"user", "assistant", "tool", "tool_result"}]
        active_ids = set(context.get("_active_message_ids") or [])
        candidate_rows = [row for row in rows if row.role in {"user", "assistant", "tool", "tool_result"}
                          and (not active_ids or str(row.id) in active_ids)]
        if len(candidate_rows) < 3:
            return {"attempted": False, "compaction_applied": False, "headroom_ok": False, "reason": "nothing_to_summarize"}
        baseline = measure({})
        control = self._control(repo.id, conversation.id)
        meta = dict(control.meta or {}) if control else {}
        now = datetime.now(timezone.utc)
        retry_at = datetime.fromisoformat(meta["retry_at"]) if meta.get("retry_at") else None
        if retry_at and retry_at > now and int(meta.get("failures") or 0) >= settings.context_max_compaction_failures:
            return {"attempted": False, "compaction_applied": False, "headroom_ok": False,
                    "reason": "circuit_breaker_open", "retry_at": meta["retry_at"]}
        attempts: list[dict[str, Any]] = []
        plan = self._compaction_plan_for_stage(stage)
        try:
            kept = self._messages_to_keep(candidate_rows, int(plan["keep_recent"]))
            keep_ids = {row.id for row in kept}
            summarized = [row for row in candidate_rows if row.id not in keep_ids]
            # 压了等于没压：可压的条数不比保留的多出足够多时，摘要的开销抵不上收益。
            if len(summarized) <= len(kept) + 1:
                return {"attempted": True, "compacted": False, "compaction_applied": False,
                        "headroom_ok": False, "reason": "not_worth_compacting", "stage": stage,
                        "messages_summarized": len(summarized), "messages_preserved": len(kept)}
            summary, mode, preservation = await self._summary_for(repo, conversation, summarized, current_message, int(plan["summary_tokens"]))
            protected = str(context.get("_context_required") or "")
            still_missing = [item for item in preservation.get("missing", [])
                             if item.removesuffix("...") not in protected]
            preservation = {**preservation, "preserved_in_working_state": int(preservation.get("missing_count") or 0) - len(still_missing),
                            "missing": still_missing, "missing_count": len(still_missing)}
            if preservation.get("missing_count"):
                attempts.append({"plan": plan, "reason": "required_facts_missing", "preservation": preservation})
                self._save_control(repo.id, conversation.id, {
                    "failures": int(meta.get("failures") or 0), "status": "target_not_met", "stage": stage,
                    "retry_at": (now + timedelta(seconds=settings.context_compaction_cooldown_seconds)).isoformat(),
                })
                return {"attempted": True, "compacted": False, "compaction_applied": False, "headroom_ok": False,
                        "reason": "required_facts_missing", "stage": stage, "compaction_attempts": attempts}
            base = "\n\n".join([str(context.get("_context_required") or ""), str(context.get("_context_memory") or "")]) if "_context_required" in context else str(context.get("_context_base") or context.get("server_context") or "")
            base += f"\n\n[Compaction Boundary]\n{summary}\n原文保留，可通过 devflow_get_thread_context 精确回查。\n[/Compaction Boundary]"
            history = [{"role": row.role, "content": row.content, "id": str(row.id)}
                       for row in kept if row.role in {"user", "assistant"}]
            coverage = {
                "source_messages": len(candidate_rows), "prompt_messages": len(history),
                "omitted_messages": len(summarized), "truncated_messages": 0,
                "exact_suffix_messages": len(history), "has_compaction_boundary": True,
                "complete_and_exact": False, "source_limit_reached": False,
            }
            stats = {**context.get("compression_stats", {}), "history_coverage": coverage,
                     "mode": "shared_request_budget"}
            updates = {"_context_base": base, "_history_tools": [],
                       "_active_message_ids": [str(row.id) for row in kept],
                       "server_context": compose_server_context(base, context.get("evidence") or [], []),
                       "server_messages": history, "compression_stats": stats}
            post_tokens = measure(updates)
            saved = baseline - post_tokens
            attempts.append({"plan": plan, "post_compact_estimated_tokens": post_tokens,
                             "tokens_saved": saved, "headroom_ok": post_tokens <= input_limit})
            if post_tokens <= input_limit and saved >= settings.context_compaction_min_savings_tokens:
                self.db.query(Conversation).filter(Conversation.id == conversation.id).with_for_update().one()
                fresh = MessageStore(self.db).active_for_context(repo.id, conversation.id)
                if [str(row.id) for row in fresh if row.role in {"user", "assistant", "tool", "tool_result"}] != source_snapshot:
                    return {"attempted": True, "compaction_applied": False, "headroom_ok": False, "reason": "history_changed_retry_next_request"}
                # Tool results created during this turn are already in the
                # runtime observations, but must survive the persisted boundary.
                unseen = [row for row in rows if row.role in {"user", "assistant", "tool", "tool_result"}
                          and active_ids and str(row.id) not in active_ids]
                with self.db.begin_nested():
                    boundary = self._insert_boundary(
                        repo, conversation, summary, mode, {"token_usage": baseline, "input_limit_tokens": input_limit, "stage": stage},
                        summarized, [*kept, *unseen], preservation, self._rehydration_state(candidate_rows))
                    self._save_control(repo.id, conversation.id, {"failures": 0, "status": "applied", "stage": stage})
                stats["compact_boundary"] = {"id": str(boundary.id)}
                return {"attempted": True, "compacted": True, "compaction_applied": True, "headroom_ok": True,
                        "reason": "applied", "stage": stage, "boundary_id": str(boundary.id), "context_updates": updates,
                        "messages_summarized": len(summarized), "messages_preserved": len(kept),
                        "rehydration": self._rehydration_state(candidate_rows),
                        "post_compact_estimated_tokens": post_tokens, "compaction_attempts": attempts}
            self._save_control(repo.id, conversation.id, {
                "failures": int(meta.get("failures") or 0), "status": "target_not_met", "stage": stage,
                "retry_at": (now + timedelta(seconds=settings.context_compaction_cooldown_seconds)).isoformat(),
            })
            return {"attempted": True, "compacted": False, "compaction_applied": False, "headroom_ok": False,
                    "reason": "target_not_met", "stage": stage, "compaction_attempts": attempts}
        except Exception as exc:
            failures = int(meta.get("failures") or 0) + 1
            self._save_control(repo.id, conversation.id, {
                "failures": failures, "status": "execution_failed", "stage": stage,
                "retry_at": (now + timedelta(seconds=settings.context_compaction_cooldown_seconds)).isoformat(),
            })
            return {"attempted": True, "compacted": False, "compaction_applied": False, "headroom_ok": False,
                    "reason": "execution_failed", "error": str(exc), "consecutive_failures": failures}

    def _control(self, repo_id, conversation_id):
        return self.db.query(ChatMessage).filter(
            ChatMessage.repo_id == repo_id, ChatMessage.conversation_id == conversation_id,
            ChatMessage.meta["subtype"].as_string() == "compact_control"
        ).order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc()).first()

    def _save_control(self, repo_id, conversation_id, values):
        row = self._control(repo_id, conversation_id)
        if row is None:
            row = ChatMessage(repo_id=repo_id, conversation_id=conversation_id, role="system",
                              route="context_compaction", content="压缩控制状态")
            self.db.add(row)
        row.meta = {"subtype": "compact_control", **values}
        row.created_at = datetime.now(timezone.utc)
        self.db.flush()

    async def _compact(self, repo, conversation, current_message, pressure, stage):
        # Compatibility entrypoint for callers/tests: no boundary can be
        # committed without a measurement of a complete candidate request.
        from app.services.agents.chat_agent import ChatAgent
        assembled = ContextAssembler(self.db).assemble(repo, conversation, current_message)
        context = assembled.runtime_fields()
        agent = ChatAgent()
        state = {"message": current_message, "context": context, "tools": {}, "observations": [], "agent_events": []}
        from app.services.request_budget import measure_request
        def measure(updates):
            state["context"] = {**context, **updates}
            return measure_request(agent._tool_calling_system_prompt(), agent._tool_calling_user_prompt(state), [])["estimated_prompt_tokens"]
        return await self.compact_request(repo, conversation, current_message, context, measure, request_budget().input_limit_tokens, stage=stage)

    def _compaction_plan_for_stage(self, stage: str) -> dict[str, Any]:
        default_keep = max(2, settings.context_compact_keep_recent_messages)
        default_summary_tokens = max(256, settings.context_compact_summary_tokens)
        if stage == "manual_path":
            return {
                "keep_recent": max(2, min(default_keep, max(2, default_keep // 3))),
                "summary_tokens": min(default_summary_tokens, max(256, int(default_summary_tokens * 0.5))),
                "mode_prefix": "manual_path",
            }
        if stage == "aggressive":
            return {
                "keep_recent": max(2, min(default_keep, max(3, default_keep // 2))),
                "summary_tokens": min(default_summary_tokens, max(256, int(default_summary_tokens * 0.67))),
                "mode_prefix": "aggressive",
            }
        return {"keep_recent": default_keep, "summary_tokens": default_summary_tokens, "mode_prefix": ""}

    def _messages_to_keep(self, rows: list[ChatMessage], keep_count: int | None = None) -> list[ChatMessage]:
        keep = max(2, keep_count if keep_count is not None else settings.context_compact_keep_recent_messages)
        start_index = max(0, len(rows) - keep)
        start_index = adjust_keep_start_to_preserve_api_invariants(
            [self._row_to_message(row) for row in rows],
            start_index,
        )
        keep_rows = rows[start_index:]
        if (
            keep_rows
            and keep_rows[0].role == "assistant"
            and start_index > 0
            and rows[start_index - 1].role == "user"
        ):
            keep_rows = [rows[start_index - 1], *keep_rows]
        return keep_rows

    async def _summary_for(self, repo, conversation, rows, current_message, max_summary_tokens):
        # Cover the whole retired interval; a previous memory snapshot is not
        # evidence that every source message has been summarized.
        messages = [self._row_to_message(row) for row in rows]
        chunk_limit = max(256, min(24000, request_budget().input_limit_tokens - 2500))
        chunks = []
        pending = ""
        for row in rows:
            text = f"[message_id={row.id} role={row.role}]\n{row.content}\n"
            while text:
                text = text.lstrip()
                piece = clip_to_token_budget(text, chunk_limit, suffix="")
                if not piece:
                    break
                if pending and estimate_tokens(pending + piece) > chunk_limit:
                    chunks.append(pending)
                    pending = ""
                pending += piece
                text = text[len(piece):]
        if pending:
            chunks.append(pending)
        summaries = []
        used_fallback = False
        for chunk in chunks:
            reply = await self.llm.chat_text(
                COMPACT_SUMMARY_SYSTEM_PROMPT,
                "总结以下有来源标识的历史。保留目标、用户约束、决定、纠正和未完成任务。"
                "不要把假设写成事实。当前请求：\n" + clip_to_token_budget(current_message, 400) + "\n\n" + chunk,
            )
            if reply.strip():
                summaries.append(format_compaction_summary(reply))
            else:
                used_fallback = True
                summaries.append(clip_to_token_budget(chunk, max_summary_tokens))
        combined = "\n\n".join(summaries)
        if estimate_tokens(combined) > max_summary_tokens and not used_fallback:
            reply = await self.llm.chat_text(COMPACT_SUMMARY_SYSTEM_PROMPT,
                "合并阶段摘要，保留明确约束和纠正：\n" + combined)
            if reply.strip():
                combined = format_compaction_summary(reply)
        if not combined.strip():
            combined = fallback_compaction_summary(messages, max_tokens=max_summary_tokens)
        summary, report = ensure_compaction_summary_preserves_key_facts(messages, combined, max_summary_tokens)
        report.update(source_message_count=len(rows), source_chunks=len(chunks), coverage="all_source_messages")
        return summary, "fallback_extractive" if used_fallback else "full_compaction", report

    def _insert_boundary(
        self,
        repo: Repository,
        conversation: Conversation,
        summary: str,
        mode: str,
        pressure: dict[str, Any],
        summarized_rows: list[ChatMessage],
        keep_rows: list[ChatMessage],
        preservation_report: dict[str, Any],
        rehydration: dict[str, Any],
    ) -> ChatMessage:
        now = datetime.now(timezone.utc)
        latest_source_time = max(
            (row.created_at.replace(tzinfo=timezone.utc) if row.created_at.tzinfo is None else row.created_at
             for row in [*summarized_rows, *keep_rows] if row.created_at),
            default=None,
        )
        if latest_source_time:
            comparable_now = now if latest_source_time.tzinfo else now.replace(tzinfo=None)
            if latest_source_time >= comparable_now:
                now = latest_source_time + timedelta(microseconds=1)
        boundary = ChatMessage(
            repo_id=repo.id,
            conversation_id=conversation.id,
            role="system",
            content=summary,
            route="context_compaction",
            tool_calls=[],
            meta={
                "subtype": COMPACT_BOUNDARY_SUBTYPE,
                "trigger": "auto",
                "mode": mode,
                "pressure": pressure,
                "pre_compact_message_id": str(summarized_rows[-1].id) if summarized_rows else None,
                "summarized_from_message_id": str(summarized_rows[0].id) if summarized_rows else None,
                "summarized_to_message_id": str(summarized_rows[-1].id) if summarized_rows else None,
                "preserved_start_message_id": str(keep_rows[0].id) if keep_rows else None,
                "preserved_message_ids": [str(row.id) for row in keep_rows],
                "transcript_ref": {
                    "repo_id": str(repo.id),
                    "conversation_id": str(conversation.id),
                    "tool": "devflow_get_thread_context",
                },
                "messages_summarized": len(summarized_rows),
                "messages_preserved": len(keep_rows),
                "summary_estimated_tokens": estimate_tokens(summary),
                "preservation": preservation_report,
                "rehydration": rehydration,
            },
            created_at=now,
        )
        self.db.add(boundary)
        self.db.flush()
        return boundary


    def _rehydration_state(self, rows: list[ChatMessage]) -> dict[str, Any]:
        working_files: list[str] = []
        active_skills: list[str] = []
        recent_tools: list[str] = []

        def remember(target: list[str], value: Any, limit: int) -> None:
            clean = str(value or "").strip()
            if not clean or clean in target:
                return
            target.append(clean)
            if len(target) > limit:
                del target[0 : len(target) - limit]

        for row in rows:
            calls = [item for item in (row.tool_calls or []) if isinstance(item, dict)]
            events = [item for item in (row.meta or {}).get("agent_events") or [] if isinstance(item, dict)]
            for call in calls:
                remember(recent_tools, call.get("tool_name"), 8)
                arguments = call.get("arguments") or {}
                if isinstance(arguments, dict):
                    remember(working_files, arguments.get("path") or arguments.get("file_path"), 5)
            for event in events:
                event_type = str(event.get("type") or "")
                if event_type in {"tool_use", "tool_result"}:
                    remember(recent_tools, event.get("tool_name"), 8)
                arguments = event.get("arguments") or {}
                if isinstance(arguments, dict):
                    remember(working_files, arguments.get("path") or arguments.get("file_path"), 5)
                if event_type == "skill_activated":
                    remember(active_skills, event.get("skill_name") or event.get("skill_title"), 6)

        return {
            "working_files": working_files,
            "active_skills": active_skills,
            "recent_tools": recent_tools,
            "recovery": {
                "exact_transcript_tool": "devflow_get_thread_context",
                "sealed_session_tool": "devflow_read_session_events",
            },
        }


    def _row_to_message(self, row: ChatMessage) -> dict[str, Any]:
        return {
            "id": str(row.id),
            "role": row.role,
            "content": row.content,
            "route": row.route,
            "tool_calls": row.tool_calls or [],
            "meta": row.meta or {},
        }




class SessionSealer:
    def __init__(self, db: Session, llm: LLMClient | None = None) -> None:
        self.db = db
        self.llm = llm or LLMClient()

    async def seal(
        self,
        *,
        repo: Repository | None,
        conversation: Conversation | None,
        user_message: ChatMessage,
        assistant_message: ChatMessage,
        route: str,
        tool_calls: list[dict[str, Any]],
        agent_events: list[dict[str, Any]] | None = None,
        model_usage: list[dict[str, Any]] | None = None,
        context_diagnostics: dict[str, Any] | None = None,
    ) -> ChatSessionModel:
        agent_events = agent_events or []
        model_usage = model_usage or []
        context_diagnostics = context_diagnostics or {}
        fallback_update = fallback_memory_update(
            user_message.content,
            assistant_message.content,
            route,
            tool_calls,
        )
        fallback_digest = str(fallback_update.get("summary") or "")
        session = ChatSession(
            repo_id=repo.id if repo else None,
            conversation_id=conversation.id if conversation else None,
            user_message_id=user_message.id,
            assistant_message_id=assistant_message.id,
            route=route,
            transcript_json={
                "messages": [
                    {"role": "user", "content": user_message.content, "created_at": self._iso(user_message.created_at)},
                    {"role": "assistant", "content": assistant_message.content, "created_at": self._iso(assistant_message.created_at)},
                ],
                "tool_calls": tool_calls,
                "agent_events": agent_events,
                "tool_use_results": [
                    event for event in agent_events if event.get("type") in {"tool_use", "tool_result"}
                ],
                "model_usage": model_usage,
                "context_diagnostics": context_diagnostics,
                "memory_update": fallback_update,
                "sealing": {"phase": "raw_persisted", "errors": []},
            },
            digest=fallback_digest,
            status="sealing",
            sealed_at=None,
        )
        self.db.add(session)
        self.db.flush()

        for event in agent_events:
            if event.get("type") != "tool_result":
                continue
            if event.get("stored_message_id"):
                continue
            raw = event.get("raw_observation", event.get("content"))
            if raw is None:
                continue
            stored = MessageStore(self.db).append(
                repo_id=repo.id if repo else None, conversation_id=conversation.id if conversation else None,
                role="tool_result", content=dumps_compact(raw), route=str(event.get("tool_name") or "tool"),
                meta={"subtype": "tool_result", "session_id": str(session.id),
                      "tool_use_id": event.get("tool_use_id"), "regenerable": True,
                      "recovery_mode": "saved_result", "source_version": (raw.get("version") if isinstance(raw, dict) else None)},
            )
            event["stored_message_id"] = str(stored.id)

        if route == "context_budget_exceeded":
            # Preserve the transcript, but do not resend the rejected input to
            # the memory model or turn a budget rejection into working facts.
            transcript = dict(session.transcript_json or {})
            transcript["memory_update"] = {}
            transcript["sealing"] = {
                "phase": "completed",
                "errors": [],
                "memory_enrichment": "skipped_request_budget",
                "evidence_refresh": "deferred_to_next_request",
            }
            session.transcript_json = transcript
            session.status = "sealed"
            session.sealed_at = datetime.now(timezone.utc)
            self.db.flush()
            return session

        errors: list[str] = []
        memory_update = fallback_update
        digest = fallback_digest
        try:
            memory_update = await self._build_memory_update(
                user_message.content,
                assistant_message.content,
                route,
                tool_calls,
                fallback_digest,
                repo_id=repo.id if repo else None,
                conversation_id=conversation.id if conversation else None,
            )
            digest = str(memory_update.pop("_digest", "") or memory_update.get("summary") or fallback_digest)
        except Exception as exc:  # model/network failures must not invalidate the completed chat turn
            errors.append(f"memory_extraction_failed: {exc}")
            memory_update = fallback_update
            digest = fallback_digest

        try:
            with self.db.begin_nested():
                if conversation is not None:
                    latest = self.db.query(ConversationMemory).filter(ConversationMemory.conversation_id == conversation.id).with_for_update().populate_existing().first()
                    last_message_id = (latest.working_state or {}).get("last_message_id") if latest else None
                    if last_message_id and last_message_id != str(user_message.id):
                        raise RuntimeError("stale_memory_update_skipped: newer user input already exists")
                updated_memory = self._update_thread_memory(
                    repo.id if repo else None,
                    conversation.id if conversation else None,
                    digest,
                    memory_update,
                )
                capture_memory_candidates(
                    self.db,
                    repo_id=repo.id if repo else None,
                    conversation_id=conversation.id if conversation else None,
                    session_id=session.id,
                    memory_update=self._memory_state(updated_memory) if updated_memory else memory_update,
                )
                self.db.flush()
        except Exception as exc:  # keep raw ChatSession even when enrichment fails
            errors.append(f"memory_persistence_failed: {exc}")

        transcript = dict(session.transcript_json or {})
        transcript["memory_update"] = {
            key: value for key, value in memory_update.items() if not key.startswith("_")
        }
        transcript["sealing"] = {
            "phase": "completed",
            "errors": errors,
            "evidence_refresh": "deferred_to_next_request",
        }
        session.transcript_json = transcript
        session.digest = digest
        session.status = "sealed_with_warnings" if errors else "sealed"
        session.sealed_at = datetime.now(timezone.utc)
        self.db.flush()
        return session

    async def _build_memory_update(
        self,
        user_message: str,
        assistant_message: str,
        route: str,
        tool_calls: list[dict[str, Any]],
        digest: str,
        *,
        repo_id: uuid.UUID | None,
        conversation_id: uuid.UUID | None,
    ) -> dict[str, Any]:
        fallback = fallback_memory_update(user_message, assistant_message, route, tool_calls)
        if repo_id and conversation_id:
            current_memory: ConversationMemory | ThreadMemory | None = (
                self.db.query(ConversationMemory)
                .filter(ConversationMemory.conversation_id == conversation_id)
                .first()
            )
        else:
            current_memory = (
                self.db.query(ThreadMemory)
                .filter(_repo_filter(ThreadMemory, repo_id))
                .first()
            )
        existing_memory = self._memory_state(current_memory) if current_memory else {}
        prompt = (
            "请增量更新软件工程 Agent 的数据库会话记忆。"
            "输入包含已有记忆和本轮原始记录；返回更新后的完整快照，而不是只返回本轮增量。"
            "只返回 JSON，包含 digest、summary、facts、decisions、open_questions、tasks、"
            "user_preferences、repo_context、citations。summary 必须描述当前工作状态和下一步；"
            "保留重要文件、函数、命令、错误修复和用户纠正，不要编造。"
        )
        parsed = await self.llm.chat_json(
            prompt,
            {
                "route": route,
                "digest": digest,
                "existing_memory": existing_memory,
                "user_message": user_message,
                "assistant_message": assistant_message,
                "tool_calls": tool_calls,
            },
        )
        update = structured_memory_from_model(parsed) if parsed else fallback
        if not update.get("summary"):
            update["summary"] = fallback["summary"]
        if parsed:
            update["_update_mode"] = "snapshot"
            update["_digest"] = str(parsed.get("digest") or parsed.get("summary") or digest)
        return update

    def _memory_state(self, memory: ConversationMemory | ThreadMemory) -> dict[str, Any]:
        return {
            "working_state": memory.working_state or {},
            "summary": memory.summary or "",
            "facts": getattr(memory, "facts", None) or [],
            "decisions": memory.decisions or [],
            "open_questions": memory.open_questions or [],
            "tasks": getattr(memory, "tasks", None) or [],
            "user_preferences": getattr(memory, "user_preferences", None) or [],
            "repo_context": getattr(memory, "repo_context", None) or [],
            "citations": getattr(memory, "citations", None) or [],
        }

    def _new_memory(self, repo_id: uuid.UUID | None, conversation_id: uuid.UUID | None) -> ConversationMemory | ThreadMemory:
        if repo_id and conversation_id:
            memory = ConversationMemory(
                repo_id=repo_id,
                conversation_id=conversation_id,
                summary="",
                facts=[],
                decisions=[],
                open_questions=[],
                tasks=[],
                user_preferences=[],
                repo_context=[],
                citations=[],
                sessions_incorporated=0,
            )
        else:
            memory = ThreadMemory(
                repo_id=repo_id,
                summary="",
                facts=[],
                decisions=[],
                open_questions=[],
                tasks=[],
                user_preferences=[],
                repo_context=[],
                citations=[],
                sessions_incorporated=0,
            )
        self.db.add(memory)
        self.db.flush()
        return memory

    def _update_thread_memory(
        self,
        repo_id: uuid.UUID | None,
        conversation_id: uuid.UUID | None,
        digest: str,
        memory_update: dict[str, Any],
    ) -> ConversationMemory | ThreadMemory | None:
        if repo_id and conversation_id:
            memory: ConversationMemory | ThreadMemory | None = self.db.query(ConversationMemory).filter(ConversationMemory.conversation_id == conversation_id).first()
        else:
            memory = self.db.query(ThreadMemory).filter(_repo_filter(ThreadMemory, repo_id)).first()
        if not memory:
            memory = self._new_memory(repo_id, conversation_id)
        if memory_update.get("_update_mode") == "snapshot":
            merged = {
                **structured_memory_from_model(memory_update),
                "summary": clip_to_token_budget(str(memory_update.get("summary") or digest), 1100),
            }
        else:
            merged = merge_memory(self._memory_state(memory), memory_update, digest, max_tokens=1100)
        merged = reconcile_memory(merged, memory.working_state or {})
        memory.summary = merged["summary"]
        memory.facts = merged["facts"]
        memory.decisions = merged["decisions"]
        memory.open_questions = merged["open_questions"]
        memory.tasks = merged["tasks"]
        memory.user_preferences = merged["user_preferences"]
        memory.repo_context = merged["repo_context"]
        memory.citations = merged["citations"]
        memory.sessions_incorporated = (memory.sessions_incorporated or 0) + 1
        return memory

    def _iso(self, value: datetime | None) -> str | None:
        return value.isoformat() if value else None


class MemoryTools:
    def __init__(self, db: Session, repo: Repository | None, conversation: Conversation | None = None) -> None:
        self.db = db
        self.repo = repo
        self.repo_id = repo.id if repo else None
        self.conversation = conversation
        self.conversation_id = conversation.id if conversation else None

    async def search_evidence(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        query = str(input_data.get("query") or input_data.get("message") or "")
        results = EvidenceStore(self.db).search(self.repo_id, self.conversation_id, query, limit=int(input_data.get("limit") or 5))
        answer = "No evidence matched this query." if not results else "\n".join(
            f"- [{item['source_type']}] {item['title']}: {item['snippet']}" for item in results
        )
        citations = [{"type": item["source_type"], "id": item["source_id"], "title": item["title"]} for item in results]
        record_recall_event(
            self.db,
            repo_id=self.repo_id,
            conversation_id=self.conversation_id,
            tool_name="devflow_search_evidence",
            query=query,
            results=results,
            citations=citations,
            scope="conversation" if self.conversation_id else "project",
            mode="hybrid",
            metadata={"limit": int(input_data.get("limit") or 5)},
        )
        return {
            "route": "memory_search",
            "tool_calls": [{"tool_name": "search_evidence", "arguments": {"query": query}, "status": "success"}],
            "answer": answer,
            "citations": citations,
            "results": results,
        }

    async def get_thread_context(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        keyword = str(input_data.get("keyword") or input_data.get("query") or "").strip().lower()
        limit = min(max(int(input_data.get("limit") or 30), 1), 100)
        message_id = self._uuid_or_none(input_data.get("message_id"))
        before_message_id = self._uuid_or_none(input_data.get("before_message_id"))
        after_message_id = self._uuid_or_none(input_data.get("after_message_id"))
        offset = max(int(input_data.get("offset") or 0), 0)
        max_chars = min(max(int(input_data.get("max_chars") or 6000), 200), 16000)

        store = MessageStore(self.db)
        if message_id:
            message = (
                self.db.query(ChatMessage)
                .filter(
                    _repo_filter(ChatMessage, self.repo_id),
                    _conversation_filter(ChatMessage, self.conversation_id),
                    ChatMessage.id == message_id,
                    ChatMessage.role.in_(["user", "assistant", "tool", "tool_result"]),
                )
                .one_or_none()
            )
            messages = [message] if message and (message.meta or {}).get("subtype") != "preserved_after_compact" else []
        else:
            messages = store.timeline(
                self.repo_id,
                self.conversation_id,
                limit=limit,
                before_message_id=before_message_id,
                after_message_id=after_message_id,
                keyword=keyword,
            )

        lines: list[str] = []
        results: list[dict[str, Any]] = []
        remaining_chars = max_chars
        for message in messages:
            content = message.content or ""
            chunk_offset = offset if message_id else 0
            chunk = content[chunk_offset : chunk_offset + remaining_chars]
            if not chunk and content:
                continue
            chunk_end = chunk_offset + len(chunk)
            truncated = chunk_end < len(content)
            created_at = message.created_at.isoformat() if message.created_at else ""
            lines.append(
                f"[message_id={message.id} role={message.role} created_at={created_at} chars={chunk_offset}:{chunk_end}/{len(content)}]"
            )
            lines.append(chunk)
            if truncated:
                lines.append(
                    f"[内容未读完；继续调用 message_id={message.id}, offset={chunk_end}]"
                )
            results.append(
                {
                    "title": f"{message.role} message",
                    "source_type": "chat_message",
                    "source_id": str(message.id),
                    "snippet": chunk,
                    "metadata": {
                        "role": message.role,
                        "created_at": created_at,
                        "offset": chunk_offset,
                        "next_offset": chunk_end if truncated else None,
                        "total_chars": len(content),
                        "truncated": truncated,
                    },
                }
            )
            remaining_chars -= len(chunk)
            if remaining_chars <= 0:
                break

        if results and not message_id:
            lines.append(
                f"[需要更早原文时使用 before_message_id={results[0]['source_id']}；"
                f"需要更晚原文时使用 after_message_id={results[-1]['source_id']}。"
                "本页没有目标内容不代表其他页不存在。]"
            )
        citations = [
            {"type": "chat_message", "id": item["source_id"], "title": item["title"]}
            for item in results
        ]
        record_recall_event(
            self.db,
            repo_id=self.repo_id,
            conversation_id=self.conversation_id,
            tool_name="devflow_get_thread_context",
            query=keyword or str(message_id or ""),
            results=results,
            citations=citations,
            scope="conversation",
            mode="exact_transcript",
            metadata={
                "limit": limit,
                "message_id": str(message_id) if message_id else None,
                "before_message_id": str(before_message_id) if before_message_id else None,
                "after_message_id": str(after_message_id) if after_message_id else None,
                "offset": offset,
                "max_chars": max_chars,
            },
        )
        return {
            "route": "thread_context",
            "tool_calls": [{"tool_name": "get_thread_context", "arguments": {"keyword": keyword, "message_id": str(message_id or "")}, "status": "success"}],
            "answer": "\n".join(lines) if lines else "当前会话还没有匹配的消息。",
            "citations": citations,
            "results": results,
        }

    async def read_session_events(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        session_id = input_data.get("session_id")
        include_transcript = bool(input_data.get("include_transcript", True))
        offset = max(int(input_data.get("offset") or 0), 0)
        max_chars = min(max(int(input_data.get("max_chars") or 8000), 400), 20000)
        query = self.db.query(ChatSession).filter(
            _repo_filter(ChatSession, self.repo_id),
            _conversation_filter(ChatSession, self.conversation_id),
        ).order_by(ChatSession.created_at.desc())
        if session_id:
            parsed_session_id = self._uuid_or_none(session_id)
            if parsed_session_id is None:
                sessions = []
            else:
                sessions = query.filter(ChatSession.id == parsed_session_id).limit(1).all()
        else:
            sessions = query.limit(min(max(int(input_data.get("limit") or 5), 1), 20)).all()
        lines: list[str] = []
        results: list[dict[str, Any]] = []
        remaining_chars = max_chars
        for session in sessions:
            header = f"Session {session.id} route={session.route} sealed_at={session.sealed_at}"
            payload = session.digest
            if include_transcript:
                payload += "\ntranscript=" + dumps_compact(session.transcript_json or {})
            chunk_offset = offset if session_id else 0
            chunk = payload[chunk_offset : chunk_offset + remaining_chars]
            chunk_end = chunk_offset + len(chunk)
            truncated = chunk_end < len(payload)
            lines.append(header)
            lines.append(chunk)
            if truncated:
                lines.append(
                    f"[会话记录未读完；继续调用 session_id={session.id}, offset={chunk_end}。]"
                )
            results.append(
                {
                    "title": f"Session {session.route}",
                    "source_type": "session",
                    "source_id": str(session.id),
                    "snippet": chunk,
                    "metadata": {
                        "sealed_at": session.sealed_at.isoformat() if session.sealed_at else None,
                        "include_transcript": include_transcript,
                        "offset": chunk_offset,
                        "next_offset": chunk_end if truncated else None,
                        "total_chars": len(payload),
                        "truncated": truncated,
                    },
                }
            )
            remaining_chars -= len(chunk)
            if remaining_chars <= 0:
                break
        citations = [
            {"type": "session", "id": item["source_id"], "title": item["title"]}
            for item in results
        ]
        record_recall_event(
            self.db,
            repo_id=self.repo_id,
            conversation_id=self.conversation_id,
            tool_name="devflow_read_session_events",
            query=str(session_id or "latest sessions"),
            results=results,
            citations=citations,
            scope="conversation",
            mode="sealed_transcript",
            metadata={"include_transcript": include_transcript, "offset": offset, "max_chars": max_chars},
        )
        return {
            "route": "session_events",
            "tool_calls": [{"tool_name": "read_session_events", "arguments": {"session_id": str(session_id or "")}, "status": "success"}],
            "answer": "\n\n".join(lines) if lines else "No sealed sessions found yet.",
            "citations": citations,
            "results": results,
        }

    @staticmethod
    def _uuid_or_none(value: Any) -> uuid.UUID | None:
        if not value:
            return None
        try:
            return uuid.UUID(str(value))
        except (TypeError, ValueError):
            return None
