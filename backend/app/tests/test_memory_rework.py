import copy
import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import ChatMessage, ChatSession, Conversation, ConversationMemory, Document, EvidenceItem, MemoryCandidate, Repository
from app.db.session import Base
from app.services.chat_memory import ContextAssembler, EvidenceStore, MemoryTools, MessageStore, ProgressiveContextManager, SessionSealer
from app.services.context_compression import merge_memory
from app.services.memory_hub import reject_memory_candidate, retire_memory_note
from app.services.request_budget import measure_request, request_budget
from app.services.tool_contracts import tool_args_schema, TOOL_FIELDS
from app.services.working_memory import update_working_state, render_working_state


@pytest.fixture
def scope(monkeypatch):
    monkeypatch.setattr(settings, "context_model_window_tokens", 12000)
    monkeypatch.setattr(settings, "context_reserved_response_tokens", 1024)
    monkeypatch.setattr(settings, "context_request_safety_tokens", 512)
    monkeypatch.setattr(settings, "context_request_safety_ratio", 0.03)
    monkeypatch.setattr(settings, "context_dynamic_model_window_enabled", True)
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        repo = Repository(owner="acme", name="memory", full_name="acme/memory")
        db.add(repo)
        db.flush()
        conversation = Conversation(repo_id=repo.id, title="Memory tests")
        db.add(conversation)
        db.commit()
        yield db, repo, conversation
    engine.dispose()


class SummaryLLM:
    calls = 0

    async def chat_text(self, system, user):
        self.calls += 1
        return "继续完成当前任务，保留用户约束，需要细节时回查原文。"


def assemble_without_search(monkeypatch, db, repo, conversation, message="continue"):
    monkeypatch.setattr(EvidenceStore, "search", lambda *a, **k: [])
    return ContextAssembler(db).assemble(repo, conversation, message)


def full_measure(context, message="continue"):
    def measure(updates):
        candidate = {**context, **updates}
        return measure_request("Rules", candidate["server_context"] + json.dumps(candidate["server_messages"], ensure_ascii=False) + message, [])["estimated_prompt_tokens"]
    return measure


def test_active_history_is_not_limited_to_thirty_or_nine_hundred(scope, monkeypatch):
    db, repo, conversation = scope
    store = MessageStore(db)
    for i in range(64):
        store.append(repo_id=repo.id, conversation_id=conversation.id, role="user" if i % 2 == 0 else "assistant", content=f"message {i} " + "detail " * 1200)
    assembled = assemble_without_search(monkeypatch, db, repo, conversation)
    assert len(assembled.history_messages) == 64
    assert assembled.history_messages[0]["content"].endswith("detail " * 1200)
    assert assembled.compression_stats["history_coverage"]["truncated_messages"] == 0
    monkeypatch.setenv("CONTEXT_RECENT_RATIO", "0.001")  # legacy env no longer controls assembly
    assert ContextAssembler(db).assemble(repo, conversation, "continue").history_messages == assembled.history_messages


@pytest.mark.asyncio
async def test_complete_request_compaction_keeps_constraints_and_exact_transcript(scope, monkeypatch):
    db, repo, conversation = scope
    store = MessageStore(db)
    first = store.append(repo_id=repo.id, conversation_id=conversation.id, role="user", content="目标：定位超时。必须保持接口格式。")
    for i in range(20):
        store.append(repo_id=repo.id, conversation_id=conversation.id, role="user" if i % 2 == 0 else "assistant", content=f"discussion {i} " + "detail " * 1000)
    assembled = assemble_without_search(monkeypatch, db, repo, conversation)
    context = assembled.runtime_fields()
    measure = full_measure(context)
    assert measure({}) > request_budget().input_limit_tokens
    manager = ProgressiveContextManager(db, SummaryLLM())
    limit = request_budget().input_limit_tokens
    result: dict = {}
    for stage in ("auto_compact", "aggressive", "manual_path"):
        result = await manager.compact_request(repo, conversation, "continue", context, measure, limit, stage=stage)
        if result.get("compaction_applied"):
            break
    assert result["compaction_applied"] and result["headroom_ok"], result.get("error") or result
    assert measure(result["context_updates"]) <= request_budget().input_limit_tokens
    assert "必须保持接口格式" in result["context_updates"]["server_context"]
    assert db.get(ChatMessage, first.id).content == "目标：定位超时。必须保持接口格式。"
    recalled = await MemoryTools(db, repo, conversation).get_thread_context({"message_id": str(first.id)}, {})
    assert "必须保持接口格式" in recalled["answer"]


@pytest.mark.asyncio
async def test_failed_target_never_commits_a_boundary(scope, monkeypatch):
    db, repo, conversation = scope
    for i in range(12):
        MessageStore(db).append(repo_id=repo.id, conversation_id=conversation.id, role="user" if i % 2 == 0 else "assistant", content=f"old {i} " * 500)
    context = assemble_without_search(monkeypatch, db, repo, conversation).runtime_fields()
    llm = SummaryLLM()
    manager = ProgressiveContextManager(db, llm)
    result = await manager.compact_request(repo, conversation, "continue", context, lambda _: 999999, 8000, stage="manual_path")
    assert result["reason"] == "target_not_met"
    assert MessageStore(db).last_compact_boundary(repo.id, conversation.id) is None


@pytest.mark.asyncio
async def test_too_few_messages_are_not_worth_compacting(scope, monkeypatch):
    """可压的消息不比保留的多出足够多时，摘要抵不上收益，本次压缩放弃。"""
    db, repo, conversation = scope
    for i in range(6):
        MessageStore(db).append(repo_id=repo.id, conversation_id=conversation.id,
                                role="user" if i % 2 == 0 else "assistant", content=f"m{i} " * 50)
    context = assemble_without_search(monkeypatch, db, repo, conversation).runtime_fields()
    manager = ProgressiveContextManager(db, SummaryLLM())
    result = await manager.compact_request(repo, conversation, "continue", context, lambda _: 100, 100, stage="auto_compact")
    assert result["reason"] == "not_worth_compacting"
    assert MessageStore(db).last_compact_boundary(repo.id, conversation.id) is None


@pytest.mark.asyncio
async def test_circuit_breaker_probes_again_after_cooldown(scope, monkeypatch):
    db, repo, conversation = scope
    for i in range(14):
        MessageStore(db).append(repo_id=repo.id, conversation_id=conversation.id, role="user" if i % 2 == 0 else "assistant", content=f"discussion {i} " * 700)
    context = assemble_without_search(monkeypatch, db, repo, conversation).runtime_fields()
    class BrokenLLM:
        async def chat_text(self, *args):
            raise TimeoutError("temporary outage")
    manager = ProgressiveContextManager(db, BrokenLLM())
    monkeypatch.setattr(settings, "context_max_compaction_failures", 1)
    first = await manager.compact_request(repo, conversation, "continue", context, full_measure(context), 10000, stage="manual_path")
    assert first["reason"] == "execution_failed"
    second = await manager.compact_request(repo, conversation, "changed request", context, full_measure(context), 10000)
    assert second["reason"] == "circuit_breaker_open"
    control = manager._control(repo.id, conversation.id)
    control.meta = {**control.meta, "retry_at": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()}
    db.flush()
    manager.llm = SummaryLLM()
    recovered = await manager.compact_request(repo, conversation, "continue", context, full_measure(context), 10000, stage="manual_path")
    assert recovered["compaction_applied"]
    assert manager._control(repo.id, conversation.id).meta["failures"] == 0


def test_explicit_correction_supersedes_old_fact_and_ambiguous_one_stays_visible():
    state = update_working_state({}, "事实：数据库有问题。必须保持接口格式。", "m1")
    state = update_working_state(state, "更正：数据库有问题 -> 数据库已排除", "m2")
    active = [item["content"] for item in state["entries"] if item["status"] == "active"]
    assert "数据库已排除" in active and "数据库有问题" not in active
    assert "必须保持接口格式" in active
    state = update_working_state(state, "之前那个判断不对", "m3")
    assert state["conflicts"][-1]["status"] == "open"
    assert "需核验冲突" in render_working_state(state)


def test_new_memory_survives_full_fallback_lists():
    merged = merge_memory({"summary": "old " * 1000, "facts": [f"old {i}" for i in range(16)]},
                          {"summary": "NEW CORRECTION", "facts": ["new fact"]}, "", 50)
    assert merged["summary"].startswith("NEW CORRECTION")
    assert merged["facts"][0] == "new fact"


def test_every_tool_has_only_its_own_schema():
    assert len(tool_args_schema("workspace.read_file").model_fields) == 3
    assert "offset" in tool_args_schema("devflow_read_session_events").model_fields
    for name in TOOL_FIELDS:
        assert "repo_id" not in tool_args_schema(name).model_fields
        assert "conversation_id" not in tool_args_schema(name).model_fields


@pytest.mark.asyncio
async def test_mcp_session_offset_is_exposed_and_forwarded(scope, monkeypatch):
    from app.mcp import memory_server
    db, repo, conversation = scope
    schema = next(t for t in memory_server.TOOLS if t["name"] == "devflow_read_session_events")
    assert "offset" in schema["inputSchema"]["properties"]
    captured = {}
    async def fake_read(self, args, context):
        captured.update(args)
        return {"answer": "second page"}
    monkeypatch.setattr(memory_server.Base.metadata, "create_all", lambda **kw: None)
    monkeypatch.setattr(memory_server, "SessionLocal", lambda: db)
    monkeypatch.setattr(MemoryTools, "read_session_events", fake_read)
    result = await memory_server._call_memory_tool("devflow_read_session_events", {
        "repo_id": str(repo.id), "conversation_id": str(conversation.id), "offset": 500,
    })
    assert captured["offset"] == 500
    assert "second page" in result["content"][0]["text"]


def test_revoked_note_is_preserved_but_not_searchable_even_with_stale_vector(scope, monkeypatch):
    from app.services.rag import retrieval
    db, repo, _ = scope
    source = uuid.uuid4()
    doc = Document(repo_id=repo.id, source_type="memory_note", source_id=source, title="retry policy", content="retry policy old", meta={"memory_status": "active"})
    db.add(doc)
    db.flush()
    stale = {"id": str(doc.id), "source_id": str(source), "source_type": "memory_note", "title": doc.title,
             "snippet": doc.content, "metadata": copy.deepcopy(doc.meta), "score": 1.0}
    retire_memory_note(db, repo, source, reason="no longer valid")
    assert db.get(Document, doc.id).content == "retry policy old"
    monkeypatch.setattr(retrieval, "_milvus_vector_candidates", lambda *a, **k: [stale])
    assert retrieval.search_similar_documents(db, repo.id, "retry policy", retrieval_method="vector", apply_rerank=False) == []
    assert retrieval.search_similar_documents(db, repo.id, "retry policy", retrieval_method="fulltext", apply_rerank=False) == []


def test_replacement_must_be_active_and_in_same_repository(scope):
    db, repo, _ = scope
    old = Document(repo_id=repo.id, source_type="memory_note", source_id=uuid.uuid4(), title="old", content="old")
    other = Repository(owner="other", name="repo", full_name="other/repo")
    db.add(other)
    db.flush()
    wrong = Document(repo_id=other.id, source_type="memory_note", source_id=uuid.uuid4(), title="other", content="other")
    db.add_all([old, wrong])
    db.flush()
    with pytest.raises(ValueError, match="same_project"):
        retire_memory_note(db, repo, old.source_id, replacement_id=wrong.source_id)
    replacement = Document(repo_id=repo.id, source_type="memory_note", source_id=uuid.uuid4(), title="new", content="new")
    db.add(replacement)
    db.flush()
    retire_memory_note(db, repo, old.source_id, replacement_id=replacement.source_id)
    assert old.meta["memory_status"] == "superseded"
    assert old.meta["replaced_by_source_id"] == str(replacement.source_id)


def test_reject_is_not_a_backdoor_for_revoking_approved_knowledge(scope):
    db, repo, conversation = scope
    candidate = MemoryCandidate(repo_id=repo.id, conversation_id=conversation.id, kind="fact", title="x", content="x", status="approved")
    db.add(candidate)
    db.flush()
    with pytest.raises(ValueError, match="use_revoke"):
        reject_memory_candidate(db, repo, candidate.id)
    assert candidate.status == "approved"


def test_evidence_refresh_reuses_unchanged_rows_and_isolates_conversations(scope):
    db, repo, conversation = scope
    other = Conversation(repo_id=repo.id, title="other")
    db.add(other)
    db.flush()
    db.add(ChatSession(repo_id=repo.id, conversation_id=other.id, digest="other private session", transcript_json={}))
    db.flush()
    EvidenceStore(db).rebuild_repo(repo, conversation)
    ids = {row.id for row in db.query(EvidenceItem).all()}
    EvidenceStore(db).rebuild_repo(repo, conversation)
    assert ids == {row.id for row in db.query(EvidenceItem).all()}
    assert not any("other private session" in row.content for row in db.query(EvidenceItem).all())


@pytest.mark.asyncio
async def test_agent_compacts_using_the_real_prompt_and_sends_only_fitting_request(scope, monkeypatch):
    from langchain_core.messages import AIMessage
    from app.services.agents.chat_agent import ChatAgent
    db, repo, conversation = scope
    store = MessageStore(db)
    store.append(repo_id=repo.id, conversation_id=conversation.id, role="user", content="目标：调查超时。必须保持接口格式。")
    for i in range(18):
        store.append(repo_id=repo.id, conversation_id=conversation.id, role="user" if i % 2 == 0 else "assistant", content=f"discussion {i} " + "detail " * 1000)
    context = assemble_without_search(monkeypatch, db, repo, conversation).runtime_fields()
    context["tools"] = {}
    context["_context_compactor"] = ProgressiveContextManager(db, SummaryLLM()).request_callback(repo, conversation)
    class Model:
        calls = []
        def bind_tools(self, tools):
            return self
        def bind(self, **kwargs):
            return self
        async def ainvoke(self, messages):
            self.calls.append(messages)
            assert measure_request(messages[0].content, messages[1].content, [])["estimated_prompt_tokens"] <= request_budget().input_limit_tokens
            assert "必须保持接口格式" in messages[1].content
            return AIMessage(content="继续验证。")
    model = Model()
    result = await ChatAgent(model=model).run({"message": "继续排查超时"}, context)
    assert len(model.calls) == 1
    assert result["progressive_compaction"]["compaction_applied"]
    assert result["completion"]["status"] == "completed"


@pytest.mark.asyncio
async def test_tool_result_is_durable_before_sealing_and_microcompact_can_recover_it(scope, monkeypatch):
    from app.api.routes.chat import _install_memory_runtime
    from app.services.context_compression import microcompact_messages
    db, repo, conversation = scope
    context = {}
    assembled = assemble_without_search(monkeypatch, db, repo, conversation)
    _install_memory_runtime(context, assembled, repo, conversation, db)
    text = "readme content " * 2000 + "EXACT_END_MARKER"
    result_id = context["_persist_tool_result"]("workspace.read_file", "call_1", {"answer": text})
    tools = ContextAssembler(db).assemble(repo, conversation, "readme").tool_messages
    assert len(tools) == 1 and tools[0]["id"] == result_id
    projected, stats = microcompact_messages(tools, "readme", keep_recent=0)
    assert stats["compacted_messages"] == 1
    assert len(projected[0]["content"]) < len(tools[0]["content"])
    raw = db.get(ChatMessage, uuid.UUID(result_id))
    answer = await MemoryTools(db, repo, conversation).get_thread_context(
        {"message_id": result_id, "offset": len(raw.content) - 100, "max_chars": 200}, {})
    assert "EXACT_END_MARKER" in answer["answer"]


@pytest.mark.asyncio
async def test_stale_memory_extraction_does_not_overwrite_a_newer_user_correction(scope):
    db, repo, conversation = scope
    store = MessageStore(db)
    old = store.append(repo_id=repo.id, conversation_id=conversation.id, role="user", content="事实：数据库有问题")
    newer = store.append(repo_id=repo.id, conversation_id=conversation.id, role="user", content="更正：数据库有问题 -> 数据库已排除")
    assistant = store.append(repo_id=repo.id, conversation_id=conversation.id, role="assistant", content="old answer")
    class OldModel:
        async def chat_json(self, *args):
            return {"summary": "数据库有问题", "facts": ["数据库有问题"]}
    session = await SessionSealer(db, OldModel()).seal(repo=repo, conversation=conversation, user_message=old,
                assistant_message=assistant, route="chat", tool_calls=[])
    memory = db.query(ConversationMemory).filter_by(conversation_id=conversation.id).one()
    assert memory.working_state["last_message_id"] == str(newer.id)
    assert "数据库有问题" not in (memory.facts or [])
    assert session.status == "sealed_with_warnings"
    assert "stale_memory_update_skipped" in str(session.transcript_json["sealing"]["errors"])


@pytest.mark.asyncio
async def test_required_memory_prefetch_can_continue_to_the_next_page():
    from langchain_core.messages import AIMessage
    from app.services.agents.chat_agent import ChatAgent
    message_id = str(uuid.uuid4())
    offsets = []
    async def read(input_data, context):
        offset = input_data.get("offset", 0)
        offsets.append(offset)
        return {"route": "thread_context", "tool_calls": [], "citations": [], "answer":
                f"原文未读完，继续读取 message_id={message_id}, offset=200" if offset == 0 else "完整约束：不得修改接口。"}
    class Model:
        calls = 0
        def bind_tools(self, tools):
            return self
        async def ainvoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content="", tool_calls=[{"name": "devflow_get_thread_context", "args": {"message_id": message_id, "offset": 200}, "id": "next_page"}])
            return AIMessage(content="原文约束是不得修改接口。")
    result = await ChatAgent(model=Model()).run({"message": "请逐字核实之前说的约束"}, {"tools": {"devflow_get_thread_context": read}})
    assert offsets == [0, 200]
    assert result["answer"] == "原文约束是不得修改接口。"


def test_expired_memory_is_not_an_active_source():
    from app.services.memory_lifecycle import memory_is_active
    assert not memory_is_active({"memory_status": "active", "valid_until": "2000-01-01T00:00:00Z"})
    assert memory_is_active({})  # existing notes remain valid until explicitly retired


def test_mcp_does_not_fall_back_to_another_conversation(scope):
    from app.mcp.memory_server import _resolve_conversation, McpError
    db, repo, _ = scope
    with pytest.raises(McpError):
        _resolve_conversation(db, repo, {"conversation_id": str(uuid.uuid4())})


def test_memory_schema_upgrade_is_additive_and_idempotent():
    from sqlalchemy import inspect, text
    from app.db.session import ensure_memory_state_schema
    engine = create_engine("sqlite+pysqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE conversation_memories (id INTEGER, summary TEXT)"))
        connection.execute(text("INSERT INTO conversation_memories VALUES (1, 'keep original')"))
    ensure_memory_state_schema(engine)
    ensure_memory_state_schema(engine)
    assert "working_state" in {column["name"] for column in inspect(engine).get_columns("conversation_memories")}
    with engine.connect() as connection:
        assert connection.execute(text("SELECT summary FROM conversation_memories")).scalar_one() == "keep original"
    engine.dispose()


def test_preservation_check_measures_the_final_clipped_summary():
    from app.services.context_compression import ensure_compaction_summary_preserves_key_facts
    messages = [{"role": "user", "content": "必须保持接口格式"}]
    summary, report = ensure_compaction_summary_preserves_key_facts(messages, "padding " * 300 + "必须保持接口格式", 40)
    assert "必须保持接口格式" not in summary
    assert report["missing_count"] == 1


@pytest.mark.asyncio
async def test_auxiliary_llm_paths_do_not_send_oversized_memory_requests(scope, monkeypatch):
    from app.services.llm.client import LLMClient
    from app.services.request_budget import RequestBudgetExceeded
    client = LLMClient()
    client.api_key = "test-only-not-a-real-key"
    monkeypatch.setattr("app.services.llm.client.httpx.AsyncClient", lambda *a, **kw: pytest.fail("must not contact provider"))
    huge = "oversized source " * 20000
    assert await client.chat_json("rules", {"source": huge}) == {}
    assert await client.chat_text("rules", huge) == ""
    with pytest.raises(RequestBudgetExceeded):
        _ = [part async for part in client.chat_text_stream("rules", [{"role": "user", "content": huge}])]
