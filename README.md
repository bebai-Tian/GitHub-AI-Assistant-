# GitHub AI Assistant (DevFlow AI)

面向 GitHub Issue、Pull Request 与 CI 的研发协作 AI Agent。系统同步仓库数据，分析 Issue、PR diff、CI 日志、项目知识库和本地工作区代码，给出结构化建议；涉及写操作时只生成草稿，必须人工确认后才会执行。

## 功能

- **仓库同步**：同步仓库中的 Issue、Pull Request、PR 文件、Review 评论、Workflow Run、Job 与日志，支持手动同步和定时同步。
- **Issue 分析**：分类、优先级、复杂度、推荐负责人、重复候选与行动项。
- **PR 审查**：变更摘要、关键变更、风险点、检查清单、测试建议和需要重点关注的文件。
- **CI 排障**：失败类型、可能原因、排查步骤与相关上下文。
- **ChatAgent**：基于 LangGraph 状态图的工具调用 Agent，注册 14 个工具，覆盖工作区、证据检索、记忆、RAG、Issue、PR、CI、周报、安全与工作流。
- **多 Agent 工作流审查**：`WorkflowOrchestrator` 构建 Planner → 专用 Agent → Observer → Synthesis 闭环，用于 PR 就绪度、阻塞项、优先级等跨领域判断。
- **Skill Runtime**：按用户指定或触发条件加载 `SKILL.md`，只向当前任务注入对应指令和受限资源，并把激活原因、指令摘要与校验结果写入 trace。仓库内置 6 个 Skill：`ci-debug`、`issue-triage`、`pr-review`、`project-investigation`、`safety-draft`、`weekly-report`。
- **工作区代码工具**：安全地列目录、读文件、搜索本地代码，路径访问受仓库根目录约束。
- **RAG 知识库**：支持 `.txt`、`.md`、`.markdown`、`.mdx`、`.pdf`、`.docx`、`.json`、`.csv`、`.log` 解析；提供切片、嵌入、向量检索、关键词检索、重排、召回测试、问答和基于证据的编号引用。重复文件按 SHA-256 跳过。
- **上下文与记忆**：统一请求预算、历史压缩、工作状态、会话事件索引、长期记忆的审核状态（待审 / 已批准 / 已拒绝 / 已撤销 / 已替代 / 到期）。
- **工程周报**与**安全风险评估**；写操作通过 `ActionDraft` 记录，需显式 `confirm`。
- **RAGAS 评测**：真实运行 ChatAgent 与工具集，计算 Context Precision / Context Recall / Faithfulness / Answer Relevancy / Agent Goal Accuracy，支持异常重试和回归对比。

## 技术栈

| 层 | 技术 |
| --- | --- |
| 后端 | Python、FastAPI 0.115.6、Pydantic 2.10、SQLAlchemy 2.0、Alembic |
| 数据 | PostgreSQL 16（`pgvector/pgvector` 镜像）、Milvus 2.5（`pymilvus`） |
| Agent / LLM | LangChain 1.2.17、LangGraph 1.1.10、OpenAI 兼容接口、Ragas 0.4.3 |
| 前端 | Next.js 15.1.2、React 19、TypeScript、Tailwind CSS、React Query |
| 基础设施 | Docker Compose、etcd、MinIO |

## 架构

```mermaid
flowchart TD
    FE["Next.js 前端"] --> API["FastAPI 后端"]
    API --> DB["PostgreSQL"]
    API --> Milvus["Milvus 向量库"]
    API --> GH["GitHub REST API"]
    API --> LLM["OpenAI 兼容 LLM API"]

    API --> Chat["ChatAgent（LangGraph）"]
    Chat --> Workspace["工作区工具"]
    Chat --> Memory["记忆 / 证据检索"]
    Chat --> RAG["RAG 工具"]
    Chat --> Issue["Issue 分析 Agent"]
    Chat --> PR["PR 审查 Agent"]
    Chat --> CI["CI 排障 Agent"]
    Chat --> Report["周报 Agent"]
    Chat --> Safety["安全 Agent"]
    Chat --> Skills["Skill 注册中心"]
    Chat --> Workflow["工作流编排器"]
    Workflow --> Planner["Planner"]
    Workflow --> Observer["Observer"]
    Workflow --> Synthesis["Synthesis"]
```

## 目录结构

```text
backend/
  app/
    api/routes/        # 18 个路由文件
    core/              # 配置、数据库、日志
    db/                # 模型与会话
    mcp/               # 记忆 MCP server
    schemas/           # Pydantic schema
    services/
      agents/          # ChatAgent、Issue/PR/CI/周报/安全/工作流 Agent
      rag/             # 解析、切片、嵌入、检索、重排、问答
      evals/           # 评测套件与 Ragas runner
      skills/          # Skill 注册与校验
      github/          # GitHub 数据同步
      llm/             # 模型客户端
    skills/            # 6 个 SKILL.md
    tests/             # 226 个测试函数
frontend/
  app/                 # Next.js App Router 页面
  components/          # 布局、RAG、通用 UI
  lib/                 # API 客户端与类型
docs/                  # 架构、API、示例文档
scripts/               # 本地启动与数据脚本
docker-compose.yml
.env.example
```

## 快速开始

前置条件：Docker、Python 3.11+、Node.js 18+、可用的 OpenAI 兼容 LLM API、GitHub Personal Access Token。

1. 准备配置：

```bash
cp .env.example .env
```

至少填写 `LLM_API_KEY`（以及需要时的 `LLM_BASE_URL`、`LLM_MODEL`）和 `GITHUB_TOKEN`。

2. 启动基础设施：

```bash
docker compose up -d postgres etcd minio milvus
```

`redis` 属于 `optional` profile，默认不启动。

3. 启动后端：

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

4. 启动前端：

```bash
cd frontend
npm install
npm run dev
```

默认地址：

- 后端：http://localhost:8000
- API 文档：http://localhost:8000/docs
- 前端：http://localhost:3000
- Milvus：http://localhost:19530
- MinIO 控制台：http://localhost:9001

### 本地演示模式

只想验证页面、结构化数据和无模型兜底回答时，Windows 下可以运行：

```powershell
.\scripts\start_rag.ps1
```

该脚本使用 SQLite、deterministic embedding 和启发式重排，向量检索会标记为 `degraded`，不能作为完整 RAG 的验证依据。要验证向量召回与重排，仍需启动完整的 PostgreSQL 与 Milvus。

## 配置

配置项以 `.env.example` 为准，常用项：

| 变量 | 说明 | 默认 |
| --- | --- | --- |
| `DATABASE_URL` | PostgreSQL 连接串 | `postgresql+psycopg://devflow:devflow@localhost:5432/devflow` |
| `GITHUB_TOKEN` | GitHub 访问令牌 | 空 |
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` | 主模型配置 | 空 / `https://api.deepseek.com` / `deepseek-v4-pro` |
| `EMBEDDING_PROVIDER` | 嵌入方式：`qwen_local` / `openai_compatible` / `deterministic` | `qwen_local` |
| `EMBEDDING_MODEL` / `EMBEDDING_DIMENSIONS` | 嵌入模型与维度 | `Qwen/Qwen3-Embedding-0.6B` / `1024` |
| `MILVUS_ENABLED` / `MILVUS_URI` / `MILVUS_COLLECTION` | 向量库配置 | `true` / `http://localhost:19530` / `devflow_rag_qwen1024` |
| `RAGAS_ENABLED` / `RAGAS_*_MIN` | Ragas 评测开关与指标门槛 | `true` / 见 `.env.example` |
| `CONTEXT_*` | 上下文预算、压缩与恢复参数 | 见 `.env.example` |
| `TOKEN_ENCRYPTION_KEY` | GitHub token 等敏感字段的加密密钥 | 演示值，部署前必须更换 |
| `AUTO_SYNC_ENABLED` / `AUTO_SYNC_INTERVAL_SECONDS` | 定时同步 | `true` / `300` |

`docs/datas/evals` 是评测套件的默认目录，仓库未附带该目录；未提供时只加载内置的 `auto-document-smoke` 冒烟集。

## API

后端共注册 101 个业务路由，另有 `GET /health`，完整文档见 `/docs`。

| 前缀 | 数量 | 用途 |
| --- | --- | --- |
| `/api/repos` | 6 | 连接、同步、选择、删除仓库 |
| `/api/issues` | 5 | Issue 列表、详情、分析、相似 Issue |
| `/api/pull-requests` | 8 | PR 列表、详情、分析、快照与检查清单 |
| `/api/ci` | 5 | Workflow Run、失败 Job、CI 分析 |
| `/api/chat` | 7 | 对话、流式对话、计划与推荐 |
| `/api/conversations` | 3 | 会话读取与删除 |
| `/api/action-drafts` | 6 | 草稿创建、确认、取消、审计 |
| `/api/rag` | 15 | 知识库仓库配置、文档、召回测试、检索、问答、重建索引 |
| `/api/evals` | 8 | 评测套件、运行、重评、对比、生成 Issue 草稿 |
| `/api/knowledge` | 17 | 项目知识、长期记忆、团队与上传 |
| `/api/reports` | 2 | 周报与仓库健康报告 |
| `/api/workspaces` | 3 | 工作区创建、读取、多仓报告 |
| `/api/code-graph` | 3 | 代码图谱重建与检索 |
| `/api/skills` | 3 | Skill 列表、详情、MCP 状态 |
| `/api/search` | 1 | 统一检索 |
| `/api/webhooks/github` | 1 | GitHub Webhook 接收 |
| `/api/chat/feedback` | 5 | 反馈与 trace |

## 前端页面

`/`、`/chat`、`/knowledge`、`/knowledge/[id]`、`/rag`、`/repos`、`/action-drafts`、`/workflow-runs`、`/workspaces`、`/code-graph`、`/evals`、`/issues`、`/pull-requests`、`/ci`、`/reports`。

其中 `/evals` 提供固定 RAGAS 评测运行、逐题轨迹查看、异常评分重试和回归对比。

## 测试

```bash
cd backend
python -m pytest app/tests -q
```

当前仓库包含 29 个测试文件、226 个测试函数，覆盖 Agent、RAG、记忆、上下文压缩、工具契约、评测与 API 路由。

## 已知限制

- **向量检索仅实现 Milvus**：PostgreSQL 使用 `pgvector` 镜像并初始化了扩展，但应用层没有使用 pgvector 检索；Milvus 不可用时 RAG 会降级，向量状态显示为 `degraded`。
- **评测基线未随仓库提供**：`docs/datas/evals` 不存在，默认只能跑内置冒烟集；接入正式基线需自行放入 `rag_cases.json` 并确保引用文档已入库。
- **文档深度不一**：`docs/architecture/memory_system.md` 较完整，其余架构文档多为提纲。
- **本地 token 计数为估算**：服务端超限时本轮最多恢复重试一次。
- **写操作不自动落库到 GitHub**：ActionDraft 必须经人工确认流程。
