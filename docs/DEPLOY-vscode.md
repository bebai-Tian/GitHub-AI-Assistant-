# 在 VSCode 中部署运行 DevFlow AI

本文档描述如何在这台 Windows 机器上，用 VSCode 把 DevFlow AI 完整跑起来（前端 + 后端 + PostgreSQL + Milvus + MinIO + etcd）。

## 一、当前环境与端口

| 组件 | 地址 | 说明 |
| --- | --- | --- |
| 前端 (Next.js) | http://localhost:3000 | 主界面，`/rag` 为知识库，`/evals` 为 RAGAS 评测 |
| 后端 (FastAPI) | http://127.0.0.1:8000 | `/docs` 为 Swagger 文档 |
| PostgreSQL (pgvector) | localhost:5432 | 用户 devflow / 密码 devflow / 库 devflow |
| Milvus | localhost:19530 | 向量库，健康检查 http://127.0.0.1:9091/healthz |
| MinIO | localhost:9000 | 控制台 :9001，账号 minioadmin / minioadmin |
| etcd | 内部 2379 | Milvus 元数据，不对外映射端口 |

本机实际使用的工具链版本：

- Python 3.11.0（`C:\Users\t1511\AppData\Local\Programs\Python\Python311`）
- Node.js v24.18.0（`D:\nodejs`）
- Docker 29.7.2 + Docker Desktop（用户目录安装）
- 本机只有 Windows PowerShell 5.1，没有 PowerShell 7

## 二、在 VSCode 中打开与运行

用 VSCode 打开项目根目录 `D:\JavaAgent\DevFlow-AI`（不要只打开 `backend` 或 `frontend` 子目录，否则任务里的 `${workspaceFolder}` 会指错）。

首次打开时 VSCode 会提示安装 `.vscode/extensions.json` 中推荐的扩展（Python、Pylance、debugpy、ESLint、Tailwind CSS、Docker），建议全部安装。其中 **debugpy** 是后端断点调试所必需的。

### 运行任务（Ctrl+Shift+P → `Tasks: Run Task`）

| 任务 | 用途 |
| --- | --- |
| `1. 基础设施：启动 PostgreSQL / Milvus / MinIO / etcd` | 自动启动 Docker Desktop 并拉起 4 个容器 |
| `2b. 基础设施：查看容器状态` | `docker compose ps` |
| `2c. 基础设施：停止容器` | `docker compose down` |
| `3. 后端：创建虚拟环境` | `python -m venv backend/.venv` |
| `4. 后端：安装 requirements.txt` | 安装后端依赖 |
| `5. 后端：安装本地向量模型依赖（可选）` | 安装 sentence-transformers，启用本地 Qwen 向量 |
| `6. 前端：安装依赖 (npm install)` | 安装前端依赖 |
| `7. 后端：启动 FastAPI (uvicorn :8000)` | 启动后端 |
| `8. 前端：启动 Next.js (:3000)` | 启动前端 |
| `9. 首次部署：按顺序完成全部准备` | 按 3 → 4 → 5 → 6 顺序执行 |
| `10. 一键启动：后端 + 前端` | 并行启动 7 和 8（默认生成任务，直接 Ctrl+Shift+B） |

### 调试（F5）

`launch.json` 提供三个配置：

- `后端：调试 FastAPI (:8000)`：以 debugpy 启动 uvicorn，支持断点。
- `前端：调试 Next.js (Edge)` / `(Chrome)`：打开浏览器并附加调试。
- 复合配置 `全栈：后端 + 前端调试`：同时启动两者。

注意：如果 `7.`/`8.` 任务已经在运行，直接 F5 会因端口占用失败，先停掉终端里的服务。

## 三、PowerShell 执行策略（重要）

本机执行策略是 `Restricted`：

```
Get-ExecutionPolicy -List   # 全部为 Undefined，实际等效 Restricted
```

这会导致：

- `npm` 命令走到 `npm.ps1` 时报“在此系统上禁止运行脚本”；
- 直接运行 `.\scripts\*.ps1` 被拒绝。

因此 `.vscode/settings.json` 已把 VSCode 终端默认配置设为 `PowerShell (Bypass)`，在 VSCode 集成终端里不会遇到这个问题。如果在**系统自带**的 PowerShell 窗口里操作，请用：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_infra.ps1
# 或本次会话内放开
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 四、.env 配置要点

`.env` 已从 `.env.example` 生成，关键项：

```ini
DATABASE_URL=postgresql+psycopg://devflow:devflow@localhost:5432/devflow
LLM_API_KEY=<DeepSeek Key>
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat

EMBEDDING_PROVIDER=qwen_local
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
EMBEDDING_DIMENSIONS=1024
MILVUS_ENABLED=true
MILVUS_COLLECTION=devflow_rag_qwen1024
```

两点需要知道：

1. **DeepSeek 不提供 Embedding / Rerank 接口**，所以向量检索必须用本地 `qwen_local`（依赖 `backend/requirements-local-embedding.txt`），或改用百炼 `text-embedding-v4`（需 `DASHSCOPE_API_KEY`）。项目自己的 `/api/rag/models` 也会返回这一提示。
2. 首次调用向量检索会从 Hugging Face 下载 `Qwen/Qwen3-Embedding-0.6B`（约 1.2 GB，缓存在 `C:\Users\t1511\.cache\huggingface`）。国内网络慢的话设置镜像：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

`RERANK_BASE_URL` / `DASHSCOPE_API_KEY` 未配置时，重排会自动降级为本地启发式重排（`heuristic`），不影响使用。

## 五、本次部署所做的改动

为了让项目能真正跑起来，改了两处**阻塞性**问题：

| 文件 | 改动 | 原因 |
| --- | --- | --- |
| `docker-compose.yml` | `minio` 镜像由 `minio/minio:latest` 改为 `bitnamilegacy/minio:latest`，环境变量改为 `MINIO_ROOT_USER/PASSWORD`，挂载路径改为 `/bitnami/minio/data` | MinIO 已停止在 Docker Hub / quay.io 免费分发官方镜像，拉取报 `error from registry: denied` |
| `backend/requirements.txt` | `pydantic-settings==2.7.0` → `2.10.1` | 与 `langchain-community 0.4.1`（要求 `>=2.10.1`）硬冲突，原文件无法安装 |

新增文件：`.vscode/{settings,tasks,launch,extensions}.json`、`scripts/start_infra.ps1`（已带 UTF-8 BOM）。

## 六、已知坑

- `scripts/start_rag.ps1`（项目原有的 SQLite 演示脚本）是 **无 BOM 的 UTF-8**，内含中文。Windows PowerShell 5.1 会按 ANSI 解析，直接运行会报语法错误。用 PowerShell 7，或先给该文件加 UTF-8 BOM，或改用 Python 启动。
- 我们自己新增的 `scripts/start_infra.ps1` 已写入 UTF-8 BOM，避免同一问题。
- 后端首次启动会在 PostgreSQL 中自动建表并做列迁移（见 `app/main.py` 的 startup 钩子），不需要手动执行 Alembic。
- 演示数据通过 `scripts/seed_demo_data.py` 注入（仓库 `course-demo/devflow-sample-api`，含 3 个 Issue、1 个 PR、1 个失败 CI）。想清空重来：`docker compose down -v` 后重启，会重建数据库卷。

## 七、验收步骤

```powershell
# 1) 容器状态
docker compose ps

# 2) 后端
Invoke-RestMethod http://127.0.0.1:8000/health
# -> {"status":"ok","app":"DevFlow AI"}

# 3) 向量库与向量模型
Invoke-RestMethod http://127.0.0.1:8000/api/rag/models
# -> embedding available=True

# 4) 已索引文档
$repos = Invoke-RestMethod http://127.0.0.1:8000/api/repos
Invoke-RestMethod "http://127.0.0.1:8000/api/rag/$($repos[0].id)/status"
# -> vector_store.status = ready

# 5) 端到端对话（会真实调用 DeepSeek）
$body = @{ repo_id = $repos[0].id; message = "总结一下当前的阻塞项" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/chat -ContentType application/json -Body $body
```

浏览器打开 http://localhost:3000 即可看到界面。
