# QwenRAG

面向 Windows 离线交付的本地法律资料检索增强问答系统。项目负责知识库构建、增量资料入库、混合检索、模型网关转发以及 Windows 安装包交付；大语言模型、Embedding 模型和 Chatbox 由部署人员单独准备。

## 系统组成

```text
Chatbox / OpenAI 兼容客户端
    -> local_rag_app        本地问答接口、RAG 路由、FAISS + FTS 混合检索、参考文件展示
    -> model_gateway        模型鉴权、请求转发、健康检查和错误处理
    -> 本地 LLM / Embedding 模型服务

Rawdata -> rag_preprocess -> rag_data
            全量构建 / 增量解析 / OCR / 向量索引 / SQLite 元数据

qwenrag_runtime -> packaging -> release
  Windows 运行时      离线安装包      最终交付介质
```

## 项目目录

```text
QwenRag/
├── local_rag_app/           本地 RAG HTTP 应用
├── model_gateway/           OpenAI 兼容模型网关
├── rag_preprocess/          知识库构建与增量入库
│   └── incremental/         任务管理、解析、OCR、增量索引与发布
├── qwenrag_runtime/         Windows 运行时、配置、进程监督和知识库快照
├── launch/                  已安装产品的 PowerShell 启动入口
├── scripts/                 开发运行、知识库构建和资料入库脚本
├── tools/                   知识库诊断、一致性校验和阶段验收工具
├── requirements/            按功能划分的 Python 依赖及离线锁定文件
├── tests/                   单元测试、集成测试、交付测试与端到端测试
├── packaging/               PyInstaller、Inno Setup 和离线交付配置
│   ├── installer/           Inno Setup 安装器定义
│   ├── manifests/           离线 wheel 完整性清单
│   ├── release_docs/        必须随安装包交付的用户文档和配置模板
│   └── scripts/             冻结运行时、快照、安装包构建与发行校验
├── docs/                    本机内部运维文档，不纳入 Git
│   ├── server/              服务器、模型部署与网关运行说明
│   └── windows/             Windows 开发验收和完整安装使用教程
├── Rawdata/                 本地法律原始资料，不纳入 Git
├── rag_data/                正式 SQLite、FAISS 索引及增量资料，不纳入 Git
├── models/                  离线 OCR 模型资源，不纳入 Git
├── wheelhouse/              已校验的离线 Python wheel，不纳入 Git
├── release/                 当前正式离线安装包和分发压缩包，不纳入 Git
└── 资料入库工作台/          本地资料导入、处理结果和归档目录联接
```

`rag_data/metadata.db`、`rag_data/vector_index/`、`rag_data/kb_deltas/`、`Rawdata/`、`models/ocr/` 和工作台中的归档资料属于有效业务资产，不能作为缓存删除。`资料入库工作台/` 包含 Windows 目录联接，不应递归删除或移动。

## 依赖清单

| 文件 | 用途 |
| --- | --- |
| `requirements/base.txt` | 知识库基础构建依赖 |
| `requirements/gateway.txt` | 模型网关依赖 |
| `requirements/local-rag.txt` | Windows 本地 RAG 服务依赖 |
| `requirements/incremental-rag.txt` | 增量入库直接依赖 |
| `requirements/incremental-rag.lock.txt` | 已验证的增量入库版本锁定 |
| `requirements/delivery.in` | Windows 离线交付依赖入口 |
| `requirements/delivery.lock.txt` | 含 SHA-256 的完整离线交付锁定清单 |

环境变量模板仍位于项目根目录：`.env.gateway.example`、`.env.local-rag.example` 和 `.env.incremental.example`。实际 `.env.*` 文件包含本机配置或密钥，不进入版本控制。

## 常用命令

在项目根目录运行：

```powershell
# 运行完整测试。
.\.venv-delivery\Scripts\python.exe -m pytest -q

# 构建或恢复知识库；完整构建会处理 Rawdata 并写入 rag_data。
.\.venv-delivery\Scripts\python.exe .\scripts\build_kb.py --stage all --resume

# 检查增量资料入库环境。
.\.venv-delivery\Scripts\python.exe .\scripts\check_incremental_environment.py

# 查看交付运行时命令。
.\.venv-delivery\Scripts\python.exe -m qwenrag_runtime --help
```

在 Linux 模型服务器上安装和启动网关：

```bash
python -m pip install -r requirements/gateway.txt
bash scripts/start_model_gateway.sh
```

## 离线打包

构建使用项目自带的 Python 3.10 交付环境、`wheelhouse/` 和本地 OCR 模型，不能在客户运行期间下载依赖或模型：

```powershell
# 构建冻结运行时。
.\packaging\scripts\build_runtime.ps1

# 从正式知识库生成待交付快照；按实际版本调整参数。
.\packaging\scripts\stage_initial_kb.ps1 `
    -SourceKnowledgeBase .\rag_data `
    -Version 1.0.0 `
    -EmbeddingRevision Qwen3-Embedding-0.6B-GGUF

# 构建安装器，或运行 build_release.ps1 完成整套发布流程。
.\packaging\scripts\build_installer.ps1
```

`packaging/build/`、`packaging/output/` 和 `packaging/payload/initial_kb/` 是可重新生成的构建目录。打包完成并确认最终发行介质有效后，可以删除这些目录；不要删除 `packaging/installer/`、`packaging/release_docs/`、`packaging/manifests/` 或 `packaging/scripts/`。
