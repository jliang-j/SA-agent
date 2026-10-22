# 文档接入

## 支持范围

DOCX已接通上传、解析、入库、列表及失败重试。解析测试为`tests/test_docx.py`，上传测试为`tests/test_documents.py`。

- TXT、Markdown（`.md`）：UTF-8文本。
- DOCX：按原顺序提取正文段落与普通表格；使用文件名和片段编号定位，不生成Word页码，不支持旧式`.doc`。
- PDF：可提取的文字，最多200页；页码从1开始，按PDF实际页面顺序编号。
- 单文件最大10 MB。暂不支持OCR、图片和加密文件；DOCX的复杂嵌套/合并表格、文本框、修订及页眉页脚不保证完整提取。
- PDF按页切分，文件内`chunk_index`跨页唯一。页码通过检索元数据进入事实及证据，不由模型决定。
- 复杂排版、表格和混合扫描页面可能提取不完整，需人工核对，不保证版面还原。

## 使用

```powershell
uv sync
uv run uvicorn sa_agent.main:app --reload
```

打开首页，点击右上角“文档库”，选择文件并“上传并导入”。成功后可发起研究。
上传只调用现有Embedding，不调用聊天模型；研究另外触发Agent模型调用。
同名文件返回409，不自动覆盖；失败文件保留，可点击“重试导入”。
新资料进入公共的本地集合，已有研究结果不自动刷新，需重新研究。

## 接口

| 接口 | 参数 | 行为 |
|---|---|---|
| GET `/documents` | 无 | 列出文件、实际片段数、导入状态 |
| POST `/documents/upload` | multipart字段`file` | 保存新文件并同步导入；成功201 |
| POST `/documents/retry` | `{"filename":"report.pdf"}` | 重新导入已保存文档；成功200 |
| POST `/documents/ingest` | 原有TXT文件名参数 | 保留原接口及其TXT限制 |

格式或文件名错误400；未找到文件404；重名或处理中409；过大413；无效编码、PDF或无正文422；服务错误500。
状态包括`processing`、`imported`、`failed`、`not_imported`。旧接口导入的文件按库中片段数兼容展示。
列表读取库元数据，不远程生成向量。刷新页面不影响已保存的文档和状态。

## 代码位置

| 文件 | 作用 |
|---|---|
| `src/sa_agent/documents.py` | 大小、文件名、重名检查，保存上传、状态记录、重试与列表 |
| `src/sa_agent/retrieval/loader.py` | TXT/MD解析、PDF逐页文字提取与切分 |
| `src/sa_agent/retrieval/ingest.py` | 复用入库；先写新片段，再清理多余旧片段 |
| `src/sa_agent/main.py` | 新文档接口，原TXT入口保留 |
| `src/sa_agent/skills/retrieve.py`、`extract.py`、`report.py` | 页码从检索传到事实、证据和简报 |
| `src/sa_agent/static/` | 文档库弹窗、上传/重试/列表、证据页码展示 |
| `pyproject.toml`、`uv.lock` | `pypdf`负责PDF、`python-docx`负责DOCX、`python-multipart`负责上传 |

## 验证

```powershell
uv run pytest tests/test_documents.py tests/test_ingest.py -v
uv run pytest -m "not integration"
uv run pytest tests/test_real_upload.py -m integration -v
```

前两条不调用模型；第三条使用真实Embedding，在测试临时目录验收上传和检索，不修改现有库。
浏览器上传验收也使用隔离目录与模拟向量，检查成功、同名冲突、失败状态及重试提示。

建议手动上传一份真实的小型文本PDF，检查列表成功，再问一个能在文件中找到答案的问题，点击证据检查页码。
上传自编示例以外的资料后，原“2025市场规模”资料不足案例的前提可能变化，不能继续盲目作为负例。

## 当前边界

- 状态在`data/documents/.document_status.json`，由`.gitignore`忽略。原文文件留在文档目录，不自动下载或公开原文。
- 采用单进程内的并发保护，按默认单个Uvicorn进程使用；不支持多进程共享写入的协调。
- 写入与清理不具备数据库事务原子性；异常中断后可重试。同名更新及版本管理未实现。
- 不支持在前端删除文件。旧TXT接口保留空文件清理旧片段的历史行为，新上传接口拒绝空文件及无正文。
