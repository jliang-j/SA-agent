
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sa_agent.embeddings import create_embeddings
from sa_agent.retrieval.ingest import ingest_text_file
from sa_agent.retrieval.qa import answer_question
from sa_agent.retrieval.store import create_persistent_vector_store
from sa_agent.workflows.research import run_research
from sa_agent.workflows.agent import run_agent
from sa_agent import documents

app = FastAPI(title="SA-agent · AI 行业研究助手", description="基于本地AI行业资料组织研究任务，输出可追溯的简报与资料缺口。")
logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

DOCUMENTS_DIR = Path("data/documents")
VECTOR_STORE_DIR = Path("data/chroma")
COLLECTION_NAME = "industry_documents"


@app.get("/", include_in_schema=False)
def research_page():
    """提供本地研究界面，静态资源路径不依赖启动目录。"""
    return FileResponse(STATIC_DIR / "index.html")


class IngestRequest(BaseModel):
    filename: str = Field(min_length=1)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=3, ge=1, le=10)


@app.get("/documents")
def document_list():
    """列出资料及实际片段数，不调用远程Embedding接口。"""
    try:
        return documents.list_documents(DOCUMENTS_DIR, VECTOR_STORE_DIR, COLLECTION_NAME)
    except Exception as exc:
        logger.exception("读取文档列表失败")
        raise HTTPException(500, "读取文档列表失败，请检查服务日志") from exc


@app.post("/documents/upload", status_code=201)
def upload_document(file: UploadFile):
    """接收单文件并同步导入，同名不覆盖，失败文件保留以便重试。"""
    try:
        return documents.upload_document(
            DOCUMENTS_DIR, file.filename or "", file.file, VECTOR_STORE_DIR, COLLECTION_NAME,
        )
    except documents.DocumentError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    except Exception as exc:
        logger.exception("文件上传失败")
        raise HTTPException(500, "文件上传失败，请检查服务日志") from exc
    finally:
        file.file.close()


@app.post("/documents/retry")
def retry_document(request: IngestRequest):
    """对已保存的支持格式文档重新执行导入，不重新上传或覆盖文件。"""
    try:
        return documents.import_document(DOCUMENTS_DIR, request.filename, VECTOR_STORE_DIR, COLLECTION_NAME)
    except documents.DocumentError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    except Exception as exc:
        logger.exception("重试文档导入失败")
        raise HTTPException(500, "重试导入失败，请检查服务日志") from exc


@app.get("/health")
def health_check():
    """检查 API 服务是否可响应，不调用模型。"""
    return {"status": "ok"}


@app.post("/documents/ingest")
def ingest_document(request: IngestRequest):
    """导入指定目录内的 TXT 文档。"""
    filename = request.filename.strip()

    if not filename or Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="请提供有效的文件名")

    if not filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="目前只支持 TXT 文件")

    base_dir = DOCUMENTS_DIR.resolve()
    file_path = (base_dir / filename).resolve()

    if not file_path.is_relative_to(base_dir):
        raise HTTPException(status_code=400, detail="不允许访问指定目录之外的文件")

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="文档不存在")

    try:
        return ingest_text_file(
            file_path=file_path,
            persist_directory=VECTOR_STORE_DIR,
            collection_name=COLLECTION_NAME,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="文档导入失败，请检查服务日志",
        ) from exc


@app.post("/ask")
def ask_question(request: QuestionRequest):
    """检索本地资料并生成基础问答。"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    try:
        vector_store = create_persistent_vector_store(
            persist_directory=VECTOR_STORE_DIR,
            collection_name=COLLECTION_NAME,
            embeddings=create_embeddings(),
        )

        return answer_question(
            vector_store=vector_store,
            question=request.question,
            k=request.k,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="问答处理失败，请检查服务日志",
        ) from exc

@app.post("/research")
def research_question(request: QuestionRequest):
    """运行固定研究工作流，保留原有接口行为。"""
    #print("收到的问题：", request.question)
    #print("问题字符编码：", [hex(ord(c)) for c in request.question])
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    try:
        
        vector_store = create_persistent_vector_store(
            persist_directory=VECTOR_STORE_DIR,
            collection_name=COLLECTION_NAME,
            embeddings=create_embeddings(),
        )

        return run_research(
            vector_store=vector_store,
            question=request.question,
            k=request.k,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="研究流程执行失败，请检查服务日志",
        ) from exc


@app.post("/agent/research")
def agent_research(request: QuestionRequest):
    """由 Agent 组织研究任务与补查，返回简报、证据和行动记录。"""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")
    try:
        vector_store = create_persistent_vector_store(
            persist_directory=VECTOR_STORE_DIR,
            collection_name=COLLECTION_NAME,
            embeddings=create_embeddings(),
        )
        return run_agent(vector_store=vector_store, question=question, k=request.k)
    except Exception as exc:
        logger.exception("Agent 研究执行失败")
        raise HTTPException(
            status_code=500,
            detail="智能体研究执行失败，请检查服务日志",
        ) from exc
