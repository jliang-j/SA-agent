
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sa_agent.embeddings import create_embeddings
from sa_agent.retrieval.ingest import ingest_text_file
from sa_agent.retrieval.qa import answer_question
from sa_agent.retrieval.store import create_persistent_vector_store


app = FastAPI(title="SA-Agent", description="可信行业研究助手")

DOCUMENTS_DIR = Path("data/documents")
VECTOR_STORE_DIR = Path("data/chroma")
COLLECTION_NAME = "industry_documents"


class IngestRequest(BaseModel):
    filename: str = Field(min_length=1)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=3, ge=1, le=10)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/documents/ingest")
def ingest_document(request: IngestRequest):
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
