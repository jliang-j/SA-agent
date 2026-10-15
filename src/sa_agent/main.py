from fastapi import FastAPI

app = FastAPI(
    title="SA Agent",
    description="基于 RAG 与 DAG 工作流的可信行业研究助手",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}
