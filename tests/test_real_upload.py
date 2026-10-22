import pytest
from fastapi.testclient import TestClient

import sa_agent.main as main
from sa_agent.embeddings import create_embeddings
from sa_agent.retrieval.store import create_persistent_vector_store


@pytest.mark.integration
def test_real_upload(tmp_path, monkeypatch):
    """用真实Embedding验收上传、持久化、列表与检索，隔离现有资料库。"""
    monkeypatch.setattr(main, "DOCUMENTS_DIR", tmp_path / "documents")
    monkeypatch.setattr(main, "VECTOR_STORE_DIR", tmp_path / "chroma")
    text = "# 文档接入验收\n测试资料要求：每年复核文档来源，并保存原文证据。"
    with TestClient(main.app) as client:
        response = client.post("/documents/upload", files={"file": ("upload_check.md", text.encode("utf-8"))})
        assert response.status_code == 201, response.text
        assert response.json()["status"] == "imported"
        listing = client.get("/documents")
        assert listing.status_code == 200
        assert listing.json()["documents"][0]["chunks_count"] > 0
    store = create_persistent_vector_store(
        persist_directory=main.VECTOR_STORE_DIR, collection_name=main.COLLECTION_NAME,
        embeddings=create_embeddings(),
    )
    results = store.similarity_search("文档来源应该多久复核？", k=1)
    assert results
    assert results[0].metadata["source"] == "upload_check.md"
    assert "每年复核" in results[0].page_content
