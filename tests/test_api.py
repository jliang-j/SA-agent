
from fastapi.testclient import TestClient

import sa_agent.main as main


client = TestClient(main.app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ingest_success(tmp_path, monkeypatch):
    (tmp_path / "demo.txt").write_text("测试文档", encoding="utf-8")
    monkeypatch.setattr(main, "DOCUMENTS_DIR", tmp_path)

    def fake_ingest(**kwargs):
        return {
            "source": "demo.txt",
            "chunks_count": 1,
            "persist_directory": "data/chroma",
        }

    monkeypatch.setattr(main, "ingest_text_file", fake_ingest)

    response = client.post(
        "/documents/ingest",
        json={"filename": "demo.txt"},
    )

    assert response.status_code == 200
    assert response.json()["source"] == "demo.txt"
    assert response.json()["chunks_count"] == 1


def test_ingest_rejects_path_traversal():
    response = client.post(
        "/documents/ingest",
        json={"filename": "../secret.txt"},
    )

    assert response.status_code == 400


def test_ingest_rejects_non_txt():
    response = client.post(
        "/documents/ingest",
        json={"filename": "demo.pdf"},
    )

    assert response.status_code == 400


def test_ingest_file_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DOCUMENTS_DIR", tmp_path)

    response = client.post(
        "/documents/ingest",
        json={"filename": "missing.txt"},
    )

    assert response.status_code == 404


def test_ingest_handles_error(tmp_path, monkeypatch):
    (tmp_path / "demo.txt").write_text("测试文档", encoding="utf-8")
    monkeypatch.setattr(main, "DOCUMENTS_DIR", tmp_path)

    def fake_ingest(**kwargs):
        raise RuntimeError("模拟导入失败")

    monkeypatch.setattr(main, "ingest_text_file", fake_ingest)

    response = client.post(
        "/documents/ingest",
        json={"filename": "demo.txt"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "文档导入失败，请检查服务日志"


def test_ask_success(monkeypatch):
    monkeypatch.setattr(main, "create_embeddings", lambda: object())
    monkeypatch.setattr(
        main,
        "create_persistent_vector_store",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        main,
        "answer_question",
        lambda **kwargs: {
            "answer": "重复数据应根据业务标识识别。",
            "sources": [{"source": "demo.txt", "chunk_index": 0}],
        },
    )

    response = client.post(
        "/ask",
        json={"question": "如何处理重复数据？", "k": 3},
    )

    assert response.status_code == 200
    assert "重复数据" in response.json()["answer"]
    assert len(response.json()["sources"]) == 1


def test_ask_rejects_blank_question():
    response = client.post(
        "/ask",
        json={"question": "   "},
    )

    assert response.status_code == 400


def test_ask_rejects_invalid_k():
    response = client.post(
        "/ask",
        json={"question": "如何处理重复数据？", "k": 0},
    )

    assert response.status_code == 422


def test_ask_handles_error(monkeypatch):
    def fake_embeddings():
        raise RuntimeError("模拟模型配置错误")

    monkeypatch.setattr(main, "create_embeddings", fake_embeddings)

    response = client.post(
        "/ask",
        json={"question": "如何处理重复数据？"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "问答处理失败，请检查服务日志"


def test_research_success(monkeypatch):
    monkeypatch.setattr(main, "create_embeddings", lambda: object())
    monkeypatch.setattr(
        main,
        "create_persistent_vector_store",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        main,
        "run_research",
        lambda **kwargs: {
            "question": "如何处理重复数据？",
            "retrieval": {"results": []},
            "extraction": {"facts": []},
            "verification": {
                "facts": [],
                "passed_count": 0,
                "failed_count": 0,
            },
        },
    )

    response = client.post(
        "/research",
        json={"question": "如何处理重复数据？", "k": 3},
    )

    assert response.status_code == 200
    assert response.json()["verification"]["passed_count"] == 0


def test_research_rejects_blank_question():
    response = client.post(
        "/research",
        json={"question": "   "},
    )

    assert response.status_code == 400
