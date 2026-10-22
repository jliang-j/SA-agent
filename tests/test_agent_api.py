import pytest
from fastapi.testclient import TestClient

import sa_agent.main as main


client = TestClient(main.app)


def test_agent_api(monkeypatch):
    """检查路由参数传递和完整研究结果的 JSON 序列化。"""
    store = object()
    embedding = object()
    monkeypatch.setattr(main, "create_embeddings", lambda: embedding)

    def create_store(**kwargs):
        """检查接口沿用现有向量库配置。"""
        assert kwargs["embeddings"] is embedding
        assert kwargs["collection_name"] == main.COLLECTION_NAME
        assert kwargs["persist_directory"] == main.VECTOR_STORE_DIR
        return store

    expected = {
        "question": "研究问题", "answer": "结论 [E1]",
        "report": {"summary": "概述", "findings": [
            {"text": "结论", "evidence_ids": ["E1"]}], "gaps": []},
        "evidence": [{"id": "E1", "fact": "事实", "source": "demo.txt",
                      "chunk_index": 0, "evidence": "原文"}],
        "tasks": [{"id": "t1", "status": "success"}],
        "trace": [{"step": "research", "call": 1}],
    }

    def run(vector_store, question, k):
        """检查去除问题空格后正确转交给 Agent。"""
        assert vector_store is store
        assert question == "研究问题"
        assert k == 4
        return expected

    monkeypatch.setattr(main, "create_persistent_vector_store", create_store)
    monkeypatch.setattr(main, "run_agent", run)
    response = client.post("/agent/research", json={"question": " 研究问题 ", "k": 4})
    assert response.status_code == 200
    assert response.json() == expected


@pytest.mark.parametrize("body,code", [
    ({"question": "   "}, 400), ({"question": ""}, 422),
    ({"question": "问题", "k": 0}, 422), ({"question": "问题", "k": 11}, 422),
    ({}, 422),
])
def test_bad_request(monkeypatch, body, code):
    """无效请求在创建模型或向量库前被拒绝。"""
    def unexpected():
        """确保请求校验失败不会触发外部服务。"""
        pytest.fail("不应调用 Embedding")

    monkeypatch.setattr(main, "create_embeddings", unexpected)
    response = client.post("/agent/research", json=body)
    assert response.status_code == code


@pytest.mark.parametrize("stage", ["store", "agent"])
def test_agent_error(monkeypatch, caplog, stage):
    """初始化或研究失败时记录异常，返回通用错误且服务继续响应。"""
    monkeypatch.setattr(main, "create_embeddings", lambda: object())
    monkeypatch.setattr(main, "create_persistent_vector_store", lambda **kwargs: object())

    def fail(**kwargs):
        """模拟内部服务异常。"""
        raise RuntimeError("内部错误详情")

    target = "create_persistent_vector_store" if stage == "store" else "run_agent"
    monkeypatch.setattr(main, target, fail)
    response = client.post("/agent/research", json={"question": "研究问题"})
    assert response.status_code == 500
    assert response.json() == {"detail": "智能体研究执行失败，请检查服务日志"}
    assert "Agent 研究执行失败" in caplog.text
    assert client.get("/health").status_code == 200
