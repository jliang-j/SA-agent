
import pytest

from sa_agent.workflows import research


def test_research_flow(monkeypatch):
    calls = []

    retrieved = {
        "question": "如何处理重复数据？",
        "results": [
            {
                "source": "demo.txt",
                "chunk_index": 0,
                "content": "重复数据应根据业务标识识别。",
            }
        ],
    }
    extracted = {
        "question": "如何处理重复数据？",
        "facts": [
            {
                "fact": "重复数据应根据业务标识进行识别。",
                "source": "demo.txt",
                "chunk_index": 0,
                "evidence": "重复数据应根据业务标识识别。",
            }
        ],
    }
    verified = {
        "question": "如何处理重复数据？",
        "facts": [{**extracted["facts"][0], "status": "passed"}],
        "passed_count": 1,
        "failed_count": 0,
    }

    def fake_retrieve(vector_store, question, k):
        calls.append("retrieve")
        assert question == "如何处理重复数据？"
        assert k == 3
        return retrieved

    def fake_extract(data):
        calls.append("extract")
        assert data == retrieved
        return extracted

    def fake_verify(extracted, retrieved):
        calls.append("verify")
        assert extracted == extracted_result
        assert retrieved == retrieved_result
        return verified

    extracted_result = extracted
    retrieved_result = retrieved

    monkeypatch.setattr(research, "run_retrieval", fake_retrieve)
    monkeypatch.setattr(research, "run_extract", fake_extract)
    monkeypatch.setattr(research, "run_verify", fake_verify)

    result = research.run_research(
        vector_store=object(),
        question="如何处理重复数据？",
    )

    assert calls == ["retrieve", "extract", "verify"]
    assert set(result) == {"question", "retrieval", "extraction", "verification"}
    assert result["question"] == "如何处理重复数据？"
    assert result["retrieval"] == retrieved
    assert result["extraction"] == extracted
    assert result["verification"]["passed_count"] == 1


def test_research_stops_if_extract_fails(monkeypatch):
    def fake_retrieve(**kwargs):
        return {"question": "测试问题", "results": []}

    def fake_extract(data):
        raise ValueError("模拟提取失败")

    def fail_verify(**kwargs):
        pytest.fail("提取失败后不应继续核验")

    monkeypatch.setattr(research, "run_retrieval", fake_retrieve)
    monkeypatch.setattr(research, "run_extract", fake_extract)
    monkeypatch.setattr(research, "run_verify", fail_verify)

    with pytest.raises(ValueError, match="模拟提取失败"):
        research.run_research(
            vector_store=object(),
            question="测试问题",
        )


@pytest.mark.parametrize("stage", ["retrieve", "extract", "verify"])
def test_stage_failure(monkeypatch, stage):
    calls = []
    error = RuntimeError(f"{stage} 失败")

    def step(name):
        calls.append(name)
        if name == stage:
            raise error
        return {}

    monkeypatch.setattr(research, "run_retrieval", lambda **kwargs: step("retrieve"))
    monkeypatch.setattr(research, "run_extract", lambda data: step("extract"))
    monkeypatch.setattr(research, "run_verify", lambda **kwargs: step("verify"))

    with pytest.raises(RuntimeError) as caught:
        research.run_research(object(), "测试问题")

    assert caught.value is error
    stages = ["retrieve", "extract", "verify"]
    assert calls == stages[:stages.index(stage) + 1]
