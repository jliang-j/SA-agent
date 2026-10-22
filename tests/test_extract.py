
import json

import pytest
from langchain_core.documents import Document

from sa_agent.skills import extract


def test_extract_ok(monkeypatch):
    content = "重复数据应根据业务标识识别。"
    output = {
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

    class FakeModel:
        def invoke(self, prompt):
            assert "如何处理重复数据？" in prompt
            return type("Response", (), {"content": json.dumps(output, ensure_ascii=False)})()

    monkeypatch.setattr(extract, "create_chat_model", lambda: FakeModel())

    result = extract.run_extract({
        "question": "如何处理重复数据？",
        "results": [
            {
                "source": "demo.txt",
                "chunk_index": 0,
                "content": content,
            }
        ],
    })

    assert result["question"] == "如何处理重复数据？"
    assert len(result["facts"]) == 1
    assert result["facts"][0]["source"] == "demo.txt"
    assert result["facts"][0]["evidence"] == content


def test_page_from_source(monkeypatch):
    """PDF页码由检索元数据确定，不接受模型臆造页码。"""
    output = {"question": "风险？", "facts": [{"fact": "风险", "source": "demo.pdf",
              "chunk_index": 0, "evidence": "原文", "page": 99}]}

    class FakeModel:
        """模拟返回错误页码的模型。"""

        def invoke(self, prompt):
            """返回合法事实和错误页码。"""
            return type("Response", (), {"content": json.dumps(output)})()

    monkeypatch.setattr(extract, "create_chat_model", lambda: FakeModel())
    result = extract.run_extract({"question": "风险？", "results": [
        {"source": "demo.pdf", "chunk_index": 0, "content": "原文", "page": 2}]})
    assert result["facts"][0]["page"] == 2


def test_extract_empty(monkeypatch):
    def fail_if_called():
        pytest.fail("没有检索结果时不应该调用模型")

    monkeypatch.setattr(extract, "create_chat_model", fail_if_called)

    result = extract.run_extract({
        "question": "资料中有相关数据吗？",
        "results": [],
    })

    assert result["facts"] == []


def test_bad_source(monkeypatch):
    output = {
        "question": "如何处理重复数据？",
        "facts": [
            {
                "fact": "重复数据应去重。",
                "source": "unknown.txt",
                "chunk_index": 0,
                "evidence": "重复数据应根据业务标识识别。",
            }
        ],
    }

    class FakeModel:
        def invoke(self, prompt):
            return type("Response", (), {"content": json.dumps(output, ensure_ascii=False)})()

    monkeypatch.setattr(extract, "create_chat_model", lambda: FakeModel())

    with pytest.raises(ValueError, match="来源"):
        extract.run_extract({
            "question": "如何处理重复数据？",
            "results": [
                {
                    "source": "demo.txt",
                    "chunk_index": 0,
                    "content": "重复数据应根据业务标识识别。",
                }
            ],
        })


def test_bad_evidence(monkeypatch):
    output = {
        "question": "如何处理重复数据？",
        "facts": [
            {
                "fact": "重复数据应自动删除。",
                "source": "demo.txt",
                "chunk_index": 0,
                "evidence": "重复数据必须自动删除。",
            }
        ],
    }

    class FakeModel:
        def invoke(self, prompt):
            return type("Response", (), {"content": json.dumps(output, ensure_ascii=False)})()

    monkeypatch.setattr(extract, "create_chat_model", lambda: FakeModel())

    with pytest.raises(ValueError, match="证据"):
        extract.run_extract({
            "question": "如何处理重复数据？",
            "results": [
                {
                    "source": "demo.txt",
                    "chunk_index": 0,
                    "content": "重复数据应根据业务标识识别。",
                }
            ],
        })


def test_bad_json(monkeypatch):
    class FakeModel:
        def invoke(self, prompt):
            return type("Response", (), {"content": "这不是有效的 JSON"})()

    monkeypatch.setattr(extract, "create_chat_model", lambda: FakeModel())

    with pytest.raises(ValueError, match="JSON"):
        extract.run_extract({
            "question": "如何处理重复数据？",
            "results": [
                {
                    "source": "demo.txt",
                    "chunk_index": 0,
                    "content": "重复数据应根据业务标识识别。",
                }
            ],
        })


@pytest.mark.parametrize("count", [0, 6, 8, 9])
def test_fact_limit(monkeypatch, count):
    """接受上限内的事实，拒绝超限响应，不截断或额外调用模型。"""
    calls = []
    facts = [{"fact": f"事实{i}", "source": "demo.txt", "chunk_index": 0,
              "evidence": f"证据{i}。"} for i in range(count)]

    class FakeModel:
        """模拟不同数量的结构化提取结果。"""

        def invoke(self, prompt):
            """检查整个任务的数量约束并返回指定数量。"""
            calls.append(prompt)
            assert "整个当前任务中最多 8 条" in prompt
            return type("Response", (), {"content": json.dumps({"question": "研究问题", "facts": facts})})()

    monkeypatch.setattr(extract, "create_chat_model", lambda: FakeModel())
    data = {"question": "研究问题", "results": [{
        "source": "demo.txt", "chunk_index": 0,
        "content": "".join(item["evidence"] for item in facts) or "无相关依据。",
    }]}
    if count > 8:
        with pytest.raises(ValueError, match="最多提取8条"):
            extract.run_extract(data)
    else:
        assert len(extract.run_extract(data)["facts"]) == count
    assert len(calls) == 1
