
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
