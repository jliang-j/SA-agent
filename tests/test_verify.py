
import pytest

from sa_agent.skills import verify


def test_verify_passed():
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

    result = verify.run_verify(extracted, retrieved)

    assert result["passed_count"] == 1
    assert result["failed_count"] == 0
    assert result["facts"][0]["status"] == "passed"


def test_verify_unknown_source():
    extracted = {
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

    result = verify.run_verify(extracted, retrieved)

    assert result["failed_count"] == 1
    assert result["facts"][0]["status"] == "failed"
    assert "来源" in result["facts"][0]["reason"]


def test_verify_bad_evidence():
    extracted = {
        "question": "如何处理重复数据？",
        "facts": [
            {
                "fact": "重复数据必须自动删除。",
                "source": "demo.txt",
                "chunk_index": 0,
                "evidence": "重复数据必须自动删除。",
            }
        ],
    }
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

    result = verify.run_verify(extracted, retrieved)

    assert result["failed_count"] == 1
    assert result["facts"][0]["status"] == "failed"
    assert "证据" in result["facts"][0]["reason"]


def test_verify_empty_facts():
    result = verify.run_verify(
        {"question": "资料中有什么事实？", "facts": []},
        {"question": "资料中有什么事实？", "results": []},
    )

    assert result["facts"] == []
    assert result["passed_count"] == 0
    assert result["failed_count"] == 0


def test_verify_rejects_invalid_input():
    with pytest.raises(ValueError):
        verify.run_verify(
            {"question": "问题", "facts": "无效数据"},
            {"question": "问题", "results": []},
        )
