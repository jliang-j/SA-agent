import json

import pytest

from sa_agent.skills.report import collect_evidence, parse_report, render_report


def fact(text="事实"):
    """构造一条通过来源核验的事实。"""
    return {"fact": text, "source": "demo.txt", "chunk_index": 0,
            "evidence": text, "status": "passed"}


def task(facts, status="success"):
    """构造简报输入所需的任务结果。"""
    return {"status": status, "verification": {"facts": facts}}


def test_evidence_ids():
    """重复证据复用编号，追加任务不改变已有编号，失败事实不进入证据池。"""
    first = [task([fact(), {**fact("不通过"), "status": "failed"}])]
    before = collect_evidence(first)
    after = collect_evidence(first + [task([fact(), fact("新增")]), task([fact("失败任务")], "failed")])
    assert after[:1] == before
    assert [item["id"] for item in after] == ["E1", "E2"]
    assert after[1]["fact"] == "新增"


def test_valid_report():
    """合法引用可解析，来源由实际证据渲染。"""
    evidence = collect_evidence([task([fact()])])
    content = {"summary": "概述", "findings": [{"text": "结论", "evidence_ids": ["E1"]}], "gaps": []}
    report = parse_report(json.dumps(content), evidence)
    answer = render_report(report, evidence)
    assert report == content
    assert "结论 [E1]" in answer
    assert "demo.txt，片段 0" in answer


@pytest.mark.parametrize("content", [
    {"summary": "概述", "findings": [{"text": "结论", "evidence_ids": ["E999"]}], "gaps": []},
    {"summary": "概述", "findings": [{"text": "结论", "evidence_ids": []}], "gaps": []},
    {"summary": "概述", "findings": [], "gaps": []},
    {"summary": " ", "findings": [], "gaps": ["资料不足"]},
    {"summary": "概述", "findings": [], "gaps": ["资料不足"], "sources": ["虚构文件"]},
])
def test_bad_report(content):
    """拒绝不存在的引用、缺失引用及无效简报格式。"""
    with pytest.raises(ValueError):
        parse_report(json.dumps(content), [])


def test_no_evidence():
    """无证据时允许只说明资料缺口，不生成研究结论。"""
    report = parse_report(json.dumps({"summary": "无法回答", "findings": [], "gaps": ["资料不足"]}), [])
    assert "无可引用证据" in render_report(report, [])


def test_bad_json():
    """拒绝自由文本回答，防止绕过结构化校验。"""
    with pytest.raises(ValueError, match="格式"):
        parse_report("不是JSON", [])


def test_pdf_page():
    """PDF页码保留到证据池和最终来源文本。"""
    evidence = collect_evidence([task([{**fact(), "page": 3}])])
    report = {"summary": "概述", "findings": [{"text": "结论", "evidence_ids": ["E1"]}], "gaps": []}
    assert evidence[0]["page"] == 3
    assert "第3页" in render_report(report, evidence)


@pytest.mark.parametrize("count", [1, 8, 9])
def test_finding_limit(count):
    """核心结论不超过8条，超限明确拒绝而非截断。"""
    evidence = collect_evidence([task([fact()])])
    content = {"summary": "概述", "findings": [
        {"text": f"结论{i}", "evidence_ids": ["E1"]} for i in range(count)
    ], "gaps": []}
    if count > 8:
        with pytest.raises(ValueError, match="最多包含8条"):
            parse_report(json.dumps(content), evidence)
    else:
        assert len(parse_report(json.dumps(content), evidence)["findings"]) == count


def test_merged_citations():
    """一条合并结论可保留多个有效引用，不删除未引用的证据记录。"""
    evidence = collect_evidence([task([fact("依据一"), fact("依据二"), fact("其他依据")])])
    content = {"summary": "概述", "findings": [
        {"text": "多项依据支持的核心结论", "evidence_ids": ["E1", "E2"]}
    ], "gaps": []}
    report = parse_report(json.dumps(content), evidence)
    assert report["findings"][0]["evidence_ids"] == ["E1", "E2"]
    assert len(evidence) == 3
