from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sa_agent.skills.extract import Fact
from sa_agent.schemas import Text


MAX_FINDINGS = 8


class Finding(BaseModel):
    """一条研究结论及其引用的证据编号。"""

    model_config = ConfigDict(extra="forbid")
    text: Text
    evidence_ids: list[Text] = Field(min_length=1)


class Report(BaseModel):
    """结构化研究简报；概述不应引入结论列表之外的新事实。"""

    model_config = ConfigDict(extra="forbid")
    summary: Text
    findings: list[Finding] = Field(max_length=MAX_FINDINGS)
    gaps: list[Text]


def evidence_key(fact: dict) -> tuple:
    """按事实、来源、片段和原文证据确定身份，仅合并完全相同的记录。"""
    return tuple(fact.get(key) for key in ("fact", "source", "chunk_index", "evidence", "page"))


def collect_evidence(tasks: list[dict]) -> list[dict]:
    """从成功任务中收集通过核验的事实，按首次出现顺序分配请求内编号。"""
    evidence, seen = [], set()
    for task in tasks:
        if task["status"] != "success":
            continue
        for fact in task["verification"]["facts"]:
            if fact["status"] != "passed":
                continue
            checked = Fact.model_validate(fact).model_dump()
            key = evidence_key(checked)
            if key not in seen:
                seen.add(key)
                evidence.append({"id": f"E{len(evidence) + 1}", **checked})
    return evidence


def parse_report(content: str, evidence: list[dict]) -> dict:
    """校验简报格式与引用存在性，不判定结论是否得到证据的语义支持。"""
    try:
        report = Report.model_validate_json(content)
    except ValidationError as exc:
        if any(error["loc"] == ("findings",) and error["type"] == "too_long" for error in exc.errors()):
            raise ValueError(f"研究简报最多包含{MAX_FINDINGS}条核心结论，模型返回超出上限") from exc
        raise ValueError("研究简报格式无效") from exc
    except (ValueError, TypeError) as exc:
        raise ValueError("研究简报格式无效") from exc
    known = {item["id"] for item in evidence}
    for finding in report.findings:
        if any(ref not in known for ref in finding.evidence_ids):
            raise ValueError("研究简报引用了不存在的证据")
    if not report.findings and not report.gaps:
        raise ValueError("没有研究结论时必须说明资料缺口")
    return report.model_dump()


def render_report(report: dict, evidence: list[dict]) -> str:
    """将已校验的简报渲染为易读文本，来源字段由程序填写。"""
    lines = ["## 概述", report["summary"], "", "## 研究结论"]
    used = set()
    for finding in report["findings"]:
        refs = finding["evidence_ids"]
        used.update(refs)
        lines.append(f"- {finding['text']} " + " ".join(f"[{ref}]" for ref in refs))
    if not report["findings"]:
        lines.append("暂无足够证据形成结论。")
    lines.extend(["", "## 来源"])
    for item in evidence:
        if item["id"] in used:
            page = f"，第{item['page']}页" if item.get("page") is not None else ""
            lines.append(f"- [{item['id']}] {item['source']}{page}，片段 {item['chunk_index']}")
    if not used:
        lines.append("无可引用证据。")
    lines.extend(["", "## 资料缺口与限制"])
    lines.extend(f"- {gap}" for gap in report["gaps"])
    lines.append("- 来源核验与引用检查不等于语义真实性认证。")
    return "\n".join(lines)
