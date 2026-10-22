
from pydantic import BaseModel, Field, ValidationError

from sa_agent.llm import create_chat_model


MAX_FACTS = 8


class Fact(BaseModel):
    """提取的事实及原文依据。"""

    fact: str = Field(min_length=1)
    source: str = Field(min_length=1)
    chunk_index: int | None = None
    evidence: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)


class ExtractOut(BaseModel):
    """信息提取结果。"""

    question: str
    facts: list[Fact] = Field(max_length=MAX_FACTS)


def run_extract(data: dict) -> dict:
    """从检索结果中提取事实并检查来源和证据。"""
    question = data.get("question", "").strip()
    results = data.get("results", [])

    if not question:
        raise ValueError("问题不能为空")

    if not isinstance(results, list):
        raise ValueError("检索结果格式无效")

    if not results:
        return {"question": question, "facts": []}

    for item in results:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("content"), str)
            or not isinstance(item.get("source"), str)
        ):
            raise ValueError("检索结果格式无效")

    context = "\n\n".join(
        f"[资料 {i}]\n"
        f"来源：{item['source']}\n"
        f"片段编号：{item.get('chunk_index')}\n"
        f"正文：{item['content']}"
        for i, item in enumerate(results, start=1)
    )

    prompt = f"""
你是AI行业研究中的信息提取助手。

请根据用户问题，从提供的资料中提取有直接依据的事实。
只输出合法 JSON，不要添加 Markdown 代码围栏或其他解释。

输出结构：
{{
  "question": "原始问题",
  "facts": [
    {{
      "fact": "提取的事实",
      "source": "来源文件名",
      "chunk_index": 0,
      "evidence": "从资料正文逐字复制的连续原文"
    }}
  ]
}}

要求：
1. 只能提取资料直接支持的事实，不得补充资料中没有的信息。
2. evidence 必须逐字引用正文，不得改写或拼接不存在的文字。
3. source 和 chunk_index 必须与对应资料一致。
4. 没有可提取的事实时，facts 返回空列表。
5. 不得把推测写成事实。
6. 涉及数字时，保留原文明确的年份、地域、单位和统计口径，不自行补全缺失信息。
7. 作者观点、预测或产品宣称须保留归属与限定语，不得改写成已证实的普遍事实。
8. facts 在整个当前任务中最多 {MAX_FACTS} 条，不是每份资料或每个片段各 {MAX_FACTS} 条。
9. 优先选择直接回答研究目标的关键事实，按重要性排列，避免重复和无关细节；不要为凑数量提取。
10. 资料充分时也不必用满上限；保留必要条件与统计口径，不得为了减少条数把无关结论拼成一条。

资料：
{context}

用户问题：
{question}
"""

    reply = create_chat_model().invoke(prompt)

    try:
        output = ExtractOut.model_validate_json(reply.content)
    except ValidationError as exc:
        if any(error["loc"] == ("facts",) and error["type"] == "too_long" for error in exc.errors()):
            raise ValueError(f"单个子任务最多提取{MAX_FACTS}条关键事实，模型返回超出上限") from exc
        raise ValueError("模型返回的 JSON 格式无效") from exc
    except (ValueError, TypeError) as exc:
        raise ValueError("模型返回的 JSON 格式无效") from exc

    if output.question != question:
        raise ValueError("模型返回的问题与原始问题不一致")

    for fact in output.facts:
        valid = any(
            item["source"] == fact.source
            and item.get("chunk_index") == fact.chunk_index
            and fact.evidence in item["content"]
            for item in results
        )

        if not valid:
            has_source = any(
                item["source"] == fact.source
                and item.get("chunk_index") == fact.chunk_index
                for item in results
            )

            if not has_source:
                raise ValueError("提取结果包含不存在的来源")

            raise ValueError("提取结果包含无法在原文中找到的证据")

        # 页码来自检索元数据，不信任模型自行生成的页码。
        matched = next(item for item in results if item["source"] == fact.source
                       and item.get("chunk_index") == fact.chunk_index and fact.evidence in item["content"])
        fact.page = matched.get("page")

    return output.model_dump()
