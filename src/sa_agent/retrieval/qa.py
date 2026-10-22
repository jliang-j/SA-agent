
from langchain_core.documents import Document

from sa_agent.llm import create_chat_model
from sa_agent.retrieval.store import search_documents


def answer_question(
    vector_store,
    question: str,
    k: int = 3,
) -> dict:
    """基于检索到的文档回答问题，并返回参考来源。"""
    if not question.strip():
        raise ValueError("问题不能为空")

    documents = search_documents(
        vector_store=vector_store,
        query=question,
        k=k,
    )

    if not documents:
        return {
            "answer": "未检索到相关资料，暂时无法回答。",
            "sources": [],
        }

    context = "\n\n".join(
        f"[资料 {index}]\n"
        f"来源：{document.metadata.get('source', '未知')}\n"
        f"内容：{document.page_content}"
        for index, document in enumerate(documents, start=1)
    )

    prompt = f"""
你是一个严谨的AI行业研究助手，主要研究技术与产品、企业应用、产业信息和风险治理。
请仅依据下面提供的资料回答用户问题。

要求：
1. 不要把资料中没有的信息编造成事实。
2. 如果资料不足以回答，请明确说明。
3. 回答应清晰、简洁。
4. 在相关结论后标注资料编号，例如 [资料 1]。
5. 不把历史资料称为最新；涉及数字时保留资料明确给出的年份、地域和统计口径。
6. 区分作者观点与事实，不根据常识补充企业能力、政策要求或市场数据。

参考资料：
{context}

用户问题：
{question}
"""

    model = create_chat_model()
    response = model.invoke(prompt)

    return {
        "answer": response.content,
        "sources": [
            {
                "source": document.metadata.get("source", "未知"),
                "chunk_index": document.metadata.get("chunk_index"),
                "content": document.page_content,
                **({"page": document.metadata["page"]} if "page" in document.metadata else {}),
            }
            for document in documents
        ],
    }
