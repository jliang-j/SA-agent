
from sa_agent.skills.extract import run_extract
from sa_agent.skills.retrieve import run_retrieval
from sa_agent.skills.verify import run_verify


def run_research(vector_store, question: str, k: int = 3) -> dict:
    """依次执行检索、提取和来源核验。"""
    retrieved = run_retrieval(
        vector_store=vector_store,
        question=question,
        k=k,
    )
    #print("检索结果 question：", repr(retrieved.get("question")))
    extracted = run_extract(retrieved)
    #print("提取结果 question：", repr(extracted.get("question")))
    verified = run_verify(
        extracted=extracted,
        retrieved=retrieved,
    )
    #print("核验结果 question：", repr(verified.get("question")))
    return {
        "question": question,
        "retrieval": retrieved,
        "extraction": extracted,
        "verification": verified,
    }
