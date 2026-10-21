
def run_verify(extracted: dict, retrieved: dict) -> dict:
    """核验提取事实的来源、片段编号和原文。"""
    facts = extracted.get("facts")
    results = retrieved.get("results")

    if not isinstance(facts, list):
        raise ValueError("提取结果格式无效")

    if not isinstance(results, list):
        raise ValueError("检索结果格式无效")

    checked = []

    for fact in facts:
        if not isinstance(fact, dict):
            raise ValueError("事实格式无效")

        source = fact.get("source")
        chunk_index = fact.get("chunk_index")
        evidence = fact.get("evidence")

        matched = [
            item
            for item in results
            if isinstance(item, dict)
            and item.get("source") == source
            and item.get("chunk_index") == chunk_index
        ]

        if not matched:
            status = "failed"
            reason = "来源或片段编号不存在于检索结果中。"
        elif (
            not isinstance(evidence, str)
            or not evidence
            or not any(
                evidence in item.get("content", "")
                for item in matched
                if isinstance(item.get("content"), str)
            )
        ):
            status = "failed"
            reason = "证据无法在对应原文中找到。"
        else:
            status = "passed"
            reason = "来源、片段编号和原文证据匹配。"

        checked.append({
            **fact,
            "status": status,
            "reason": reason,
        })

    passed = sum(item["status"] == "passed" for item in checked)
    failed = len(checked) - passed

    return {
        "question": extracted.get("question", ""),
        "facts": checked,
        "passed_count": passed,
        "failed_count": failed,
    }
