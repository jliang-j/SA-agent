from sa_agent.retrieval.loader import load_text_file, split_text


def ts_text_returns_chunks():
    text = "这是第一段内容。" * 100

    chunks = split_text(text, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(chunks)


def ts_empty_text():
    assert split_text("   ") == []


def load_missing_file():
    import pytest

    with pytest.raises(FileNotFoundError):
        load_text_file("not_exists.txt")

def test_load_and_split_document(tmp_path):
    from sa_agent.retrieval.loader import load_and_split_document

    file_path = tmp_path / "industry_report.txt"
    file_path.write_text("行业报告内容。" * 100, encoding="utf-8")

    documents = load_and_split_document(file_path)

    assert len(documents) > 1
    assert documents[0].page_content
    assert documents[0].metadata["source"] == "industry_report.txt"
    assert documents[0].metadata["chunk_index"] == 0
    assert documents[1].metadata["chunk_index"] == 1
