
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_text_file(file_path: str | Path) -> str:
    """读取 UTF-8 编码的文本文件。"""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"文档不存在：{path}")
    return path.read_text(encoding="utf-8")

def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[str]:
    """将长文本拆分为多个有重叠的文本片段。"""
    if not text.strip():
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须大于等于 0 且小于 chunk_size")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    return splitter.split_text(text)

def load_and_split_document(file_path: str | Path) -> list[Document]:
    """读取文本文件，将其切分为带来源元数据的文档片段。"""
    path = Path(file_path)
    text = load_text_file(path)
    chunks = split_text(text)

    documents = []
    for index, chunk in enumerate(chunks):
        document = Document(
            page_content=chunk,
            metadata={
                "source": path.name,
                "chunk_index": index,
            },
        )
        documents.append(document)

    return documents