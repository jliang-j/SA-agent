
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
    """解析文本、DOCX或PDF，复用切分并保留来源；仅PDF提供页码。"""
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        pages = [(None, load_text_file(path))]
    elif suffix == ".pdf":
        pages = load_pdf_pages(path)
    elif suffix == ".docx":
        pages = [(None, load_docx_text(path))]
    else:
        raise ValueError("只支持 TXT、MD、DOCX 和文本 PDF")
    documents = []
    for page, text in pages:
        for chunk in split_text(text):
            metadata = {"source": path.name, "chunk_index": len(documents)}
            if page is not None:
                metadata["page"] = page
            documents.append(Document(page_content=chunk, metadata=metadata))

    return documents


def load_docx_text(path: Path) -> str:
    """按正文顺序提取段落和普通表格，表格以制表符分列、换行分行。"""
    from zipfile import BadZipFile

    from docx import Document as WordDocument
    from docx.opc.exceptions import PackageNotFoundError
    from docx.text.paragraph import Paragraph
    from lxml.etree import XMLSyntaxError

    if not path.is_file():
        raise FileNotFoundError(f"文档不存在：{path}")
    try:
        document = WordDocument(path)
        blocks = []
        # 使用公开的顺序迭代接口，避免先读全部段落、再读全部表格而打乱上下文。
        for block in document.iter_inner_content():
            if isinstance(block, Paragraph):
                text = block.text
            else:
                text = "\n".join("\t".join(cell.text for cell in row.cells)
                                 for row in block.rows)
            if text.strip():
                blocks.append(text)
        return "\n\n".join(blocks)
    except (PackageNotFoundError, BadZipFile, XMLSyntaxError, KeyError, ValueError, OSError) as exc:
        raise ValueError("DOCX解析失败，请检查文件是否完整且为有效DOCX；不支持旧式DOC或加密文件") from exc


def load_pdf_pages(path: Path) -> list[tuple[int, str]]:
    """提取PDF各页正文；拒绝加密文件和超大页数，不提供OCR。"""
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise ValueError("暂不支持加密PDF，请先提供未加密文件")
        if len(reader.pages) > 200:
            raise ValueError("PDF最多支持200页")
        return [(number, page.extract_text() or "")
                for number, page in enumerate(reader.pages, start=1)]
    except (PdfReadError, OSError) as exc:
        raise ValueError("PDF解析失败，请检查文件是否完整且为有效PDF") from exc
