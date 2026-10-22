import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4

from sa_agent.retrieval.ingest import ingest_document_file
from sa_agent.retrieval.store import create_persistent_vector_store


MAX_BYTES = 10 * 1024 * 1024
SUFFIXES = {".txt", ".md", ".pdf", ".docx"}
_lock = RLock()
_active: set[str] = set()
logger = logging.getLogger(__name__)


class DocumentError(Exception):
    """携带HTTP状态码的文档业务错误，不包含内部异常详情。"""

    def __init__(self, status_code: int, message: str):
        """保存调用方可展示的错误信息。"""
        super().__init__(message)
        self.status_code = status_code


def document_path(directory: Path, filename: str) -> Path:
    """只接受安全的单文件名，拒绝目录访问、Windows设备名和不支持的格式。"""
    if (not filename or filename != filename.strip() or filename.startswith(".")
            or filename.endswith((".", " ")) or len(filename) > 180
            or re.search(r'[\\/:*?"<>|\x00-\x1f]', filename)
            or re.match(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", filename, re.I)):
        raise DocumentError(400, "请使用有效的单文件名，不要包含路径或特殊字符")
    if Path(filename).suffix.lower() not in SUFFIXES:
        raise DocumentError(400, "只支持 TXT、MD、DOCX 和可提取文字的PDF")
    root = directory.resolve()
    path = (root / filename).resolve()
    if not path.is_relative_to(root):
        raise DocumentError(400, "不允许访问文档目录之外的文件")
    return path


def _read_status(directory: Path) -> dict:
    """读取本地导入状态；损坏时明确报错，避免误报成功。"""
    path = directory / ".document_status.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("文档状态记录格式无效")
    return data


def _set_status(directory: Path, filename: str, status: str, **fields) -> dict:
    """在单进程内互斥并原子替换状态文件，保留每个文件的最新状态。"""
    with _lock:
        data = _read_status(directory)
        record = {"filename": filename, "status": status,
                  "updated_at": datetime.now(timezone.utc).isoformat(), **fields}
        data[filename] = record
        temp = directory / f".status-{uuid4().hex}.tmp"
        try:
            temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            temp.replace(directory / ".document_status.json")
        finally:
            temp.unlink(missing_ok=True)
        return record


def import_document(directory: Path, filename: str, store_dir: Path, collection: str) -> dict:
    """导入或重试已有文档，持久化状态；同一文件不同时处理。"""
    path = document_path(directory, filename)
    with _lock:
        if not path.is_file():
            raise DocumentError(404, "文档不存在")
        if path.stat().st_size > MAX_BYTES:
            raise DocumentError(413, "文件不能超过10 MB")
        if str(path) in _active:
            raise DocumentError(409, "该文档正在处理中，请稍后刷新")
        _active.add(str(path))
    try:
        _set_status(directory, filename, "processing")
        try:
            result = ingest_document_file(path, persist_directory=store_dir, collection_name=collection)
        except (ValueError, UnicodeError) as exc:
            message = "文本编码或内容无效，请使用UTF-8文本" if isinstance(exc, UnicodeError) else str(exc)
            _set_status(directory, filename, "failed", message=message)
            raise DocumentError(422, message) from exc
        except Exception as exc:
            logger.exception("文档导入失败")
            _set_status(directory, filename, "failed", message="导入失败，可重试；详情见服务日志")
            raise DocumentError(500, "文档导入失败，文件已保留，可重试导入") from exc
        return _set_status(directory, filename, "imported", chunks_count=result["chunks_count"])
    finally:
        with _lock:
            _active.discard(str(path))


def upload_document(directory: Path, filename: str, stream, store_dir: Path, collection: str) -> dict:
    """限制大小并独占保存新文件，然后同步导入；同名文件不覆盖。"""
    path = document_path(directory, filename)
    content = stream.read(MAX_BYTES + 1)
    if not content:
        raise DocumentError(422, "不能上传空文件")
    if len(content) > MAX_BYTES:
        raise DocumentError(413, "文件不能超过10 MB")
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as output:
            output.write(content)
    except FileExistsError as exc:
        raise DocumentError(409, "同名文件已存在，请重命名上传；失败文档可直接重试导入") from exc
    return import_document(directory, filename, store_dir, collection)


def list_documents(directory: Path, store_dir: Path, collection: str) -> dict:
    """列出本地文档并读取实际片段数，兼容旧接口导入且没有状态记录的文件。"""
    with _lock:
        records = _read_status(directory)
        active = set(_active)
    counts: dict[str, int] = {}
    if store_dir.exists():
        store = create_persistent_vector_store(persist_directory=store_dir, collection_name=collection)
        for metadata in store.get(include=["metadatas"])["metadatas"]:
            key = metadata.get("source_path", "")
            counts[key] = counts.get(key, 0) + 1
    items = []
    if directory.exists():
        for path in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
            if not path.is_file() or path.suffix.lower() not in SUFFIXES or path.name.startswith("."):
                continue
            # 不展示指向指定目录之外的链接文件。
            if not path.resolve().is_relative_to(directory.resolve()):
                continue
            count = counts.get(str(path.resolve()), 0)
            record = records.get(path.name, {})
            status = record.get("status", "imported" if count else "not_imported")
            message = record.get("message", "")
            if status == "imported" and count == 0:
                status, message = "not_imported", "未找到对应入库片段，请重新导入"
            if status == "processing" and str(path.resolve()) not in active:
                status, message = "failed", "上次导入未正常完成，请重试导入"
            items.append({"filename": path.name, "format": path.suffix[1:].upper(),
                          "size_bytes": path.stat().st_size, "status": status,
                          "chunks_count": count, "updated_at": record.get("updated_at"), "message": message})
    return {"documents": items}
