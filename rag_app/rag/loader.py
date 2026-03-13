import json
import os
from typing import List, Dict


def load_documents(docs_dir: str) -> List[dict]:
    """从目录中加载文本/Markdown文档并返回统一结构列表。

    Args:
        docs_dir: 文档目录路径。

    Returns:
        List[dict]: 文档列表。
    """
    docs: List[dict] = []
    if not os.path.isdir(docs_dir):
        return docs
    for root, _, files in os.walk(docs_dir):
        for name in files:
            lower_name = name.lower()
            path = os.path.join(root, name)
            if lower_name.endswith((".txt", ".md")):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                except UnicodeDecodeError:
                    with open(path, "r", encoding="gb18030", errors="ignore") as f:
                        text = f.read().strip()
                if not text:
                    continue
                docs.append({"id": path, "text": text, "source": name})
                continue
            if lower_name.endswith(".pdf"):
                pages = _load_pdf_pages(path)
                if not pages:
                    continue
                docs.append({"id": path, "pages": pages, "source": name})
    return docs


def _load_pdf_pages(path: str) -> List[dict]:
    try:
        import pdfplumber
    except ImportError:
        return []
    pages: List[dict] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            pages.append({"page": i, "text": text})
    return pages


def load_chunks_json(docs_dir: str) -> List[dict]:
    """读取chunks.json并规范化为chunk列表。

    Args:
        docs_dir: 文档目录路径。

    Returns:
        List[dict]: chunk列表。
    """
    path = os.path.join(docs_dir, "chunks.json")
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks: List[dict] = []
    for item in data:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        source_doc_id = str(item.get("doc_id", "")).strip()
        chunk_index = item.get("chunk_index")
        chunk_id = str(item.get("chunk_id", "")).strip()
        if not chunk_id and source_doc_id and chunk_index is not None:
            chunk_id = f"{source_doc_id}::chunk_{chunk_index}"
        if not chunk_id:
            continue
        chunks.append(
            {
                "doc_id": chunk_id,
                "chunk_id": chunk_id,
                "source_doc_id": source_doc_id,
                "chunk_index": chunk_index,
                "text": text,
                "source": str(item.get("source", "")).strip(),
            }
        )
    return chunks


def load_doc_source_map(docs_dir: str) -> Dict[str, dict]:
    """读取doc_source_map.json并构建doc_id到来源信息的映射。

    Args:
        docs_dir: 文档目录路径。

    Returns:
        Dict[str, dict]: doc_id到来源信息的映射。
    """
    path = os.path.join(docs_dir, "doc_source_map.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping: Dict[str, dict] = {}
    if isinstance(data, dict):
        for source, item in data.items():
            if not isinstance(item, dict):
                continue
            doc_id = str(item.get("doc_id", "")).strip()
            if not doc_id:
                continue
            mapping[doc_id] = {**item, "source": source}
        return mapping
    for item in data:
        doc_id = str(item.get("doc_id", "")).strip()
        if not doc_id:
            continue
        mapping[doc_id] = item
    return mapping


def load_chunk_to_kg(docs_dir: str) -> Dict[str, dict]:
    """读取chunk_to_kg.json并构建chunk_id到KG映射。

    Args:
        docs_dir: 文档目录路径。

    Returns:
        Dict[str, dict]: chunk_id到KG实体/关系映射。
    """
    path = os.path.join(docs_dir, "chunk_to_kg.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = data.get("chunks", []) if isinstance(data, dict) else data
    mapping: Dict[str, dict] = {}
    for item in chunks:
        chunk_id = str(item.get("chunk_id", "")).strip()
        if not chunk_id:
            continue
        mapping[chunk_id] = {
            "chunk_id": chunk_id,
            "doc_id": str(item.get("doc_id", "")).strip(),
            "source": str(item.get("source", "")).strip(),
            "chunk_index": item.get("chunk_index"),
            "kg_entities": item.get("kg_entities", []) or [],
            "kg_relations": item.get("kg_relations", []) or [],
        }
    return mapping
