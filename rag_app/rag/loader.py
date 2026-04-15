import json
import os
from typing import List, Dict

from .config import SETTINGS


def _infer_domain_by_text(text: str) -> str:
    value = str(text or "").lower()
    if not value:
        return ""
    rules = getattr(SETTINGS, "domain_routing_rules", {}) or {}
    scores: Dict[str, int] = {}
    for domain, keywords in rules.items():
        hit = 0
        for kw in keywords:
            if kw and kw in value:
                hit += 1
        if hit > 0:
            scores[domain] = hit
    if not scores:
        return ""
    best = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[0]
    return best[0]


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


def _resolve_chunks_path(base_dir: str) -> str:
    candidates = [
        os.path.join(base_dir, "KG", "chunks", "chunk.json"),
        os.path.join(base_dir, "KG", "chunks", "chunks.json"),
        os.path.join(base_dir, "chunk.json"),
        os.path.join(base_dir, "chunks.json"),
        os.path.join(base_dir, "chunks", "chunk.json"),
        os.path.join(base_dir, "chunks", "chunks.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return ""


def _resolve_doc_source_map_path(base_dir: str) -> str:
    candidates = [
        os.path.join(base_dir, "KG", "chunks", "doc_source_map.json"),
        os.path.join(base_dir, "doc_source_map.json"),
        os.path.join(base_dir, "chunks", "doc_source_map.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return ""


def _resolve_chunk_to_kg_path(base_dir: str) -> str:
    candidates = [
        os.path.join(base_dir, "KG", "chunks", "chunk_to_kg.json"),
        os.path.join(base_dir, "chunk_to_kg.json"),
        os.path.join(base_dir, "chunks", "chunk_to_kg.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return ""


def load_chunks_json(base_dir: str) -> List[dict]:
    """读取chunks.json并规范化为chunk列表。

    Args:
        base_dir: chunks所在目录路径。

    Returns:
        List[dict]: chunk列表。
    """
    path = _resolve_chunks_path(base_dir)
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks: List[dict] = []
    source_domain_cache: Dict[str, str] = {}
    for item in data:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        source_doc_id = str(item.get("source_doc_id", "") or "").strip()
        if not source_doc_id:
            source_doc_id = str(item.get("doc_id", "") or "").strip()
        chunk_index = item.get("chunk_index")
        chunk_id = str(item.get("chunk_id", "")).strip()
        if not chunk_id and source_doc_id and chunk_index is not None:
            chunk_id = f"{source_doc_id}::chunk_{chunk_index}"
        if not source_doc_id and "::chunk_" in chunk_id:
            source_doc_id = chunk_id.split("::chunk_", 1)[0].strip()
        if not source_doc_id:
            source = str(item.get("source", "") or "").strip()
            if source.lower().endswith(".md"):
                source_doc_id = source[:-3]
        if not chunk_id:
            continue
        source = str(item.get("source", "")).strip()
        source_doc_id_for_domain = source_doc_id or source
        domain = ""
        if source_doc_id_for_domain in source_domain_cache:
            domain = source_domain_cache[source_doc_id_for_domain]
        else:
            domain = _infer_domain_by_text(
                " ".join(
                    [
                        source_doc_id_for_domain,
                        source,
                        str(item.get("heading_context", "") or ""),
                        text[:240],
                    ]
                )
            )
            source_domain_cache[source_doc_id_for_domain] = domain
        chunks.append(
            {
                "doc_id": chunk_id,
                "chunk_id": chunk_id,
                "source_doc_id": source_doc_id,
                "chunk_index": chunk_index,
                "text": text,
                "source": source,
                "heading_context": str(item.get("heading_context", "") or "").strip(),
                "domain": domain,
            }
        )
    return chunks


def load_doc_source_map(base_dir: str) -> Dict[str, dict]:
    """读取doc_source_map.json并构建doc_id到来源信息的映射。

    Args:
        base_dir: doc_source_map.json所在目录路径。

    Returns:
        Dict[str, dict]: doc_id到来源信息的映射。
    """
    path = _resolve_doc_source_map_path(base_dir)
    if not path:
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


def load_chunk_to_kg(base_dir: str) -> Dict[str, dict]:
    """读取chunk_to_kg.json并构建chunk_id到KG映射。

    Args:
        base_dir: chunk_to_kg.json所在目录路径。

    Returns:
        Dict[str, dict]: chunk_id到KG实体/关系映射。
    """
    path = _resolve_chunk_to_kg_path(base_dir)
    if not path:
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
