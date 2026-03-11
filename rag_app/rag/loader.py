import json
import os
from typing import List, Dict


def load_documents(docs_dir: str) -> List[dict]:
    docs: List[dict] = []
    if not os.path.isdir(docs_dir):
        return docs
    for root, _, files in os.walk(docs_dir):
        for name in files:
            if not name.lower().endswith((".txt", ".md")):
                continue
            path = os.path.join(root, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
            except UnicodeDecodeError:
                with open(path, "r", encoding="gb18030", errors="ignore") as f:
                    text = f.read().strip()
            if not text:
                continue
            docs.append({"id": path, "text": text, "source": name})
    return docs


def load_chunks_json(docs_dir: str) -> List[dict]:
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
    path = os.path.join(docs_dir, "doc_source_map.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping: Dict[str, dict] = {}
    for item in data:
        doc_id = str(item.get("doc_id", "")).strip()
        if not doc_id:
            continue
        mapping[doc_id] = item
    return mapping
