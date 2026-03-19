import json
import os
from typing import Dict, List, Tuple

from rag.config import SETTINGS
from rag.chunking import TextChunker
from rag.loader import load_documents, load_chunks_json
from rag.indexer import build_or_load_chroma


def _read_chunks(path: str) -> List[dict]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _read_doc_source_map(path: str) -> List[dict]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        items = []
        for source, item in data.items():
            if not isinstance(item, dict):
                continue
            items.append({"source": source, **item})
        return items
    return []


def _doc_id_from_source(source: str) -> str:
    name = os.path.basename(source)
    stem, _ = os.path.splitext(name)
    return stem.strip() or name


def _rel_path(path: str, root: str) -> str:
    try:
        return os.path.relpath(path, root)
    except Exception:
        return path


def _build_chunks_for_doc(doc: dict, chunker: TextChunker, doc_id: str) -> List[dict]:
    chunks: List[dict] = []
    source = doc.get("source", "")
    if "pages" in doc and isinstance(doc["pages"], list):
        for page in doc["pages"]:
            page_text = str(page.get("text", "")).strip()
            if not page_text:
                continue
            page_num = page.get("page")
            for i, chunk in enumerate(chunker.split_text(page_text)):
                chunk_id = f"{doc_id}::page_{page_num}::chunk_{i}"
                src = source
                if page_num is not None:
                    src = f"{source}#p{page_num}" if source else f"p{page_num}"
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "chunk_index": i,
                        "text": chunk,
                        "source": src,
                        "char_count": len(chunk),
                    }
                )
        return chunks

    text = str(doc.get("text", "")).strip()
    for i, chunk in enumerate(chunker.split_text(text)):
        chunk_id = f"{doc_id}::chunk_{i}"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "chunk_index": i,
                "text": chunk,
                "source": source,
                "char_count": len(chunk),
            }
        )
    return chunks


def _update_doc_source_map(
    entries: List[dict], new_docs: List[Tuple[str, str]], num_chunks_map: Dict[str, int]
) -> List[dict]:
    existing = {str(item.get("doc_id", "")).strip(): item for item in entries}
    for doc_id, rel_path in new_docs:
        entry = existing.get(doc_id, {})
        entry.update(
            {
                "doc_id": doc_id,
                "source": os.path.basename(rel_path),
                "path": rel_path,
                "num_chunks": num_chunks_map.get(doc_id, 0),
            }
        )
        existing[doc_id] = entry
    return list(existing.values())


def main() -> None:
    docs_dir = SETTINGS.docs_dir
    index_dir = SETTINGS.index_dir
    chunks_path = os.path.join(docs_dir, "chunks.json")
    doc_map_path = os.path.join(docs_dir, "doc_source_map.json")

    all_docs = load_documents(docs_dir)
    chunker = TextChunker.from_settings(SETTINGS)

    existing_chunks_raw = _read_chunks(chunks_path)
    existing_chunk_ids = {
        str(item.get("chunk_id", "")).strip() for item in existing_chunks_raw
    }
    existing_chunks_norm = load_chunks_json(docs_dir)
    existing_doc_map = _read_doc_source_map(doc_map_path)
    existing_doc_ids = {
        str(item.get("doc_id", "")).strip() for item in existing_doc_map
    }

    new_chunks_raw: List[dict] = []
    new_chunks_norm: List[dict] = []
    new_docs: List[Tuple[str, str]] = []
    num_chunks_map: Dict[str, int] = {}

    for doc in all_docs:
        source = str(doc.get("source", "")).strip()
        if not source:
            continue
        doc_id = _doc_id_from_source(source)
        if doc_id in existing_doc_ids:
            continue
        built = _build_chunks_for_doc(doc, chunker, doc_id)
        built = [c for c in built if c.get("chunk_id") not in existing_chunk_ids]
        if not built:
            continue
        new_chunks_raw.extend(built)
        for item in built:
            normalized = {**item, "doc_id": item.get("chunk_id", "")}
            new_chunks_norm.append(normalized)
        num_chunks_map[doc_id] = len(built)
        rel_path = _rel_path(doc.get("id", ""), os.path.dirname(docs_dir))
        new_docs.append((doc_id, rel_path))

    if not new_chunks_raw:
        print("No new documents found. Nothing to update.")
        return

    store = build_or_load_chroma(existing_chunks_norm + new_chunks_norm, index_dir)
    if existing_chunks_norm:
        texts = [c["text"] for c in new_chunks_norm]
        metadatas = [
            {"doc_id": c["chunk_id"], "source": c.get("source", "")}
            for c in new_chunks_norm
        ]
        store.add_texts(texts=texts, metadatas=metadatas)
        store.persist()

    updated_chunks = existing_chunks_raw + new_chunks_raw
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(updated_chunks, f, ensure_ascii=False, indent=2)

    updated_doc_map = _update_doc_source_map(existing_doc_map, new_docs, num_chunks_map)
    with open(doc_map_path, "w", encoding="utf-8") as f:
        json.dump(updated_doc_map, f, ensure_ascii=False, indent=2)

    print(
        f"Updated chunks.json (+{len(new_chunks_raw)} chunks) and doc_source_map.json (+{len(new_docs)} docs)."
    )


if __name__ == "__main__":
    main()
