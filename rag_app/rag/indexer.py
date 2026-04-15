import os
from typing import List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from .config import SETTINGS


def build_or_load_chroma(
    chunks: List[dict], index_dir: str, allow_build: bool = True
) -> Chroma:
    """构建或加载Chroma向量库。

    Args:
        chunks: chunk列表。
        index_dir: 索引目录路径。

    Returns:
        Chroma: 向量库实例。
    """
    os.makedirs(index_dir, exist_ok=True)
    persist_dir = os.path.join(index_dir, "chroma")
    embeddings = HuggingFaceEmbeddings(model_name=SETTINGS.embedding_model)
    if os.path.isdir(persist_dir) and os.listdir(persist_dir):
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="rag_chunks",
        )
    if not allow_build:
        raise RuntimeError(
            f"Chroma index not found under '{persist_dir}'. Please run indexing first."
        )
    texts = [c["text"] for c in chunks]
    metadatas = [
        {
            "doc_id": c["doc_id"],
            "source": c["source"],
            "source_doc_id": c.get("source_doc_id", ""),
            "chunk_index": c.get("chunk_index", -1),
            "heading_context": c.get("heading_context", ""),
        }
        for c in chunks
    ]
    store = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=persist_dir,
        collection_name="rag_chunks",
    )
    if hasattr(store, "persist"):
        store.persist()
    return store
