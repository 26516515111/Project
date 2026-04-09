import os
from typing import List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from .config import SETTINGS


def build_or_load_chroma(chunks: List[dict], index_dir: str) -> Chroma:
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
    texts = [c["text"] for c in chunks]
    metadatas = [{"doc_id": c["doc_id"], "source": c["source"]} for c in chunks]
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
