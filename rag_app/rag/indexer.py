import os
from typing import List

import numpy as np
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


def get_dense_scores(store: Chroma, query: str, top_k: int) -> List[dict]:
    """使用向量库计算dense检索得分。

    Args:
        store: 向量库实例。
        query: 查询文本。
        top_k: 返回结果数量。

    Returns:
        List[dict]: 带得分的检索结果列表。
    """
    docs_and_scores = store.similarity_search_with_score(query, k=top_k)
    results: List[dict] = []
    for doc, score in docs_and_scores:
        results.append(
            {
                "doc_id": doc.metadata.get("doc_id", ""),
                "text": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "score": float(1.0 / (1.0 + score)),
            }
        )
    return results


def normalize_scores(items: List[dict]) -> List[dict]:
    """将分数归一化到0-1范围。

    Args:
        items: 带score字段的结果列表。

    Returns:
        List[dict]: 归一化后的结果列表。
    """
    if not items:
        return items
    scores = np.array([i["score"] for i in items], dtype=np.float32)
    min_s = float(scores.min())
    max_s = float(scores.max())
    if max_s - min_s < 1e-6:
        for i in items:
            i["score"] = 1.0
        return items
    for i in items:
        i["score"] = (i["score"] - min_s) / (max_s - min_s)
    return items
