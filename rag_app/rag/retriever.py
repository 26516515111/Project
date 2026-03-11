from typing import List

from .schema import Passage, RetrievedContext
from .config import SETTINGS
from .indexer import get_dense_scores, normalize_scores
from .bm25 import get_bm25_scores


def hybrid_retrieve(
    store,
    bm25,
    chunks: List[dict],
    query: str,
    top_k: int,
) -> RetrievedContext:
    """融合dense与sparse分数，返回排序后的检索上下文。

    Args:
        store: 向量库实例。
        bm25: BM25检索器实例或None。
        chunks: chunk列表。
        query: 查询文本。
        top_k: 返回结果数量。

    Returns:
        RetrievedContext: 检索上下文结果。
    """
    dense = get_dense_scores(store, query, top_k=top_k)
    sparse = get_bm25_scores(bm25, chunks, query, top_k=top_k)
    dense = normalize_scores(dense)
    sparse = normalize_scores(sparse)

    merged = {}
    for item in dense:
        merged[item["doc_id"]] = {
            **item,
            "score": item["score"] * SETTINGS.vector_weight,
        }
    for item in sparse:
        if item["doc_id"] in merged:
            merged[item["doc_id"]]["score"] += item["score"] * SETTINGS.bm25_weight
        else:
            merged[item["doc_id"]] = {
                **item,
                "score": item["score"] * SETTINGS.bm25_weight,
            }

    ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    passages = [
        Passage(
            doc_id=r["doc_id"],
            text=r["text"],
            source=r.get("source", ""),
            score=r["score"],
        )
        for r in ranked
    ]
    return RetrievedContext(passages=passages)
