from typing import List, Union
import logging

from .schema import Passage, RetrievedContext
from .config import SETTINGS
from .indexer import get_dense_scores, normalize_scores
from .bm25 import get_bm25_scores
from .reranker import get_reranker

logger = logging.getLogger(__name__)


def hybrid_retrieve(
    store,
    bm25,
    chunks: List[dict],
    query: Union[str, List[str]],
    top_k: int,
    max_subqueries: int = None,
    per_query_top_k: int = None,
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
    queries = query if isinstance(query, list) else [query]
    queries = [q for q in queries if str(q).strip()]
    if not queries:
        return RetrievedContext(passages=[])
    if max_subqueries is not None:
        queries = queries[:max_subqueries]

    merged = {}
    per_k = per_query_top_k or max(1, top_k)
    for q in queries:
        dense = get_dense_scores(store, q, top_k=per_k)
        sparse = get_bm25_scores(bm25, chunks, q, top_k=per_k)
        dense = normalize_scores(dense)
        sparse = normalize_scores(sparse)
        for item in dense:
            entry = merged.get(item["doc_id"])
            score = item["score"] * SETTINGS.vector_weight
            if entry is None or score > entry["score"]:
                merged[item["doc_id"]] = {**item, "score": score}
        for item in sparse:
            entry = merged.get(item["doc_id"])
            score = item["score"] * SETTINGS.bm25_weight
            if entry is None:
                merged[item["doc_id"]] = {**item, "score": score}
            else:
                entry["score"] = max(entry["score"], score) + score * 0.5

    ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
    max_per_source = max(1, top_k // 3)
    source_counts = {}
    filtered = []
    for item in ranked:
        source_key = (
            item.get("source_doc_id") or item.get("source") or item.get("doc_id")
        )
        count = source_counts.get(source_key, 0)
        if count >= max_per_source:
            continue
        source_counts[source_key] = count + 1
        filtered.append(item)
        if len(filtered) >= top_k:
            break
    passages = [
        Passage(
            doc_id=r["doc_id"],
            text=r["text"],
            source=r.get("source", ""),
            score=r["score"],
        )
        for r in filtered
    ]

    # Apply reranking if enabled
    reranker = get_reranker()
    if reranker:
        logger.info("Applying reranking...")
        rerank_query = queries[0]
        reranked = reranker.rerank(rerank_query, passages)
        return reranked

    return RetrievedContext(passages=passages)
