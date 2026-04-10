from typing import List, Union, Dict, Optional
import logging
import re

from langchain.retrievers import EnsembleRetriever
from langchain_core.runnables import RunnableLambda

from .schema import Passage, RetrievedContext
from .config import SETTINGS
from .reranker import get_reranker

logger = logging.getLogger(__name__)


def _normalize_for_match(text: str) -> str:
    value = " ".join(str(text or "").strip().split()).lower()
    return re.sub(r"\s+", "", value)


def _extract_entity_candidates(query: str) -> List[str]:
    q = str(query or "").strip()
    if not q:
        return []
    candidates = set()
    patterns = [
        r"[A-Za-z]{2,}\d*(?:-[A-Za-z0-9]+)+[\u4e00-\u9fffA-Za-z0-9_-]*",
        r"[A-Za-z]{2,}\d+[A-Za-z0-9_-]*[\u4e00-\u9fffA-Za-z0-9_-]*",
    ]
    for pat in patterns:
        for hit in re.findall(pat, q):
            token = " ".join(str(hit).split())
            if len(token) >= 3:
                candidates.add(token)
    for sep in ["的", "作用", "功能", "有哪些", "是什么", "如何", "和", "与"]:
        if sep in q:
            left = q.split(sep)[0].strip(" ，,。:：")
            if len(left) >= 3 and re.search(r"[A-Za-z0-9-]", left):
                candidates.add(left)
    return sorted(candidates, key=lambda x: len(x), reverse=True)


def _count_entity_hits(text: str, entities: List[str]) -> int:
    if not entities:
        return 0
    haystack = _normalize_for_match(text)
    if not haystack:
        return 0
    hits = 0
    for ent in entities:
        needle = _normalize_for_match(ent)
        if needle and needle in haystack:
            hits += 1
    return hits


def _is_technical_question(query: str) -> bool:
    q = str(query or "").strip().lower()
    if not q:
        return False
    keywords = (
        "技术",
        "参数",
        "规格",
        "配置",
        "电源",
        "电压",
        "电流",
        "功率",
        "接口",
        "协议",
        "防护",
        "等级",
        "温度",
        "湿度",
    )
    return any(k in q for k in keywords)


def _prepare_payload(
    store,
    bm25,
    query: Union[str, List[str]],
    top_k: int,
    max_subqueries: int = None,
    per_query_top_k: int = None,
    use_reranker: bool = True,
    allowed_source_doc_ids: Optional[List[str]] = None,
) -> Dict:
    queries = query if isinstance(query, list) else [query]
    queries = [q for q in queries if str(q).strip()]
    if max_subqueries is not None:
        queries = queries[:max_subqueries]
    return {
        "store": store,
        "bm25": bm25,
        "queries": queries,
        "top_k": top_k,
        "per_query_top_k": per_query_top_k,
        "use_reranker": use_reranker,
        "allowed_source_doc_ids": allowed_source_doc_ids or [],
    }


def _run_search(payload: Dict) -> Dict:
    queries = payload["queries"]
    if not queries:
        payload["merged"] = {}
        return payload
    merged: Dict[str, dict] = {}
    per_k = payload["per_query_top_k"] or max(1, payload["top_k"])
    allowed_source_doc_ids = set(payload.get("allowed_source_doc_ids") or [])
    vector_retriever = payload["store"].as_retriever(search_kwargs={"k": per_k})
    bm25_retriever = payload.get("bm25")
    if bm25_retriever is not None:
        bm25_retriever.k = per_k
        retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[SETTINGS.vector_weight, SETTINGS.bm25_weight],
        )
    else:
        retriever = vector_retriever

    for q in queries:
        docs = retriever.invoke(q)
        if allowed_source_doc_ids:
            filtered_docs = []
            for d in docs:
                metadata = d.metadata or {}
                source_doc_id = str(metadata.get("source_doc_id", "") or "").strip()
                if not source_doc_id:
                    doc_id = str(metadata.get("doc_id", "") or "").strip()
                    if "::chunk_" in doc_id:
                        source_doc_id = doc_id.split("::chunk_", 1)[0].strip()
                if source_doc_id in allowed_source_doc_ids:
                    filtered_docs.append(d)
            docs = filtered_docs
        total = max(1, len(docs))
        for idx, doc in enumerate(docs):
            metadata = doc.metadata or {}
            doc_id = str(metadata.get("doc_id", "") or "")
            if not doc_id:
                continue
            source_doc_id = str(metadata.get("source_doc_id", "") or "").strip()
            if not source_doc_id and "::chunk_" in doc_id:
                source_doc_id = doc_id.split("::chunk_", 1)[0].strip()
            score = float((total - idx) / total)
            item = {
                "doc_id": doc_id,
                "text": str(doc.page_content or ""),
                "source": str(metadata.get("source", "") or ""),
                "source_doc_id": source_doc_id,
                "heading_context": str(metadata.get("heading_context", "") or ""),
                "score": score,
            }
            entry = merged.get(doc_id)
            if entry is None or score > entry["score"]:
                merged[doc_id] = item
    payload["merged"] = merged
    return payload


def _apply_entity_priority(payload: Dict) -> Dict:
    if not SETTINGS.entity_priority_enabled:
        payload["entity_candidates"] = []
        return payload
    queries = payload.get("queries") or []
    merged = payload.get("merged") or {}
    if not queries or not merged:
        payload["entity_candidates"] = []
        return payload
    entity_candidates = _extract_entity_candidates(queries[0])
    payload["entity_candidates"] = entity_candidates
    if not entity_candidates:
        return payload

    entity_hit_ids = set()
    for doc_id, item in merged.items():
        text = str(item.get("text", "") or "")
        source = str(item.get("source", "") or "")
        source_doc_id = str(item.get("source_doc_id", "") or "")
        match_hits = max(
            _count_entity_hits(text, entity_candidates),
            _count_entity_hits(source, entity_candidates),
            _count_entity_hits(source_doc_id, entity_candidates),
            _count_entity_hits(doc_id, entity_candidates),
        )
        if match_hits > 0:
            item["entity_hit"] = True
            item["entity_hits"] = match_hits
            entity_hit_ids.add(doc_id)
        else:
            item["entity_hit"] = False
            item["entity_hits"] = 0

    # 先做实体过滤：当识别到实体且存在命中时，仅保留实体命中的候选。
    if SETTINGS.entity_strict_filter and entity_hit_ids:
        merged = {
            doc_id: item for doc_id, item in merged.items() if doc_id in entity_hit_ids
        }

    # 再做加分：仅在过滤后的候选中提升实体命中项分数。
    for item in merged.values():
        hits = int(item.get("entity_hits", 0))
        if hits > 0:
            item["score"] = float(
                item.get("score", 0.0)
            ) + SETTINGS.entity_boost * float(hits)

    payload["merged"] = merged
    return payload


def _apply_technical_heading_boost(payload: Dict) -> Dict:
    if not SETTINGS.tech_heading_boost_enabled:
        return payload
    queries = payload.get("queries") or []
    merged = payload.get("merged") or {}
    if not queries or not merged:
        return payload
    if not _is_technical_question(queries[0]):
        return payload

    boost = float(getattr(SETTINGS, "tech_heading_boost", 0.0) or 0.0)
    if boost <= 0.0:
        return payload
    for item in merged.values():
        heading_context = str(item.get("heading_context", "") or "")
        if "技术" in heading_context:
            item["score"] = float(item.get("score", 0.0)) + boost
    payload["merged"] = merged
    return payload


def _build_context(payload: Dict) -> Dict:
    merged = payload.get("merged") or {}
    ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
    entity_candidates = payload.get("entity_candidates") or []
    max_per_source = max(1, payload["top_k"] // 3)
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
        if len(filtered) >= payload["top_k"]:
            break
    if SETTINGS.entity_priority_enabled and entity_candidates:
        min_entity = min(SETTINGS.entity_min_in_topk, payload["top_k"])
        entity_count = sum(1 for item in filtered if item.get("entity_hit"))
        if entity_count < min_entity:
            selected_ids = {item.get("doc_id", "") for item in filtered}
            for item in ranked:
                if not item.get("entity_hit"):
                    continue
                item_doc_id = item.get("doc_id", "")
                if item_doc_id in selected_ids:
                    continue
                replaced = False
                for idx in range(len(filtered) - 1, -1, -1):
                    if not filtered[idx].get("entity_hit"):
                        selected_ids.discard(filtered[idx].get("doc_id", ""))
                        filtered[idx] = item
                        selected_ids.add(item_doc_id)
                        replaced = True
                        break
                if not replaced and len(filtered) < payload["top_k"]:
                    filtered.append(item)
                    selected_ids.add(item_doc_id)
                entity_count = sum(1 for row in filtered if row.get("entity_hit"))
                if entity_count >= min_entity:
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
    payload["context"] = RetrievedContext(passages=passages)
    return payload


def _apply_rerank(payload: Dict) -> RetrievedContext:
    context = payload["context"]
    queries = payload["queries"]
    if not queries:
        return RetrievedContext(passages=[])
    if not payload.get("use_reranker", True):
        return context
    reranker = get_reranker()
    if reranker:
        logger.info("Applying reranking...")
        return reranker.rerank(queries[0], context.passages)
    return context


def hybrid_retrieve(
    store,
    bm25,
    query: Union[str, List[str]],
    top_k: int,
    max_subqueries: int = None,
    per_query_top_k: int = None,
    use_reranker: bool = True,
    allowed_source_doc_ids: Optional[List[str]] = None,
) -> RetrievedContext:
    """融合dense与sparse分数，返回排序后的检索上下文。

    Args:
        store: 向量库实例。
        bm25: BM25检索器实例或None。
        query: 查询文本。
        top_k: 返回结果数量。

    Returns:
        RetrievedContext: 检索上下文结果。
    """
    chain = (
        RunnableLambda(
            lambda _: _prepare_payload(
                store,
                bm25,
                query,
                top_k,
                max_subqueries=max_subqueries,
                per_query_top_k=per_query_top_k,
                use_reranker=use_reranker,
                allowed_source_doc_ids=allowed_source_doc_ids,
            )
        )
        | RunnableLambda(_run_search)
        | RunnableLambda(_apply_entity_priority)
        | RunnableLambda(_apply_technical_heading_boost)
        | RunnableLambda(_build_context)
        | RunnableLambda(_apply_rerank)
    )
    return chain.invoke(None)
