from typing import List, Union, Dict, Optional, Tuple
import logging
import re

from langchain.retrievers import EnsembleRetriever
from langchain_core.runnables import RunnableLambda

from .schema import Passage, RetrievedContext
from .config import SETTINGS
from .reranker import get_reranker
from .model import build_embeddings

logger = logging.getLogger(__name__)

_COMPRESSION_RETRIEVER_CACHE = None

try:
    from langchain.retrievers.contextual_compression import (
        ContextualCompressionRetriever,
    )
    from langchain.retrievers.document_compressors import (
        DocumentCompressorPipeline,
        EmbeddingsFilter,
        EmbeddingsRedundantFilter,
    )
    from langchain_text_splitters import CharacterTextSplitter

    _CONTEXT_COMPRESSION_AVAILABLE = True
except Exception:
    ContextualCompressionRetriever = None
    DocumentCompressorPipeline = None
    EmbeddingsFilter = None
    EmbeddingsRedundantFilter = None
    CharacterTextSplitter = None
    _CONTEXT_COMPRESSION_AVAILABLE = False


def _get_compression_retriever(base_retriever):
    global _COMPRESSION_RETRIEVER_CACHE
    if not _CONTEXT_COMPRESSION_AVAILABLE:
        raise RuntimeError("Context compression dependencies are not available")
    cache_key = (
        id(base_retriever),
        SETTINGS.embedding_model,
        SETTINGS.context_compression_chunk_size,
        SETTINGS.context_compression_chunk_overlap,
        SETTINGS.context_compression_separator,
        SETTINGS.context_compression_similarity_threshold,
    )
    if _COMPRESSION_RETRIEVER_CACHE and _COMPRESSION_RETRIEVER_CACHE[0] == cache_key:
        return _COMPRESSION_RETRIEVER_CACHE[1]
    embeddings_model = build_embeddings(SETTINGS.embedding_model)
    splitter = CharacterTextSplitter(
        chunk_size=max(1, SETTINGS.context_compression_chunk_size),
        chunk_overlap=max(0, SETTINGS.context_compression_chunk_overlap),
        separator=SETTINGS.context_compression_separator,
    )
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings_model)
    relevant_filter = EmbeddingsFilter(
        embeddings=embeddings_model,
        similarity_threshold=SETTINGS.context_compression_similarity_threshold,
    )
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter]
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor,
        base_retriever=base_retriever,
    )
    _COMPRESSION_RETRIEVER_CACHE = (cache_key, compression_retriever)
    return compression_retriever


def _should_enable_context_compression(queries: List[str]) -> bool:
    if not SETTINGS.context_compression_enabled:
        return False
    non_empty_queries = [
        str(q or "").strip() for q in (queries or []) if str(q or "").strip()
    ]
    if not non_empty_queries:
        return False
    min_query_count = max(1, SETTINGS.context_compression_min_query_count)
    if len(non_empty_queries) >= min_query_count:
        return True
    first_query = non_empty_queries[0]
    min_query_length = max(1, SETTINGS.context_compression_min_query_length)
    return len(first_query) >= min_query_length


def _normalize_for_match(text: str) -> str:
    value = " ".join(str(text or "").strip().split()).lower()
    return re.sub(r"\s+", "", value)


def _normalize_text_for_dedup(text: str) -> str:
    value = str(text or "").strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


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


def _infer_query_domains(query: str) -> List[str]:
    if not getattr(SETTINGS, "domain_routing_enabled", False):
        return []
    q = str(query or "").strip().lower()
    if not q:
        return []
    rules = getattr(SETTINGS, "domain_routing_rules", {}) or {}
    scores: List[tuple[str, int]] = []
    for domain, keywords in rules.items():
        hit = 0
        for kw in keywords:
            if kw and kw in q:
                hit += 1
        if hit > 0:
            scores.append((domain, hit))
    if not scores:
        return []
    scores = sorted(scores, key=lambda x: (-x[1], x[0]))
    max_hit = scores[0][1]
    return [d for d, h in scores if h == max_hit]


def _prepare_payload(
    store,
    bm25,
    query: Union[str, List[str]],
    top_k: int,
    max_subqueries: int = None,
    per_query_top_k: int = None,
    use_reranker: bool = True,
    allowed_source_doc_ids: Optional[List[str]] = None,
    parent_source_doc_ids: Optional[List[str]] = None,
    parent_route_mode: str = "hard",
    parent_source_soft_boost: float = 0.0,
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
        "parent_source_doc_ids": parent_source_doc_ids or [],
        "parent_route_mode": str(parent_route_mode or "hard").strip().lower(),
        "parent_source_soft_boost": float(parent_source_soft_boost or 0.0),
        "routed_domains": _infer_query_domains(queries[0] if queries else ""),
    }


def _run_search(payload: Dict) -> Dict:
    queries = payload["queries"]
    if not queries:
        payload["merged"] = {}
        return payload
    merged: Dict[str, dict] = {}
    per_k = payload["per_query_top_k"] or max(1, payload["top_k"])
    parent_route_mode = str(payload.get("parent_route_mode", "hard") or "hard").lower()
    allowed_source_doc_ids = set(payload.get("allowed_source_doc_ids") or [])
    parent_source_doc_ids = set(payload.get("parent_source_doc_ids") or [])
    parent_source_soft_boost = float(
        payload.get("parent_source_soft_boost", 0.0) or 0.0
    )
    routed_domains = payload.get("routed_domains") or []
    domain_route_mode = str(
        getattr(SETTINGS, "domain_route_mode", "soft") or "soft"
    ).lower()
    domain_soft_boost = float(getattr(SETTINGS, "domain_soft_boost", 0.0) or 0.0)
    strict_min_hits = max(1, int(getattr(SETTINGS, "domain_strict_min_hits", 2) or 2))
    use_context_compression = _should_enable_context_compression(queries)

    def extract_source_doc_id(metadata: Dict) -> str:
        source_doc_id = str(metadata.get("source_doc_id", "") or "").strip()
        if source_doc_id:
            return source_doc_id
        doc_id = str(metadata.get("doc_id", "") or "").strip()
        if "::chunk_" in doc_id:
            return doc_id.split("::chunk_", 1)[0].strip()
        return ""

    def build_retriever(run_per_k: int, bm25_only: bool = False):
        vector_retriever = payload["store"].as_retriever(search_kwargs={"k": run_per_k})
        bm25_retriever = payload.get("bm25")
        if bm25_only:
            if bm25_retriever is None:
                return vector_retriever
            bm25_retriever.k = run_per_k
            return bm25_retriever
        if bm25_retriever is not None:
            bm25_retriever.k = run_per_k
            return EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[SETTINGS.vector_weight, SETTINGS.bm25_weight],
            )
        return vector_retriever

    def run_once(
        run_per_k: int,
        run_parent_route_mode: str,
        run_domain_route_mode: str,
        bm25_only: bool = False,
    ) -> Tuple[Dict[str, dict], int]:
        active_merged: Dict[str, dict] = {}
        total_docs_after_filter = 0
        retriever = build_retriever(run_per_k, bm25_only=bm25_only)
        for q in queries:
            active_retriever = retriever
            if use_context_compression and not bm25_only:
                try:
                    active_retriever = _get_compression_retriever(retriever)
                except Exception as exc:
                    logger.warning(
                        "Init context compression retriever failed: %s",
                        type(exc).__name__,
                    )
                    active_retriever = retriever
            docs = active_retriever.invoke(q)
            if routed_domains:
                matched_docs = []
                for d in docs:
                    md = d.metadata or {}
                    if str(md.get("domain", "") or "") in routed_domains:
                        matched_docs.append(d)
                if (
                    run_domain_route_mode == "hard"
                    and len(matched_docs) >= strict_min_hits
                ):
                    docs = matched_docs
            if run_parent_route_mode == "hard" and allowed_source_doc_ids:
                filtered_docs = []
                for d in docs:
                    metadata = d.metadata or {}
                    source_doc_id = extract_source_doc_id(metadata)
                    if source_doc_id in allowed_source_doc_ids:
                        filtered_docs.append(d)
                docs = filtered_docs

            total_docs_after_filter += len(docs)
            total = max(1, len(docs))
            for idx, doc in enumerate(docs):
                metadata = doc.metadata or {}
                doc_id = str(metadata.get("doc_id", "") or "")
                if not doc_id:
                    continue
                source_doc_id = extract_source_doc_id(metadata)
                domain = str(metadata.get("domain", "") or "").strip()
                score = float((total - idx) / total)
                if (
                    run_parent_route_mode == "soft"
                    and parent_source_soft_boost > 0.0
                    and source_doc_id
                    and source_doc_id in parent_source_doc_ids
                ):
                    score += parent_source_soft_boost
                if (
                    routed_domains
                    and run_domain_route_mode == "soft"
                    and domain_soft_boost > 0.0
                    and domain in routed_domains
                ):
                    score += domain_soft_boost
                item = {
                    "doc_id": doc_id,
                    "text": str(doc.page_content or ""),
                    "source": str(metadata.get("source", "") or ""),
                    "source_doc_id": source_doc_id,
                    "domain": domain,
                    "heading_context": str(metadata.get("heading_context", "") or ""),
                    "score": score,
                }
                entry = active_merged.get(doc_id)
                if entry is None or score > entry["score"]:
                    active_merged[doc_id] = item
        return active_merged, total_docs_after_filter

    fallback_enabled = bool(getattr(SETTINGS, "retrieval_fallback_enabled", True))
    fallback_min_results = max(
        1,
        int(
            getattr(
                SETTINGS,
                "retrieval_fallback_min_results",
                max(1, payload["top_k"]),
            )
            or max(1, payload["top_k"])
        ),
    )
    relax_hard_filters = bool(
        getattr(SETTINGS, "retrieval_fallback_relax_hard_filters", True)
    )

    merged, total_docs_after_filter = run_once(
        run_per_k=per_k,
        run_parent_route_mode=parent_route_mode,
        run_domain_route_mode=domain_route_mode,
    )
    fallback_reasons: List[str] = []

    if (
        fallback_enabled
        and relax_hard_filters
        and len(merged) < fallback_min_results
        and (
            (parent_route_mode == "hard" and total_docs_after_filter < 3)
            or (domain_route_mode == "hard" and total_docs_after_filter < 3)
        )
    ):
        relaxed_parent_mode = (
            "soft" if parent_route_mode == "hard" else parent_route_mode
        )
        relaxed_domain_mode = (
            "soft" if domain_route_mode == "hard" else domain_route_mode
        )
        relaxed_merged, _ = run_once(
            run_per_k=per_k,
            run_parent_route_mode=relaxed_parent_mode,
            run_domain_route_mode=relaxed_domain_mode,
        )
        if relaxed_merged:
            for doc_id, item in relaxed_merged.items():
                if doc_id not in merged or float(item["score"]) > float(
                    merged[doc_id]["score"]
                ):
                    merged[doc_id] = item
            fallback_reasons.append("relax_hard_filters")

    if fallback_enabled and len(merged) < fallback_min_results:
        bm25_only_k = max(
            per_k,
            int(getattr(SETTINGS, "retrieval_fallback_bm25_only_k", 20) or 20),
        )
        bm25_merged, _ = run_once(
            run_per_k=bm25_only_k,
            run_parent_route_mode="soft",
            run_domain_route_mode="soft",
            bm25_only=True,
        )
        if bm25_merged:
            for doc_id, item in bm25_merged.items():
                if doc_id not in merged:
                    merged[doc_id] = item
            fallback_reasons.append("bm25_only")

    payload["fallback_applied"] = "true" if fallback_reasons else "false"
    payload["fallback_reasons"] = ",".join(fallback_reasons)
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
        strict_filtered = {
            doc_id: item for doc_id, item in merged.items() if doc_id in entity_hit_ids
        }
        min_after_filter = 2
        if len(strict_filtered) >= min_after_filter:
            merged = strict_filtered
        else:
            payload["entity_strict_relaxed"] = "true"

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

    deduped_ranked = []
    seen_signatures = set()
    for item in ranked:
        signature = (
            _normalize_text_for_dedup(item.get("text", "")),
            str(item.get("source_doc_id", "") or "").strip().lower(),
        )
        if not signature[0] or signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        deduped_ranked.append(item)

    score_threshold = float(getattr(SETTINGS, "retrieval_score_threshold", 0.0) or 0.0)
    min_after_threshold = max(
        1,
        int(getattr(SETTINGS, "retrieval_min_results_after_threshold", 1) or 1),
    )
    if score_threshold > 0.0 and len(deduped_ranked) > min_after_threshold:
        thresholded_ranked = [
            item
            for item in deduped_ranked
            if float(item.get("score", 0.0)) >= score_threshold
        ]
        ranked = (
            thresholded_ranked
            if len(thresholded_ranked) >= min_after_threshold
            else deduped_ranked
        )
    else:
        ranked = deduped_ranked
    entity_candidates = payload.get("entity_candidates") or []
    configured_max_per_source = int(getattr(SETTINGS, "max_per_source", 0) or 0)
    if configured_max_per_source > 0:
        max_per_source = configured_max_per_source
    else:
        max_per_source = max(1, payload["top_k"] // 2)
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
        max_rerank_input = max(
            1,
            int(
                getattr(
                    SETTINGS,
                    "reranker_max_input",
                    max(1, len(context.passages)),
                )
                or max(1, len(context.passages))
            ),
        )
        rerank_input = list(context.passages[:max_rerank_input])
        reranked = reranker.rerank(queries[0], rerank_input)

        reranked_passages = list(reranked.passages)
        rerank_threshold = float(
            getattr(SETTINGS, "reranker_score_threshold", 0.0) or 0.0
        )
        rerank_min_after_threshold = max(
            1,
            int(getattr(SETTINGS, "reranker_min_results_after_threshold", 1) or 1),
        )
        if (
            rerank_threshold > 0.0
            and len(reranked_passages) > rerank_min_after_threshold
        ):
            filtered_reranked = [
                p
                for p in reranked_passages
                if float(getattr(p, "score", 0.0)) >= rerank_threshold
            ]
            if len(filtered_reranked) >= rerank_min_after_threshold:
                reranked_passages = filtered_reranked

        target_k = max(1, int(payload.get("top_k", len(context.passages)) or 1))
        if len(reranked_passages) >= target_k:
            return RetrievedContext(passages=reranked_passages[:target_k])
        selected = list(reranked_passages)
        selected_ids = {p.doc_id for p in selected}
        for p in context.passages:
            if p.doc_id in selected_ids:
                continue
            selected.append(p)
            selected_ids.add(p.doc_id)
            if len(selected) >= target_k:
                break
        return RetrievedContext(passages=selected[:target_k])
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
    parent_source_doc_ids: Optional[List[str]] = None,
    parent_route_mode: str = "hard",
    parent_source_soft_boost: float = 0.0,
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
                parent_source_doc_ids=parent_source_doc_ids,
                parent_route_mode=parent_route_mode,
                parent_source_soft_boost=parent_source_soft_boost,
            )
        )
        | RunnableLambda(_run_search)
        | RunnableLambda(_apply_entity_priority)
        | RunnableLambda(_apply_technical_heading_boost)
        | RunnableLambda(_build_context)
        | RunnableLambda(_apply_rerank)
    )
    return chain.invoke(None)
