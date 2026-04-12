from typing import Dict, List, Set, Tuple, Iterable, Optional
import time
import logging

from langchain_core.runnables import RunnableLambda, RunnableParallel

from .config import SETTINGS
from .schema import Answer, QueryRequest, Passage, RetrievedContext
from .loader import (
    load_documents,
    load_chunks_json,
    load_doc_source_map,
    load_chunk_to_kg,
)
from .indexer import build_or_load_chroma
from .chunking import TextChunker
from .bm25 import build_bm25
from .retriever import hybrid_retrieve
from .parent_retriever import (
    build_parent_document_retriever,
    retrieve_parent_source_doc_ids,
    retrieve_parent_documents,
)
from .generator import generate_answer, stream_answer
from .kg_interface import query_knowledge_graph
from .decomposer import DecompositionQueryRetriever, decompose_question
from .model import build_chat_llm


logger = logging.getLogger(__name__)


class RagPipeline:
    def __init__(self):
        """初始化RAG流水线，加载文档、索引与检索器。

        Args:
            None

        Returns:
            None
        """
        self.docs = load_documents(SETTINGS.docs_dir)
        self.chunks = load_chunks_json(SETTINGS.data_dir)
        if not self.chunks:
            self.chunks = TextChunker.from_settings(SETTINGS).split_documents(self.docs)
        self.doc_source_map = load_doc_source_map(SETTINGS.data_dir)
        self.chunk_kg_map = load_chunk_to_kg(SETTINGS.data_dir)
        self._validate_chunk_kg_alignment(self.chunks, self.chunk_kg_map)
        self.chunk_text_map = {c["chunk_id"]: c.get("text", "") for c in self.chunks}
        self.chunks_by_source_doc_id = self._build_chunks_by_source_doc_id(self.chunks)
        self.chunks_by_chunk_id = {c.get("chunk_id", ""): c for c in self.chunks}
        self.chunk_pos_by_chunk_id = self._build_chunk_pos_map(
            self.chunks_by_source_doc_id
        )
        self.store = build_or_load_chroma(self.chunks, SETTINGS.index_dir)
        self.bm25 = build_bm25(self.chunks)
        self.parent_retriever = build_parent_document_retriever(
            self.chunks,
            embedding_model=SETTINGS.embedding_model,
            search_k=max(1, SETTINGS.parent_retriever_k),
        )
        self.decomposer = self._build_decomposer()
        self._chain = self._build_chain()

    @staticmethod
    def _validate_chunk_kg_alignment(
        chunks: List[dict], chunk_kg_map: Dict[str, dict]
    ) -> None:
        if not chunks or not chunk_kg_map:
            return
        chunk_ids = {
            str(item.get("chunk_id", "") or "").strip()
            for item in chunks
            if str(item.get("chunk_id", "") or "").strip()
        }
        kg_chunk_ids = {
            str(key or "").strip() for key in chunk_kg_map.keys() if str(key or "").strip()
        }
        if not chunk_ids or not kg_chunk_ids:
            return
        overlap = chunk_ids & kg_chunk_ids
        overlap_ratio = len(overlap) / max(1, len(kg_chunk_ids))
        minimum_overlap = min(len(kg_chunk_ids), max(5, int(len(kg_chunk_ids) * 0.2)))
        if len(overlap) < minimum_overlap:
            raise RuntimeError(
                "RAG chunks 与 chunk_to_kg 映射不一致，已拒绝启动。"
                f" chunks={len(chunk_ids)}"
                f" chunk_to_kg={len(kg_chunk_ids)}"
                f" overlap={len(overlap)}"
                f" overlap_ratio={overlap_ratio:.3f}"
                f" data_dir={SETTINGS.data_dir}"
            )

    def _build_decomposer(self) -> Optional[DecompositionQueryRetriever]:
        if SETTINGS.decomposer_method != "llm":
            return None
        if SETTINGS.llm_provider == "none":
            return None
        try:
            llm = build_chat_llm(
                temperature=SETTINGS.decompose_llm_temperature,
                max_tokens=SETTINGS.decompose_llm_max_tokens,
            )
            retriever = self.store.as_retriever(
                search_kwargs={"k": max(1, SETTINGS.decompose_per_query_top_k)}
            )
            return DecompositionQueryRetriever.from_llm(
                retriever=retriever,
                llm=llm,
            )
        except Exception as exc:
            logger.info("Init decomposition retriever failed: %s", type(exc).__name__)
            return None

    @staticmethod
    def _build_chunks_by_source_doc_id(chunks: List[dict]) -> Dict[str, List[dict]]:
        mapping: Dict[str, List[dict]] = {}
        for chunk in chunks:
            source_doc_id = str(chunk.get("source_doc_id", "") or "").strip()
            if not source_doc_id:
                chunk_id = str(chunk.get("chunk_id", "") or "").strip()
                if "::chunk_" in chunk_id:
                    source_doc_id = chunk_id.split("::chunk_", 1)[0].strip()
            if not source_doc_id:
                continue
            mapping.setdefault(source_doc_id, []).append(chunk)
        for source_doc_id, items in mapping.items():
            mapping[source_doc_id] = sorted(
                items,
                key=lambda x: (
                    x.get("chunk_index") is None,
                    x.get("chunk_index") if x.get("chunk_index") is not None else 0,
                ),
            )
        return mapping

    @staticmethod
    def _build_chunk_pos_map(
        chunks_by_source_doc_id: Dict[str, List[dict]],
    ) -> Dict[str, Tuple[str, int]]:
        pos_map: Dict[str, Tuple[str, int]] = {}
        for source_doc_id, items in chunks_by_source_doc_id.items():
            for idx, chunk in enumerate(items):
                chunk_id = str(chunk.get("chunk_id", "") or "").strip()
                if not chunk_id:
                    continue
                pos_map[chunk_id] = (source_doc_id, idx)
        return pos_map

    def _augment_retrieved_with_neighbor_chunks(
        self, retrieved_chunks: List[Passage], question: str = ""
    ) -> List[Passage]:
        if not retrieved_chunks:
            return []
        if not SETTINGS.neighbor_context_enabled:
            return list(retrieved_chunks)

        expanded: List[Passage] = []
        seen_doc_ids: Set[str] = set()

        for passage in retrieved_chunks:
            if passage.doc_id in seen_doc_ids:
                continue
            expanded.append(passage)
            seen_doc_ids.add(passage.doc_id)

        max_extra = max(0, SETTINGS.neighbor_context_max_chunks)
        if max_extra == 0:
            return expanded

        seed_limit = max(1, SETTINGS.neighbor_context_seed_limit)
        window = max(1, SETTINGS.neighbor_context_window)
        if self._is_parameter_question(question):
            window = max(window, max(1, SETTINGS.neighbor_context_param_window))
        extra_added = 0
        for passage in expanded[:seed_limit]:
            chunk_pos = self.chunk_pos_by_chunk_id.get(passage.doc_id)
            if not chunk_pos:
                continue
            source_doc_id, pos = chunk_pos
            seed_source = str(passage.source or "").strip()
            if not seed_source:
                continue
            source_chunks = self.chunks_by_source_doc_id.get(source_doc_id, [])
            if not source_chunks:
                continue

            for step in range(1, window + 1):
                neighbor_pos = pos + step
                if neighbor_pos >= len(source_chunks):
                    break
                neighbor_chunk = source_chunks[neighbor_pos]
                neighbor_doc_id = str(neighbor_chunk.get("chunk_id", "") or "").strip()
                if not neighbor_doc_id or neighbor_doc_id in seen_doc_ids:
                    continue
                neighbor_source = str(neighbor_chunk.get("source", "") or "").strip()
                if not neighbor_source or neighbor_source != seed_source:
                    continue
                neighbor_text = str(neighbor_chunk.get("text", "") or "").strip()
                if not neighbor_text:
                    continue

                expanded.append(
                    Passage(
                        doc_id=neighbor_doc_id,
                        text=neighbor_text,
                        source=str(neighbor_chunk.get("source", "") or ""),
                        score=max(0.0, float(passage.score) - 0.01 * step),
                    )
                )
                seen_doc_ids.add(neighbor_doc_id)
                extra_added += 1
                if extra_added >= max_extra:
                    return expanded
        return expanded

    @staticmethod
    def _is_parameter_question(question: str) -> bool:
        q = str(question or "").strip().lower()
        if not q:
            return False
        keywords = (
            "参数",
            "规格",
            "配置",
            "电源",
            "电压",
            "电流",
            "功率",
            "防护",
            "等级",
            "输入",
            "输出",
        )
        return any(k in q for k in keywords)

    def _augment_context_with_kg_doc_chunks(
        self, context, kg_triplets: List[Dict[str, str]], top_k: int
    ):
        if not kg_triplets:
            return context, []
        existing_doc_ids: Set[str] = {p.doc_id for p in context.passages}
        kg_doc_ids: List[str] = []
        for triplet in kg_triplets:
            for key in ("head_doc_id", "tail_doc_id", "rel_doc_id"):
                doc_id = str(triplet.get(key, "") or "").strip()
                if doc_id and doc_id not in kg_doc_ids:
                    kg_doc_ids.append(doc_id)
        extra_passages: List[Passage] = []
        for doc_id in kg_doc_ids:
            chunks = self.chunks_by_source_doc_id.get(doc_id)
            if not chunks:
                chunk = self.chunks_by_chunk_id.get(doc_id)
                chunks = [chunk] if chunk else []
            for chunk in chunks:
                chunk_id = str(chunk.get("chunk_id", "") or "").strip()
                if not chunk_id or chunk_id in existing_doc_ids:
                    continue
                text = str(chunk.get("text", "") or "").strip()
                if not text:
                    continue
                extra_passages.append(
                    Passage(
                        doc_id=chunk_id,
                        text=text,
                        source=str(chunk.get("source", "") or ""),
                        score=0.0,
                    )
                )
                existing_doc_ids.add(chunk_id)
                if len(extra_passages) >= top_k:
                    break
            if len(extra_passages) >= top_k:
                break
        if not extra_passages:
            return context, []
        return type(context)(passages=context.passages + extra_passages), extra_passages

    def _build_chain(self):
        prepare = RunnableLambda(self._prepare_request)
        retrieve = RunnableLambda(self._run_retrieve)
        fetch_kg = RunnableLambda(self._run_kg)
        parallel = RunnableParallel(retrieve=retrieve, fetch_kg=fetch_kg)
        merge_prompt = RunnableLambda(self._merge_prompt)
        generate = RunnableLambda(self._run_generate)
        finalize = RunnableLambda(self._build_answer)
        return (
            prepare
            | parallel
            | RunnableLambda(self._merge_parallel)
            | merge_prompt
            | generate
            | finalize
        )

    def _merge_prompt_passages(
        self,
        retrieved_chunks: List[Passage],
        kg_chunks: List[Passage],
        top_k: int,
        parent_chunks: Optional[List[Passage]] = None,
    ) -> List[Passage]:
        merged: List[Passage] = []
        seen_doc_ids: Set[str] = set()

        def append_unique(items: List[Passage], limit: int = None) -> None:
            added = 0
            for item in items:
                if item.doc_id in seen_doc_ids:
                    continue
                seen_doc_ids.add(item.doc_id)
                merged.append(item)
                added += 1
                if limit is not None and added >= limit:
                    return

        kg_quota = 0
        if kg_chunks:
            kg_quota = min(len(kg_chunks), max(1, top_k // 2))
        parent_quota = 0
        if parent_chunks:
            parent_quota = min(
                len(parent_chunks),
                max(1, min(SETTINGS.parent_prompt_top_k, top_k // 3)),
            )
        hybrid_quota = max(0, top_k - kg_quota - parent_quota)

        append_unique(retrieved_chunks, limit=hybrid_quota)
        if parent_chunks:
            append_unique(parent_chunks, limit=parent_quota)
        append_unique(kg_chunks, limit=kg_quota)
        if len(merged) < top_k:
            append_unique(retrieved_chunks)
        if parent_chunks and len(merged) < top_k:
            append_unique(parent_chunks)
        if len(merged) < top_k:
            append_unique(kg_chunks)
        return merged[:top_k]

    def _build_parent_prompt_chunks(self, payload: Dict) -> List[Passage]:
        if not payload.get("use_parent_retriever", True):
            return []
        mode = str(getattr(SETTINGS, "parent_prompt_mode", "route") or "route").lower()
        if mode != "hybrid":
            return []
        parent_docs = retrieve_parent_documents(
            self.parent_retriever,
            payload["req"].question,
            top_k=max(1, SETTINGS.parent_prompt_top_k),
        )
        passages: List[Passage] = []
        for idx, doc in enumerate(parent_docs):
            md = doc.metadata or {}
            source_doc_id = str(md.get("source_doc_id", "") or "").strip()
            if not source_doc_id:
                continue
            text = str(doc.page_content or "").strip()
            if not text:
                continue
            passages.append(
                Passage(
                    doc_id=f"{source_doc_id}::parent",
                    text=text,
                    source=str(md.get("source", "") or ""),
                    score=max(0.0, 0.8 - idx * 0.05),
                )
            )
        return passages

    def _merge_prompt(self, payload: Dict) -> Dict:
        context = payload["context"]
        kg_triplets = payload.get("kg_triplets") or []
        top_k = payload["top_k"]
        context, kg_chunks = self._augment_context_with_kg_doc_chunks(
            context, kg_triplets, top_k=top_k
        )
        retrieved_chunks = payload.get("retrieved_chunks") or list(context.passages)
        retrieved_chunks = self._augment_retrieved_with_neighbor_chunks(
            retrieved_chunks,
            question=payload["req"].question,
        )
        parent_chunks = self._build_parent_prompt_chunks(payload)
        merged_passages = self._merge_prompt_passages(
            retrieved_chunks=retrieved_chunks,
            kg_chunks=kg_chunks,
            top_k=top_k,
            parent_chunks=parent_chunks,
        )
        payload["retrieved_chunks"] = retrieved_chunks
        payload["kg_chunks"] = kg_chunks
        payload["parent_chunks"] = parent_chunks
        payload["prompt_passages"] = merged_passages
        payload["context"] = RetrievedContext(passages=merged_passages)
        return payload

    def _prepare_request(self, req: QueryRequest) -> Dict:
        top_k = req.top_k or SETTINGS.top_k
        retrieval_optimization_enabled = (
            req.enable_retrieval_optimization
            if req.enable_retrieval_optimization is not None
            else True
        )
        default_decompose = SETTINGS.use_query_decomposition
        use_decompose = (
            req.enable_decompose
            if req.enable_decompose is not None
            else default_decompose
        ) and retrieval_optimization_enabled
        use_reranker = SETTINGS.use_reranker and retrieval_optimization_enabled
        questions = [req.question]
        if use_decompose:
            if self.decomposer is not None:
                sub_questions = self.decomposer.generate_queries(req.question)
            else:
                sub_questions = decompose_question(req.question)
            if sub_questions:
                questions.extend(sub_questions)
        return {
            "req": req,
            "top_k": top_k,
            "hybrid_top_k": SETTINGS.hybrid_top_k,
            "use_reranker": use_reranker,
            "use_parent_retriever": (
                req.enable_parent_retriever
                if req.enable_parent_retriever is not None
                else SETTINGS.use_parent_retriever
            ),
            "use_decompose": use_decompose,
            "enable_retrieval_optimization": retrieval_optimization_enabled,
            "questions": questions,
            "timing": {"t0": time.perf_counter()},
        }

    def _run_retrieve(self, payload: Dict) -> Dict:
        parent_source_doc_ids: List[str] = []
        if payload.get("use_parent_retriever", True):
            parent_source_doc_ids = retrieve_parent_source_doc_ids(
                self.parent_retriever,
                payload["questions"][0],
                top_k=max(1, SETTINGS.parent_retriever_k),
            )
        route_mode = str(
            getattr(SETTINGS, "parent_retriever_route_mode", "soft") or "soft"
        ).lower()
        hard_allowed_source_doc_ids = (
            parent_source_doc_ids if route_mode == "hard" else []
        )
        context = hybrid_retrieve(
            self.store,
            self.bm25,
            payload["questions"],
            payload["hybrid_top_k"],
            max_subqueries=SETTINGS.decompose_max_subqueries,
            per_query_top_k=SETTINGS.decompose_per_query_top_k,
            use_reranker=payload["use_reranker"],
            allowed_source_doc_ids=hard_allowed_source_doc_ids,
            parent_source_doc_ids=parent_source_doc_ids,
            parent_route_mode=route_mode,
            parent_source_soft_boost=getattr(SETTINGS, "parent_source_soft_boost", 0.0),
        )
        payload["context"] = context
        payload["retrieved_chunks"] = list(context.passages)
        payload["parent_source_doc_ids"] = parent_source_doc_ids
        payload["parent_route_mode"] = route_mode
        payload["timing"]["t1"] = time.perf_counter()
        return payload

    def _run_kg(self, payload: Dict) -> Dict:
        req = payload["req"]
        top_k = payload["top_k"]
        kg_triplets = (
            query_knowledge_graph(
                req.question,
                self.doc_source_map,
                self.chunk_kg_map,
                self.chunk_text_map,
                top_k=top_k,
            )
            if req.use_kg
            else []
        )
        payload["kg_triplets"] = kg_triplets
        payload["timing"]["t2"] = time.perf_counter()
        return payload

    def _merge_parallel(self, results: Dict[str, Dict]) -> Dict:
        base = results.get("retrieve") or {}
        kg_payload = results.get("fetch_kg") or {}
        if "kg_triplets" in kg_payload:
            base["kg_triplets"] = kg_payload["kg_triplets"]
        if "timing" in kg_payload and "t2" in kg_payload["timing"]:
            base.setdefault("timing", {})["t2"] = kg_payload["timing"]["t2"]
        return base

    def _run_generate(self, payload: Dict) -> Dict:
        req = payload["req"]
        context = payload["context"]
        kg_triplets = payload["kg_triplets"]
        answer_text = generate_answer(
            req.question,
            context.passages,
            kg_triplets,
            use_llm=req.use_llm,
            use_history=req.use_history,
            session_id=req.session_id or "default",
            llm_provider=req.llm_provider,
            llm_model=req.llm_model,
            llm_api_key=req.llm_api_key,
            llm_api_base=req.llm_api_base,
        )
        payload["answer_text"] = answer_text
        payload["timing"]["t3"] = time.perf_counter()
        return payload

    def _build_answer(self, payload: Dict) -> Answer:
        req = payload["req"]
        top_k = payload["top_k"]
        hybrid_top_k = payload["hybrid_top_k"]
        use_decompose = payload["use_decompose"]
        context = payload["context"]
        retrieved_chunks = payload.get("retrieved_chunks") or []
        kg_chunks = payload.get("kg_chunks") or []
        parent_chunks = payload.get("parent_chunks") or []
        kg_triplets = payload["kg_triplets"]
        answer_text = payload["answer_text"]
        timing = payload["timing"]
        print(
            f"[timing] retrieve={timing['t1'] - timing['t0']:.3f}s "
            f"kg={timing['t2'] - timing['t1']:.3f}s "
            f"generate={timing['t3'] - timing['t2']:.3f}s "
            f"total={timing['t3'] - timing['t0']:.3f}s"
        )
        return Answer(
            question=req.question,
            answer=answer_text,
            citations=context.passages,
            retrieved_chunks=retrieved_chunks,
            kg_chunks=kg_chunks,
            kg_triplets=kg_triplets,
            meta={
                "retriever": "hybrid",
                "top_k": str(top_k),
                "hybrid_top_k": str(hybrid_top_k),
                "decompose": str(use_decompose),
                "retrieval_optimization": str(
                    payload.get("enable_retrieval_optimization", True)
                ),
                "reranker": str(payload.get("use_reranker", False)),
                "parent_retriever": str(payload.get("use_parent_retriever", True)),
                "parent_route_mode": str(payload.get("parent_route_mode", "soft")),
                "parent_prompt_mode": str(
                    getattr(SETTINGS, "parent_prompt_mode", "route")
                ),
                "parent_chunks": str(len(parent_chunks)),
                "parent_chunk_doc_ids": ",".join(
                    [p.doc_id for p in parent_chunks if getattr(p, "doc_id", "")]
                ),
                "parent_source_doc_ids": ",".join(
                    payload.get("parent_source_doc_ids") or []
                ),
                "llm": str(req.use_llm),
            },
        )

    def query(self, req: QueryRequest) -> Answer:
        """执行一次问答检索与生成，返回答案与引用。

        Args:
            req: 查询请求参数。

        Returns:
            Answer: 生成的答案对象。
        """
        return self._chain.invoke(req)

    def stream_query(self, req: QueryRequest):
        stream, _ = self.stream_query_with_payload(req)
        return stream

    def stream_query_with_payload(
        self, req: QueryRequest
    ) -> Tuple[Iterable[str], Dict]:
        payload = self._prepare_request(req)
        payload = self._run_retrieve(payload)
        payload = self._run_kg(payload)
        payload = self._merge_prompt(payload)
        stream = stream_answer(
            req.question,
            payload["context"].passages,
            payload.get("kg_triplets") or [],
            use_llm=req.use_llm,
            use_history=req.use_history,
            session_id=req.session_id or "default",
            llm_provider=req.llm_provider,
            llm_model=req.llm_model,
            llm_api_key=req.llm_api_key,
            llm_api_base=req.llm_api_base,
        )
        return stream, payload

    def export_answer(self, answer: Answer) -> Dict:
        """导出答案为可序列化的字典。

        Args:
            answer: 答案对象。

        Returns:
            Dict: 可序列化的答案字典。
        """
        return answer.model_dump()
