from typing import Dict
import time

from .config import SETTINGS
from .schema import Answer, QueryRequest
from .loader import load_documents, load_chunks_json, load_doc_source_map
from .indexer import split_documents, build_or_load_chroma
from .bm25 import build_bm25
from .retriever import hybrid_retrieve
from .generator import generate_answer
from .kg_interface import query_knowledge_graph


class RagPipeline:
    def __init__(self):
        """初始化RAG流水线，加载文档、索引与检索器。

        Args:
            None

        Returns:
            None
        """
        self.docs = load_documents(SETTINGS.docs_dir)
        self.chunks = load_chunks_json(SETTINGS.docs_dir)
        if not self.chunks:
            self.chunks = split_documents(self.docs)
        self.doc_source_map = load_doc_source_map(SETTINGS.docs_dir)
        self.store = build_or_load_chroma(self.chunks, SETTINGS.index_dir)
        self.bm25 = build_bm25(self.chunks)

    def query(self, req: QueryRequest) -> Answer:
        """执行一次问答检索与生成，返回答案与引用。

        Args:
            req: 查询请求参数。

        Returns:
            Answer: 生成的答案对象。
        """
        top_k = req.top_k or SETTINGS.top_k
        t0 = time.perf_counter()
        context = hybrid_retrieve(
            self.store, self.bm25, self.chunks, req.question, top_k
        )
        t1 = time.perf_counter()
        kg_triplets = (
            query_knowledge_graph(req.question, self.doc_source_map)
            if req.use_kg
            else []
        )
        t2 = time.perf_counter()
        answer_text = generate_answer(
            req.question, context.passages, kg_triplets, use_llm=True
        )
        t3 = time.perf_counter()
        print(
            f"[timing] retrieve={t1 - t0:.3f}s kg={t2 - t1:.3f}s generate={t3 - t2:.3f}s total={t3 - t0:.3f}s"
        )
        return Answer(
            question=req.question,
            answer=answer_text,
            citations=context.passages,
            kg_triplets=kg_triplets,
            meta={"retriever": "hybrid", "top_k": str(top_k)},
        )

    def export_answer(self, answer: Answer) -> Dict:
        """导出答案为可序列化的字典。

        Args:
            answer: 答案对象。

        Returns:
            Dict: 可序列化的答案字典。
        """
        return answer.model_dump()
