import os
from typing import Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .model import build_embeddings
from .config import SETTINGS


_PARENT_RETRIEVER_CACHE: Optional[Tuple[str, ParentDocumentRetriever]] = None


def _group_parent_documents(chunks: List[dict]) -> List[Document]:
    grouped: Dict[str, List[str]] = {}
    source_by_doc: Dict[str, str] = {}
    domain_by_doc: Dict[str, str] = {}
    for chunk in chunks:
        source_doc_id = str(chunk.get("source_doc_id", "") or "").strip()
        if not source_doc_id:
            chunk_id = str(chunk.get("chunk_id", "") or "").strip()
            if "::chunk_" in chunk_id:
                source_doc_id = chunk_id.split("::chunk_", 1)[0].strip()
        if not source_doc_id:
            source = str(chunk.get("source", "") or "").strip()
            if source.lower().endswith(".md"):
                source_doc_id = source[:-3]
        text = str(chunk.get("text", "") or "").strip()
        if not source_doc_id or not text:
            continue
        grouped.setdefault(source_doc_id, []).append(text)
        source_by_doc[source_doc_id] = str(chunk.get("source", "") or "").strip()
        domain = str(chunk.get("domain", "") or "").strip()
        if domain and source_doc_id not in domain_by_doc:
            domain_by_doc[source_doc_id] = domain

    docs: List[Document] = []
    for source_doc_id, parts in grouped.items():
        docs.append(
            Document(
                page_content="\n\n".join(parts),
                metadata={
                    "source_doc_id": source_doc_id,
                    "source": source_by_doc.get(source_doc_id, ""),
                    "domain": domain_by_doc.get(source_doc_id, ""),
                },
            )
        )
    return docs


def build_parent_document_retriever(
    chunks: List[dict],
    embedding_model: Optional[str] = None,
    search_k: int = 4,
    persist_dir: Optional[str] = None,
) -> Optional[ParentDocumentRetriever]:
    global _PARENT_RETRIEVER_CACHE
    parent_docs = _group_parent_documents(chunks)
    if not parent_docs:
        return None
    cache_key = (
        f"{len(parent_docs)}::{len(chunks)}::{embedding_model or ''}::{int(search_k)}"
    )
    if _PARENT_RETRIEVER_CACHE and _PARENT_RETRIEVER_CACHE[0] == cache_key:
        return _PARENT_RETRIEVER_CACHE[1]

    chroma_kwargs = {
        "collection_name": "rag_parent_child_runtime",
        "embedding_function": build_embeddings(embedding_model),
    }
    if persist_dir:
        os.makedirs(persist_dir, exist_ok=True)
        chroma_kwargs["persist_directory"] = persist_dir
    child_store = Chroma(**chroma_kwargs)
    docstore = InMemoryStore()
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=80,
    )
    retriever = ParentDocumentRetriever(
        vectorstore=child_store,
        docstore=docstore,
        parent_splitter=parent_splitter,
        child_splitter=child_splitter,
        search_kwargs={"k": max(1, int(search_k))},
    )
    retriever.add_documents(parent_docs)
    if hasattr(child_store, "persist"):
        child_store.persist()
    _PARENT_RETRIEVER_CACHE = (cache_key, retriever)
    return retriever


def retrieve_parent_source_doc_ids(
    retriever: Optional[ParentDocumentRetriever],
    question: str,
    top_k: int,
) -> List[str]:
    if retriever is None:
        return []
    docs = retriever.invoke(question)
    routed_domains = _infer_query_domains(question)
    route_mode = str(getattr(SETTINGS, "domain_route_mode", "soft") or "soft").lower()
    strict_min_hits = max(1, int(getattr(SETTINGS, "domain_strict_min_hits", 2) or 2))
    if routed_domains:
        matched = [
            doc
            for doc in docs
            if str((doc.metadata or {}).get("domain", "") or "") in routed_domains
        ]
        if route_mode == "hard" and len(matched) >= strict_min_hits:
            docs = matched
        elif route_mode == "soft":
            docs = matched + [doc for doc in docs if doc not in matched]
    source_doc_ids: List[str] = []
    for doc in docs:
        source_doc_id = str((doc.metadata or {}).get("source_doc_id", "") or "").strip()
        if not source_doc_id or source_doc_id in source_doc_ids:
            continue
        source_doc_ids.append(source_doc_id)
        if len(source_doc_ids) >= max(1, int(top_k)):
            break
    return source_doc_ids


def _infer_query_domains(question: str) -> List[str]:
    if not getattr(SETTINGS, "domain_routing_enabled", False):
        return []
    q = str(question or "").strip().lower()
    if not q:
        return []
    rules = getattr(SETTINGS, "domain_routing_rules", {}) or {}
    scores: List[Tuple[str, int]] = []
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
    return [domain for domain, hit in scores if hit == max_hit]


def retrieve_parent_documents(
    retriever: Optional[ParentDocumentRetriever],
    question: str,
    top_k: int,
) -> List[Document]:
    if retriever is None:
        return []
    docs = retriever.invoke(question)
    if not docs:
        return []
    return docs[: max(1, int(top_k))]
