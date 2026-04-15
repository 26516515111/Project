import os
from typing import Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .model import build_embeddings


_PARENT_RETRIEVER_CACHE: Optional[Tuple[str, ParentDocumentRetriever]] = None


def _group_parent_documents(chunks: List[dict]) -> List[Document]:
    grouped: Dict[str, List[str]] = {}
    source_by_doc: Dict[str, str] = {}
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

    docs: List[Document] = []
    for source_doc_id, parts in grouped.items():
        docs.append(
            Document(
                page_content="\n\n".join(parts),
                metadata={
                    "source_doc_id": source_doc_id,
                    "source": source_by_doc.get(source_doc_id, ""),
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
    source_doc_ids: List[str] = []
    for doc in docs:
        source_doc_id = str((doc.metadata or {}).get("source_doc_id", "") or "").strip()
        if not source_doc_id or source_doc_id in source_doc_ids:
            continue
        source_doc_ids.append(source_doc_id)
        if len(source_doc_ids) >= max(1, int(top_k)):
            break
    return source_doc_ids


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
