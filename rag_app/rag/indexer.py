import os
from typing import List

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from .config import SETTINGS


def split_documents(docs: List[dict]) -> List[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    chunks: List[dict] = []
    for d in docs:
        for i, chunk in enumerate(splitter.split_text(d["text"])):
            chunks.append(
                {
                    "doc_id": f"{d['id']}::chunk_{i}",
                    "text": chunk,
                    "source": d["source"],
                }
            )
    return chunks


def build_or_load_faiss(chunks: List[dict], index_dir: str) -> FAISS:
    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, "faiss_index")
    embeddings = HuggingFaceEmbeddings(model_name=SETTINGS.embedding_model)
    if os.path.isdir(index_path):
        return FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )
    texts = [c["text"] for c in chunks]
    metadatas = [{"doc_id": c["doc_id"], "source": c["source"]} for c in chunks]
    store = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    store.save_local(index_path)
    return store


def get_dense_scores(store: FAISS, query: str, top_k: int) -> List[dict]:
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
