import os
from typing import List, Optional

from langchain_chroma import Chroma
from .config import SETTINGS
from .model import build_embeddings


class ChromaStore:
    def __init__(
        self,
        index_dir: str,
        collection_name: str = "rag_chunks",
        embedding_model: Optional[str] = None,
    ) -> None:
        self.index_dir = index_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model or SETTINGS.embedding_model
        self.persist_dir = os.path.join(index_dir, "chroma")
        self._embeddings = build_embeddings(self.embedding_model)
        self._store: Optional[Chroma] = None

    @property
    def store(self) -> Chroma:
        if self._store is None:
            raise RuntimeError(
                "Chroma store is not initialized. Call build_or_load() first."
            )
        return self._store

    def build_or_load(self, chunks: List[dict]) -> Chroma:
        """构建或加载Chroma向量库。"""
        os.makedirs(self.index_dir, exist_ok=True)
        if os.path.isdir(self.persist_dir) and os.listdir(self.persist_dir):
            self._store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self._embeddings,
                collection_name=self.collection_name,
            )
            return self._store

        texts = [c["text"] for c in chunks]
        metadatas = [
            {
                "doc_id": c["doc_id"],
                "source": c["source"],
                "source_doc_id": c.get("source_doc_id", ""),
                "chunk_index": c.get("chunk_index", -1),
                "heading_context": c.get("heading_context", ""),
            }
            for c in chunks
        ]
        self._store = Chroma.from_texts(
            texts=texts,
            embedding=self._embeddings,
            metadatas=metadatas,
            persist_directory=self.persist_dir,
            collection_name=self.collection_name,
        )
        self.persist()
        return self._store

    def add_texts(self, texts: List[str], metadatas: List[dict]) -> None:
        self.store.add_texts(texts=texts, metadatas=metadatas)
        self.persist()

    def persist(self) -> None:
        if hasattr(self.store, "persist"):
            self.store.persist()
