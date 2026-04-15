from typing import List

from langchain_core.documents import Document

try:
    from langchain_community.retrievers import BM25Retriever
except Exception:  # pragma: no cover
    BM25Retriever = None


def tokenize(text: str) -> List[str]:
    """将文本按空格与换行做简单分词。

    Args:
        text: 输入文本。

    Returns:
        List[str]: 分词结果列表。
    """
    return [t for t in text.replace("\n", " ").split(" ") if t]


def build_bm25(chunks: List[dict]):
    """构建LangChain BM25Retriever，若依赖不可用返回None。

    Args:
        chunks: chunk列表。

    Returns:
        BM25Retriever | None: BM25检索器或None。
    """
    if BM25Retriever is None:
        return None
    docs = []
    for c in chunks:
        docs.append(
            Document(
                page_content=str(c.get("text", "") or ""),
                metadata={
                    "doc_id": str(c.get("doc_id", "") or ""),
                    "source": str(c.get("source", "") or ""),
                    "source_doc_id": str(c.get("source_doc_id", "") or ""),
                    "domain": str(c.get("domain", "") or ""),
                },
            )
        )
    if not docs:
        return None
    return BM25Retriever.from_documents(docs, preprocess_func=tokenize)
