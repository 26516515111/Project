from typing import List, Dict

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover
    BM25Okapi = None


def tokenize(text: str) -> List[str]:
    """将文本按空格与换行做简单分词。

    Args:
        text: 输入文本。

    Returns:
        List[str]: 分词结果列表。
    """
    return [t for t in text.replace("\n", " ").split(" ") if t]


def build_bm25(chunks: List[dict]):
    """构建BM25索引，若依赖不可用返回None。

    Args:
        chunks: chunk列表。

    Returns:
        BM25Okapi | None: BM25实例或None。
    """
    if BM25Okapi is None:
        return None
    corpus = [tokenize(c["text"]) for c in chunks]
    return BM25Okapi(corpus)


def get_bm25_scores(bm25, chunks: List[dict], query: str, top_k: int) -> List[Dict]:
    """计算BM25得分并返回Top-K结果。

    Args:
        bm25: BM25检索器实例或None。
        chunks: chunk列表。
        query: 查询文本。
        top_k: 返回结果数量。

    Returns:
        List[Dict]: 带得分的检索结果列表。
    """
    if bm25 is None:
        # Simple overlap scorer
        q_tokens = set(tokenize(query))
        scored = []
        for c in chunks:
            c_tokens = set(tokenize(c["text"]))
            overlap = len(q_tokens.intersection(c_tokens))
            scored.append({**c, "score": float(overlap)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
    scores = bm25.get_scores(tokenize(query))
    items = []
    for idx, s in enumerate(scores):
        items.append({**chunks[idx], "score": float(s)})
    items.sort(key=lambda x: x["score"], reverse=True)
    return items[:top_k]
