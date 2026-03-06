from typing import List, Dict

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover
    BM25Okapi = None


def tokenize(text: str) -> List[str]:
    return [t for t in text.replace("\n", " ").split(" ") if t]


def build_bm25(chunks: List[dict]):
    if BM25Okapi is None:
        return None
    corpus = [tokenize(c["text"]) for c in chunks]
    return BM25Okapi(corpus)


def get_bm25_scores(bm25, chunks: List[dict], query: str, top_k: int) -> List[Dict]:
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
