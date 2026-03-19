import json
import logging
import re
from typing import List

from .config import SETTINGS

logger = logging.getLogger(__name__)

try:
    from langchain_community.chat_models import ChatOllama

    OLLAMA_AVAILABLE = True
except Exception:
    OLLAMA_AVAILABLE = False


def _normalize(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _heuristic_decompose(question: str) -> List[str]:
    text = _normalize(question)
    if not text:
        return []

    parts: List[str] = []
    for seg in re.split(r"[。！？?!；;]", text):
        seg = _normalize(seg)
        if not seg:
            continue
        subparts = re.split(r"(?:以及|并且|并|同时|而且|还有|或者|或|及|和)", seg)
        for sub in subparts:
            sub = _normalize(sub)
            if sub:
                parts.append(sub)

    min_len = SETTINGS.decompose_min_length
    max_num = SETTINGS.decompose_max_subquestions
    filtered = [p for p in parts if len(p) >= min_len and p != text]
    filtered = _dedupe(filtered)
    return filtered[:max_num]


def _llm_decompose(question: str) -> List[str]:
    if not OLLAMA_AVAILABLE or SETTINGS.llm_provider != "ollama":
        return []
    prompt = (
        "请将用户问题拆解为最多{n}个子问题，便于检索。"
        "要求：1) 子问题要简短可检索；2) 不能重复；3) 只输出JSON数组。\n"
        "问题：{q}"
    )
    llm = ChatOllama(
        model=SETTINGS.llm_model,
        base_url=SETTINGS.ollama_base_url,
        temperature=SETTINGS.decompose_llm_temperature,
        model_kwargs={"num_predict": SETTINGS.decompose_llm_max_tokens},
    )
    try:
        response = llm.invoke(
            prompt.format(n=SETTINGS.decompose_max_subquestions, q=question)
        )
        content = str(getattr(response, "content", response)).strip()
        items = json.loads(content)
        if not isinstance(items, list):
            return []
        cleaned = [_normalize(x) for x in items if isinstance(x, str)]
        cleaned = [x for x in cleaned if x]
        return cleaned[: SETTINGS.decompose_max_subquestions]
    except Exception as exc:
        logger.info("LLM decomposition failed: %s", type(exc).__name__)
        return []


def decompose_question(question: str) -> List[str]:
    """将用户问题拆解为子问题列表。"""
    text = _normalize(question)
    if len(text) < SETTINGS.decompose_min_length:
        return []
    if SETTINGS.decomposer_method == "llm":
        items = _llm_decompose(text)
        if items:
            return items
    return _heuristic_decompose(text)
