import re
from typing import List

from .config import SETTINGS


def _normalize(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        key = _normalize(item)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def _parse_alias_pairs() -> List[tuple[str, str]]:
    rows = getattr(SETTINGS, "kg_entity_aliases", []) or []
    pairs: List[tuple[str, str]] = []
    for row in rows:
        value = str(row or "").strip()
        if not value or "=" not in value:
            continue
        alias, canonical = value.split("=", 1)
        alias = alias.strip()
        canonical = canonical.strip()
        if alias and canonical and alias != canonical:
            pairs.append((alias, canonical))
    return pairs


def _alias_variants(question: str) -> List[str]:
    q = _normalize(question)
    if not q:
        return []
    variants: List[str] = []
    q_lower = q.lower()
    for alias, canonical in _parse_alias_pairs():
        alias_lower = alias.lower()
        canonical_lower = canonical.lower()
        if alias_lower in q_lower and canonical_lower not in q_lower:
            variants.append(q.replace(alias, canonical))
    return variants


def _model_token_variants(question: str) -> List[str]:
    q = _normalize(question)
    if not q:
        return []
    variants: List[str] = []
    tokens = set(re.findall(r"[A-Za-z]{2,}\d*(?:-[A-Za-z0-9]+)+", q))
    for token in tokens:
        no_dash = token.replace("-", "")
        spaced = token.replace("-", " ")
        if no_dash != token:
            variants.append(q.replace(token, no_dash))
        if spaced != token:
            variants.append(q.replace(token, spaced))
    return variants


def _intent_simplified_variant(question: str) -> List[str]:
    q = _normalize(question)
    if not q:
        return []
    compact = q
    for pattern in [
        r"^请问",
        r"^请教",
        r"^想了解",
        r"^咨询",
        r"是什么$",
        r"有哪些$",
        r"怎么做$",
        r"如何处理$",
        r"吗$",
        r"[？?]$",
    ]:
        compact = re.sub(pattern, "", compact).strip()
    if compact and compact != q:
        return [compact]
    return []


def generate_query_variants(question: str, max_variants: int = 2) -> List[str]:
    q = _normalize(question)
    if not q:
        return []
    variants: List[str] = []
    variants.extend(_alias_variants(q))
    variants.extend(_model_token_variants(q))
    variants.extend(_intent_simplified_variant(q))
    variants = [v for v in _dedupe(variants) if v != q]
    return variants[: max(0, int(max_variants))]
