from typing import Dict, List, Optional
from collections import Counter
import re
import os
import json

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from .config import SETTINGS


def _parse_alias_pairs(raw_pairs: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for raw in raw_pairs:
        line = str(raw or "").strip()
        if not line or "=" not in line:
            continue
        alias, canonical = line.split("=", 1)
        alias = _normalize_text(alias)
        canonical = _normalize_text(canonical)
        if not alias or not canonical:
            continue
        mapping[alias] = canonical
    return mapping


def _build_alias_map_from_chunks(
    chunk_kg_map: Optional[Dict[str, dict]], chunk_text_map: Optional[Dict[str, str]]
) -> Dict[str, str]:
    if not chunk_kg_map or not chunk_text_map:
        return {}
    roots_by_doc: Dict[str, List[str]] = {}
    for item in chunk_kg_map.values():
        doc_id = _normalize_text(item.get("doc_id", ""))
        if not doc_id:
            continue
        for ent in item.get("kg_entities", []) or []:
            ent_text = _normalize_text(ent)
            if not ent_text:
                continue
            root = ent_text.split("_", 1)[0].strip()
            if not root:
                continue
            roots_by_doc.setdefault(doc_id, []).append(root)

    canonical_by_doc: Dict[str, str] = {}
    for doc_id, roots in roots_by_doc.items():
        if not roots:
            continue
        counts = Counter(roots)
        canonical_by_doc[doc_id] = sorted(
            counts.items(), key=lambda x: (-x[1], -len(x[0]), x[0])
        )[0][0]

    alias_map: Dict[str, str] = {}
    pat = re.compile(
        r"\b[A-Za-z]{2,}[A-Za-z0-9]*(?:\s+[A-Za-z0-9]{1,})*-[A-Za-z0-9][A-Za-z0-9-]*\b"
        r"|\b[A-Za-z]{2,}\d+[A-Za-z0-9-]*\b"
    )
    blocked_prefixes = {
        "IP",
        "DC",
        "AC",
        "RH",
        "PT",
        "OUT",
        "SK",
        "YV",
        "CO",
        "IMG",
        "VDR",
        "RS",
        "IEC",
        "SOLAS",
        "EIA",
        "LCD",
        "MPU",
        "PLC",
        "CCS",
    }
    for chunk_id, item in chunk_kg_map.items():
        doc_id = _normalize_text(item.get("doc_id", ""))
        canonical = canonical_by_doc.get(doc_id)
        if not canonical:
            continue
        text = _normalize_text(chunk_text_map.get(chunk_id, ""))
        if not text:
            continue
        for alias in set(pat.findall(text)):
            alias = _normalize_text(alias)
            if len(alias) < 4 or len(alias) > 40:
                continue
            alias_key = _normalize_for_match(alias)
            if not alias_key:
                continue
            prefix = alias_key.split("-", 1)[0].upper()
            if prefix in blocked_prefixes:
                continue
            alias_map[alias] = canonical
    return alias_map


def _expand_entities_with_aliases(
    entities: List[str], question: str, extra_alias_map: Optional[Dict[str, str]] = None
) -> List[str]:
    q_norm = _normalize_text(question)
    if not q_norm:
        return entities
    alias_map = _parse_alias_pairs(getattr(SETTINGS, "kg_entity_aliases", []))
    if extra_alias_map:
        alias_map.update(extra_alias_map)
    if not alias_map:
        return entities
    resolved = list(entities)
    seen = {e for e in entities if e}
    for alias, canonical in alias_map.items():
        if alias in q_norm and canonical not in seen:
            resolved.append(canonical)
            seen.add(canonical)
    return resolved


def _get_driver():
    """创建并返回Neo4j驱动。"""
    return GraphDatabase.driver(
        SETTINGS.neo4j_uri,
        auth=(SETTINGS.neo4j_user, SETTINGS.neo4j_password),
    )


def _get_driver_with_override(
    neo4j_uri: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None,
):
    uri = str(neo4j_uri or SETTINGS.neo4j_uri).strip()
    user = str(neo4j_user or SETTINGS.neo4j_user).strip()
    password = str(neo4j_password or SETTINGS.neo4j_password)
    return GraphDatabase.driver(uri, auth=(user, password))


def _pick_rel_id_field(session) -> str:
    keys = []
    try:
        rows = session.run("CALL db.propertyKeys()")
        keys = [row.get("propertyKey") for row in rows if row.get("propertyKey")]
    except Exception:
        return ""
    for candidate in ["rel_id", "id", "relation_id", "kg_id"]:
        if candidate in keys:
            return candidate
    return ""


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _normalize_for_match(text: str) -> str:
    value = _normalize_text(text).lower()
    return re.sub(r"\s+", "", value)


def _is_kg_id(text: str) -> bool:
    return re.fullmatch(r"(ENT|REL)_\d+", str(text or "").strip()) is not None


def _tokenize(text: str) -> List[str]:
    value = _normalize_text(text)
    if not value:
        return []
    tokens: List[str] = []
    parts = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+", value)
    for part in parts:
        if re.fullmatch(r"[A-Za-z0-9_]+", part):
            tokens.append(part.lower())
            continue
        if len(part) <= 2:
            tokens.append(part)
            continue
        max_len = min(4, len(part))
        for size in range(2, max_len + 1):
            for i in range(0, len(part) - size + 1):
                tokens.append(part[i : i + size])
    seen = set()
    deduped: List[str] = []
    for token in tokens:
        token = _normalize_text(token)
        if not token or len(token) < 2:
            continue
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def _resolve_kg_merged_path() -> str:
    candidates = [
        os.path.join(SETTINGS.kg_dir, "kg_merged.json"),
        os.path.join(SETTINGS.kg_dir, "delivery", "kg_merged.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return ""


def _load_entity_name_map() -> Dict[str, str]:
    path = _resolve_kg_merged_path()
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    mapping: Dict[str, str] = {}
    entities = data.get("entities", []) if isinstance(data, dict) else []
    for idx, ent in enumerate(entities, start=1):
        name = _normalize_text(ent.get("name", ""))
        if not name:
            continue
        ent_id = _normalize_text(ent.get("entity_id", ""))
        if not ent_id:
            ent_id = f"ENT_{idx:06d}"
        mapping[ent_id] = name
    return mapping


def _extract_entities(
    question: str, chunk_kg_map: Optional[Dict[str, dict]], ent_name_map: Dict[str, str]
) -> List[str]:
    if not chunk_kg_map:
        return []
    q = _normalize_text(question)
    if not q:
        return []
    candidates = set()
    for item in chunk_kg_map.values():
        for ent in item.get("kg_entities", []) or []:
            ent_text = _normalize_text(ent)
            if not ent_text:
                continue
            if _is_kg_id(ent_text):
                ent_text = _normalize_text(ent_name_map.get(ent_text, ""))
                if not ent_text:
                    continue
            if len(ent_text) < 2:
                continue
            if ent_text == q or ent_text in q:
                candidates.add(ent_text)
    return list(candidates)


def _collect_entity_names(
    ent_name_map: Dict[str, str], chunk_kg_map: Optional[Dict[str, dict]]
) -> List[str]:
    names = set()
    for name in ent_name_map.values():
        n = _normalize_text(name)
        if len(n) >= 2:
            names.add(n)
    if chunk_kg_map:
        for item in chunk_kg_map.values():
            for ent in item.get("kg_entities", []) or []:
                ent_text = _normalize_text(ent)
                if not ent_text:
                    continue
                if _is_kg_id(ent_text):
                    ent_text = _normalize_text(ent_name_map.get(ent_text, ""))
                if len(ent_text) >= 2:
                    names.add(ent_text)
    return list(names)


def _extract_exact_entities(
    question: str, ent_name_map: Dict[str, str], chunk_kg_map: Optional[Dict[str, dict]]
) -> List[str]:
    q_match = _normalize_for_match(question)
    if not q_match:
        return []
    entity_names = _collect_entity_names(ent_name_map, chunk_kg_map)

    candidates = []
    for name in entity_names:
        n_match = _normalize_for_match(name)
        if not n_match:
            continue
        pos = q_match.find(n_match)
        if pos < 0:
            continue
        candidates.append((name, len(n_match), pos))
    candidates.sort(key=lambda x: (-x[1], x[2]))
    deduped: List[str] = []
    seen = set()
    for name, _, _ in candidates:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)

    # 回退匹配：优先用型号/代号做实体字符串匹配，缓解
    # “SMP Series Hydraulic Pump_xxx” vs “SMP气动泵/WREN SMP”这类命名差异。
    if len(deduped) >= 5:
        return deduped[:5]

    q_lower = _normalize_text(question).lower()
    q_model_tokens = set(
        re.findall(r"[a-z]{2,}\d*[a-z0-9-]*", q_lower, flags=re.IGNORECASE)
    )
    q_model_tokens = {
        t.lower()
        for t in q_model_tokens
        if len(t) >= 2 and t.lower() not in {"series", "model", "manual"}
    }
    q_has_pump = any(k in q_lower for k in ("泵", "液压泵", "气动泵", "pump"))
    q_has_valve = any(k in q_lower for k in ("阀", "刀闸阀", "法兰", "valve", "flange"))

    fuzzy_candidates = []
    for name in entity_names:
        if name in seen:
            continue
        n_lower = _normalize_text(name).lower()
        n_match = _normalize_for_match(name)
        if not n_match:
            continue
        n_model_tokens = set(
            re.findall(r"[a-z]{2,}\d*[a-z0-9-]*", n_lower, flags=re.IGNORECASE)
        )
        n_model_tokens = {
            t.lower()
            for t in n_model_tokens
            if len(t) >= 2 and t.lower() not in {"series", "model", "manual"}
        }
        shared_tokens = q_model_tokens & n_model_tokens
        score = 0
        if shared_tokens:
            score += 10 * len(shared_tokens)
        if q_has_pump and "pump" in n_lower:
            score += 3
        if q_has_valve and ("valve" in n_lower or "flange" in n_lower):
            score += 3
        if score <= 0:
            continue
        fuzzy_candidates.append((name, score, len(n_match)))

    fuzzy_candidates.sort(key=lambda x: (-x[1], -x[2], x[0]))
    for name, _, _ in fuzzy_candidates:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
        if len(deduped) >= 5:
            break

    return deduped[:5]


def _extract_intent_terms(question: str, entities: List[str]) -> List[str]:
    q = _normalize_text(question)
    if not q:
        return []
    remainder = q
    for ent in entities:
        if not ent:
            continue
        remainder = remainder.replace(ent, " ")
    remainder = re.sub(r"[\s,，。；;：:、()（）\[\]【】'\"“”]+", " ", remainder)
    tokens = _tokenize(remainder)
    stopwords = {
        "什么",
        "如何",
        "怎么",
        "怎样",
        "以及",
        "有关",
        "关于",
        "的",
        "和",
        "与",
        "及",
    }
    return [t for t in tokens if t not in stopwords]


def query_knowledge_graph(
    question: str,
    doc_source_map: Optional[Dict[str, dict]] = None,
    chunk_kg_map: Optional[Dict[str, dict]] = None,
    chunk_text_map: Optional[Dict[str, str]] = None,
    top_k: int = 5,
    neo4j_uri: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None,
) -> List[Dict[str, str]]:
    """查询知识图谱并返回扩展三元组列表。

    Args:
        question: 用户问题文本。
        doc_source_map: 可选的doc_id到来源信息映射。
        chunk_kg_map: 可选的chunk到KG实体/关系映射。
        chunk_text_map: 可选的chunk_id到文本映射，用于相关性评分。
        top_k: 返回结果数量。

    Returns:
        List[Dict[str, str]]: 三元组列表，字段包含head、rel、tail等信息。
    """
    results: List[Dict[str, str]] = []
    chunk_rel_ids: List[str] = []
    entities: List[str] = []
    ent_name_map = _load_entity_name_map()
    exact_entities = _extract_exact_entities(question, ent_name_map, chunk_kg_map)
    chunk_alias_map = _build_alias_map_from_chunks(chunk_kg_map, chunk_text_map)
    exact_entities = _expand_entities_with_aliases(
        exact_entities, question, extra_alias_map=chunk_alias_map
    )[:5]
    intent_terms = _extract_intent_terms(question, exact_entities)
    query_terms = _tokenize(question)
    if chunk_kg_map and chunk_text_map:
        query_tokens = set(query_terms)
        scored = []
        for chunk_id, item in chunk_kg_map.items():
            text = str(chunk_text_map.get(chunk_id, "")).strip()
            if not text:
                continue
            score = float(sum(1 for token in query_tokens if token and token in text))
            norm = max(1.0, float(len(query_tokens)))
            score = score / norm
            if score < SETTINGS.kg_score_threshold:
                continue
            scored.append({"chunk_id": chunk_id, "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        for hit in scored[: top_k * 3]:
            item = chunk_kg_map.get(hit["chunk_id"], {})
            for rel_id in item.get("kg_relations", []) or []:
                if rel_id not in chunk_rel_ids:
                    chunk_rel_ids.append(rel_id)
        if len(chunk_rel_ids) > SETTINGS.kg_rel_limit:
            chunk_rel_ids = chunk_rel_ids[: SETTINGS.kg_rel_limit]
        entities = _extract_entities(question, chunk_kg_map, ent_name_map)
    driver = _get_driver_with_override(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
    )
    try:
        with driver.session() as session:
            rel_id_field = _pick_rel_id_field(session)
            if exact_entities:
                rel_id_expr = (
                    f"coalesce(r.`{rel_id_field}`, '')" if rel_id_field else "''"
                )
                cypher = (
                    "MATCH (seed) "
                    "WHERE coalesce(seed.name, '') IN $entities "
                    "MATCH p=(seed)-[*1..2]-(x) "
                    "UNWIND relationships(p) AS r "
                    "WITH DISTINCT r, seed, startNode(r) AS h, endNode(r) AS t "
                    "WITH r, seed, h, t, "
                    "reduce(s = 0, tn IN $intent_terms | s + CASE "
                    "WHEN coalesce(h.name, '') CONTAINS tn THEN 1 "
                    "WHEN coalesce(t.name, '') CONTAINS tn THEN 1 "
                    "WHEN type(r) CONTAINS tn THEN 1 "
                    "WHEN coalesce(r.description, '') CONTAINS tn THEN 1 "
                    "ELSE 0 END) AS intent_score, "
                    "CASE WHEN h = seed OR t = seed THEN 2 ELSE 1 END AS hop_score "
                    "WHERE size($intent_terms) = 0 OR intent_score > 0 "
                    "RETURN "
                    "h.name AS head, "
                    "coalesce(head(labels(h)), '') AS head_label, "
                    "type(r) AS rel, "
                    "t.name AS tail, "
                    "coalesce(head(labels(t)), '') AS tail_label, "
                    "coalesce(r.description, '') AS description, "
                    "coalesce(h.source_section, '') AS head_source_section, "
                    "coalesce(t.source_section, '') AS tail_source_section, "
                    "coalesce(h.doc_id, '') AS head_doc_id, "
                    "coalesce(t.doc_id, '') AS tail_doc_id, "
                    "coalesce(r.doc_id, '') AS rel_doc_id, "
                    f"{rel_id_expr} AS rel_id "
                    "ORDER BY intent_score DESC, hop_score DESC "
                    "LIMIT $limit"
                )
                rows = session.run(
                    cypher,
                    entities=exact_entities,
                    intent_terms=intent_terms,
                    limit=min(top_k, SETTINGS.kg_rel_limit),
                )
            elif chunk_rel_ids and rel_id_field:
                cypher = (
                    "MATCH (h)-[r]->(t) "
                    f"WHERE coalesce(r.`{rel_id_field}`, '') IN $rel_ids "
                    "RETURN "
                    "h.name AS head, "
                    "coalesce(head(labels(h)), '') AS head_label, "
                    "type(r) AS rel, "
                    "t.name AS tail, "
                    "coalesce(head(labels(t)), '') AS tail_label, "
                    "coalesce(r.description, '') AS description, "
                    "coalesce(h.source_section, '') AS head_source_section, "
                    "coalesce(t.source_section, '') AS tail_source_section, "
                    "coalesce(h.doc_id, '') AS head_doc_id, "
                    "coalesce(t.doc_id, '') AS tail_doc_id, "
                    "coalesce(r.doc_id, '') AS rel_doc_id, "
                    f"coalesce(r.`{rel_id_field}`, '') AS rel_id "
                    "LIMIT $limit"
                )
                rows = session.run(
                    cypher,
                    rel_ids=chunk_rel_ids,
                    limit=min(top_k, SETTINGS.kg_rel_limit),
                )
            elif entities:
                cypher = (
                    "MATCH (h)-[r]->(t) "
                    "WHERE h.name IN $entities OR t.name IN $entities "
                    "RETURN "
                    "h.name AS head, "
                    "coalesce(head(labels(h)), '') AS head_label, "
                    "type(r) AS rel, "
                    "t.name AS tail, "
                    "coalesce(head(labels(t)), '') AS tail_label, "
                    "coalesce(r.description, '') AS description, "
                    "coalesce(h.source_section, '') AS head_source_section, "
                    "coalesce(t.source_section, '') AS tail_source_section, "
                    "coalesce(h.doc_id, '') AS head_doc_id, "
                    "coalesce(t.doc_id, '') AS tail_doc_id, "
                    "coalesce(r.doc_id, '') AS rel_doc_id, "
                    "'' AS rel_id "
                    "LIMIT $limit"
                )
                rows = session.run(
                    cypher,
                    entities=entities,
                    limit=min(top_k, SETTINGS.kg_rel_limit),
                )
            elif query_terms:
                cypher = (
                    "MATCH (h)-[r]->(t) "
                    "WHERE any(tn IN $terms WHERE h.name CONTAINS tn OR t.name CONTAINS tn OR coalesce(r.description, '') CONTAINS tn) "
                    "RETURN "
                    "h.name AS head, "
                    "coalesce(head(labels(h)), '') AS head_label, "
                    "type(r) AS rel, "
                    "t.name AS tail, "
                    "coalesce(head(labels(t)), '') AS tail_label, "
                    "coalesce(r.description, '') AS description, "
                    "coalesce(h.source_section, '') AS head_source_section, "
                    "coalesce(t.source_section, '') AS tail_source_section, "
                    "coalesce(h.doc_id, '') AS head_doc_id, "
                    "coalesce(t.doc_id, '') AS tail_doc_id, "
                    "coalesce(r.doc_id, '') AS rel_doc_id, "
                    "'' AS rel_id "
                    "LIMIT $limit"
                )
                rows = session.run(
                    cypher,
                    terms=query_terms,
                    limit=min(top_k, SETTINGS.kg_rel_limit),
                )
            else:
                cypher = (
                    "MATCH (h)-[r]->(t) "
                    "WHERE h.name CONTAINS $q OR t.name CONTAINS $q OR coalesce(r.description, '') CONTAINS $q "
                    "RETURN "
                    "h.name AS head, "
                    "coalesce(head(labels(h)), '') AS head_label, "
                    "type(r) AS rel, "
                    "t.name AS tail, "
                    "coalesce(head(labels(t)), '') AS tail_label, "
                    "coalesce(r.description, '') AS description, "
                    "coalesce(h.source_section, '') AS head_source_section, "
                    "coalesce(t.source_section, '') AS tail_source_section, "
                    "coalesce(h.doc_id, '') AS head_doc_id, "
                    "coalesce(t.doc_id, '') AS tail_doc_id, "
                    "coalesce(r.doc_id, '') AS rel_doc_id, "
                    "'' AS rel_id "
                    "LIMIT $limit"
                )
                rows = session.run(
                    cypher,
                    q=question,
                    limit=min(top_k, SETTINGS.kg_rel_limit),
                )
            for row in rows:
                item = {
                    "head": row.get("head", ""),
                    "head_label": row.get("head_label", ""),
                    "rel": row.get("rel", ""),
                    "relation": row.get("rel", ""),
                    "tail": row.get("tail", ""),
                    "tail_label": row.get("tail_label", ""),
                    "description": row.get("description", ""),
                    "head_source_section": row.get("head_source_section", ""),
                    "tail_source_section": row.get("tail_source_section", ""),
                    "rel_id": row.get("rel_id", ""),
                }
                head_doc_id = str(row.get("head_doc_id", "") or "").strip()
                tail_doc_id = str(row.get("tail_doc_id", "") or "").strip()
                rel_doc_id = str(row.get("rel_doc_id", "") or "").strip()
                if head_doc_id:
                    item["head_doc_id"] = head_doc_id
                if tail_doc_id:
                    item["tail_doc_id"] = tail_doc_id
                if rel_doc_id:
                    item["rel_doc_id"] = rel_doc_id
                if doc_source_map:
                    if head_doc_id in doc_source_map:
                        item["head_source"] = doc_source_map[head_doc_id].get(
                            "source", ""
                        )
                    if tail_doc_id in doc_source_map:
                        item["tail_source"] = doc_source_map[tail_doc_id].get(
                            "source", ""
                        )
                    if rel_doc_id in doc_source_map:
                        item["rel_source"] = doc_source_map[rel_doc_id].get(
                            "source", ""
                        )
                results.append(item)
    except ServiceUnavailable:
        return []
    finally:
        driver.close()
    return results
