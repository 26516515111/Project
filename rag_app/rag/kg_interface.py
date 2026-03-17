from typing import Dict, List, Optional
import re
import os
import json

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from .config import SETTINGS


def _get_driver():
    """创建并返回Neo4j驱动。"""
    return GraphDatabase.driver(
        SETTINGS.neo4j_uri,
        auth=(SETTINGS.neo4j_user, SETTINGS.neo4j_password),
    )


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


def _load_entity_name_map() -> Dict[str, str]:
    path = os.path.join(SETTINGS.kg_dir, "kg_merged.json")
    if not os.path.isfile(path):
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
        ent_id = _normalize_text(ent.get("id", ""))
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


def query_knowledge_graph(
    question: str,
    doc_source_map: Optional[Dict[str, dict]] = None,
    chunk_kg_map: Optional[Dict[str, dict]] = None,
    chunk_text_map: Optional[Dict[str, str]] = None,
    top_k: int = 5,
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
    driver = _get_driver()
    try:
        with driver.session() as session:
            rel_id_field = _pick_rel_id_field(session)
            if chunk_rel_ids and rel_id_field:
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
