from typing import Dict, List, Optional

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
    if chunk_kg_map and chunk_text_map:
        query_tokens = set(question.replace("\n", " ").split(" "))
        scored = []
        for chunk_id, item in chunk_kg_map.items():
            text = str(chunk_text_map.get(chunk_id, "")).strip()
            if not text:
                continue
            tokens = set(text.replace("\n", " ").split(" "))
            score = float(len(tokens.intersection(query_tokens)))
            if score <= 0:
                continue
            scored.append({"chunk_id": chunk_id, "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        for hit in scored[: top_k * 3]:
            item = chunk_kg_map.get(hit["chunk_id"], {})
            for rel_id in item.get("kg_relations", []) or []:
                if rel_id not in chunk_rel_ids:
                    chunk_rel_ids.append(rel_id)
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
                rows = session.run(cypher, rel_ids=chunk_rel_ids, limit=top_k)
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
                rows = session.run(cypher, q=question, limit=top_k)
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
