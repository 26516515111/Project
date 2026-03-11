from typing import List, Dict, Optional

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from .config import SETTINGS


def _get_driver():
    return GraphDatabase.driver(
        SETTINGS.neo4j_uri,
        auth=(SETTINGS.neo4j_user, SETTINGS.neo4j_password),
    )


def query_knowledge_graph(
    question: str, doc_source_map: Optional[Dict[str, dict]] = None
) -> List[Dict[str, str]]:
    """
    Neo4j query template for extended triplets.
    Returns items like:
    {
        "head": "...",
        "head_label": "...",
        "rel": "...",
        "tail": "...",
        "tail_label": "...",
        "description": "...",
        "head_source_section": "...",
        "tail_source_section": "...",
    }
    """
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
        "coalesce(t.source_section, '') AS tail_source_section "
        "LIMIT 5"
    )
    results: List[Dict[str, str]] = []
    driver = _get_driver()
    try:
        with driver.session() as session:
            rows = session.run(cypher, q=question)
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
                }
                head_doc_id = row.get("head_doc_id", "")
                tail_doc_id = row.get("tail_doc_id", "")
                if doc_source_map:
                    if head_doc_id in doc_source_map:
                        item["head_source"] = doc_source_map[head_doc_id].get(
                            "source", ""
                        )
                    if tail_doc_id in doc_source_map:
                        item["tail_source"] = doc_source_map[tail_doc_id].get(
                            "source", ""
                        )
                results.append(item)
    except ServiceUnavailable:
        return []
    finally:
        driver.close()
    return results
