from typing import List, Dict

from neo4j import GraphDatabase

from .config import SETTINGS


def _get_driver():
    return GraphDatabase.driver(
        SETTINGS.neo4j_uri,
        auth=(SETTINGS.neo4j_user, SETTINGS.neo4j_password),
    )


def query_knowledge_graph(question: str) -> List[Dict[str, str]]:
    """
    Basic Neo4j query template.
    Adjust labels and relationship types to your KG schema.
    Returns triplets like {"head": "故障现象", "rel": "原因", "tail": "..."}.
    """
    cypher = (
        "MATCH (s:Symptom)-[r:CAUSES]->(c:Cause) "
        "WHERE s.name CONTAINS $q OR c.name CONTAINS $q "
        "RETURN s.name AS head, type(r) AS rel, c.name AS tail "
        "LIMIT 5"
    )
    results: List[Dict[str, str]] = []
    driver = _get_driver()
    try:
        with driver.session() as session:
            rows = session.run(cypher, q=question)
            for row in rows:
                results.append(
                    {
                        "head": row.get("head", ""),
                        "rel": row.get("rel", ""),
                        "tail": row.get("tail", ""),
                    }
                )
    finally:
        driver.close()
    return results
