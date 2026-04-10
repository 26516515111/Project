try:
    from .chunking import TextChunker, LanggraphyChunkerAdapter
except ImportError:
    # Placeholder to allow importing reranker without full dependencies
    # This is expected if langchain dependencies are missing
    TextChunker = None
    LanggraphyChunkerAdapter = None

try:
    from .langgraphy import LanggraphyPipelineAdapter
except Exception:  # pragma: no cover - optional dependency
    LanggraphyPipelineAdapter = None

__all__ = [
    "TextChunker",
    "LanggraphyChunkerAdapter",
    "LanggraphyPipelineAdapter",
]
"""
docker run --interactive --tty --rm --volume=D:/work/AI/neo4j/data:/data --volume=D:/work/AI/Project/rag_app/data/KG/delivery:/backups neo4j:latest neo4j-admin database load --from-path=/backups --verbose --overwrite-destination neo4j

docker run -d --name neo4j -p 17474:7474 -p 7687:7687 -v d:/work/AI/neo4j/data:/data -v d:/work/AI/neo4j/logs:/logs -v d:/work/AI/neo4j/conf:/var/lib/neo4j/conf -v d:/work/AI/neo4j/import:/var/lib/neo4j/import -v d:/work/AI/neo4j/plugins:/var/lib/neo4j/plugins -e NEO4J_AUTH=neo4j/neo4j1234 neo4j:latest
"""
