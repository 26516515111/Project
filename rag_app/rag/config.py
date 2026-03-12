import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


def env(key: str, default: str) -> str:
    """读取环境变量，若为空则返回默认值。

    Args:
        key: 环境变量名。
        default: 默认值。

    Returns:
        str: 环境变量值或默认值。
    """
    value = os.getenv(key)
    if value is None or value.strip() == "":
        return default
    return value


class Settings:
    # Paths
    docs_dir = env(
        "RAG_DOCS_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "docs")
    )
    index_dir = env(
        "RAG_INDEX_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "index")
    )

    # Retrieval
    top_k = int(env("RAG_TOP_K", "5"))
    bm25_weight = float(env("RAG_BM25_WEIGHT", "0.45"))
    vector_weight = float(env("RAG_VECTOR_WEIGHT", "0.55"))

    # Embeddings
    embedding_model = env(
        "RAG_EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )

    # LLM (optional)
    llm_provider = env("RAG_LLM_PROVIDER", "ollama")  # ollama|none
    llm_model = env("RAG_LLM_MODEL", "qwen2.5:3b")
    llm_max_tokens = int(env("RAG_LLM_MAX_TOKENS", "512"))
    llm_temperature = float(env("RAG_LLM_TEMPERATURE", "0.2"))
    use_history = env("RAG_USE_HISTORY", "true").lower() in {"1", "true", "yes", "y"}

    # Ollama
    ollama_base_url = env("RAG_OLLAMA_BASE_URL", "http://localhost:11434")

    # Neo4j
    neo4j_uri = env("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = env("NEO4J_USER", "neo4j")
    neo4j_password = env("NEO4J_PASSWORD", "neo4j1234")


SETTINGS = Settings()
