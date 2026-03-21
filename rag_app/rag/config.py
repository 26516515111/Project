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
    kg_dir = env(
        "RAG_KG_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "KG")
    )

    # 检索模块
    # RAG_TOP_K：最终用于回答的检索数量
    top_k = int(env("RAG_TOP_K", "5"))
    # RAG_HYBRID_TOP_K：混合检索候选数量（重排序前）
    hybrid_top_k = int(env("RAG_HYBRID_TOP_K", "20"))
    # RAG_BM25_WEIGHT：BM25在混合检索中的权重
    bm25_weight = float(env("RAG_BM25_WEIGHT", "0.45"))
    # RAG_VECTOR_WEIGHT：向量检索在混合检索中的权重
    vector_weight = float(env("RAG_VECTOR_WEIGHT", "0.55"))

    # 向量嵌入模块
    # RAG_EMBEDDING_MODEL：向量模型名称
    embedding_model = env(
        "RAG_EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )

    # LLM模块（可选）
    # RAG_LLM_PROVIDER：ollama|none
    llm_provider = env("RAG_LLM_PROVIDER", "ollama")  # ollama|none
    # RAG_LLM_MODEL：Ollama模型名称
    llm_model = env("RAG_LLM_MODEL", "qwen2.5:3b")
    # RAG_LLM_MAX_TOKENS：最大输出token数
    llm_max_tokens = int(env("RAG_LLM_MAX_TOKENS", "512"))
    # RAG_LLM_TEMPERATURE：生成温度
    llm_temperature = float(env("RAG_LLM_TEMPERATURE", "0.2"))
    # RAG_USE_HISTORY：是否启用历史对话
    use_history = env("RAG_USE_HISTORY", "true").lower() in {"1", "true", "yes", "y"}

    # Ollama模块
    # RAG_OLLAMA_BASE_URL：Ollama服务地址
    ollama_base_url = env("RAG_OLLAMA_BASE_URL", "http://localhost:11434")

    # Neo4j模块
    # NEO4J_URI：Neo4j连接地址
    neo4j_uri = env("NEO4J_URI", "bolt://localhost:7687")
    # NEO4J_USER：Neo4j用户名
    neo4j_user = env("NEO4J_USER", "neo4j")
    # NEO4J_PASSWORD：Neo4j密码
    neo4j_password = env("NEO4J_PASSWORD", "neo4j1234")

    # KG检索模块
    # RAG_KG_REL_LIMIT：每次查询KG关系上限
    kg_rel_limit = int(env("RAG_KG_REL_LIMIT", "50"))
    # RAG_KG_SCORE_THRESHOLD：KG检索的chunk相关性阈值
    kg_score_threshold = float(env("RAG_KG_SCORE_THRESHOLD", "0.1"))

    # 重排序模块
    # RAG_USE_RERANKER：是否启用交叉编码器重排序
    use_reranker = env("RAG_USE_RERANKER", "true").lower() in {"1", "true", "yes", "y"}
    # RAG_RERANKER_MODEL：重排序模型名称
    reranker_model = env("RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    # RAG_RERANKER_TOP_K：重排序后保留的passage数量
    reranker_top_k = int(env("RAG_RERANKER_TOP_K", "5"))

    # 问题分解模块
    # RAG_USE_QUERY_DECOMPOSITION：是否启用问题分解
    use_query_decomposition = env("RAG_USE_QUERY_DECOMPOSITION", "true").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    # RAG_DECOMPOSER_METHOD：heuristic|llm
    decomposer_method = env("RAG_DECOMPOSER_METHOD", "heuristic")
    # RAG_DECOMPOSE_MIN_LENGTH：触发问题分解的最小长度
    decompose_min_length = int(env("RAG_DECOMPOSE_MIN_LENGTH", "8"))
    # RAG_DECOMPOSE_MAX_SUBQUESTIONS：最大子问题数量
    decompose_max_subquestions = int(env("RAG_DECOMPOSE_MAX_SUBQUESTIONS", "3"))
    # RAG_DECOMPOSE_MAX_SUBQUERIES：用于检索的最大查询数（含原问题）
    decompose_max_subqueries = int(env("RAG_DECOMPOSE_MAX_SUBQUERIES", "4"))
    # RAG_DECOMPOSE_PER_QUERY_TOP_K：每个子问题的检索数量
    decompose_per_query_top_k = int(env("RAG_DECOMPOSE_PER_QUERY_TOP_K", "4"))
    # RAG_DECOMPOSE_LLM_MAX_TOKENS：分解LLM最大输出token数
    decompose_llm_max_tokens = int(env("RAG_DECOMPOSE_LLM_MAX_TOKENS", "128"))
    # RAG_DECOMPOSE_LLM_TEMPERATURE：分解LLM温度
    decompose_llm_temperature = float(env("RAG_DECOMPOSE_LLM_TEMPERATURE", "0.1"))

    # 切块模块
    # RAG_CHUNK_SIZE：单块长度
    chunk_size = int(env("RAG_CHUNK_SIZE", "1500"))
    # RAG_CHUNK_OVERLAP：相邻块重叠长度
    chunk_overlap = int(env("RAG_CHUNK_OVERLAP", "80"))
    # RAG_HEADING_MERGE_ENABLED：是否启用标题合并规则
    heading_merge_enabled = env("RAG_HEADING_MERGE_ENABLED", "true").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    # RAG_HEADING_MERGE_LEVEL：合并到的标题层级（如2代表1.1）
    heading_merge_level = int(env("RAG_HEADING_MERGE_LEVEL", "3"))
    separators = ["\n\n", "\n", "。", ".", " ", "，", ",", ""]


SETTINGS = Settings()
