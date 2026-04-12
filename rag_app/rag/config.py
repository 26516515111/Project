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


def env_list(key: str, default: str) -> list[str]:
    """读取逗号分隔环境变量并返回字符串列表。"""
    raw = env(key, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


class Settings:
    # Paths
    data_dir = env(
        "RAG_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data")
    )
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
    # RAG_ENTITY_PRIORITY_ENABLED：是否启用实体优先检索
    entity_priority_enabled = env("RAG_ENTITY_PRIORITY_ENABLED", "true").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    # RAG_ENTITY_BOOST：实体命中时的附加分
    entity_boost = float(env("RAG_ENTITY_BOOST", "0.35"))
    # RAG_ENTITY_MIN_IN_TOPK：Top-K中至少保留的实体命中数量
    entity_min_in_topk = int(env("RAG_ENTITY_MIN_IN_TOPK", "2"))
    # RAG_ENTITY_STRICT_FILTER：命中实体时是否仅保留实体相关chunk
    entity_strict_filter = env("RAG_ENTITY_STRICT_FILTER", "true").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    # RAG_TECH_HEADING_BOOST_ENABLED：技术性问题是否提升标题含“技术”的chunk
    tech_heading_boost_enabled = env(
        "RAG_TECH_HEADING_BOOST_ENABLED", "true"
    ).lower() in {"1", "true", "yes", "y"}
    # RAG_TECH_HEADING_BOOST：技术性问题的标题匹配加分系数
    tech_heading_boost = float(env("RAG_TECH_HEADING_BOOST", "0.35"))
    # RAG_NEIGHBOR_CONTEXT_ENABLED：是否为命中chunk补充后续相邻chunk
    neighbor_context_enabled = env("RAG_NEIGHBOR_CONTEXT_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    # RAG_NEIGHBOR_CONTEXT_WINDOW：每个命中chunk向后补充的相邻窗口大小
    neighbor_context_window = int(env("RAG_NEIGHBOR_CONTEXT_WINDOW", "1"))
    # RAG_NEIGHBOR_CONTEXT_PARAM_WINDOW：参数/规格类问题时的相邻补充窗口大小
    neighbor_context_param_window = int(env("RAG_NEIGHBOR_CONTEXT_PARAM_WINDOW", "3"))
    # RAG_NEIGHBOR_CONTEXT_SEED_LIMIT：最多对前N个高分命中chunk做相邻扩展
    neighbor_context_seed_limit = int(env("RAG_NEIGHBOR_CONTEXT_SEED_LIMIT", "3"))
    # RAG_NEIGHBOR_CONTEXT_MAX_CHUNKS：单次查询最多补充的相邻chunk数量
    neighbor_context_max_chunks = int(env("RAG_NEIGHBOR_CONTEXT_MAX_CHUNKS", "6"))
    # RAG_CONTEXT_COMPRESSION_ENABLED：是否启用上下文压缩检索
    context_compression_enabled = env(
        "RAG_CONTEXT_COMPRESSION_ENABLED", "false"
    ).lower() in {"1", "true", "yes", "y"}
    # RAG_CONTEXT_COMPRESSION_CHUNK_SIZE：压缩前切分块大小
    context_compression_chunk_size = int(
        env("RAG_CONTEXT_COMPRESSION_CHUNK_SIZE", "300")
    )
    # RAG_CONTEXT_COMPRESSION_CHUNK_OVERLAP：压缩前切分块重叠
    context_compression_chunk_overlap = int(
        env("RAG_CONTEXT_COMPRESSION_CHUNK_OVERLAP", "0")
    )
    # RAG_CONTEXT_COMPRESSION_SEPARATOR：压缩切分分隔符
    context_compression_separator = env("RAG_CONTEXT_COMPRESSION_SEPARATOR", ". ")
    # RAG_CONTEXT_COMPRESSION_SIMILARITY_THRESHOLD：相关性过滤阈值
    context_compression_similarity_threshold = float(
        env("RAG_CONTEXT_COMPRESSION_SIMILARITY_THRESHOLD", "0.66")
    )
    # RAG_CONTEXT_COMPRESSION_MIN_QUERY_LENGTH：触发压缩的最小查询长度
    context_compression_min_query_length = int(
        env("RAG_CONTEXT_COMPRESSION_MIN_QUERY_LENGTH", "20")
    )
    # RAG_CONTEXT_COMPRESSION_MIN_QUERY_COUNT：触发压缩的最小查询数量
    context_compression_min_query_count = int(
        env("RAG_CONTEXT_COMPRESSION_MIN_QUERY_COUNT", "2")
    )

    # 向量嵌入模块
    # RAG_EMBEDDING_MODEL：向量模型名称
    embedding_model = env(
        "RAG_EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    # RAG_EMBEDDING_LOCAL_ONLY：是否强制仅从本地加载embedding模型
    embedding_local_only = env("RAG_EMBEDDING_LOCAL_ONLY", "true").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    # RAG_EMBEDDING_CACHE_DIR：embedding模型缓存目录（可选）
    embedding_cache_dir = env("RAG_EMBEDDING_CACHE_DIR", "").strip()

    # LLM模块（可选）
    # RAG_LLM_PROVIDER：ollama|modelscope|none
    llm_provider = env("RAG_LLM_PROVIDER", "modelscope")  # ollama|modelscope|none
    # RAG_LLM_MODEL：聊天模型名称
    llm_model_name = env("RAG_LLM_MODEL", "qwen2.5:3b")
    # RAG_MODELSCOPE_MODEL：ModelScope聊天模型名称
    llm_modelscope_model = env("RAG_MODELSCOPE_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
    # llm_model_name = env("RAG_LLM_MODEL", "Qwen/Qwen3-4B")
    # 兼容旧命名
    llm_model = llm_model_name
    # RAG_LLM_MAX_TOKENS：最大输出token数
    llm_max_tokens = int(env("RAG_LLM_MAX_TOKENS", "512"))
    # RAG_LLM_TEMPERATURE：生成温度
    llm_temperature = float(env("RAG_LLM_TEMPERATURE", "0.2"))
    # RAG_USE_HISTORY：是否启用历史对话
    use_history = env("RAG_USE_HISTORY", "true").lower() in {"1", "true", "yes", "y"}
    # RAG_DOMAIN_KEYWORDS：领域相关性判定关键词（逗号分隔）
    domain_keywords = env_list(
        "RAG_DOMAIN_KEYWORDS",
        "船,船舶,舰,机舱,主机,辅机,柴油机,燃油,润滑,冷却,泵,阀,轴,轴承,齿轮,振动,温度,压力,报警,故障,检修,维修,维护,排查,推进,发电,舵机,海水,淡水,设备,ship,marine,engine,pump,valve,alarm,fault,maintenance",
    )

    # LLM连接模块
    # RAG_OLLAMA_BASE_URL：Ollama服务地址（历史兼容）
    llm_base_url = env("RAG_OLLAMA_BASE_URL", "http://localhost:11434")
    # RAG_MODELSCOPE_BASE_URL：ModelScope OpenAI兼容服务地址
    llm_modelscope_base_url = env(
        "RAG_MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1"
    )

    # 兼容旧命名
    ollama_base_url = llm_base_url

    # RAG_LLM_API_KEY / MODELSCOPE_API_KEY / LLM_API_KEY：LLM鉴权密钥
    llm_api_key = env(
        "RAG_LLM_API_KEY", env("MODELSCOPE_API_KEY", env("LLM_API_KEY", "ms-49891fe4-7e57-48d8-836e-1160b8f89ac9"))
    )

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
    # RAG_KG_ENTITY_ALIASES：KG实体别名映射，格式 alias=canonical，逗号分隔
    kg_entity_aliases = env_list(
        "RAG_KG_ENTITY_ALIASES",
        "GCWJ-01=船舶监测报警系统,GCWJ01=船舶监测报警系统,GCJB0-5=机舱总线制监测报警系统,GCJB-0-5=机舱总线制监测报警系统,GCJB24=机舱总线制监测报警系统,GCJB-24=机舱总线制监测报警系统,GC WAS-01=驾驶台航行值班报警系统,WAS-01=驾驶台航行值班报警系统,GC WAS-01-03=驾驶台航行值班报警系统,WAS-01-03=驾驶台航行值班报警系统,CDQY2A-PC2-6=CDQY2A-PC2-6 型 船用主柴油机电气遥控系统,CDQY2A-PC2=CDQY2A-PC2-6 型 船用主柴油机电气遥控系统,PC2-6=CDQY2A-PC2-6 型 船用主柴油机电气遥控系统,GCJB-1=船舶监测报警系统,GCJB1=船舶监测报警系统",
    )
    # RAG_USE_PARENT_RETRIEVER：是否启用父子索引（父文档预检索）
    use_parent_retriever = env("RAG_USE_PARENT_RETRIEVER", "true").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    # RAG_PARENT_RETRIEVER_K：父文档预检索数量
    parent_retriever_k = int(env("RAG_PARENT_RETRIEVER_K", "3"))
    # RAG_PARENT_RETRIEVER_ROUTE_MODE：父检索路由模式 hard|soft
    parent_retriever_route_mode = (
        env("RAG_PARENT_RETRIEVER_ROUTE_MODE", "soft").strip().lower()
    )
    # RAG_PARENT_SOURCE_SOFT_BOOST：soft路由时父来源加分系数
    parent_source_soft_boost = float(env("RAG_PARENT_SOURCE_SOFT_BOOST", "0.2"))
    # RAG_PARENT_PROMPT_MODE：父子检索结果合并到Prompt的模式 route|hybrid
    parent_prompt_mode = env("RAG_PARENT_PROMPT_MODE", "hybrid").strip().lower()
    # RAG_PARENT_PROMPT_TOP_K：混合模式下最多加入的父文档片段数
    parent_prompt_top_k = int(env("RAG_PARENT_PROMPT_TOP_K", "2"))

    # 重排序模块
    # RAG_USE_RERANKER：是否启用交叉编码器重排序
    use_reranker = env("RAG_USE_RERANKER", "false").lower() in {"1", "true", "yes", "y"}
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
    decomposer_method = env("RAG_DECOMPOSER_METHOD", "llm")
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
