import os
from typing import Dict, List

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


def env_domain_rules(key: str, default: str) -> Dict[str, List[str]]:
    raw = env(key, default)
    rules: Dict[str, List[str]] = {}
    for item in raw.split(";"):
        segment = item.strip()
        if not segment or "=" not in segment:
            continue
        domain, keywords_raw = segment.split("=", 1)
        domain = domain.strip()
        if not domain:
            continue
        keywords = [kw.strip().lower() for kw in keywords_raw.split(",") if kw.strip()]
        if keywords:
            rules[domain] = keywords
    return rules


class Settings:
    # Paths
    data_dir = env(
        "RAG_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "docs")
    )
    docs_dir = env(
        "RAG_DOCS_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "docs")
    )
    index_dir = env(
        "RAG_INDEX_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "index")
    )
    parent_index_cache_dir = env(
        "RAG_PARENT_INDEX_CACHE_DIR",
        os.path.join(os.path.dirname(__file__), "..", "data", "index", "parent_child"),
    )
    kg_dir = env(
        "RAG_KG_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "KG")
    )

    # 检索模块
    # RAG_TOP_K：最终用于回答的检索数量
    top_k = int(env("RAG_TOP_K", "6"))
    # RAG_HYBRID_TOP_K：混合检索候选数量（重排序前）
    hybrid_top_k = int(env("RAG_HYBRID_TOP_K", "24"))
    # RAG_BM25_WEIGHT：BM25在混合检索中的权重
    bm25_weight = float(env("RAG_BM25_WEIGHT", "0.60"))
    # RAG_VECTOR_WEIGHT：向量检索在混合检索中的权重
    vector_weight = float(env("RAG_VECTOR_WEIGHT", "0.40"))
    # RAG_QUERY_REWRITE_ENABLED：是否启用查询改写扩召回
    query_rewrite_enabled = env("RAG_QUERY_REWRITE_ENABLED", "true").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    # RAG_QUERY_REWRITE_MAX_VARIANTS：每次查询最多生成的改写条数
    query_rewrite_max_variants = int(env("RAG_QUERY_REWRITE_MAX_VARIANTS", "2"))
    # RAG_RETRIEVAL_FALLBACK_ENABLED：是否启用低召回兜底检索
    retrieval_fallback_enabled = env(
        "RAG_RETRIEVAL_FALLBACK_ENABLED", "true"
    ).lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    # RAG_RETRIEVAL_FALLBACK_MIN_RESULTS：触发兜底的最小候选数量阈值
    retrieval_fallback_min_results = int(env("RAG_RETRIEVAL_FALLBACK_MIN_RESULTS", "8"))
    # RAG_RETRIEVAL_FALLBACK_BM25_ONLY_K：BM25兜底时每个query的检索数量
    retrieval_fallback_bm25_only_k = int(
        env("RAG_RETRIEVAL_FALLBACK_BM25_ONLY_K", "20")
    )
    # RAG_RETRIEVAL_SCORE_THRESHOLD：融合排序后的最小分数阈值（0表示关闭）
    retrieval_score_threshold = float(env("RAG_RETRIEVAL_SCORE_THRESHOLD", "0.35"))
    # RAG_RETRIEVAL_MIN_RESULTS_AFTER_THRESHOLD：阈值过滤后最少保留数量
    retrieval_min_results_after_threshold = int(
        env("RAG_RETRIEVAL_MIN_RESULTS_AFTER_THRESHOLD", "4")
    )
    # RAG_RETRIEVAL_FALLBACK_RELAX_HARD_FILTERS：hard路由候选过少时是否自动回退soft
    retrieval_fallback_relax_hard_filters = env(
        "RAG_RETRIEVAL_FALLBACK_RELAX_HARD_FILTERS", "true"
    ).lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
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
    entity_strict_filter = env("RAG_ENTITY_STRICT_FILTER", "false").lower() in {
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
    neighbor_context_enabled = env("RAG_NEIGHBOR_CONTEXT_ENABLED", "true").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    # RAG_NEIGHBOR_CONTEXT_WINDOW：每个命中chunk向后补充的相邻窗口大小
    neighbor_context_window = int(env("RAG_NEIGHBOR_CONTEXT_WINDOW", "2"))
    # RAG_NEIGHBOR_CONTEXT_PARAM_WINDOW：参数/规格类问题时的相邻补充窗口大小
    neighbor_context_param_window = int(env("RAG_NEIGHBOR_CONTEXT_PARAM_WINDOW", "5"))
    # RAG_NEIGHBOR_CONTEXT_SEED_LIMIT：最多对前N个高分命中chunk做相邻扩展
    neighbor_context_seed_limit = int(env("RAG_NEIGHBOR_CONTEXT_SEED_LIMIT", "3"))
    # RAG_NEIGHBOR_CONTEXT_MAX_CHUNKS：单次查询最多补充的相邻chunk数量
    neighbor_context_max_chunks = int(env("RAG_NEIGHBOR_CONTEXT_MAX_CHUNKS", "10"))
    # RAG_MAX_PER_SOURCE：最终候选中同一source_doc最多保留数量（0表示自动）
    max_per_source = int(env("RAG_MAX_PER_SOURCE", "4"))
    # RAG_DOMAIN_ROUTING_ENABLED：是否启用文档域路由
    domain_routing_enabled = env("RAG_DOMAIN_ROUTING_ENABLED", "true").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    # RAG_DOMAIN_ROUTE_MODE：域路由模式 soft|hard
    domain_route_mode = env("RAG_DOMAIN_ROUTE_MODE", "soft").strip().lower()
    # RAG_DOMAIN_SOFT_BOOST：soft模式下同域chunk加分
    domain_soft_boost = float(env("RAG_DOMAIN_SOFT_BOOST", "0.30"))
    # RAG_DOMAIN_STRICT_MIN_HITS：hard模式启用前，至少需要命中的同域候选数量
    domain_strict_min_hits = int(env("RAG_DOMAIN_STRICT_MIN_HITS", "2"))
    # RAG_DOMAIN_ROUTING_RULES：域分类关键词规则，格式 domain=a,b,c;domain2=d,e
    domain_routing_rules = env_domain_rules(
        "RAG_DOMAIN_ROUTING_RULES",
        (
            "engine_maintenance=发动机,主机,辅机,机舱,维保,维护,检修,保养,润滑,润滑系统,"
            "机油,油压,冷却,冷却液,增压,增压空气冷却器,海水泵,叶轮,扭矩,气缸,电解液,"
            "电池酸,bms,evc,故障降速,故障停车,燃油,通海塞,预滤器;"
            "alarm_system=报警,告警,监测,监控,报警系统,监测报警系统,机舱总线制监测报警系统,"
            "指示灯,报警显示,显示单元,数据采集模块,采集模块,扩展监测单元,监测单元,"
            "延伸监测单元,工作电压,防护等级,ip,rs-485,rs485,总线,gcjb,gcjb0-5,gcjb-0-5,"
            "gcjb24,gcjb-24,gcwj,gcwj-01,bnwas,was-01,gc was-01,gc was-01-03;"
            "hydraulic_system=液压,液压系统,液压油,液压泵,气动泵,泵站,升压,泄压,保压,"
            "换向阀,手动阀,柱塞,油箱,注油,排油,配管,软管,快速接头,漏油,气源,气动马达,"
            "噪声,震动,异味,smp,hydraulic,pump,operation and maintenance manual,"
            "vanessa,tov,三偏心阀,刀闸阀,zp300,clarkson,rosemount,3051s;"
            "navigation_system=航行,驾驶台,值班,驾驶室,航向,操舵,舵机,推进,螺旋桨,错向,"
            "换档,换挡,倒档,紧急变速,控制杆,压浪板,trim,避碰,航行条件"
        ),
    )
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
        "RAG_LLM_API_KEY",
        env(
            "MODELSCOPE_API_KEY",
            env("LLM_API_KEY", "ms-49891fe4-7e57-48d8-836e-1160b8f89ac9"),
        ),
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
    # RAG_KG_TOP_K：KG检索返回条数（独立于回答top_k）
    kg_top_k = int(env("RAG_KG_TOP_K", "12"))
    # RAG_KG_SCORE_THRESHOLD：KG检索的chunk相关性阈值
    kg_score_threshold = float(env("RAG_KG_SCORE_THRESHOLD", "0.1"))
    # RAG_KG_ENTITY_ALIASES：KG实体别名映射，格式 alias=canonical，逗号分隔
    kg_entity_aliases = env_list(
        "RAG_KG_ENTITY_ALIASES",
        "GCWJ-01=船舶监测报警系统,GCWJ01=船舶监测报警系统,GCJB0-5=机舱总线制监测报警系统,GCJB-0-5=机舱总线制监测报警系统,GCJB24=机舱总线制监测报警系统,GCJB-24=机舱总线制监测报警系统,GC WAS-01=驾驶台航行值班报警系统,WAS-01=驾驶台航行值班报警系统,GC WAS-01-03=驾驶台航行值班报警系统,WAS-01-03=驾驶台航行值班报警系统,CDQY2A-PC2-6=CDQY2A-PC2-6 型 船用主柴油机电气遥控系统,CDQY2A-PC2=CDQY2A-PC2-6 型 船用主柴油机电气遥控系统,PC2-6=CDQY2A-PC2-6 型 船用主柴油机电气遥控系统,GCJB-1=船舶监测报警系统,GCJB1=船舶监测报警系统,Clarkson ZP300=manuals-刀闸阀-zp300型-clarkson-zh-zh-cn-5197946,ZP300型刀闸阀=manuals-刀闸阀-zp300型-clarkson-zh-zh-cn-5197946,刀闸阀ZP300=manuals-刀闸阀-zp300型-clarkson-zh-zh-cn-5197946,WREN SMP系列气动泵=Operation and Maintenance Manual for SMP Series Hydraulic Pu,SMP系列气动泵=Operation and Maintenance Manual for SMP Series Hydraulic Pu,SMP气动泵=Operation and Maintenance Manual for SMP Series Hydraulic Pu,WREN气动泵=Operation and Maintenance Manual for SMP Series Hydraulic Pu,配对法兰=manuals-刀闸阀-zp300型-clarkson-zh-zh-cn-5197946,额定压力=Operation and Maintenance Manual for SMP Series Hydraulic Pu",
    )
    # RAG_USE_PARENT_RETRIEVER：是否启用父子索引（父文档预检索）
    use_parent_retriever = env("RAG_USE_PARENT_RETRIEVER", "true").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    # RAG_PARENT_RETRIEVER_K：父文档预检索数量
    parent_retriever_k = int(env("RAG_PARENT_RETRIEVER_K", "6"))
    # RAG_PARENT_RETRIEVER_ROUTE_MODE：父检索路由模式 hard|soft
    parent_retriever_route_mode = (
        env("RAG_PARENT_RETRIEVER_ROUTE_MODE", "soft").strip().lower()
    )
    # RAG_PARENT_SOURCE_SOFT_BOOST：soft路由时父来源加分系数
    parent_source_soft_boost = float(env("RAG_PARENT_SOURCE_SOFT_BOOST", "0.35"))
    # RAG_PARENT_PROMPT_MODE：父子检索结果合并到Prompt的模式 route|hybrid
    parent_prompt_mode = env("RAG_PARENT_PROMPT_MODE", "hybrid").strip().lower()
    # RAG_PARENT_PROMPT_TOP_K：混合模式下最多加入的父文档片段数
    parent_prompt_top_k = int(env("RAG_PARENT_PROMPT_TOP_K", "2"))

    # 重排序模块
    # RAG_USE_RERANKER：是否启用交叉编码器重排序
    use_reranker = env("RAG_USE_RERANKER", "true").lower() in {"1", "true", "yes", "y"}
    # RAG_RERANKER_PROVIDER：重排序模型提供者 local|dashscope
    reranker_provider = env("RAG_RERANKER_PROVIDER", "local").strip().lower()
    # RAG_RERANKER_MODEL：本地重排序模型名称
    reranker_model = env("RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    # RAG_RERANKER_TOP_K：重排序后保留的passage数量
    reranker_top_k = int(env("RAG_RERANKER_TOP_K", "6"))
    # RAG_RERANKER_MAX_INPUT：送入重排序模型的最大候选数量
    reranker_max_input = int(env("RAG_RERANKER_MAX_INPUT", "24"))
    # RAG_RERANKER_SCORE_THRESHOLD：重排序分数阈值（0表示关闭）
    reranker_score_threshold = float(env("RAG_RERANKER_SCORE_THRESHOLD", "0.25"))
    # RAG_RERANKER_MIN_RESULTS_AFTER_THRESHOLD：重排序阈值过滤后最少保留数量
    reranker_min_results_after_threshold = int(
        env("RAG_RERANKER_MIN_RESULTS_AFTER_THRESHOLD", "3")
    )
    # DASHSCOPE_API_KEY：阿里云百炼 DashScope API密钥
    dashscope_api_key = env("DASHSCOPE_API_KEY", "")
    # RAG_DASHSCOPE_RERANKER_MODEL：DashScope重排序模型名称
    dashscope_reranker_model = env("RAG_DASHSCOPE_RERANKER_MODEL", "gte-rerank-v2")

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
    decompose_per_query_top_k = int(env("RAG_DECOMPOSE_PER_QUERY_TOP_K", "6"))
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
