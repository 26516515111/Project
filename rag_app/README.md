# Ship Equipment Fault Diagnosis RAG (Skeleton)

Minimal, offline-friendly RAG core with placeholder interfaces for Knowledge Graph (KG)
and frontend integration. This is a starter scaffold that you can extend with your
actual data extraction, Neo4j/GraphDB queries, and UI.

## Structure

- rag/: core RAG modules (retrieval, generation, pipeline)
- data/docs/: place unstructured fault cases and manuals (txt/md)
- app_streamlit.py: simple Streamlit UI
- run_cli.py: CLI for quick testing

## Quick start

1. Create a virtual environment and install requirements (`pip install -r rag_app/requirements.txt`).
2. Put some .txt/.md docs in data/docs.
3. Run `python run_cli.py` or `streamlit run app_streamlit.py`.
4. To enable reranking, set `RAG_USE_RERANKER=true` in your environment.

Example:
```bash
export RAG_USE_RERANKER=true
python run_cli.py
```

Note: Reranking requires the `transformers` library. Install it via `pip install transformers torch`. If you encounter NumPy version conflicts, downgrade to `numpy<2` (already handled in `requirements.txt`). The first time you run with reranking enabled, the model will be downloaded (approx. 1GB). Ensure you have sufficient disk space and network access. The reranking step is disabled by default to maintain speed.

## LangGraph 流水线

基于 LangGraph 的 KG 构建入口：`run_langgraph_pipeline.py`。

这套实现只使用 `Project/rag_app` 内部代码，不再依赖仓库根目录 `./scripts`。

KG 目录结构：

- `data/KG/raw/`: 原始文档
- `data/KG/cleaned/`: 清洗结果
- `data/KG/chunks/`: `chunks.json`、`doc_source_map.json`、`chunk_to_kg.json`
- `data/KG/extracted/`: `kg_raw.json`、checkpoint
- `data/KG/delivery/`: `kg_merged.json`、`entity_merge_log.json`、Neo4j CSV、dump、可视化 HTML

流水线步骤：

- `1` 语料清洗
- `2` 文本切块
- `3` KG 三元组抽取
- `4` 实体合并
- `5` Neo4j 导入与导出
- `6` 图谱可视化

示例：

- `python run_langgraph_pipeline.py --dry-run`
- `python run_langgraph_pipeline.py --from 4`
- `python run_langgraph_pipeline.py --only 3 --only-doc 船舶电气设备及系统`
- `python run_langgraph_pipeline.py --skip-neo4j`
- `python run_langgraph_pipeline.py --no-neo4j-import`

## RAG Flow Summary

This project implements a complete RAG pipeline with the following stages:

### 1. Retrieval (Recall)
- **Dense Retrieval**: Vector similarity search using ChromaDB and HuggingFace embeddings.
- **Sparse Retrieval**: Keyword matching using BM25 (or overlap scoring fallback).
- **Hybrid Retrieval**: Combines both methods using weighted scores (vector + BM25).

### 2. Reranking (Re-ranking)
- **Cross-Encoder Reranker**: Optional precision step that reorders retrieved passages using a transformer-based cross-encoder (e.g., BAAI/bge-reranker-v2-m3). This model performs full attention between the query and each document, offering higher accuracy but requiring significant compute and time (especially the first time it downloads the model).
- **Configuration**: Enable via `RAG_USE_RERANKER=true` and optionally set `RAG_RERANKER_TOP_K`.
- **Note**: This step requires the `transformers` and `torch` libraries. The model will be downloaded automatically (approx. 1GB) upon first use.

### 3. Generation
- **LLM Integration**: Uses Ollama for local LLM inference (qwen2.5:3b).
- **Fallback**: Extractive answer generation when LLM is unavailable.

### 4. Knowledge Graph Integration
- **Neo4j Storage**: Stores entities and relationships for semantic querying.
- **KG Retrieval**: Extends context with graph-based relationships.

## Configuration

Edit `rag/config.py` or set environment variables described there.

### Retrieval
- `RAG_TOP_K` (default 5): Number of results to retrieve.
- `RAG_HYBRID_TOP_K` (default 5): Hybrid retrieval top-k before reranking/answering.
- `RAG_BM25_WEIGHT` (default 0.45): Weight for BM25 scores in hybrid retrieval.
- `RAG_VECTOR_WEIGHT` (default 0.55): Weight for vector search scores in hybrid retrieval.

### Embeddings
- `RAG_EMBEDDING_MODEL` (default `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`).

### Ollama (LLM)
- `RAG_OLLAMA_BASE_URL` (default http://localhost:11434)
- `RAG_LLM_MODEL` (default qwen2.5:3b)
- `RAG_LLM_MAX_TOKENS` (default 512)
- `RAG_LLM_TEMPERATURE` (default 0.2)

### Knowledge Graph (Neo4j)
- `NEO4J_URI` (default bolt://localhost:7687)
- `NEO4J_USER` (default neo4j)
- `NEO4J_PASSWORD` (default neo4j)
- `RAG_KG_REL_LIMIT` (default 50)
- `RAG_KG_SCORE_THRESHOLD` (default 0.1)

### Reranker (Optional)
- `RAG_USE_RERANKER` (default false): Enable cross-encoder reranking.
- `RAG_RERANKER_MODEL` (default BAAI/bge-reranker-v2-m3): HuggingFace model name for reranking.
- `RAG_RERANKER_TOP_K` (default 3): Number of results to keep after reranking.

### Query Decomposition
- `RAG_USE_QUERY_DECOMPOSITION` (default true): Enable query decomposition.
- `RAG_DECOMPOSER_METHOD` (default heuristic): `heuristic` or `llm`.
- `RAG_DECOMPOSE_MIN_LENGTH` (default 8): Minimum question length to trigger decomposition.
- `RAG_DECOMPOSE_MAX_SUBQUESTIONS` (default 3): Max sub-questions produced by decomposer.
- `RAG_DECOMPOSE_MAX_SUBQUERIES` (default 4): Max queries used for retrieval (includes original).
- `RAG_DECOMPOSE_PER_QUERY_TOP_K` (default 4): Per-query retrieval size before merging.
- `RAG_DECOMPOSE_LLM_MAX_TOKENS` (default 128): LLM output token limit for decomposition.
- `RAG_DECOMPOSE_LLM_TEMPERATURE` (default 0.1): LLM temperature for decomposition.

### Chunking
- `RAG_CHUNK_SIZE` (default 200)
- `RAG_CHUNK_OVERLAP` (default 20)

## Knowledge graph import规范

数据文件（示例路径）：

- 文档块：`data/docs/chunks.json`
- 文档映射：`data/docs/doc_source_map.json`

实体记录格式（节点）：

```
name
label
description
source_section
```

关系记录格式（扩展三元组）：

```
head
head_label
relation
tail
tail_label
description  # 可选
```

最小导入约束：

- `name` 必填，用于检索
- `label` 用于节点标签
- `relation` 用作关系类型
- `source_section` 用于和文档块建立关联

联合检索：

- 应在节点中保存 `doc_id` 字段（对应 `doc_source_map.json` 的 `doc_id`）
- 查询时会用 `doc_id` 回填 `source` 信息
