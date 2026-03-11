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
1) Create a virtual environment and install requirements.
2) Put some .txt/.md docs in data/docs.
3) Run `python run_cli.py` or `streamlit run app_streamlit.py`.

## Notes
- Uses LangChain + Chroma for vector retrieval.
- Keyword retrieval uses rank_bm25 if installed, otherwise a simple overlap scorer.
- Generation uses local Ollama (qwen2.5vl:7b) if available. Otherwise, it returns
  an extractive answer built from retrieved snippets.
 - Knowledge graph supports extended triplets with head/tail labels and optional
   relation description.

## Configuration
Edit `rag/config.py` or set environment variables described there.

Ollama
- `RAG_OLLAMA_BASE_URL` (default http://localhost:11434)
- `RAG_LLM_MODEL` (default qwen2.5vl:7b)

Neo4j
- `NEO4J_URI` (default bolt://localhost:7687)
- `NEO4J_USER` (default neo4j)
- `NEO4J_PASSWORD` (default neo4j)

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
