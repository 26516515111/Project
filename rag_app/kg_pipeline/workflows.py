from __future__ import annotations

import csv
import html
import json
import os
import shutil
import subprocess
import time
from collections import defaultdict
from typing import Any, TypedDict

import networkx as nx
import torch
from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from neo4j import GraphDatabase
from pyvis.network import Network

from .config import (
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_QUALITY_MAX_TOKENS,
    LLM_QUALITY_SCORE_THRESHOLD,
    LLM_RETRY_LIMIT,
    LLM_RETRY_BACKOFF,
    LLM_MIN_CHUNK_CHARS,
    LLM_SLEEP_SECONDS,
    LLM_CHECKPOINT_EVERY_CHUNKS,
    LLM_MOCK_ENV,
    LLM_REQUIRE_API_KEY,
    KG_CHUNK_SIZE,
    KG_CHUNK_OVERLAP,
)
from .constants import RELATION_TYPES
from .paths import PipelinePaths
from .steps import (
    LABEL_COLORS,
    SYSTEM_PROMPT,
    LABEL_THRESHOLDS,
    DEFAULT_THRESHOLD,
    DEFAULT_COLOR,
    UnionFind,
    _neo4j_executable,
    _raw_documents,
    _serialize,
    _build_chunk_to_kg,
    _resolve_api_key,
    _resolve_api_base_url,
    _resolve_api_model,
    build_quality_score_request,
    build_context_snapshot,
    clean_text,
    count_chunks,
    extract_heading_path,
    extract_score_json,
    extract_chunk_attachments,
    extract_json_from_response,
    extract_heading_context,
    heuristic_chunk_score,
    infer_semantic_group,
    infer_system_root,
    load_embedding_model,
    merge_chunk_ids,
    merge_doc_ids,
    merge_entities,
    build_semantic_chunks,
    _semantic_family,
    _chunk_filter_reason,
    _chunk_keep_score,
    split_text_by_heading_tags,
    validate_extracted,
    QUALITY_SCORE_PROMPT,
)
from .utils import apply_local_envs, read_json, write_json
from .utils import numbered_path


_CHUNK_FILTER_MODEL: Any | None = None
_CHUNK_FILTER_MODEL_KEY = ""


def _get_chunk_filter_model(paths: PipelinePaths) -> tuple[Any | None, str]:
    global _CHUNK_FILTER_MODEL
    global _CHUNK_FILTER_MODEL_KEY

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = os.environ.get("BGE_MODEL_NAME", "BAAI/bge-m3").strip()
    cache_dir = os.environ.get("BGE_CACHE_DIR", "").strip()
    key = f"{model_name}|{cache_dir}|{device}"
    if _CHUNK_FILTER_MODEL is not None and _CHUNK_FILTER_MODEL_KEY == key:
        return _CHUNK_FILTER_MODEL, device

    model = load_embedding_model(paths, device)
    _CHUNK_FILTER_MODEL = model
    _CHUNK_FILTER_MODEL_KEY = key
    return model, device


class CleanState(TypedDict, total=False):
    paths: PipelinePaths
    raw_files: list[str]
    moved_stale_cleaned: list[str]
    file_cursor: int
    cleaned_files: list[dict]


class ChunkState(TypedDict, total=False):
    paths: PipelinePaths
    cleaned_files: list[str]
    file_cursor: int
    chunks: list[dict]
    doc_source_map: list[dict]
    dropped_chunks: int
    dropped_by_doc: dict[str, int]
    dropped_reason_counts: dict[str, int]


class ExtractState(TypedDict, total=False):
    paths: PipelinePaths
    only_doc_id: str | None
    use_context: bool
    checkpoint_enabled: bool
    logger: Any
    target_chunks: list[dict]
    total_target_chunks: int
    chunk_positions: dict[str, int]
    all_entities: list[dict]
    all_relations: list[dict]
    done_chunk_ids: list[str]
    docs_order: list[str]
    doc_chunk_map: dict[str, list[dict]]
    doc_cursor: int
    chunk_cursor: int
    current_doc_id: str | None
    current_doc_entities: list[dict]
    current_doc_relations: list[dict]
    client: Any
    model: str
    base_url: str
    env_files_loaded: list[str]
    total_filtered_entities: int
    total_filtered_relations: int
    total_cross_chunk_relations: int
    skipped_low_value_chunks: int
    chunks_since_checkpoint: int
    bundled_extract_requests: int
    failed: bool
    error: str | None


class MergeState(TypedDict, total=False):
    paths: PipelinePaths
    raw_entities: list[dict]
    raw_relations: list[dict]
    merge_log: list[dict]
    entities_dedup: list[dict]
    entities_unified: list[dict]
    merged_entities: list[dict]
    clean_relations: list[dict]
    semantic_merges: int
    name_remap: dict[str, str]
    failed: bool
    error: str | None


class Neo4jState(TypedDict, total=False):
    paths: PipelinePaths
    import_to_neo4j: bool
    export_dump: bool
    entities: list[dict]
    relations: list[dict]
    imported: bool
    dumped: bool


class VisualizeState(TypedDict, total=False):
    paths: PipelinePaths
    top_n: int
    filter_label: str | None
    entities: list[dict]
    relations: list[dict]
    output: str
    nodes: int
    edges: int


class _MockMessage:
    def __init__(self, content: str):
        self.content = content


class _MockChoice:
    def __init__(self, content: str):
        self.message = _MockMessage(content)


class _MockResponse:
    def __init__(self, content: str):
        self.choices = [_MockChoice(content)]


class _MockChatCompletions:
    def create(self, **kwargs):
        payload = {
            "entities": [
                {
                    "name": "MOCK_SYSTEM",
                    "label": "System",
                    "description": "mock extracted entity",
                }
            ],
            "relations": [],
        }
        return _MockResponse(json.dumps(payload, ensure_ascii=False))


class _MockChat:
    def __init__(self):
        self.completions = _MockChatCompletions()


class MockOpenAI:
    def __init__(self):
        self.chat = _MockChat()


def _log(state: dict[str, Any], message: str, level: str = "info") -> None:
    logger = state.get("logger")
    if logger is None:
        return
    if level == "error":
        logger.error(message)
    else:
        logger.info(message)


def prepare_clean_node(state: CleanState) -> CleanState:
    raw_files = [str(path.name) for path in _raw_documents(state["paths"])]
    if not raw_files:
        raise RuntimeError("data/KG/raw 中没有可处理的 .md/.txt 文档")
    raw_names = set(raw_files)
    moved_stale_cleaned: list[str] = []
    for pattern in ("*.md", "*.txt"):
        for path in sorted(state["paths"].cleaned_dir.glob(pattern)):
            if path.name in raw_names:
                continue
            target = numbered_path(state["paths"].cleaned_backups_dir / path.name)
            shutil.move(str(path), str(target))
            moved_stale_cleaned.append(target.name)
    return {
        "raw_files": raw_files,
        "moved_stale_cleaned": moved_stale_cleaned,
        "file_cursor": 0,
        "cleaned_files": [],
    }


def _route_clean_file(state: CleanState) -> str:
    return (
        "process"
        if state.get("file_cursor", 0) < len(state.get("raw_files", []))
        else "done"
    )


def clean_file_node(state: CleanState) -> CleanState:
    paths = state["paths"]
    file_cursor = state.get("file_cursor", 0)
    source_name = state["raw_files"][file_cursor]
    source = paths.raw_dir / source_name
    target = paths.cleaned_dir / source_name
    raw_text = source.read_text(encoding="utf-8")
    doc_id = source.stem
    output, saved_images = clean_text(raw_text, paths, doc_id)
    target.write_text(output, encoding="utf-8")
    cleaned_files = list(state.get("cleaned_files", []))
    cleaned_files.append(
        {
            "source": source_name,
            "raw_chars": len(raw_text),
            "cleaned_chars": len(output),
            "images": len(saved_images),
        }
    )
    return {"cleaned_files": cleaned_files, "file_cursor": file_cursor + 1}


def done_clean_node(state: CleanState) -> CleanState:
    return state


def build_clean_graph():
    graph = StateGraph(CleanState)
    graph.add_node("prepare", prepare_clean_node)
    graph.add_node("process", clean_file_node)
    graph.add_node("done", done_clean_node)
    graph.add_edge(START, "prepare")
    graph.add_conditional_edges(
        "prepare", _route_clean_file, {"process": "process", "done": "done"}
    )
    graph.add_conditional_edges(
        "process", _route_clean_file, {"process": "process", "done": "done"}
    )
    graph.add_edge("done", END)
    return graph.compile()


def prepare_chunk_node(state: ChunkState) -> ChunkState:
    paths = state["paths"]
    cleaned_files = [
        str(path.name)
        for path in sorted(paths.cleaned_dir.glob("*.md"))
        + sorted(paths.cleaned_dir.glob("*.txt"))
    ]
    return {
        "cleaned_files": cleaned_files,
        "file_cursor": 0,
        "chunks": [],
        "doc_source_map": [],
        "dropped_chunks": 0,
        "dropped_by_doc": {},
        "dropped_reason_counts": {},
    }


def _route_chunk_file(state: ChunkState) -> str:
    return (
        "process"
        if state.get("file_cursor", 0) < len(state.get("cleaned_files", []))
        else "persist"
    )


def chunk_file_node(state: ChunkState) -> ChunkState:
    paths = state["paths"]
    file_cursor = state.get("file_cursor", 0)
    source_name = state["cleaned_files"][file_cursor]
    source = paths.cleaned_dir / source_name
    text = source.read_text(encoding="utf-8")
    semantic_chunks = build_semantic_chunks(
        text,
        chunk_size=KG_CHUNK_SIZE,
        chunk_overlap=KG_CHUNK_OVERLAP,
    )
    chunks = list(state.get("chunks", []))
    doc_source_map = list(state.get("doc_source_map", []))
    dropped_chunks = int(state.get("dropped_chunks", 0))
    dropped_by_doc = dict(state.get("dropped_by_doc", {}))
    dropped_reason_counts = dict(state.get("dropped_reason_counts", {}))
    doc_id = source.stem
    ranked_chunks: list[tuple[int, dict[str, Any]]] = []
    for chunk in semantic_chunks:
        fast_score = _chunk_keep_score(chunk)
        chunk["pre_prune_score"] = fast_score
        reason = _chunk_filter_reason(paths, chunk, fast_score)
        if reason:
            dropped_chunks += 1
            dropped_by_doc[doc_id] = dropped_by_doc.get(doc_id, 0) + 1
            dropped_reason_counts[reason] = dropped_reason_counts.get(reason, 0) + 1
            continue
        ranked_chunks.append((fast_score, chunk))

    enable_embed_filter = os.environ.get("CHUNK_EMBED_FILTER", "0") == "1"
    if enable_embed_filter and ranked_chunks:
        try:
            model, device = _get_chunk_filter_model(paths)
            if model is not None:
                query = os.environ.get(
                    "CHUNK_EMBED_QUERY",
                    "hydraulic system maintenance troubleshooting pressure valve pump actuator fault diagnosis repair procedure installation",
                ).strip()
                threshold = float(os.environ.get("CHUNK_EMBED_THRESHOLD", "0.36"))
                min_keep = int(os.environ.get("CHUNK_EMBED_MIN_KEEP", "110"))

                texts = [
                    str(item[1].get("text", ""))[:1600]
                    for item in ranked_chunks
                ]
                query_emb = model.encode(
                    [query],
                    convert_to_tensor=True,
                    device=device,
                )
                chunk_embs = model.encode(
                    texts,
                    convert_to_tensor=True,
                    device=device,
                )
                query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)
                chunk_embs = torch.nn.functional.normalize(chunk_embs, p=2, dim=1)
                similarities = torch.matmul(chunk_embs, query_emb.T).squeeze(1).tolist()

                kept: list[tuple[int, dict[str, Any]]] = []
                sim_ranked: list[tuple[float, tuple[int, dict[str, Any]]]] = []
                for pair, sim in zip(ranked_chunks, similarities):
                    pair[1]["embed_score"] = float(sim)
                    sim_ranked.append((float(sim), pair))
                    if sim >= threshold:
                        kept.append(pair)

                keep_floor = max(0, min(min_keep, len(ranked_chunks)))
                if len(kept) < keep_floor:
                    sim_ranked.sort(key=lambda item: item[0], reverse=True)
                    kept = [item[1] for item in sim_ranked[:keep_floor]]

                if len(kept) < len(ranked_chunks):
                    pruned_by_embed = len(ranked_chunks) - len(kept)
                    dropped_chunks += pruned_by_embed
                    dropped_by_doc[doc_id] = dropped_by_doc.get(doc_id, 0) + pruned_by_embed
                    ranked_chunks = kept
        except Exception:
            # Embedding filter is optional; fallback to heuristic-only pruning on any runtime issue.
            pass

    original_ranked_count = len(ranked_chunks)
    if ranked_chunks:
        grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for score, chunk in ranked_chunks:
            root = chunk.get("system_root") or infer_system_root(
                chunk.get("heading_path", []),
                chunk.get("text", ""),
            )
            grouped[str(root or "global")].append((score, chunk))

        if original_ranked_count > 500:
            target_keep = min(180, int(original_ranked_count * 0.28))
        elif original_ranked_count > 320:
            target_keep = min(170, int(original_ranked_count * 0.35))
        elif original_ranked_count > 220:
            target_keep = min(150, int(original_ranked_count * 0.45))
        else:
            target_keep = original_ranked_count

        if target_keep < original_ranked_count:
            selected: list[tuple[int, dict[str, Any]]] = []
            leftovers: list[tuple[int, dict[str, Any]]] = []
            roots_count = max(1, len(grouped))
            root_soft_cap = max(8, int(target_keep / roots_count * 1.8))
            for root_items in grouped.values():
                ordered = sorted(root_items, key=lambda item: item[0], reverse=True)
                quota = min(
                    len(ordered),
                    max(4, min(root_soft_cap, int(len(ordered) * 0.55) + 2)),
                )
                selected.extend(ordered[:quota])
                leftovers.extend(ordered[quota:])

            if len(selected) > target_keep:
                selected = sorted(selected, key=lambda item: item[0], reverse=True)[:target_keep]
            elif len(selected) < target_keep and leftovers:
                need = target_keep - len(selected)
                selected.extend(sorted(leftovers, key=lambda item: item[0], reverse=True)[:need])

            ranked_chunks = sorted(selected, key=lambda item: item[1].get("chunk_index", 0))

    if len(ranked_chunks) < original_ranked_count:
        pruned_by_cap = original_ranked_count - len(ranked_chunks)
        dropped_chunks += pruned_by_cap
        dropped_by_doc[doc_id] = dropped_by_doc.get(doc_id, 0) + pruned_by_cap

    semantic_chunks = [item[1] for item in ranked_chunks]

    for index, chunk in enumerate(semantic_chunks):
        normalized_chunk_text, attachments = extract_chunk_attachments(chunk["text"])
        chunks.append(
            {
                "chunk_id": f"{doc_id}::chunk_{index}",
                "doc_id": doc_id,
                "chunk_index": index,
                "text": normalized_chunk_text,
                "source": source.name,
                "char_count": len(normalized_chunk_text),
                "heading_context": chunk.get("heading_context") or extract_heading_context(normalized_chunk_text),
                "heading_path": chunk.get("heading_path", extract_heading_path(normalized_chunk_text)),
                "semantic_group": chunk.get("semantic_group", infer_semantic_group(normalized_chunk_text, extract_heading_path(normalized_chunk_text))),
                "system_root": chunk.get("system_root", infer_system_root(chunk.get("heading_path", []), normalized_chunk_text)),
                "pre_prune_score": int(chunk.get("pre_prune_score") or _chunk_keep_score(chunk)),
                "embed_score": float(chunk.get("embed_score", 0.0)),
                "attachments": attachments,
            }
        )
    doc_source_map.append(
        {
            "doc_id": doc_id,
            "source": source.name,
            "path": f"data/KG/raw/{source.name}",
            "num_chunks": len(semantic_chunks),
            "dropped_chunks": dropped_by_doc.get(doc_id, 0),
        }
    )
    return {
        "chunks": chunks,
        "doc_source_map": doc_source_map,
        "dropped_chunks": dropped_chunks,
        "dropped_by_doc": dropped_by_doc,
        "dropped_reason_counts": dropped_reason_counts,
        "file_cursor": file_cursor + 1,
    }


def persist_chunk_node(state: ChunkState) -> ChunkState:
    chunk_ids = [item["chunk_id"] for item in state.get("chunks", [])]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise RuntimeError("chunk_id 不唯一")
    write_json(state["paths"].chunks_path, state.get("chunks", []))
    write_json(state["paths"].doc_map_path, state.get("doc_source_map", []))
    return state


def build_chunk_graph():
    graph = StateGraph(ChunkState)
    graph.add_node("prepare", prepare_chunk_node)
    graph.add_node("process", chunk_file_node)
    graph.add_node("persist", persist_chunk_node)
    graph.add_edge(START, "prepare")
    graph.add_conditional_edges(
        "prepare", _route_chunk_file, {"process": "process", "persist": "persist"}
    )
    graph.add_conditional_edges(
        "process", _route_chunk_file, {"process": "process", "persist": "persist"}
    )
    graph.add_edge("persist", END)
    return graph.compile()


def prepare_neo4j_node(state: Neo4jState) -> Neo4jState:
    paths = state["paths"]
    apply_local_envs(paths.env_paths)
    kg = read_json(paths.kg_merged_path, {"entities": [], "relations": []})
    return {
        "entities": kg.get("entities", []),
        "relations": kg.get("relations", []),
        "imported": False,
        "dumped": False,
    }


def write_neo4j_csv_node(state: Neo4jState) -> Neo4jState:
    paths = state["paths"]
    with paths.neo4j_entities_csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "entity_id",
                "name",
                "label",
                "description",
                "doc_id",
                "chunk_id",
                "all_labels",
                "source",
            ]
        )
        for entity in state.get("entities", []):
            writer.writerow(
                [
                    entity.get("entity_id", ""),
                    entity["name"],
                    entity.get("label", ""),
                    entity.get("description", ""),
                    _serialize(entity.get("doc_id", "")),
                    _serialize(entity.get("chunk_id", "")),
                    _serialize(entity.get("all_labels", [])),
                    entity.get("source", ""),
                ]
            )
    with paths.neo4j_relations_csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "rel_id",
                "head",
                "head_label",
                "relation",
                "tail",
                "tail_label",
                "description",
                "doc_id",
                "chunk_id",
            ]
        )
        for relation in state.get("relations", []):
            writer.writerow(
                [
                    relation.get("rel_id", ""),
                    relation["head"],
                    relation.get("head_label", ""),
                    relation["relation"],
                    relation["tail"],
                    relation.get("tail_label", ""),
                    relation.get("description", ""),
                    _serialize(relation.get("doc_id", "")),
                    _serialize(relation.get("chunk_id", "")),
                ]
            )
    return {
        "imported": state.get("imported", False),
        "dumped": state.get("dumped", False),
    }


def _route_after_neo4j_csv(state: Neo4jState) -> str:
    return "import" if state.get("import_to_neo4j", True) else "done"


def import_neo4j_node(state: Neo4jState) -> Neo4jState:
    neo4j_password = os.environ.get(
        "NEO4J_PASSWORD", os.environ.get("NEO4J_PASS", "")
    ).strip()
    if not neo4j_password:
        raise RuntimeError("NEO4J_PASSWORD/NEO4J_PASS 未设置")
    driver = GraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.environ.get("NEO4J_USER", "neo4j"), neo4j_password),
    )
    with driver.session(database=os.environ.get("NEO4J_DB", "neo4j")) as session:
        session.run("CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.name)").consume()
        tx = session.begin_transaction()
        try:
            tx.run("MATCH (n) DETACH DELETE n")
            tx.run(
                """
                UNWIND $rows AS row
                MERGE (n:Entity {name: row.name})
                SET n.entity_id = row.entity_id,
                    n.label = row.label,
                    n.description = row.description,
                    n.doc_id = row.doc_id,
                    n.chunk_id = row.chunk_id,
                    n.all_labels = row.all_labels,
                    n.source = row.source
                """,
                rows=[
                    {
                        "entity_id": entity.get("entity_id", ""),
                        "name": entity["name"],
                        "label": entity.get("label", ""),
                        "description": entity.get("description", ""),
                        "doc_id": _serialize(entity.get("doc_id", "")),
                        "chunk_id": _serialize(entity.get("chunk_id", "")),
                        "all_labels": _serialize(entity.get("all_labels", [])),
                        "source": entity.get("source", ""),
                    }
                    for entity in state.get("entities", [])
                ],
            )
            grouped_relations: dict[str, list[dict]] = defaultdict(list)
            for relation in state.get("relations", []):
                grouped_relations[relation["relation"]].append(
                    {
                        "rel_id": relation.get("rel_id", ""),
                        "head": relation["head"],
                        "tail": relation["tail"],
                        "description": relation.get("description", ""),
                        "doc_id": _serialize(relation.get("doc_id", "")),
                        "chunk_id": _serialize(relation.get("chunk_id", "")),
                    }
                )
            for relation_type, rows in grouped_relations.items():
                if relation_type not in RELATION_TYPES:
                    continue
                tx.run(
                    f"""
                    UNWIND $rows AS row
                    MATCH (h:Entity {{name: row.head}})
                    MATCH (t:Entity {{name: row.tail}})
                    CREATE (h)-[r:`{relation_type}`]->(t)
                    SET r.rel_id = row.rel_id,
                        r.description = row.description,
                        r.doc_id = row.doc_id,
                        r.chunk_id = row.chunk_id
                    """,
                    rows=rows,
                )
            tx.commit()
        except Exception:
            tx.rollback()
            raise
        finally:
            driver.close()
    return {"imported": True}


def _route_after_neo4j_import(state: Neo4jState) -> str:
    return "dump" if state.get("export_dump", True) else "done"


def dump_neo4j_node(state: Neo4jState) -> Neo4jState:
    env = os.environ.copy()
    java_home = os.environ.get("JAVA_HOME", "").strip()
    if java_home:
        env["JAVA_HOME"] = java_home
    neo4j_cmd = _neo4j_executable("neo4j")
    neo4j_admin_cmd = _neo4j_executable("neo4j-admin")
    dump_target = state["paths"].neo4j_dump_path
    if dump_target.exists():
        backup_target = numbered_path(
            state["paths"].delivery_backups_dir / dump_target.name
        )
        backup_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(dump_target), str(backup_target))
    subprocess.run([neo4j_cmd, "stop"], check=False, env=env)
    time.sleep(3)
    try:
        subprocess.run(
            [
                neo4j_admin_cmd,
                "database",
                "dump",
                "neo4j",
                f"--to-path={state['paths'].delivery_dir}",
            ],
            check=True,
            env=env,
        )
    finally:
        subprocess.run([neo4j_cmd, "start"], check=False, env=env)
    return {"dumped": True}


def done_neo4j_node(state: Neo4jState) -> Neo4jState:
    return state


def build_neo4j_graph():
    graph = StateGraph(Neo4jState)
    graph.add_node("prepare", prepare_neo4j_node)
    graph.add_node("write_csv", write_neo4j_csv_node)
    graph.add_node("import", import_neo4j_node)
    graph.add_node("dump", dump_neo4j_node)
    graph.add_node("done", done_neo4j_node)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "write_csv")
    graph.add_conditional_edges(
        "write_csv", _route_after_neo4j_csv, {"import": "import", "done": "done"}
    )
    graph.add_conditional_edges(
        "import", _route_after_neo4j_import, {"dump": "dump", "done": "done"}
    )
    graph.add_edge("dump", "done")
    graph.add_edge("done", END)
    return graph.compile()


def prepare_visualize_node(state: VisualizeState) -> VisualizeState:
    kg = read_json(state["paths"].kg_merged_path, {"entities": [], "relations": []})
    entities = kg.get("entities", [])
    relations = kg.get("relations", [])
    filter_label = state.get("filter_label")
    if filter_label:
        entities = [
            entity for entity in entities if entity.get("label") == filter_label
        ]
        valid_names = {entity["name"] for entity in entities}
        relations = [
            relation
            for relation in relations
            if relation["head"] in valid_names and relation["tail"] in valid_names
        ]
    return {"entities": entities, "relations": relations}


def build_visual_graph_node(state: VisualizeState) -> VisualizeState:
    entity_info = {entity["name"]: entity for entity in state.get("entities", [])}
    graph_obj = nx.DiGraph()
    graph_obj.add_nodes_from(entity_info.keys())
    for relation in state.get("relations", []):
        if relation["head"] in entity_info and relation["tail"] in entity_info:
            graph_obj.add_edge(
                relation["head"],
                relation["tail"],
                relation=relation["relation"],
                description=relation.get("description", ""),
            )
    degrees = dict(graph_obj.degree())
    top_n = state.get("top_n", 300)
    if graph_obj.number_of_nodes() > top_n:
        top_nodes = sorted(degrees, key=lambda name: degrees[name], reverse=True)[
            :top_n
        ]
        graph_obj = graph_obj.subgraph(top_nodes).copy()
        entity_info = {name: entity_info[name] for name in graph_obj.nodes()}
        degrees = dict(graph_obj.degree())

    net = Network(
        height="900px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#333333",
        directed=True,
        notebook=False,
    )
    net.barnes_hut(
        gravity=-3000,
        central_gravity=0.3,
        spring_length=200,
        spring_strength=0.01,
        damping=0.09,
    )
    for name in graph_obj.nodes():
        entity = entity_info[name]
        degree = degrees.get(name, 1)
        title = (
            f"<b>{html.escape(name)}</b><br>"
            f"标签: {html.escape(entity.get('label', ''))}<br>"
            f"描述: {html.escape(entity.get('description', '')[:200])}<br>"
            f"doc_id: {html.escape(_serialize(entity.get('doc_id', '')))}<br>"
            f"chunk_id: {html.escape(_serialize(entity.get('chunk_id', '')))}<br>"
            f"度数: {degree}"
        )
        net.add_node(
            name,
            label=name,
            color=LABEL_COLORS.get(entity.get("label", ""), DEFAULT_COLOR),
            size=max(8, min(40, 8 + degree * 3)),
            title=title,
        )
    for head, tail, data in graph_obj.edges(data=True):
        rel_type = data.get("relation", "")
        desc = data.get("description", "")
        net.add_edge(
            head,
            tail,
            title=f"{rel_type}: {desc}" if desc else rel_type,
            label=rel_type,
            arrows="to",
        )
    net.save_graph(str(state["paths"].visualization_path))
    return {
        "nodes": graph_obj.number_of_nodes(),
        "edges": graph_obj.number_of_edges(),
        "output": str(state["paths"].visualization_path),
    }


def persist_visualize_node(state: VisualizeState) -> VisualizeState:
    legend = "".join(
        (
            f'<div style="margin:2px 0"><span style="display:inline-block;width:14px;'
            f'height:14px;background:{color};border-radius:50%;margin-right:6px;vertical-align:middle"></span>{label}</div>'
        )
        for label, color in LABEL_COLORS.items()
    )
    content = state["paths"].visualization_path.read_text(encoding="utf-8")
    state["paths"].visualization_path.write_text(
        content.replace(
            "</body>",
            (
                '<div style="position:fixed;top:10px;right:10px;background:rgba(255,255,255,0.95);'
                "padding:12px 16px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.15);"
                'font-family:Arial,sans-serif;font-size:13px;z-index:9999">'
                '<div style="font-weight:bold;margin-bottom:6px;font-size:14px">图例</div>'
                f"{legend}</div></body>"
            ),
        ),
        encoding="utf-8",
    )
    return state


def build_visualize_graph():
    graph = StateGraph(VisualizeState)
    graph.add_node("prepare", prepare_visualize_node)
    graph.add_node("build", build_visual_graph_node)
    graph.add_node("persist", persist_visualize_node)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "build")
    graph.add_edge("build", "persist")
    graph.add_edge("persist", END)
    return graph.compile()


def _extract_route_after_prepare(state: ExtractState) -> str:
    return "fail" if state.get("failed") else "load_doc"


def _extract_route_after_load_doc(state: ExtractState) -> str:
    if state.get("failed"):
        return "fail"
    if state.get("current_doc_id") is None:
        return "finalize"
    return "process_chunk"


def _extract_route_after_chunk(state: ExtractState) -> str:
    if state.get("failed"):
        return "fail"
    current_doc_id = state.get("current_doc_id")
    if current_doc_id is None:
        return "finalize"
    doc_chunk_map = state.get("doc_chunk_map", {})
    chunk_cursor = state.get("chunk_cursor", 0)
    if chunk_cursor >= len(doc_chunk_map.get(current_doc_id, [])):
        return "load_doc"
    return "process_chunk"


def _same_root_heading(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_path = left.get("heading_path") or []
    right_path = right.get("heading_path") or []
    return bool(left_path) and bool(right_path) and left_path[:1] == right_path[:1]


def _same_semantic_family(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return _semantic_family(left.get("semantic_group", "")) == _semantic_family(
        right.get("semantic_group", "")
    )


def _build_extract_bundle(
    doc_chunks: list[dict[str, Any]],
    start_index: int,
    done_chunk_ids: set[str],
) -> list[dict[str, Any]]:
    bundle_small_chars = int(os.environ.get("KG_EXTRACT_BUNDLE_SMALL_CHARS", "1600"))
    bundle_max_chars = int(os.environ.get("KG_EXTRACT_BUNDLE_MAX_CHARS", "5200"))
    bundle_max_chunks = int(os.environ.get("KG_EXTRACT_BUNDLE_MAX_CHUNKS", "4"))

    first = doc_chunks[start_index]
    first_text = str(first.get("text", "")).strip()
    if len(first_text) >= bundle_small_chars:
        return [first]

    bundle = [first]
    total_chars = len(first_text)

    for index in range(start_index + 1, len(doc_chunks)):
        if len(bundle) >= bundle_max_chunks:
            break
        candidate = doc_chunks[index]
        candidate_id = candidate.get("chunk_id", "")
        candidate_text = str(candidate.get("text", "")).strip()
        if candidate_id in done_chunk_ids or len(candidate_text) < LLM_MIN_CHUNK_CHARS:
            break
        if len(candidate_text) >= bundle_small_chars:
            break
        if not _same_root_heading(first, candidate):
            break
        if not _same_semantic_family(first, candidate):
            break
        if total_chars + 2 + len(candidate_text) > bundle_max_chars:
            break
        bundle.append(candidate)
        total_chars += 2 + len(candidate_text)

    return bundle


def _build_bundle_prompt(
    bundle: list[dict[str, Any]],
    doc_entities: list[dict[str, Any]],
    doc_relations: list[dict[str, Any]],
    use_context: bool,
) -> str:
    user_parts = ["请从以下文本中抽取知识图谱三元组："]
    if use_context and doc_entities:
        user_parts.append(build_context_snapshot(doc_entities, doc_relations))
    if len(bundle) == 1:
        chunk = bundle[0]
        if chunk.get("heading_context"):
            user_parts.append(f"所属章节：{chunk['heading_context']}")
        user_parts.append(f"待抽取文本：\n{chunk['text']}")
        return "\n\n".join(user_parts)

    user_parts.append(
        "以下是同一文档、同一语义系群下的连续片段。请联合理解后抽取，但不要编造跨片段关系。"
    )
    for idx, chunk in enumerate(bundle, start=1):
        header_parts = [f"片段{idx}", f"chunk_id={chunk.get('chunk_id', '')}"]
        if chunk.get("heading_context"):
            header_parts.append(f"heading={chunk['heading_context']}")
        header_parts.append(f"semantic_group={chunk.get('semantic_group', '')}")
        user_parts.append(" | ".join(header_parts))
        user_parts.append(str(chunk.get("text", "")))
    return "\n\n".join(user_parts)


def prepare_extract_node(state: ExtractState) -> ExtractState:
    paths = state["paths"]
    apply_local_envs(paths.env_paths)
    all_chunks = read_json(paths.chunks_path, [])
    only_doc_id = state.get("only_doc_id")
    target_chunks = (
        [chunk for chunk in all_chunks if chunk["doc_id"] == only_doc_id]
        if only_doc_id
        else all_chunks
    )
    if only_doc_id and not target_chunks:
        return {"failed": True, "error": f"未找到 doc_id={only_doc_id} 的 chunks"}

    if only_doc_id and paths.kg_raw_path.exists():
        existing = read_json(paths.kg_raw_path, {"entities": [], "relations": []})
        all_entities = [
            entity
            for entity in existing.get("entities", [])
            if entity.get("doc_id") != only_doc_id
        ]
        all_relations = [
            relation
            for relation in existing.get("relations", [])
            if relation.get("doc_id") != only_doc_id
        ]
    else:
        all_entities = []
        all_relations = []

    done_chunk_ids: set[str] = set()
    if state.get("checkpoint_enabled", True) and paths.checkpoint_path.exists():
        checkpoint = read_json(
            paths.checkpoint_path,
            {"done_chunk_ids": [], "entities": [], "relations": []},
        )
        done_chunk_ids = set(checkpoint.get("done_chunk_ids", []))
        if not only_doc_id:
            all_entities = checkpoint.get("entities", [])
            all_relations = checkpoint.get("relations", [])

    docs_order: list[str] = []
    doc_chunk_map: dict[str, list[dict]] = defaultdict(list)
    chunk_positions: dict[str, int] = {}
    for index, chunk in enumerate(target_chunks, start=1):
        if chunk["doc_id"] not in doc_chunk_map:
            docs_order.append(chunk["doc_id"])
        doc_chunk_map[chunk["doc_id"]].append(chunk)
        chunk_positions[chunk["chunk_id"]] = index

    if os.environ.get(LLM_MOCK_ENV, "0") == "1":
        client = MockOpenAI()
        base_url = "mock://local"
        model = "mock"
    else:
        api_key = _resolve_api_key(paths)
        base_url = _resolve_api_base_url(paths)
        model = _resolve_api_model(paths)
        client = OpenAI(
            api_key=api_key or "ollama",
            base_url=base_url,
        )
    _log(
        state,
        "Step 3 准备完成 | docs=%s | chunks=%s | checkpoint_enabled=%s | resumed_done_chunks=%s | base_url=%s | model=%s"
        % (
            len(docs_order),
            len(target_chunks),
            state.get("checkpoint_enabled", True),
            len(done_chunk_ids),
            base_url,
            model,
        ),
    )
    return {
        "target_chunks": target_chunks,
        "total_target_chunks": len(target_chunks),
        "chunk_positions": chunk_positions,
        "all_entities": all_entities,
        "all_relations": all_relations,
        "done_chunk_ids": sorted(done_chunk_ids),
        "docs_order": docs_order,
        "doc_chunk_map": dict(doc_chunk_map),
        "doc_cursor": 0,
        "chunk_cursor": 0,
        "current_doc_id": None,
        "current_doc_entities": [],
        "current_doc_relations": [],
        "client": client,
        "model": model,
        "base_url": base_url,
        "env_files_loaded": [
            str(path)
            for path in paths.env_paths
            if path.exists()
        ],
        "total_filtered_entities": 0,
        "total_filtered_relations": 0,
        "total_cross_chunk_relations": 0,
        "skipped_low_value_chunks": 0,
        "chunks_since_checkpoint": 0,
        "bundled_extract_requests": 0,
        "failed": False,
        "error": None,
    }


def load_extract_doc_node(state: ExtractState) -> ExtractState:
    docs_order = state.get("docs_order", [])
    doc_cursor = state.get("doc_cursor", 0)
    if doc_cursor >= len(docs_order):
        return {"current_doc_id": None}
    current_doc_id = docs_order[doc_cursor]
    _log(
        state,
        f"Step 3 切换文档 | doc_id={current_doc_id} | doc_index={doc_cursor + 1}/{len(docs_order)} | doc_chunks={len(state.get('doc_chunk_map', {}).get(current_doc_id, []))}",
    )
    return {
        "current_doc_id": current_doc_id,
        "doc_cursor": doc_cursor + 1,
        "chunk_cursor": 0,
        "current_doc_entities": [],
        "current_doc_relations": [],
    }


def _save_extract_checkpoint(state: ExtractState) -> None:
    if not state.get("checkpoint_enabled", True):
        return
    paths = state["paths"]
    write_json(
        paths.checkpoint_path,
        {
            "done_chunk_ids": sorted(state.get("done_chunk_ids", [])),
            "entities": state.get("all_entities", []),
            "relations": state.get("all_relations", []),
        },
    )
    _log(
        state,
        "Step 3 保存 checkpoint | path=%s | done_chunks=%s | entities=%s | relations=%s"
        % (
            paths.checkpoint_path,
            len(state.get("done_chunk_ids", [])),
            len(state.get("all_entities", [])),
            len(state.get("all_relations", [])),
        ),
    )


def process_extract_chunk_node(state: ExtractState) -> ExtractState:
    current_doc_id = state.get("current_doc_id")
    if current_doc_id is None:
        return {}

    chunk_cursor = state.get("chunk_cursor", 0)
    doc_chunks = state["doc_chunk_map"][current_doc_id]
    chunk = doc_chunks[chunk_cursor]
    done_chunk_ids = set(state.get("done_chunk_ids", []))
    all_entities = list(state.get("all_entities", []))
    all_relations = list(state.get("all_relations", []))
    doc_entities = list(state.get("current_doc_entities", []))
    doc_relations = list(state.get("current_doc_relations", []))
    total_filtered_entities = state.get("total_filtered_entities", 0)
    total_filtered_relations = state.get("total_filtered_relations", 0)
    total_cross_chunk_relations = state.get("total_cross_chunk_relations", 0)
    skipped_low_value_chunks = state.get("skipped_low_value_chunks", 0)
    chunks_since_checkpoint = state.get("chunks_since_checkpoint", 0)
    bundled_extract_requests = state.get("bundled_extract_requests", 0)

    chunk_id = chunk["chunk_id"]
    chunk_text_chars = len(chunk["text"])
    global_index = state.get("chunk_positions", {}).get(chunk_id, chunk_cursor + 1)
    total_target_chunks = state.get("total_target_chunks", 0)
    already_done = chunk_id in done_chunk_ids
    too_short = len(chunk["text"].strip()) < LLM_MIN_CHUNK_CHARS
    if already_done or too_short:
        done_chunk_ids.add(chunk_id)
        _log(
            state,
            "Step 3 跳过 chunk | chunk=%s/%s | doc_id=%s | chunk_id=%s | text_chars=%s | reason=%s"
            % (
                global_index,
                total_target_chunks,
                current_doc_id,
                chunk_id,
                chunk_text_chars,
                "already_done" if already_done else "too_short",
            ),
        )
        return {
            "done_chunk_ids": sorted(done_chunk_ids),
            "chunk_cursor": chunk_cursor + 1,
        }

    bundle = _build_extract_bundle(doc_chunks, chunk_cursor, done_chunk_ids)
    bundle_chunk_ids = [item["chunk_id"] for item in bundle]
    bundle_chunk_id = merge_chunk_ids(bundle_chunk_ids)
    bundle_text_chars = sum(len(str(item.get("text", ""))) for item in bundle)
    user_content = _build_bundle_prompt(
        bundle,
        doc_entities,
        doc_relations,
        state.get("use_context", True),
    )
    doc_entity_names = {entity["name"] for entity in doc_entities}
    _log(
        state,
        "Step 3 请求 chunk%s | chunk=%s/%s | doc_id=%s | doc_chunk=%s/%s | chunk_ids=%s | text_chars=%s | request_chars=%s | context_entities=%s | context_relations=%s | semantic_group=%s"
        % (
            "(batched)" if len(bundle) > 1 else "",
            global_index,
            total_target_chunks,
            current_doc_id,
            chunk_cursor + 1,
            len(doc_chunks),
            bundle_chunk_ids,
            bundle_text_chars,
            len(user_content),
            len(doc_entities),
            len(doc_relations),
            chunk.get("semantic_group", ""),
        ),
    )

    quality_chunk = dict(chunk)
    if len(bundle) > 1:
        quality_chunk["text"] = "\n\n".join(str(item.get("text", "")) for item in bundle)
        quality_chunk["heading_context"] = " / ".join(
            [
                str(item.get("heading_context", "")).strip()
                for item in bundle
                if str(item.get("heading_context", "")).strip()
            ]
        )
        quality_chunk["semantic_group"] = chunk.get("semantic_group", "")
    quality_payload = heuristic_chunk_score(quality_chunk)
    if os.environ.get(LLM_MOCK_ENV, "0") != "1":
        quality_request = build_quality_score_request(quality_chunk)
        try:
            quality_response = state["client"].chat.completions.create(
                model=state["model"],
                messages=[
                    {"role": "system", "content": QUALITY_SCORE_PROMPT},
                    {"role": "user", "content": quality_request},
                ],
                temperature=0,
                max_tokens=LLM_QUALITY_MAX_TOKENS,
            )
            quality_payload = extract_score_json(
                quality_response.choices[0].message.content or ""
            )
        except Exception as exc:
            _log(
                state,
                "Step 3 chunk 评分失败，回退 heuristic | chunk=%s/%s | chunk_id=%s | error=%s"
                % (
                    global_index,
                    total_target_chunks,
                    bundle_chunk_ids,
                    f"{type(exc).__name__}: {exc}",
                ),
                "error",
            )

    _log(
        state,
        "Step 3 chunk 评分 | chunk=%s/%s | chunk_ids=%s | score=%s | decision=%s | category=%s | reasons=%s"
        % (
            global_index,
            total_target_chunks,
            bundle_chunk_ids,
            quality_payload.get("score", 0),
            quality_payload.get("decision", ""),
            quality_payload.get("category", ""),
            "|".join(quality_payload.get("reasons", [])),
        ),
    )
    if quality_payload.get("decision") != "extract":
        done_chunk_ids.update(bundle_chunk_ids)
        skipped_low_value_chunks += len(bundle)
        _log(
            state,
            "Step 3 跳过低价值 chunk%s | chunk=%s/%s | chunk_ids=%s | score=%s | threshold=%s"
            % (
                "(batched)" if len(bundle) > 1 else "",
                global_index,
                total_target_chunks,
                bundle_chunk_ids,
                quality_payload.get("score", 0),
                LLM_QUALITY_SCORE_THRESHOLD,
            ),
        )
        return {
            "done_chunk_ids": sorted(done_chunk_ids),
            "chunk_cursor": chunk_cursor + len(bundle),
            "skipped_low_value_chunks": skipped_low_value_chunks,
        }

    retries = 0
    while retries < LLM_RETRY_LIMIT:
        try:
            response = state["client"].chat.completions.create(
                model=state["model"],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
            result = extract_json_from_response(
                response.choices[0].message.content or ""
            )
            entities, relations = validate_extracted(
                result, doc_entity_names, state.get("use_context", True)
            )
            total_filtered_entities += len(result.get("entities", [])) - len(entities)
            total_filtered_relations += len(result.get("relations", [])) - len(
                relations
            )
            new_names = {entity["name"] for entity in entities}
            total_cross_chunk_relations += len(
                [
                    rel
                    for rel in relations
                    if rel["head"] not in new_names or rel["tail"] not in new_names
                ]
            )
            for entity in entities:
                entity["doc_id"] = current_doc_id
                entity["chunk_id"] = bundle_chunk_id
                entity["source"] = chunk.get("source", "")
            for relation in relations:
                relation["doc_id"] = current_doc_id
                relation["chunk_id"] = bundle_chunk_id
            all_entities.extend(entities)
            all_relations.extend(relations)
            doc_entities.extend(entities)
            doc_relations.extend(relations)
            done_chunk_ids.update(bundle_chunk_ids)
            chunks_since_checkpoint += len(bundle)
            bundled_extract_requests += 1 if len(bundle) > 1 else 0
            _log(
                state,
                "Step 3 完成 chunk%s | chunk=%s/%s | chunk_ids=%s | new_entities=%s | new_relations=%s | total_entities=%s | total_relations=%s"
                % (
                    "(batched)" if len(bundle) > 1 else "",
                    global_index,
                    total_target_chunks,
                    bundle_chunk_ids,
                    len(entities),
                    len(relations),
                    len(all_entities),
                    len(all_relations),
                ),
            )
            if chunks_since_checkpoint >= LLM_CHECKPOINT_EVERY_CHUNKS:
                snapshot = {
                    "paths": state["paths"],
                    "checkpoint_enabled": state.get("checkpoint_enabled", True),
                    "logger": state.get("logger"),
                    "done_chunk_ids": sorted(done_chunk_ids),
                    "all_entities": all_entities,
                    "all_relations": all_relations,
                }
                _save_extract_checkpoint(snapshot)
                chunks_since_checkpoint = 0
            break
        except Exception as exc:
            retries += 1
            _log(
                state,
                "Step 3 chunk 请求失败 | chunk=%s/%s | chunk_id=%s | retry=%s/%s | error=%s"
                % (
                    global_index,
                    total_target_chunks,
                    bundle_chunk_ids,
                    retries,
                    LLM_RETRY_LIMIT,
                    f"{type(exc).__name__}: {exc}",
                ),
                "error",
            )
            if retries >= LLM_RETRY_LIMIT:
                detail = (
                    f"{type(exc).__name__}: {exc} | "
                    f"base_url={state.get('base_url')} | "
                    f"model={state.get('model')} | "
                    f"env_files={state.get('env_files_loaded', [])}"
                )
                return {"failed": True, "error": detail}
            time.sleep(LLM_RETRY_BACKOFF * retries)

    if os.environ.get(LLM_MOCK_ENV, "0") != "1":
        time.sleep(LLM_SLEEP_SECONDS)
    return {
        "all_entities": all_entities,
        "all_relations": all_relations,
        "current_doc_entities": doc_entities,
        "current_doc_relations": doc_relations,
        "done_chunk_ids": sorted(done_chunk_ids),
        "chunk_cursor": chunk_cursor + len(bundle),
        "total_filtered_entities": total_filtered_entities,
        "total_filtered_relations": total_filtered_relations,
        "total_cross_chunk_relations": total_cross_chunk_relations,
        "skipped_low_value_chunks": skipped_low_value_chunks,
        "chunks_since_checkpoint": chunks_since_checkpoint,
        "bundled_extract_requests": bundled_extract_requests,
    }


def finalize_extract_node(state: ExtractState) -> ExtractState:
    paths = state["paths"]
    write_json(
        paths.kg_raw_path,
        {
            "entities": state.get("all_entities", []),
            "relations": state.get("all_relations", []),
        },
    )
    if paths.checkpoint_path.exists():
        paths.checkpoint_path.unlink()
        _log(state, f"Step 3 清理 checkpoint | path={paths.checkpoint_path}")
    return {
        "all_entities": state.get("all_entities", []),
        "all_relations": state.get("all_relations", []),
        "bundled_extract_requests": state.get("bundled_extract_requests", 0),
    }


def fail_extract_node(state: ExtractState) -> ExtractState:
    return state


def build_extract_graph():
    graph = StateGraph(ExtractState)
    graph.add_node("prepare", prepare_extract_node)
    graph.add_node("load_doc", load_extract_doc_node)
    graph.add_node("process_chunk", process_extract_chunk_node)
    graph.add_node("finalize", finalize_extract_node)
    graph.add_node("fail", fail_extract_node)
    graph.add_edge(START, "prepare")
    graph.add_conditional_edges(
        "prepare",
        _extract_route_after_prepare,
        {"load_doc": "load_doc", "fail": "fail"},
    )
    graph.add_conditional_edges(
        "load_doc",
        _extract_route_after_load_doc,
        {"process_chunk": "process_chunk", "finalize": "finalize", "fail": "fail"},
    )
    graph.add_conditional_edges(
        "process_chunk",
        _extract_route_after_chunk,
        {
            "process_chunk": "process_chunk",
            "load_doc": "load_doc",
            "finalize": "finalize",
            "fail": "fail",
        },
    )
    graph.add_edge("finalize", END)
    graph.add_edge("fail", END)
    return graph.compile()


def prepare_merge_node(state: MergeState) -> MergeState:
    raw = read_json(state["paths"].kg_raw_path, {"entities": [], "relations": []})
    return {
        "raw_entities": raw.get("entities", []),
        "raw_relations": raw.get("relations", []),
        "merge_log": [],
        "semantic_merges": 0,
        "failed": False,
        "error": None,
    }


def exact_dedup_node(state: MergeState) -> MergeState:
    dedup_map: dict[tuple[str, str], dict] = {}
    merge_log = list(state.get("merge_log", []))
    for entity in state.get("raw_entities", []):
        key = (entity["name"].strip(), entity.get("label", ""))
        if key in dedup_map:
            existing = dedup_map[key]
            existing["doc_id"] = merge_doc_ids(
                [existing.get("doc_id", ""), entity.get("doc_id", "")]
            )
            existing["chunk_id"] = merge_chunk_ids(
                [existing.get("chunk_id", ""), entity.get("chunk_id", "")]
            )
            if entity.get("description") and entity["description"] not in existing.get(
                "description", ""
            ):
                existing["description"] = (
                    f"{existing.get('description', '')}；{entity['description']}".strip(
                        "；"
                    )
                )
            merge_log.append(
                {
                    "step": "exact_dedup",
                    "merged_into": existing["name"],
                    "merged_from": entity["name"],
                }
            )
        else:
            entity_copy = dict(entity)
            entity_copy["doc_id"] = merge_doc_ids([entity.get("doc_id", "")])
            entity_copy["chunk_id"] = merge_chunk_ids([entity.get("chunk_id", "")])
            dedup_map[key] = entity_copy
    return {"entities_dedup": list(dedup_map.values()), "merge_log": merge_log}


def unify_labels_node(state: MergeState) -> MergeState:
    grouped_entities: dict[str, list[dict]] = defaultdict(list)
    merge_log = list(state.get("merge_log", []))
    for entity in state.get("entities_dedup", []):
        grouped_entities[entity["name"].strip()].append(entity)
    entities_unified = []
    for name, group in grouped_entities.items():
        if len(group) == 1:
            entities_unified.append(group[0])
            continue
        merged = merge_entities(group)
        label_weights: dict[str, int] = defaultdict(int)
        all_labels = set()
        for entity in group:
            if entity.get("label"):
                label_weights[entity["label"]] += count_chunks(entity)
                all_labels.add(entity["label"])
        merged["label"] = (
            max(label_weights, key=label_weights.get) if label_weights else ""
        )
        merged["all_labels"] = sorted(all_labels)
        entities_unified.append(merged)
        merge_log.append(
            {"step": "label_unify", "name": name, "labels_merged": sorted(all_labels)}
        )
    return {"entities_unified": entities_unified, "merge_log": merge_log}


def semantic_merge_node(state: MergeState) -> MergeState:
    entities_unified = list(state.get("entities_unified", []))
    merge_log = list(state.get("merge_log", []))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_embedding_model(state["paths"], device)
    uf = UnionFind(len(entities_unified))
    semantic_merges = 0
    if model is not None:
        labels: dict[str, list[int]] = defaultdict(list)
        for index, entity in enumerate(entities_unified):
            labels[entity.get("label", "Unknown")].append(index)
        for label, indices in labels.items():
            if len(indices) < 2:
                continue
            threshold = LABEL_THRESHOLDS.get(label, DEFAULT_THRESHOLD)
            names = [entities_unified[index]["name"] for index in indices]
            embeddings = model.encode(names, convert_to_tensor=True, device=device)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            similarities = torch.mm(embeddings, embeddings.T)
            for left in range(len(indices)):
                for right in range(left + 1, len(indices)):
                    score = similarities[left][right].item()
                    if score >= threshold and uf.union(indices[left], indices[right]):
                        semantic_merges += 1
                        merge_log.append(
                            {
                                "step": "semantic_merge",
                                "entity_a": entities_unified[indices[left]]["name"],
                                "entity_b": entities_unified[indices[right]]["name"],
                                "label": label,
                                "threshold": threshold,
                                "similarity": round(score, 4),
                            }
                        )
    merged_entities = []
    name_remap = {}
    for members in uf.groups().values():
        if len(members) == 1:
            merged_entities.append(entities_unified[members[0]])
            continue
        group = [entities_unified[index] for index in members]
        merged = merge_entities(group)
        for entity in group:
            if entity["name"] != merged["name"]:
                name_remap[entity["name"]] = merged["name"]
        merged_entities.append(merged)
    return {
        "merged_entities": merged_entities,
        "name_remap": name_remap,
        "merge_log": merge_log,
        "semantic_merges": semantic_merges,
    }


def remap_relations_node(state: MergeState) -> MergeState:
    updated_relations = []
    for relation in state.get("raw_relations", []):
        relation_copy = dict(relation)
        relation_copy["head"] = state.get("name_remap", {}).get(
            relation_copy["head"], relation_copy["head"]
        )
        relation_copy["tail"] = state.get("name_remap", {}).get(
            relation_copy["tail"], relation_copy["tail"]
        )
        updated_relations.append(relation_copy)
    seen_relations = set()
    deduped_relations = []
    for relation in updated_relations:
        key = (relation["head"], relation["relation"], relation["tail"])
        if key in seen_relations:
            continue
        seen_relations.add(key)
        deduped_relations.append(relation)
    valid_names = {entity["name"] for entity in state.get("merged_entities", [])}
    clean_relations = [
        relation
        for relation in deduped_relations
        if relation["head"] in valid_names and relation["tail"] in valid_names
    ]
    return {"clean_relations": clean_relations}


def persist_merge_node(state: MergeState) -> MergeState:
    merged_entities = list(state.get("merged_entities", []))
    clean_relations = list(state.get("clean_relations", []))
    merged_entities.sort(key=lambda entity: (entity.get("label", ""), entity["name"]))
    clean_relations.sort(
        key=lambda relation: (relation["head"], relation["relation"], relation["tail"])
    )
    for index, entity in enumerate(merged_entities, start=1):
        entity["entity_id"] = f"ENT_{index:06d}"
    for index, relation in enumerate(clean_relations, start=1):
        relation["rel_id"] = f"REL_{index:06d}"
    paths = state["paths"]
    write_json(
        paths.kg_merged_path,
        {"entities": merged_entities, "relations": clean_relations},
    )
    write_json(paths.merge_log_path, state.get("merge_log", []))
    write_json(
        paths.chunk_to_kg_path,
        _build_chunk_to_kg(
            read_json(paths.chunks_path, []), merged_entities, clean_relations
        ),
    )
    return {"merged_entities": merged_entities, "clean_relations": clean_relations}


def build_merge_graph():
    graph = StateGraph(MergeState)
    graph.add_node("prepare", prepare_merge_node)
    graph.add_node("exact_dedup", exact_dedup_node)
    graph.add_node("unify_labels", unify_labels_node)
    graph.add_node("semantic_merge", semantic_merge_node)
    graph.add_node("remap_relations", remap_relations_node)
    graph.add_node("persist", persist_merge_node)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "exact_dedup")
    graph.add_edge("exact_dedup", "unify_labels")
    graph.add_edge("unify_labels", "semantic_merge")
    graph.add_edge("semantic_merge", "remap_relations")
    graph.add_edge("remap_relations", "persist")
    graph.add_edge("persist", END)
    return graph.compile()


def run_extract_workflow(
    paths: PipelinePaths,
    only_doc_id: str | None = None,
    use_context: bool = True,
    checkpoint_enabled: bool = True,
    logger: Any | None = None,
) -> dict[str, Any]:
    final_state = build_extract_graph().invoke(
        {
            "paths": paths,
            "only_doc_id": only_doc_id,
            "use_context": use_context,
            "checkpoint_enabled": checkpoint_enabled,
            "logger": logger,
        },
        config={"recursion_limit": 10000},
    )
    if final_state.get("failed"):
        raise RuntimeError(final_state.get("error", "KG 抽取失败"))
    return {
        "entities": len(final_state.get("all_entities", [])),
        "relations": len(final_state.get("all_relations", [])),
        "cross_chunk_relations": final_state.get("total_cross_chunk_relations", 0),
        "filtered_entities": final_state.get("total_filtered_entities", 0),
        "filtered_relations": final_state.get("total_filtered_relations", 0),
        "bundled_extract_requests": final_state.get("bundled_extract_requests", 0),
    }


def run_merge_workflow(paths: PipelinePaths) -> dict[str, Any]:
    final_state = build_merge_graph().invoke(
        {"paths": paths}, config={"recursion_limit": 1000}
    )
    return {
        "entities": len(final_state.get("merged_entities", [])),
        "relations": len(final_state.get("clean_relations", [])),
        "semantic_merges": final_state.get("semantic_merges", 0),
        "merge_log_entries": len(final_state.get("merge_log", [])),
    }


def run_clean_workflow(paths: PipelinePaths) -> dict[str, Any]:
    final_state = build_clean_graph().invoke(
        {"paths": paths}, config={"recursion_limit": 1000}
    )
    return {
        "documents": len(final_state.get("cleaned_files", [])),
        "files": final_state.get("cleaned_files", []),
        "moved_stale_cleaned": len(final_state.get("moved_stale_cleaned", [])),
    }


def run_chunk_workflow(paths: PipelinePaths) -> dict[str, Any]:
    final_state = build_chunk_graph().invoke(
        {"paths": paths}, config={"recursion_limit": 1000}
    )
    return {
        "documents": len(final_state.get("doc_source_map", [])),
        "chunks": len(final_state.get("chunks", [])),
        "dropped_chunks": final_state.get("dropped_chunks", 0),
        "dropped_reason_counts": final_state.get("dropped_reason_counts", {}),
    }


def run_neo4j_workflow(
    paths: PipelinePaths, import_to_neo4j: bool = True, export_dump: bool = True
) -> dict[str, Any]:
    final_state = build_neo4j_graph().invoke(
        {
            "paths": paths,
            "import_to_neo4j": import_to_neo4j,
            "export_dump": export_dump,
        },
        config={"recursion_limit": 1000},
    )
    return {
        "entities": len(final_state.get("entities", [])),
        "relations": len(final_state.get("relations", [])),
        "imported": final_state.get("imported", False),
        "dumped": final_state.get("dumped", False),
    }


def run_visualize_workflow(
    paths: PipelinePaths, top_n: int = 300, filter_label: str | None = None
) -> dict[str, Any]:
    final_state = build_visualize_graph().invoke(
        {"paths": paths, "top_n": top_n, "filter_label": filter_label},
        config={"recursion_limit": 1000},
    )
    return {
        "nodes": final_state.get("nodes", 0),
        "edges": final_state.get("edges", 0),
        "output": final_state.get("output", str(paths.visualization_path)),
    }
