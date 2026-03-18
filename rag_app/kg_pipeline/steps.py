from __future__ import annotations

import csv
import html
import json
import os
import re
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import torch
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from neo4j import GraphDatabase
from openai import OpenAI
from pyvis.network import Network
from sentence_transformers import SentenceTransformer

from .config import (
    LLM_API_KEY_ENV,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_RETRY_LIMIT,
    LLM_RETRY_BACKOFF,
    LLM_MIN_CHUNK_CHARS,
    LLM_SLEEP_SECONDS,
    LLM_CHECKPOINT_EVERY_CHUNKS,
    LLM_MOCK_ENV,
    LLM_REQUIRE_API_KEY,
)
from .constants import ENTITY_LABELS, RELATION_TYPES
from .paths import PipelinePaths
from .utils import (
    apply_local_env,
    numbered_path,
    read_json,
    same_file_content,
    write_json,
)

MAX_CONTEXT_ENTITIES = 200
MAX_CONTEXT_RELATIONS = 100

LABEL_THRESHOLDS = {
    "Equipment": 0.96,
    "Component": 0.96,
    "System": 0.95,
    "Parameter": 0.95,
    "FaultPhenomenon": 0.93,
    "FaultCause": 0.93,
    "DiagnosticMethod": 0.92,
    "RepairMethod": 0.92,
    "SafetyNote": 0.92,
}
DEFAULT_THRESHOLD = 0.94

LABEL_COLORS = {
    "Equipment": "#e6194b",
    "Component": "#3cb44b",
    "FaultPhenomenon": "#ffe119",
    "FaultCause": "#f58231",
    "DiagnosticMethod": "#4363d8",
    "RepairMethod": "#911eb4",
    "System": "#42d4f4",
    "Parameter": "#f032e6",
    "SafetyNote": "#bfef45",
}
DEFAULT_COLOR = "#aaaaaa"

SYSTEM_PROMPT = f"""你是船舶电气设备领域的知识图谱专家。请从给定文本中抽取实体和关系三元组。

实体标签（label）必须从以下选取：
{", ".join(ENTITY_LABELS)}

关系类型（relation）必须从以下选取：
{", ".join(RELATION_TYPES)}

输出严格 JSON 格式，不要输出任何其他文字、解释或 markdown 代码块：
{{
  "entities": [
    {{"name": "实体名", "label": "标签", "description": "简要描述"}}
  ],
  "relations": [
    {{"head": "头实体名", "head_label": "头标签", "relation": "关系类型", "tail": "尾实体名", "tail_label": "尾标签", "description": "关系描述"}}
  ]
}}
"""


LEGACY_KG_FILE_MAP = {
    "chunks.json": "chunks/chunks.json",
    "doc_source_map.json": "chunks/doc_source_map.json",
    "chunk_to_kg.json": "chunks/chunk_to_kg.json",
    "kg_raw.json": "extracted/kg_raw.json",
    "kg_raw_checkpoint.json": "extracted/kg_raw_checkpoint.json",
    "kg_merged.json": "delivery/kg_merged.json",
    "entity_merge_log.json": "delivery/entity_merge_log.json",
    "neo4j_entities.csv": "delivery/neo4j_entities.csv",
    "neo4j_relations.csv": "delivery/neo4j_relations.csv",
    "neo4j.dump": "delivery/neo4j.dump",
    "kg_visualization.html": "delivery/kg_visualization.html",
}


def migrate_legacy_files(paths: PipelinePaths) -> dict[str, Any]:
    moved = []
    removed_duplicates = []
    for legacy_name, relative_target in LEGACY_KG_FILE_MAP.items():
        source = paths.kg_dir / legacy_name
        if not source.exists():
            continue
        target = paths.kg_dir / relative_target
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if same_file_content(source, target):
                source.unlink()
                removed_duplicates.append(source.name)
                continue
            target = numbered_path(target)
        shutil.move(str(source), str(target))
        moved.append({"from": legacy_name, "to": str(target.relative_to(paths.kg_dir))})
    return {"moved": moved, "removed_duplicates": removed_duplicates}


def _copy_docs_into_raw(paths: PipelinePaths) -> int:
    copied = 0
    for pattern in ("*.md", "*.txt"):
        for source in sorted(paths.docs_dir.glob(pattern)):
            target = paths.raw_dir / source.name
            if target.exists():
                if same_file_content(source, target):
                    continue
                target = numbered_path(target)
            target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            copied += 1
    return copied


def _raw_documents(paths: PipelinePaths) -> list[Path]:
    files: list[Path] = []
    for pattern in ("*.md", "*.txt"):
        files.extend(sorted(paths.raw_dir.glob(pattern)))
    return files


def html_table_to_text(table_html: str) -> str:
    soup = BeautifulSoup(table_html, "html.parser")
    rows = []
    for tr in soup.find_all("tr"):
        cells = []
        for td in tr.find_all(["td", "th"]):
            text = td.get_text(separator=" ", strip=True)
            cells.append(text)
        if any(cell.strip() for cell in cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def clean_text(text: str) -> str:
    text = re.sub(
        r"<table[^>]*>.*?</table>",
        lambda match: html_table_to_text(match.group(0)),
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(r"IMAGE_PLACEHOLDER\s*#?\d*", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"!\[.*?\]\(data:image/[^)]+\)", "[图片]", text)
    text = re.sub(
        r"^(#{1,6})\s+(.+)$",
        lambda match: f"[H{len(match.group(1))}] {match.group(2).strip()}",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^\)]*\)", r"\1", text)
    text = re.sub(r".*ISBN[\s\-\d]+.*", "", text)
    text = re.sub(r".*定价[：:]\s*[\d.]+元.*", "", text)
    text = re.sub(r"^[A-Z]{20,}$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\$\s*\)?\s*\$", "", text)
    text = re.sub(r"\$([^$]{1,60})\$", r"\1", text)

    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and len(stripped) <= 4 and re.match(r"^[\d.]+$", stripped):
            continue
        lines.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def clean_documents(paths: PipelinePaths) -> dict[str, Any]:
    copied = _copy_docs_into_raw(paths)
    files = _raw_documents(paths)
    if not files:
        raise RuntimeError("data/KG/raw 中没有可处理的 .md/.txt 文档")

    cleaned = []
    for source in files:
        raw_text = source.read_text(encoding="utf-8")
        output = clean_text(raw_text)
        target = paths.cleaned_dir / source.name
        target.write_text(output, encoding="utf-8")
        cleaned.append(
            {
                "source": source.name,
                "raw_chars": len(raw_text),
                "cleaned_chars": len(output),
            }
        )
    return {"copied_into_raw": copied, "documents": len(cleaned), "files": cleaned}


def make_doc_id(filename: str) -> str:
    return Path(filename).stem


def extract_heading_context(text: str) -> str:
    matches = re.findall(r"^\[H\d\] .+$", text, flags=re.MULTILINE)
    return matches[-1] if matches else ""


def chunk_documents(
    paths: PipelinePaths, chunk_size: int = 1200, chunk_overlap: int = 150
) -> dict[str, Any]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = []
    doc_source_map = []
    cleaned_files = sorted(paths.cleaned_dir.glob("*.md")) + sorted(
        paths.cleaned_dir.glob("*.txt")
    )
    for source in cleaned_files:
        text = source.read_text(encoding="utf-8")
        doc_id = make_doc_id(source.name)
        valid_chunks = [
            part.strip()
            for part in splitter.split_text(text)
            if len(part.strip()) >= 20
        ]
        for index, chunk_text in enumerate(valid_chunks):
            chunks.append(
                {
                    "chunk_id": f"{doc_id}::chunk_{index}",
                    "doc_id": doc_id,
                    "chunk_index": index,
                    "text": chunk_text,
                    "source": source.name,
                    "char_count": len(chunk_text),
                    "heading_context": extract_heading_context(chunk_text),
                }
            )
        doc_source_map.append(
            {
                "doc_id": doc_id,
                "source": source.name,
                "path": f"data/KG/raw/{source.name}",
                "num_chunks": len(valid_chunks),
            }
        )

    chunk_ids = [item["chunk_id"] for item in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise RuntimeError("chunk_id 不唯一")

    write_json(paths.chunks_path, chunks)
    write_json(paths.doc_map_path, doc_source_map)
    return {"documents": len(doc_source_map), "chunks": len(chunks)}


def build_context_snapshot(doc_entities: list[dict], doc_relations: list[dict]) -> str:
    if not doc_entities:
        return ""

    entity_freq: dict[str, int] = {}
    for entity in doc_entities:
        entity_freq[entity["name"]] = entity_freq.get(entity["name"], 0) + 1

    seen_names: set[str] = set()
    unique_entities: list[dict] = []
    for entity in reversed(doc_entities):
        if entity["name"] not in seen_names:
            seen_names.add(entity["name"])
            unique_entities.append(entity)
    unique_entities.sort(
        key=lambda entity: entity_freq.get(entity["name"], 0), reverse=True
    )
    top_entities = unique_entities[:MAX_CONTEXT_ENTITIES]
    top_names = {entity["name"] for entity in top_entities}

    seen_relations: set[tuple[str, str, str]] = set()
    candidate_relations: list[dict] = []
    for relation in doc_relations:
        key = (relation["head"], relation["relation"], relation["tail"])
        if key in seen_relations:
            continue
        if relation["head"] in top_names and relation["tail"] in top_names:
            seen_relations.add(key)
            candidate_relations.append(relation)
    candidate_relations.sort(
        key=lambda relation: entity_freq.get(relation["head"], 0)
        + entity_freq.get(relation["tail"], 0),
        reverse=True,
    )
    top_relations = candidate_relations[:MAX_CONTEXT_RELATIONS]

    payload = [
        "已有实体（%d条）：%s"
        % (
            len(top_entities),
            json.dumps(
                [
                    {"name": entity["name"], "label": entity.get("label", "")}
                    for entity in top_entities
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
    ]
    if top_relations:
        payload.append(
            "已有关系（%d条）：%s"
            % (
                len(top_relations),
                json.dumps(
                    [
                        {
                            "h": relation["head"],
                            "r": relation["relation"],
                            "t": relation["tail"],
                        }
                        for relation in top_relations
                    ],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
        )
    return "\n".join(payload)


def _fix_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([\]\}])", r"\1", text)


def extract_json_from_response(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text).strip()
    for candidate in (text, _fix_trailing_commas(text)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start < 0:
        return {"entities": [], "relations": []}

    depth = 0
    end = -1
    for index, char in enumerate(text[start:], start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index
                break
    if end < 0:
        return {"entities": [], "relations": []}
    try:
        return json.loads(_fix_trailing_commas(text[start : end + 1]))
    except json.JSONDecodeError:
        return {"entities": [], "relations": []}


def validate_extracted(
    result: dict, doc_entity_names: set[str], use_context: bool
) -> tuple[list, list]:
    entities = []
    for entity in result.get("entities", []):
        if not isinstance(entity, dict):
            continue
        if entity.get("label") not in ENTITY_LABELS:
            continue
        if not str(entity.get("name", "")).strip():
            continue
        entities.append(entity)

    names = {entity["name"] for entity in entities}
    allowed_names = names | (doc_entity_names if use_context else set())

    relations = []
    for relation in result.get("relations", []):
        if not isinstance(relation, dict):
            continue
        if relation.get("relation") not in RELATION_TYPES:
            continue
        if (
            relation.get("head") not in allowed_names
            or relation.get("tail") not in allowed_names
        ):
            continue
        if relation.get("head") == relation.get("tail"):
            continue
        relations.append(relation)
    return entities, relations


def _resolve_api_key(paths: PipelinePaths) -> str:
    apply_local_env(paths.env_path)
    api_key = os.environ.get(LLM_API_KEY_ENV, "").strip()
    if not api_key and not LLM_REQUIRE_API_KEY:
        return ""
    if not api_key:
        raise RuntimeError(
            f"{LLM_API_KEY_ENV} 未设置，请在环境变量或 Project/rag_app/.env 中配置"
        )
    return api_key


def extract_kg(
    paths: PipelinePaths,
    only_doc_id: str | None = None,
    use_context: bool = True,
    checkpoint_enabled: bool = True,
) -> dict[str, Any]:
    api_key = _resolve_api_key(paths)
    client = OpenAI(
        api_key=api_key or "ollama",
        base_url=os.environ.get("DEEPSEEK_BASE_URL", LLM_BASE_URL),
    )
    model = os.environ.get("DEEPSEEK_MODEL", LLM_MODEL)
    all_chunks = read_json(paths.chunks_path, [])
    target_chunks = (
        [chunk for chunk in all_chunks if chunk["doc_id"] == only_doc_id]
        if only_doc_id
        else all_chunks
    )
    if only_doc_id and not target_chunks:
        raise ValueError(f"未找到 doc_id={only_doc_id} 的 chunks")

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
    if checkpoint_enabled and paths.checkpoint_path.exists():
        checkpoint = read_json(
            paths.checkpoint_path,
            {"done_chunk_ids": [], "entities": [], "relations": []},
        )
        done_chunk_ids = set(checkpoint.get("done_chunk_ids", []))
        if not only_doc_id:
            all_entities = checkpoint.get("entities", [])
            all_relations = checkpoint.get("relations", [])

    total_filtered_entities = 0
    total_filtered_relations = 0
    total_cross_chunk_relations = 0
    chunks_since_checkpoint = 0

    def save_checkpoint() -> None:
        if checkpoint_enabled:
            write_json(
                paths.checkpoint_path,
                {
                    "done_chunk_ids": sorted(done_chunk_ids),
                    "entities": all_entities,
                    "relations": all_relations,
                },
            )

    docs_order: list[str] = []
    grouped_chunks: dict[str, list[dict]] = defaultdict(list)
    for chunk in target_chunks:
        if chunk["doc_id"] not in grouped_chunks:
            docs_order.append(chunk["doc_id"])
        grouped_chunks[chunk["doc_id"]].append(chunk)

    for doc_id in docs_order:
        doc_entities: list[dict] = []
        doc_relations: list[dict] = []
        for chunk in grouped_chunks[doc_id]:
            chunk_id = chunk["chunk_id"]
            if chunk_id in done_chunk_ids:
                continue
            if len(chunk["text"].strip()) < LLM_MIN_CHUNK_CHARS:
                done_chunk_ids.add(chunk_id)
                continue

            user_parts = ["请从以下文本中抽取知识图谱三元组："]
            if use_context and doc_entities:
                user_parts.append(build_context_snapshot(doc_entities, doc_relations))
            if chunk.get("heading_context"):
                user_parts.append(f"所属章节：{chunk['heading_context']}")
            user_parts.append(f"待抽取文本：\n{chunk['text']}")
            user_content = "\n\n".join(user_parts)
            doc_entity_names = {entity["name"] for entity in doc_entities}

            retries = 0
            while retries < LLM_RETRY_LIMIT:
                try:
                    response = client.chat.completions.create(
                        model=model,
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
                        result, doc_entity_names, use_context
                    )
                    total_filtered_entities += len(result.get("entities", [])) - len(
                        entities
                    )
                    total_filtered_relations += len(result.get("relations", [])) - len(
                        relations
                    )

                    new_names = {entity["name"] for entity in entities}
                    total_cross_chunk_relations += len(
                        [
                            rel
                            for rel in relations
                            if rel["head"] not in new_names
                            or rel["tail"] not in new_names
                        ]
                    )

                    for entity in entities:
                        entity["doc_id"] = doc_id
                        entity["chunk_id"] = chunk_id
                        entity["source"] = chunk.get("source", "")
                    for relation in relations:
                        relation["doc_id"] = doc_id
                        relation["chunk_id"] = chunk_id

                    all_entities.extend(entities)
                    all_relations.extend(relations)
                    doc_entities.extend(entities)
                    doc_relations.extend(relations)
                    done_chunk_ids.add(chunk_id)
                    chunks_since_checkpoint += 1
                    if chunks_since_checkpoint >= LLM_CHECKPOINT_EVERY_CHUNKS:
                        save_checkpoint()
                        chunks_since_checkpoint = 0
                    break
                except Exception:
                    retries += 1
                    if retries >= LLM_RETRY_LIMIT:
                        raise
                    time.sleep(LLM_RETRY_BACKOFF * retries)
            time.sleep(LLM_SLEEP_SECONDS)

    write_json(
        paths.kg_raw_path, {"entities": all_entities, "relations": all_relations}
    )
    if paths.checkpoint_path.exists():
        paths.checkpoint_path.unlink()
    return {
        "entities": len(all_entities),
        "relations": len(all_relations),
        "cross_chunk_relations": total_cross_chunk_relations,
        "filtered_entities": total_filtered_entities,
        "filtered_relations": total_filtered_relations,
    }


def merge_doc_ids(items: list[Any]) -> list[str]:
    values = set()
    for item in items:
        if isinstance(item, list):
            values.update(part for part in item if part)
        elif item:
            values.add(item)
    return sorted(values)


def merge_chunk_ids(items: list[Any]) -> list[str]:
    values = set()
    for item in items:
        if isinstance(item, list):
            values.update(part for part in item if part)
        elif item:
            values.add(item)
    return sorted(values)


def count_chunks(entity: dict) -> int:
    chunk_id = entity.get("chunk_id", "")
    if isinstance(chunk_id, list):
        return len(chunk_id)
    return 1 if chunk_id else 0


def merge_entities(member_entities: list[dict]) -> dict:
    representative = max(
        member_entities, key=lambda entity: (count_chunks(entity), len(entity["name"]))
    )
    merged = dict(representative)
    descriptions = []
    labels = set()
    sources = []
    for entity in member_entities:
        descriptions.extend(
            [part for part in str(entity.get("description", "")).split("；") if part]
        )
        if entity.get("label"):
            labels.add(entity["label"])
        if entity.get("all_labels"):
            labels.update(entity["all_labels"])
        if entity.get("source"):
            sources.append(entity["source"])
    merged["doc_id"] = merge_doc_ids(
        [entity.get("doc_id", "") for entity in member_entities]
    )
    merged["chunk_id"] = merge_chunk_ids(
        [entity.get("chunk_id", "") for entity in member_entities]
    )
    merged["description"] = "；".join(dict.fromkeys(descriptions))
    merged["source"] = next(
        (source for source in sources if source), merged.get("source", "")
    )
    if len(labels) > 1:
        merged["all_labels"] = sorted(labels)
    return merged


def load_embedding_model(
    paths: PipelinePaths, device: str
) -> SentenceTransformer | None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    model_name = os.environ.get("BGE_MODEL_NAME", "BAAI/bge-small-zh-v1.5")
    try:
        return SentenceTransformer(
            model_name,
            device=device,
            cache_folder=str(paths.cache_dir),
            local_files_only=True,
        )
    except Exception:
        pass
    if os.environ.get("ALLOW_ONLINE_MODEL_DOWNLOAD", "1") != "1":
        return None
    try:
        return SentenceTransformer(
            model_name, device=device, cache_folder=str(paths.cache_dir)
        )
    except Exception:
        return None


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return True

    def groups(self) -> dict[int, list[int]]:
        grouped: dict[int, list[int]] = defaultdict(list)
        for index in range(len(self.parent)):
            grouped[self.find(index)].append(index)
        return dict(grouped)


def _build_chunk_to_kg(
    chunks: list[dict], entities: list[dict], relations: list[dict]
) -> dict[str, list[dict]]:
    names_by_chunk: dict[str, set[str]] = defaultdict(set)
    ids_by_chunk: dict[str, set[str]] = defaultdict(set)
    rels_by_chunk: dict[str, set[str]] = defaultdict(set)
    for entity in entities:
        chunk_ids = entity.get("chunk_id", [])
        if isinstance(chunk_ids, str):
            chunk_ids = [chunk_ids]
        for chunk_id in chunk_ids:
            names_by_chunk[chunk_id].add(entity["name"])
            if entity.get("entity_id"):
                ids_by_chunk[chunk_id].add(entity["entity_id"])
    for relation in relations:
        chunk_ids = relation.get("chunk_id", [])
        if isinstance(chunk_ids, str):
            chunk_ids = [chunk_ids]
        for chunk_id in chunk_ids:
            if relation.get("rel_id"):
                rels_by_chunk[chunk_id].add(relation["rel_id"])
    return {
        "chunks": [
            {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "source": chunk.get("source", ""),
                "chunk_index": chunk.get("chunk_index"),
                "kg_entities": sorted(names_by_chunk.get(chunk["chunk_id"], set())),
                "kg_entity_ids": sorted(ids_by_chunk.get(chunk["chunk_id"], set())),
                "kg_relations": sorted(rels_by_chunk.get(chunk["chunk_id"], set())),
            }
            for chunk in chunks
        ]
    }


def merge_kg(paths: PipelinePaths) -> dict[str, Any]:
    raw = read_json(paths.kg_raw_path, {"entities": [], "relations": []})
    entities = raw.get("entities", [])
    relations = raw.get("relations", [])
    merge_log = []

    dedup_map: dict[tuple[str, str], dict] = {}
    for entity in entities:
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

    entities_dedup = list(dedup_map.values())
    grouped_entities: dict[str, list[dict]] = defaultdict(list)
    for entity in entities_dedup:
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_embedding_model(paths, device)
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

    updated_relations = []
    for relation in relations:
        relation_copy = dict(relation)
        relation_copy["head"] = name_remap.get(
            relation_copy["head"], relation_copy["head"]
        )
        relation_copy["tail"] = name_remap.get(
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

    valid_names = {entity["name"] for entity in merged_entities}
    clean_relations = [
        relation
        for relation in deduped_relations
        if relation["head"] in valid_names and relation["tail"] in valid_names
    ]

    merged_entities.sort(key=lambda entity: (entity.get("label", ""), entity["name"]))
    clean_relations.sort(
        key=lambda relation: (relation["head"], relation["relation"], relation["tail"])
    )
    for index, entity in enumerate(merged_entities, start=1):
        entity["entity_id"] = f"ENT_{index:06d}"
    for index, relation in enumerate(clean_relations, start=1):
        relation["rel_id"] = f"REL_{index:06d}"

    write_json(
        paths.kg_merged_path,
        {"entities": merged_entities, "relations": clean_relations},
    )
    write_json(paths.merge_log_path, merge_log)
    write_json(
        paths.chunk_to_kg_path,
        _build_chunk_to_kg(
            read_json(paths.chunks_path, []), merged_entities, clean_relations
        ),
    )
    return {
        "entities": len(merged_entities),
        "relations": len(clean_relations),
        "semantic_merges": semantic_merges,
        "merge_log_entries": len(merge_log),
    }


def _serialize(value: Any) -> str:
    if isinstance(value, list):
        return ";".join(str(item) for item in value if item)
    return str(value or "")


def _neo4j_executable(command: str) -> str:
    neo4j_home = os.environ.get("NEO4J_HOME", "").strip()
    if neo4j_home:
        candidate = Path(neo4j_home) / "bin" / command
        if candidate.exists():
            return str(candidate)
    discovered = shutil.which(command)
    if discovered:
        return discovered
    raise RuntimeError(f"未找到 {command}，请配置 NEO4J_HOME 或将其加入 PATH")


def generate_neo4j_artifacts(
    paths: PipelinePaths,
    import_to_neo4j: bool = True,
    export_dump: bool = True,
) -> dict[str, Any]:
    apply_local_env(paths.env_path)
    kg = read_json(paths.kg_merged_path, {"entities": [], "relations": []})
    entities = kg.get("entities", [])
    relations = kg.get("relations", [])

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
        for entity in entities:
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
        for relation in relations:
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

    if not import_to_neo4j:
        return {
            "entities": len(entities),
            "relations": len(relations),
            "imported": False,
            "dumped": False,
        }

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
                    for entity in entities
                ],
            )
            grouped_relations: dict[str, list[dict]] = defaultdict(list)
            for relation in relations:
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

    dumped = False
    if export_dump:
        env = os.environ.copy()
        java_home = os.environ.get("JAVA_HOME", "").strip()
        if java_home:
            env["JAVA_HOME"] = java_home
        neo4j_cmd = _neo4j_executable("neo4j")
        neo4j_admin_cmd = _neo4j_executable("neo4j-admin")
        subprocess.run([neo4j_cmd, "stop"], check=False, env=env)
        time.sleep(3)
        subprocess.run(
            [
                neo4j_admin_cmd,
                "database",
                "dump",
                "neo4j",
                f"--to-path={paths.delivery_dir}",
            ],
            check=True,
            env=env,
        )
        subprocess.run([neo4j_cmd, "start"], check=False, env=env)
        dumped = True

    return {
        "entities": len(entities),
        "relations": len(relations),
        "imported": True,
        "dumped": dumped,
    }


def visualize_kg(
    paths: PipelinePaths, top_n: int = 300, filter_label: str | None = None
) -> dict[str, Any]:
    kg = read_json(paths.kg_merged_path, {"entities": [], "relations": []})
    entities = kg.get("entities", [])
    relations = kg.get("relations", [])
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

    entity_info = {entity["name"]: entity for entity in entities}
    graph = nx.DiGraph()
    graph.add_nodes_from(entity_info.keys())
    for relation in relations:
        if relation["head"] in entity_info and relation["tail"] in entity_info:
            graph.add_edge(
                relation["head"],
                relation["tail"],
                relation=relation["relation"],
                description=relation.get("description", ""),
            )

    degrees = dict(graph.degree())
    if graph.number_of_nodes() > top_n:
        top_nodes = sorted(degrees, key=lambda name: degrees[name], reverse=True)[
            :top_n
        ]
        graph = graph.subgraph(top_nodes).copy()
        entity_info = {name: entity_info[name] for name in graph.nodes()}
        degrees = dict(graph.degree())

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

    for name in graph.nodes():
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

    for head, tail, data in graph.edges(data=True):
        rel_type = data.get("relation", "")
        desc = data.get("description", "")
        net.add_edge(
            head,
            tail,
            title=f"{rel_type}: {desc}" if desc else rel_type,
            label=rel_type,
            arrows="to",
        )

    net.save_graph(str(paths.visualization_path))
    legend = "".join(
        (
            f'<div style="margin:2px 0"><span style="display:inline-block;width:14px;'
            f'height:14px;background:{color};border-radius:50%;margin-right:6px;vertical-align:middle"></span>{label}</div>'
        )
        for label, color in LABEL_COLORS.items()
    )
    content = paths.visualization_path.read_text(encoding="utf-8")
    paths.visualization_path.write_text(
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
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "output": str(paths.visualization_path),
    }
