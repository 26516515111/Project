#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
from collections import defaultdict
from collections import Counter
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

from kg_pipeline.constants import RELATION_TYPES
from kg_pipeline.paths import PipelinePaths
from kg_pipeline.steps import write_json
from kg_pipeline.utils import apply_local_envs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将多个 KG_Build 子图打包为最终 KG")
    parser.add_argument(
        "--kg",
        dest="kg_names",
        action="append",
        default=[],
        help="要打包的 KG 名称，可重复传入",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="packaged",
        help="最终包名称，仅用于 manifest 记录",
    )
    parser.add_argument(
        "--import-backups",
        action="store_true",
        help="先将 data/KG/backups 下的 tar.gz 解包到 KG_Build/<kg_name>",
    )
    return parser.parse_args()


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_json_from_candidates(
    candidates: list[Path], default: Any, *, required_name: str | None = None
) -> Any:
    for path in candidates:
        if path.exists():
            return read_json(path, default)
    if required_name:
        joined = ", ".join(str(path) for path in candidates)
        raise SystemExit(f"缺少 {required_name}，已尝试路径: {joined}")
    return default


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _serialize(value: Any) -> str:
    if isinstance(value, list):
        return ";".join(str(item) for item in value if item)
    return str(value or "")


HEADING_TAG_PATTERN = re.compile(r"(?m)^\s*\[H([1-6])\]\s*(.+?)\s*$")
HEADING_INLINE_TAG_PATTERN = re.compile(r"(?<!\S)\[H([1-6])\]\s*")


def render_heading_tags_as_markdown(text: Any) -> Any:
    if not isinstance(text, str) or "[H" not in text:
        return text

    def replace(match: re.Match[str]) -> str:
        level = int(match.group(1))
        title = match.group(2).strip()
        return f"{'#' * level} {title}"

    return HEADING_TAG_PATTERN.sub(replace, text)


def render_heading_path_string_as_markdown(text: Any) -> Any:
    if not isinstance(text, str) or "[H" not in text:
        return text

    def replace(match: re.Match[str]) -> str:
        level = int(match.group(1))
        return f"{'#' * level} "

    # heading_context 常见格式为 "A -> [H2] B -> [H4] C"，这里处理行内/后置标签。
    return HEADING_INLINE_TAG_PATTERN.sub(replace, text)


def normalize_release_chunk_fields(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    normalized["text"] = render_heading_tags_as_markdown(normalized.get("text", ""))
    normalized["heading_context"] = render_heading_path_string_as_markdown(
        normalized.get("heading_context", "")
    )
    heading_path = normalized.get("heading_path")
    if isinstance(heading_path, list):
        normalized["heading_path"] = [
            render_heading_tags_as_markdown(item) for item in heading_path
        ]
    return normalized


def namespace_doc_id(kg_name: str, doc_id: str, enabled: bool) -> str:
    value = str(doc_id or "").strip()
    if not value or not enabled:
        return ""
    prefix = f"{kg_name}::"
    return value if value.startswith(prefix) else f"{prefix}{value}"


def namespace_chunk_id(kg_name: str, chunk_id: str, enabled: bool) -> str:
    value = str(chunk_id or "").strip()
    if not value or not enabled:
        return ""
    prefix = f"{kg_name}::"
    return value if value.startswith(prefix) else f"{prefix}{value}"


def normalize_build_kg_name(name: str) -> str:
    return PipelinePaths.normalize_kg_name(name)


def materialize_backup_kgs(paths: PipelinePaths) -> list[str]:
    extracted: list[str] = []
    for archive in sorted(paths.backups_dir.glob("*.tar.gz")):
        kg_name = archive.name[: -len(".tar.gz")]
        target_dir = paths.build_root_dir / normalize_build_kg_name(kg_name)
        if target_dir.exists():
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(target_dir)
        extracted.append(kg_name)
    return extracted


def migrate_flat_build_root(paths: PipelinePaths) -> str | None:
    legacy_items = ["raw", "cleaned", "chunks", "extracted", "images", "logs", "delivery"]
    state_path = paths.build_root_dir / "kg_state.json"
    has_legacy = state_path.exists() or any((paths.build_root_dir / name).exists() for name in legacy_items)
    if not has_legacy:
        return None
    state = read_json(state_path, {})
    kg_name = normalize_build_kg_name(state.get("kg_name", "legacy"))
    target_dir = paths.build_root_dir / kg_name
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in legacy_items:
        source = paths.build_root_dir / name
        target = target_dir / name
        if not source.exists():
            continue
        if target.exists():
            continue
        source.rename(target)
    if state_path.exists():
        target_state = target_dir / "kg_state.json"
        if not target_state.exists():
            state_path.rename(target_state)
    return kg_name


def _collision_key(entity: dict[str, Any]) -> tuple[str, str]:
    return (str(entity.get("name", "")).strip(), str(entity.get("label", "")).strip())


def collect_collision_keys(build_dirs: list[tuple[str, Path]]) -> set[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    for _, build_dir in build_dirs:
        merged = read_json(build_dir / "delivery" / "kg_merged.json", {"entities": []})
        for entity in merged.get("entities", []):
            key = _collision_key(entity)
            if key[0] and key[1]:
                keys.append(key)
    counts = Counter(keys)
    return {key for key, count in counts.items() if count > 1}


def rename_entity_name(
    kg_name: str,
    name: str,
    label: str,
    collision_keys: set[tuple[str, str]],
    namespace_enabled: bool,
) -> str:
    key = (str(name or "").strip(), str(label or "").strip())
    if namespace_enabled and key in collision_keys:
        return f"{kg_name}::{key[0]}"
    return key[0]


def transform_chunks(
    kg_name: str, chunks: list[dict[str, Any]], namespace_enabled: bool
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in chunks:
        row = normalize_release_chunk_fields(item)
        row["doc_id"] = namespace_doc_id(kg_name, row.get("doc_id", ""), namespace_enabled) or str(row.get("doc_id", "")).strip()
        row["source_doc_id"] = namespace_doc_id(
            kg_name,
            row.get("source_doc_id", row.get("doc_id", "")),
            namespace_enabled,
        ) or str(row.get("source_doc_id", row.get("doc_id", ""))).strip()
        row["chunk_id"] = namespace_chunk_id(kg_name, row.get("chunk_id", ""), namespace_enabled) or str(row.get("chunk_id", "")).strip()
        output.append(row)
    return output


def transform_doc_map(
    kg_name: str, items: list[dict[str, Any]], namespace_enabled: bool
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in items:
        row = dict(item)
        row["doc_id"] = namespace_doc_id(kg_name, row.get("doc_id", ""), namespace_enabled) or str(row.get("doc_id", "")).strip()
        output.append(row)
    return output


def transform_chunk_to_kg(
    kg_name: str,
    data: dict[str, Any] | list[dict[str, Any]],
    renamed_entities: dict[tuple[str, str], str],
    namespace_enabled: bool,
) -> list[dict[str, Any]]:
    chunks = data.get("chunks", []) if isinstance(data, dict) else data
    output: list[dict[str, Any]] = []
    flat_name_map = {
        original_name: renamed_name
        for (_, original_name), renamed_name in renamed_entities.items()
        if original_name != renamed_name
    }
    for item in chunks:
        row = dict(item)
        row["doc_id"] = namespace_doc_id(kg_name, row.get("doc_id", ""), namespace_enabled) or str(row.get("doc_id", "")).strip()
        row["chunk_id"] = namespace_chunk_id(kg_name, row.get("chunk_id", ""), namespace_enabled) or str(row.get("chunk_id", "")).strip()
        row["kg_entities"] = [
            flat_name_map.get(str(name).strip(), str(name).strip())
            for name in ensure_list(row.get("kg_entities"))
        ]
        row["kg_entity_ids"] = []
        output.append(row)
    return output


def transform_entities(
    kg_name: str,
    entities: list[dict[str, Any]],
    collision_keys: set[tuple[str, str]],
    namespace_enabled: bool,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], str]]:
    renamed: dict[tuple[str, str], str] = {}
    output: list[dict[str, Any]] = []
    for item in entities:
        row = dict(item)
        original_name = str(row.get("name", "")).strip()
        label = str(row.get("label", "")).strip()
        new_name = rename_entity_name(
            kg_name, original_name, label, collision_keys, namespace_enabled
        )
        renamed[(label, original_name)] = new_name
        row["name"] = new_name
        row["doc_id"] = [
            namespace_doc_id(kg_name, value, namespace_enabled) or str(value).strip()
            for value in ensure_list(row.get("doc_id"))
            if str(value).strip()
        ]
        row["chunk_id"] = [
            namespace_chunk_id(kg_name, value, namespace_enabled) or str(value).strip()
            for value in ensure_list(row.get("chunk_id"))
            if str(value).strip()
        ]
        row.pop("entity_id", None)
        output.append(row)
    return output, renamed


def rename_by_label(name: str, label: str, renamed_entities: dict[tuple[str, str], str]) -> str:
    return renamed_entities.get((str(label).strip(), str(name).strip()), str(name).strip())


def transform_relations(
    kg_name: str,
    relations: list[dict[str, Any]],
    renamed_entities: dict[tuple[str, str], str],
    namespace_enabled: bool,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in relations:
        row = dict(item)
        row["head"] = rename_by_label(row.get("head", ""), row.get("head_label", ""), renamed_entities)
        row["tail"] = rename_by_label(row.get("tail", ""), row.get("tail_label", ""), renamed_entities)
        row["doc_id"] = [
            namespace_doc_id(kg_name, value, namespace_enabled) or str(value).strip()
            for value in ensure_list(row.get("doc_id"))
            if str(value).strip()
        ]
        row["chunk_id"] = [
            namespace_chunk_id(kg_name, value, namespace_enabled) or str(value).strip()
            for value in ensure_list(row.get("chunk_id"))
            if str(value).strip()
        ]
        row.pop("rel_id", None)
        output.append(row)
    return output


def transform_merge_log(
    kg_name: str, items: list[dict[str, Any]], namespace_enabled: bool
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in items:
        row = dict(item)
        if namespace_enabled:
            row["kg_name"] = kg_name
        output.append(row)
    return output


def assign_entity_ids(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, item in enumerate(entities, start=1):
        row = dict(item)
        row["entity_id"] = f"ENT_{index:06d}"
        output.append(row)
    return output


def assign_relation_ids(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, item in enumerate(relations, start=1):
        row = dict(item)
        row["rel_id"] = f"REL_{index:06d}"
        output.append(row)
    return output


def ensure_package_dirs(paths: PipelinePaths) -> None:
    (paths.kg_dir / "chunks").mkdir(parents=True, exist_ok=True)
    (paths.kg_dir / "extracted").mkdir(parents=True, exist_ok=True)
    (paths.kg_dir / "delivery").mkdir(parents=True, exist_ok=True)


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


def generate_release_neo4j_artifacts(
    paths: PipelinePaths,
    merged_delivery: dict[str, Any],
    import_to_neo4j: bool,
    export_dump: bool,
) -> dict[str, Any]:
    apply_local_envs(paths.env_paths)
    entities = merged_delivery.get("entities", [])
    relations = merged_delivery.get("relations", [])

    entities_csv = paths.kg_dir / "delivery" / "neo4j_entities.csv"
    relations_csv = paths.kg_dir / "delivery" / "neo4j_relations.csv"
    dump_target = paths.kg_dir / "delivery" / "neo4j.dump"

    with entities_csv.open("w", encoding="utf-8", newline="") as fp:
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
                    entity.get("name", ""),
                    entity.get("label", ""),
                    entity.get("description", ""),
                    _serialize(entity.get("doc_id", "")),
                    _serialize(entity.get("chunk_id", "")),
                    _serialize(entity.get("all_labels", [])),
                    entity.get("source", ""),
                ]
            )

    with relations_csv.open("w", encoding="utf-8", newline="") as fp:
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
                    relation.get("head", ""),
                    relation.get("head_label", ""),
                    relation.get("relation", ""),
                    relation.get("tail", ""),
                    relation.get("tail_label", ""),
                    relation.get("description", ""),
                    _serialize(relation.get("doc_id", "")),
                    _serialize(relation.get("chunk_id", "")),
                ]
            )

    if not import_to_neo4j:
        return {
            "entities_csv": str(entities_csv),
            "relations_csv": str(relations_csv),
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
                    n.all_labels = row.all_labels,
                    n.source = row.source
                """,
                rows=[
                    {
                        "entity_id": entity.get("entity_id", ""),
                        "name": entity.get("name", ""),
                        "label": entity.get("label", ""),
                        "description": entity.get("description", ""),
                        "all_labels": _serialize(entity.get("all_labels", [])),
                        "source": str(entity.get("source", "")),
                    }
                    for entity in entities
                ],
            )
            grouped_relations: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for relation in relations:
                relation_type = str(relation.get("relation", "")).strip()
                if relation_type:
                    grouped_relations[relation_type].append(
                        {
                            "rel_id": relation.get("rel_id", ""),
                            "head": relation.get("head", ""),
                            "tail": relation.get("tail", ""),
                            "description": relation.get("description", ""),
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
                        r.description = row.description
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
        if dump_target.exists():
            dump_target.unlink()
        subprocess.run(
            [
                neo4j_admin_cmd,
                "database",
                "dump",
                "neo4j",
                f"--to-path={paths.kg_dir / 'delivery'}",
            ],
            check=True,
            env=env,
        )
        subprocess.run([neo4j_cmd, "start"], check=False, env=env)
        dumped = True

    return {
        "entities_csv": str(entities_csv),
        "relations_csv": str(relations_csv),
        "imported": True,
        "dumped": dumped,
        "dump_path": str(dump_target) if dumped else "",
    }


def package_kgs(
    paths: PipelinePaths,
    kg_names: list[str],
    output_name: str,
    neo4j_import: bool = True,
    neo4j_dump: bool = True,
) -> dict[str, Any]:
    build_dirs = [
        (normalize_build_kg_name(name), paths.build_root_dir / normalize_build_kg_name(name))
        for name in kg_names
    ]
    missing = [name for name, directory in build_dirs if not directory.exists()]
    if missing:
        raise SystemExit(f"未找到以下 KG_Build 子目录: {', '.join(missing)}")

    namespace_enabled = len(build_dirs) > 1
    collision_keys = collect_collision_keys(build_dirs) if namespace_enabled else set()
    merged_chunks: list[dict[str, Any]] = []
    merged_doc_map: list[dict[str, Any]] = []
    merged_chunk_to_kg: list[dict[str, Any]] = []
    merged_raw_entities: list[dict[str, Any]] = []
    merged_raw_relations: list[dict[str, Any]] = []
    merged_entities: list[dict[str, Any]] = []
    merged_relations: list[dict[str, Any]] = []
    merged_merge_log: list[dict[str, Any]] = []

    for kg_name, build_dir in build_dirs:
        chunks = read_json_from_candidates(
            [
                build_dir / "chunks" / "chunks.json",
                build_dir / "chunks" / "docs" / "chunks.json",
            ],
            [],
            required_name=f"{kg_name} 的 chunks.json",
        )
        doc_map = read_json_from_candidates(
            [
                build_dir / "chunks" / "doc_source_map.json",
                build_dir / "chunks" / "docs" / "doc_source_map.json",
            ],
            [],
            required_name=f"{kg_name} 的 doc_source_map.json",
        )
        chunk_to_kg = read_json_from_candidates(
            [
                build_dir / "chunks" / "chunk_to_kg.json",
                build_dir / "chunks" / "docs" / "chunk_to_kg.json",
            ],
            {"chunks": []},
            required_name=f"{kg_name} 的 chunk_to_kg.json",
        )
        delivery = read_json(build_dir / "delivery" / "kg_merged.json", {"entities": [], "relations": []})
        merge_log = read_json(build_dir / "delivery" / "entity_merge_log.json", [])

        raw_payload = {"entities": [], "relations": []}
        extracted_dir = build_dir / "extracted"
        preferred_raw = extracted_dir / "kg_raw.json"
        if preferred_raw.exists():
            raw_payload = read_json(preferred_raw, raw_payload)
        else:
            matches = sorted(extracted_dir.glob("kg_raw*.json"))
            if matches:
                raw_payload = read_json(matches[0], raw_payload)

        transformed_entities, renamed_entities = transform_entities(
            kg_name, delivery.get("entities", []), collision_keys, namespace_enabled
        )
        transformed_relations = transform_relations(
            kg_name, delivery.get("relations", []), renamed_entities, namespace_enabled
        )
        transformed_raw_entities, _ = transform_entities(
            kg_name, raw_payload.get("entities", []), collision_keys, namespace_enabled
        )
        transformed_raw_relations = transform_relations(
            kg_name, raw_payload.get("relations", []), renamed_entities, namespace_enabled
        )

        merged_chunks.extend(transform_chunks(kg_name, chunks, namespace_enabled))
        merged_doc_map.extend(transform_doc_map(kg_name, doc_map, namespace_enabled))
        merged_chunk_to_kg.extend(
            transform_chunk_to_kg(kg_name, chunk_to_kg, renamed_entities, namespace_enabled)
        )
        merged_raw_entities.extend(transformed_raw_entities)
        merged_raw_relations.extend(transformed_raw_relations)
        merged_entities.extend(transformed_entities)
        merged_relations.extend(transformed_relations)
        merged_merge_log.extend(transform_merge_log(kg_name, merge_log, namespace_enabled))

    merged_raw = {
        "entities": assign_entity_ids(merged_raw_entities),
        "relations": assign_relation_ids(merged_raw_relations),
    }
    merged_delivery = {
        "entities": assign_entity_ids(merged_entities),
        "relations": assign_relation_ids(merged_relations),
    }

    ensure_package_dirs(paths)
    write_json(paths.kg_dir / "chunks" / "chunks.json", merged_chunks)
    write_json(paths.kg_dir / "chunks" / "doc_source_map.json", merged_doc_map)
    write_json(paths.kg_dir / "chunks" / "chunk_to_kg.json", {"chunks": merged_chunk_to_kg})
    write_json(paths.kg_dir / "extracted" / "kg_raw.json", merged_raw)
    write_json(paths.kg_dir / "delivery" / "kg_merged.json", merged_delivery)
    write_json(paths.kg_dir / "delivery" / "entity_merge_log.json", merged_merge_log)
    neo4j_result = generate_release_neo4j_artifacts(
        paths,
        merged_delivery,
        import_to_neo4j=neo4j_import,
        export_dump=neo4j_dump,
    )
    write_json(
        paths.kg_dir / "package_manifest.json",
        {
            "output_name": output_name,
            "mode": "package" if namespace_enabled else "single",
            "kg_names": [name for name, _ in build_dirs],
            "entities": len(merged_delivery["entities"]),
            "relations": len(merged_delivery["relations"]),
            "chunks": len(merged_chunks),
            "neo4j_imported": neo4j_result.get("imported", False),
            "neo4j_dumped": neo4j_result.get("dumped", False),
        },
    )
    return {
        "kg_names": [name for name, _ in build_dirs],
        "entities": len(merged_delivery["entities"]),
        "relations": len(merged_delivery["relations"]),
        "chunks": len(merged_chunks),
        "neo4j_imported": neo4j_result.get("imported", False),
        "neo4j_dumped": neo4j_result.get("dumped", False),
    }


def main() -> int:
    args = parse_args()
    paths = PipelinePaths.discover()
    paths.build_root_dir.mkdir(parents=True, exist_ok=True)
    paths.kg_dir.mkdir(parents=True, exist_ok=True)
    paths.backups_dir.mkdir(parents=True, exist_ok=True)

    migrated = migrate_flat_build_root(paths)
    extracted = materialize_backup_kgs(paths) if args.import_backups else []

    kg_names = [normalize_build_kg_name(name) for name in args.kg_names]
    if not kg_names:
        if migrated:
            kg_names = [migrated]
        else:
            raise SystemExit("请至少通过 --kg 传入一个 KG 名称")

    result = package_kgs(paths, kg_names, args.output_name)
    if migrated:
        result["migrated_legacy_build"] = migrated
    if extracted:
        result["imported_backups"] = extracted
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
