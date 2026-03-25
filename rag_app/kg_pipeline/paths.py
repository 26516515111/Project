from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelinePaths:
    project_root: Path
    app_root: Path
    data_dir: Path
    docs_dir: Path
    kg_dir: Path
    backups_dir: Path
    cleaned_backups_dir: Path
    logs_dir: Path
    raw_dir: Path
    cleaned_dir: Path
    chunks_dir: Path
    extracted_dir: Path
    delivery_dir: Path
    env_path: Path
    cache_dir: Path

    @classmethod
    def discover(cls) -> "PipelinePaths":
        app_root = Path(__file__).resolve().parents[1]
        project_root = app_root.parent
        data_dir = app_root / "data"
        kg_dir = data_dir / "KG"
        return cls(
            project_root=project_root,
            app_root=app_root,
            data_dir=data_dir,
            docs_dir=data_dir / "docs",
            kg_dir=kg_dir,
            backups_dir=kg_dir / "backups",
            cleaned_backups_dir=kg_dir / "backups" / "cleaned",
            logs_dir=kg_dir / "logs",
            raw_dir=kg_dir / "raw",
            cleaned_dir=kg_dir / "cleaned",
            chunks_dir=kg_dir / "chunks",
            extracted_dir=kg_dir / "extracted",
            delivery_dir=kg_dir / "delivery",
            env_path=project_root / ".env",
            cache_dir=app_root / ".cache",
        )

    @property
    def env_paths(self) -> list[Path]:
        candidates = [self.project_root / ".env", self.app_root / ".env"]
        ordered: list[Path] = []
        seen: set[Path] = set()
        for path in candidates:
            if path in seen:
                continue
            ordered.append(path)
            seen.add(path)
        return ordered

    @property
    def chunks_path(self) -> Path:
        return self.chunks_dir / "chunks.json"

    @property
    def doc_map_path(self) -> Path:
        return self.chunks_dir / "doc_source_map.json"

    @property
    def kg_raw_path(self) -> Path:
        return self.extracted_dir / "kg_raw.json"

    @property
    def checkpoint_path(self) -> Path:
        return self.extracted_dir / "kg_raw_checkpoint.json"

    @property
    def kg_merged_path(self) -> Path:
        return self.delivery_dir / "kg_merged.json"

    @property
    def merge_log_path(self) -> Path:
        return self.delivery_dir / "entity_merge_log.json"

    @property
    def chunk_to_kg_path(self) -> Path:
        return self.chunks_dir / "chunk_to_kg.json"

    @property
    def neo4j_entities_csv_path(self) -> Path:
        return self.delivery_dir / "neo4j_entities.csv"

    @property
    def neo4j_relations_csv_path(self) -> Path:
        return self.delivery_dir / "neo4j_relations.csv"

    @property
    def neo4j_dump_path(self) -> Path:
        return self.delivery_dir / "neo4j.dump"

    @property
    def visualization_path(self) -> Path:
        return self.delivery_dir / "kg_visualization.html"

    def ensure_dirs(self) -> None:
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.kg_dir.mkdir(parents=True, exist_ok=True)
        self.backups_dir.mkdir(parents=True, exist_ok=True)
        self.cleaned_backups_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        self.delivery_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
