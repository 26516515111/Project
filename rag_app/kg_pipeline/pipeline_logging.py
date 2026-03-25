from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any


class PipelineLogger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self._logger = logging.getLogger(f"kg_pipeline.{log_path.stem}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._logger.handlers.clear()

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        self._logger.addHandler(file_handler)
        self._logger.addHandler(stream_handler)

    def info(self, message: str) -> None:
        self._logger.info(message)

    def error(self, message: str) -> None:
        self._logger.error(message)

    def close(self) -> None:
        handlers = list(self._logger.handlers)
        for handler in handlers:
            handler.close()
            self._logger.removeHandler(handler)


def build_run_log_path(logs_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = logs_dir / f"pipeline_{timestamp}.log"
    if not base.exists():
        return base
    index = 1
    while True:
        candidate = logs_dir / f"pipeline_{timestamp}_{index:03d}.log"
        if not candidate.exists():
            return candidate
        index += 1


def summarize_output(output: Any) -> str:
    if not isinstance(output, dict) or not output:
        return ""
    parts: list[str] = []
    preferred_keys = [
        "documents",
        "files",
        "chunks",
        "entities",
        "relations",
        "cross_chunk_relations",
        "filtered_entities",
        "filtered_relations",
        "semantic_merges",
        "merge_log_entries",
        "imported",
        "dumped",
        "nodes",
        "edges",
        "output",
        "copied_into_raw",
        "moved_stale_cleaned",
    ]
    used_keys: set[str] = set()
    for key in preferred_keys:
        if key not in output:
            continue
        value = output[key]
        used_keys.add(key)
        if isinstance(value, list):
            parts.append(f"{key}={len(value)}")
        else:
            parts.append(f"{key}={value}")
    for key, value in output.items():
        if key in used_keys:
            continue
        if isinstance(value, list):
            parts.append(f"{key}={len(value)}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)
