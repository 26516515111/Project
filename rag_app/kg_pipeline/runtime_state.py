from __future__ import annotations

import shutil
import tarfile
from pathlib import Path
from typing import Any

from .paths import PipelinePaths
from .utils import numbered_path, read_json, write_json

ALL_STEPS = [1, 2, 3, 4, 5, 6]
DERIVED_DIR_NAMES = ["cleaned", "chunks", "extracted", "delivery"]


def load_kg_state(paths: PipelinePaths) -> dict[str, Any]:
    state = read_json(
        paths.state_path,
        {
            "kg_name": "",
            "used_files": [],
            "completed_steps": [],
            "last_archive": "",
        },
    )
    state["used_files"] = sorted({str(name) for name in state.get("used_files", [])})
    state["completed_steps"] = sorted(
        {
            int(step)
            for step in state.get("completed_steps", [])
            if isinstance(step, int) or str(step).isdigit()
        }
    )
    state.setdefault("kg_name", "")
    state.setdefault("last_archive", "")
    return state


def save_kg_state(paths: PipelinePaths, state: dict[str, Any]) -> None:
    payload = {
        "kg_name": state.get("kg_name", ""),
        "used_files": sorted({str(name) for name in state.get("used_files", [])}),
        "completed_steps": sorted(
            {
                int(step)
                for step in state.get("completed_steps", [])
                if isinstance(step, int) or str(step).isdigit()
            }
        ),
        "last_archive": state.get("last_archive", ""),
    }
    write_json(paths.state_path, payload)


def current_raw_files(paths: PipelinePaths) -> list[str]:
    names: list[str] = []
    for pattern in ("*.md", "*.txt"):
        names.extend(path.name for path in sorted(paths.raw_dir.glob(pattern)))
    return sorted(names)


def next_incomplete_step(completed_steps: list[int]) -> int | None:
    completed = set(completed_steps)
    for step in ALL_STEPS:
        if step not in completed:
            return step
    return None


def clear_derived_outputs(paths: PipelinePaths) -> None:
    derived_directories = [
        paths.cleaned_dir,
        paths.chunks_dir,
        paths.extracted_dir,
        paths.delivery_dir,
    ]
    for directory in derived_directories:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)


def archive_current_kg(paths: PipelinePaths, archive_name: str) -> Path:
    safe_name = archive_name.strip() or "KG"
    target = numbered_path(paths.archives_dir / f"{safe_name}.tar.gz")
    exclude_roots = {paths.backups_dir.resolve()}
    include_paths = [
        paths.build_dir,
    ]
    with tarfile.open(target, "w:gz") as tar:
        for item in include_paths:
            if not item.exists():
                continue
            resolved = item.resolve()
            if any(
                resolved == root or root in resolved.parents for root in exclude_roots
            ):
                continue
            arcname = item.relative_to(paths.data_dir)
            tar.add(item, arcname=str(arcname))
    return target


def merge_completed_steps(
    previous_steps: list[int], selected_steps: list[int], failed_step: int | None
) -> list[int]:
    merged = set(previous_steps)
    for step in selected_steps:
        if failed_step is not None and step >= failed_step:
            break
        merged.add(step)
    return sorted(merged)
