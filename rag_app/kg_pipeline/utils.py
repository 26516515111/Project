from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_env_file(env_path: Path) -> dict[str, str]:
    env_vars: dict[str, str] = {}
    if not env_path.exists():
        return env_vars
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env_vars[key.strip()] = value.strip().strip('"').strip("'")
    return env_vars


def apply_local_env(env_path: Path) -> dict[str, str]:
    loaded = load_env_file(env_path)
    for key, value in loaded.items():
        os.environ.setdefault(key, value)
    return loaded


def apply_local_envs(env_paths: list[Path]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for env_path in env_paths:
        loaded = load_env_file(env_path)
        for key, value in loaded.items():
            merged.setdefault(key, value)
            os.environ.setdefault(key, value)
    return merged


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def numbered_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    index = 1
    while True:
        candidate = parent / f"{stem}_{index:03d}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def same_file_content(left: Path, right: Path) -> bool:
    if not left.exists() or not right.exists():
        return False
    if left.stat().st_size != right.stat().st_size:
        return False
    return left.read_bytes() == right.read_bytes()
