from __future__ import annotations

from pathlib import Path

"""LangGraph pipeline configuration and defaults."""

SHIP_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = SHIP_ROOT / "scripts"

NODES: list[tuple[str, str, str]] = []

NODE_ORDER = [node_id for node_id, _, _ in NODES]

DEFAULT_START = 0
DEFAULT_END = 0
DEFAULT_EXECUTION_MODE = "inline"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
