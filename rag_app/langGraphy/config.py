from __future__ import annotations

from pathlib import Path

"""LangGraph pipeline configuration and defaults."""

SHIP_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = SHIP_ROOT / "scripts"

NODES: list[tuple[int, str, str]] = [
    (1, "clean", "语料清洗"),
    (2, "chunk", "文本切块"),
    (3, "extract", "KG 三元组抽取"),
    (4, "merge", "实体合并"),
    (5, "neo4j", "Neo4j 导入与导出"),
    (6, "visualize", "图谱可视化"),
]

NODE_ORDER = [node_id for node_id, _, _ in NODES]

DEFAULT_START = 1
DEFAULT_END = 6
DEFAULT_EXECUTION_MODE = "inline"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
