from __future__ import annotations

import base64
import csv
import hashlib
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
from huggingface_hub import snapshot_download
from langchain_text_splitters import RecursiveCharacterTextSplitter
from neo4j import GraphDatabase
from openai import OpenAI
from pyvis.network import Network
from sentence_transformers import SentenceTransformer

from .config import (
    LLM_API_KEY_ENV,
    LLM_BASE_URL_ENV,
    LLM_MODEL_ENV,
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
    KG_MIN_CHUNK_SIZE,
)
from .constants import ENTITY_LABELS, RELATION_TYPES
from .paths import PipelinePaths
from .utils import (
    apply_local_env,
    apply_local_envs,
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

SYSTEM_PROMPT = f"""你是船舶电气设备维护与故障检修领域的知识图谱专家。本次任务包含解决系统级图谱层级与颗粒度混乱的专项要求。请从给定文本中抽取高价值实体和关系三元组。

实体标签（label）必须从以下选取：
{", ".join(ENTITY_LABELS)}

关系类型（relation）必须从以下选取：
{", ".join(RELATION_TYPES)}

【重要】避免细粒度爆炸与冗杂零件：
- 【强制底线】：禁止抽取任何细碎的基础支撑元器件、接口或外围引脚。坚决丢弃所有的“继电器”、“保险丝”、“电阻器”、“接线端子”、“各个插头(如J8/COM1)”、“指示灯”或“单一开关/按钮”等微观零件名称。
- 只有具备独立运算、控制或完整封装的【单板】、【控制箱】或【核心模块】才能被提取为节点。
- 所有低于单板/模块级别的零件维修信息（比如XX继电器损坏），必须折叠塞入所属单板节点（或者故障节点）的 `description` 描述文本内，绝不允许建立独立的树枝状子节点，以此降低图谱的密集度并提升检索性能！

【重要】清晰划分层级系统边界（消歧）：
- [System] 即整体大系统（如“机舱总线制监测报警系统”、“驾驶台航行值班报警系统”）。
- [Equipment] 是系统内独立成套的主体设备（如“控制箱”、“灯箱”）。
- [Component] 是设备内的关键部件（如“主控制板”、“电源模块”）。
- 绝不允许“大系统”和“内部组件”平级出现。在提取所有设备和组件名时，**必须带上其所属的上一级环境前缀**。例如不要提取孤立的“继电器”、“数据采集板”，必须提取为“机舱微机监测系统_数据采集板”、“控制箱_继电器”。这对于消灭成环、防毛线球极度重要！

标签判定要求：
- [System] 和 [Equipment]：用于整套系统或独立箱体。
- [Component]：用于单板、核心元器件或内部模块。
- [FaultPhenomenon]：明确描述异常症状、报警。
- [FaultCause]：诱发故障的原因。
- [DiagnosticMethod], [RepairMethod], [SafetyNote] 分别为检测诊断、维修处理和安全禁忌事宜。

关系抽取要求与语义丰富性：
- `BELONGS_TO` / `HAS_COMPONENT`：用于表示【System -> Equipment -> Component】的严格组成树结构，**绝不能出现互相包含或子包父的逻辑悖论（成环）**！
- `CAUSED_BY`：故障现象由某个原因导致。
- `DIAGNOSED_BY` / `REPAIRED_BY`：故障/设备的诊断和修复方法。
- `REQUIRES`：系统或设备“必须依赖”某项环境或参数才能工作。
- `PREVENTS`：某项安全措施或修补“防止”了某个故障。
- `CONTROLS` / `MONITORS`：用于表达逻辑上某主控设备“控制”或“监测”另一设备。
- `CONNECTS_TO`：用于设备模块之间的物理、电气或通讯链接。

输出约束：
- 只输出文本中明确出现或可以直接落地到维修语义的实体。
- 实体名称尽量具体、带有所属系统层级前缀、可复用。
- description 用一句中文短语说明该实体或关系在检修语境中的含义（请勿在此字段里塞入标签、度数或id等死数据）。
- 输出严格 JSON 格式，不要输出任何文字、解释或 markdown 代码块。

输出格式：
{{
  "entities": [
    {{"name": "带有前缀的规范实体名", "label": "标签", "description": "纯文字简要描述"}}
  ],
  "relations": [
    {{"head": "头实体名", "head_label": "头标签", "relation": "关系类型", "tail": "尾实体名", "tail_label": "尾标签", "description": "精准的业务级描述，而非简单重复标签"}}
  ]
}}
"""

QUALITY_SCORE_PROMPT = """你是船舶设备厂商说明书的知识抽取前置筛选器。

你的任务不是抽实体，而是判断当前 chunk 是否值得进入 KG 抽取。

判断标准：
1. 高分 chunk 应该能直接支持问答或检修知识检索，例如：
   - 设备/部件结构与组成
   - 参数、阈值、报警点、接线定义
   - 故障现象、故障原因、诊断方法、维修步骤
   - 操作限制、安全注意事项
   - 安装、调试、使用、维护流程
2. 低分 chunk 通常是：
   - 空泛概述、宣传性描述、厂商介绍
   - 没有明确对象和操作含义的泛化表述
   - 纯目录、纯封面、纯版权信息
3. 如果 chunk 信息密度低、概念空泛、无法稳定落到问答知识点，应判定为 skip。

输出严格 JSON，不要输出其他文字：
{
  "score": 0-100,
  "decision": "extract" 或 "skip",
  "category": "该 chunk 的主要知识类别",
  "reasons": ["原因1", "原因2"]
}
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
        target_root = paths.kg_dir if relative_target.startswith("delivery/") else paths.build_dir
        target = target_root / relative_target
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if same_file_content(source, target):
                source.unlink()
                removed_duplicates.append(source.name)
                continue
            target = numbered_path(target)
        shutil.move(str(source), str(target))
        moved.append({"from": legacy_name, "to": str(target.relative_to(paths.data_dir))})
    return {"moved": moved, "removed_duplicates": removed_duplicates}


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


def _image_extension_from_mime(mime_subtype: str) -> str:
    subtype = mime_subtype.lower().strip()
    if subtype in {"jpeg", "jpg"}:
        return ".jpg"
    if subtype == "png":
        return ".png"
    if subtype == "gif":
        return ".gif"
    if subtype == "webp":
        return ".webp"
    if subtype == "bmp":
        return ".bmp"
    return ".bin"


def _extract_base64_images(
    text: str, paths: PipelinePaths, doc_id: str
) -> tuple[str, list[dict[str, str]]]:
    image_dir = paths.images_dir / doc_id
    image_dir.mkdir(parents=True, exist_ok=True)
    saved_images: list[dict[str, str]] = []

    pattern = re.compile(
        r"!\[(?P<alt>.*?)\]\(data:image/(?P<mime>[a-zA-Z0-9.+-]+);base64,(?P<data>[^)]+)\)"
    )

    def replace(match: re.Match[str]) -> str:
        alt = (match.group("alt") or "").strip() or "未命名图片"
        mime_subtype = match.group("mime")
        encoded = match.group("data").strip()
        binary = base64.b64decode(encoded)
        digest = hashlib.sha256(binary).hexdigest()
        filename = f"{digest}{_image_extension_from_mime(mime_subtype)}"
        target = image_dir / filename
        if not target.exists():
            target.write_bytes(binary)
        rel_path = target.relative_to(paths.app_root)
        saved_images.append(
            {
                "path": str(rel_path),
                "alt": alt,
            }
        )
        return f"\n[IMG path={rel_path} alt={alt}]\n"

    return pattern.sub(replace, text), saved_images


def _normalize_manual_structure(text: str) -> str:
    value = text
    value = re.sub(
        r"^(#{1,6})\s+(.+)$",
        lambda match: f"[H{len(match.group(1))}] {match.group(2).strip()}",
        value,
        flags=re.MULTILINE,
    )
    value = re.sub(
        r"^\s*([0-9]+)[、\.．]\s*(.+)$",
        lambda match: f"[H2] {match.group(1)}、{match.group(2).strip()}",
        value,
        flags=re.MULTILINE,
    )
    value = re.sub(
        r"^\s*([①②③④⑤⑥⑦⑧⑨⑩])\s*[、．.]?\s*(.+)$",
        lambda match: f"[H3] {match.group(1)} {match.group(2).strip()}",
        value,
        flags=re.MULTILINE,
    )
    value = re.sub(
        r"^\s*\(([0-9]+)\)\s*(.+)$",
        lambda match: f"[H4] ({match.group(1)}) {match.group(2).strip()}",
        value,
        flags=re.MULTILINE,
    )
    value = re.sub(
        r"^\s*（([0-9]+)）\s*(.+)$",
        lambda match: f"[H4] （{match.group(1)}） {match.group(2).strip()}",
        value,
        flags=re.MULTILINE,
    )
    value = re.sub(
        r"^\s*([一二三四五六七八九十]+)、\s*(.+)$",
        lambda match: f"[H3] {match.group(1)}、{match.group(2).strip()}",
        value,
        flags=re.MULTILINE,
    )
    value = _rebalance_heading_levels(value)
    return value


def _is_real_chapter_heading(title: str) -> bool:
    stripped = title.strip()
    return bool(
        re.match(r"^[一二三四五六七八九十]+、.+$", stripped)
        or re.match(r"^第[一二三四五六七八九十0-9]+[章节部分篇].+$", stripped)
    )


def _looks_like_subsection_heading(title: str) -> bool:
    stripped = title.strip()
    return bool(
        re.match(r"^[0-9]+[、\.．:].+$", stripped)
        or re.match(r"^[a-zA-Z][、\.．:].+$", stripped)
        or re.match(r"^\([0-9]+\)\s*.+$", stripped)
        or re.match(r"^（[0-9]+）\s*.+$", stripped)
        or re.match(r"^[ivxlcdmIVXLCDM]+[、\.．:].+$", stripped)
    )


def _rebalance_heading_levels(text: str) -> str:
    lines = text.splitlines()
    result: list[str] = []
    seen_real_h1 = False
    warning_titles = {"重要事项", "注意事项", "注意", "警告", "危险", "小心"}
    preface_titles = {"欢迎", "欢迎!", "前言", "preface", "introduction"}

    for line in lines:
        match = re.match(r"^\s*\[H([1-4])\]\s+(.+)$", line.strip())
        if not match:
            result.append(line)
            continue

        level = int(match.group(1))
        title = match.group(2).strip()
        new_level = level

        if title in warning_titles:
            new_level = max(level, 3)
        if title.lower() in preface_titles:
            new_level = 2
        if level == 1:
            if _is_real_chapter_heading(title):
                seen_real_h1 = True
            elif seen_real_h1 or _looks_like_subsection_heading(title):
                new_level = 2

        result.append(f"[H{new_level}] {title}")

    return "\n".join(result)


def _trim_manual_front_matter(text: str) -> str:
    lines = text.splitlines()

    def is_heading(value: str) -> bool:
        return bool(re.match(r"^\[H[1-4]\]\s+.+$", value))

    def is_title_only_heading(value: str) -> bool:
        plain = re.sub(r"^\[H[1-4]\]\s+", "", value).strip()
        normalized = re.sub(r"\s+", "", plain)
        if normalized in {"目录", "目次", "目錄", "目錄表", "目录表", "目录"}:
            return True
        if "产品说明书" in plain or "使用说明书" in plain:
            return True
        if plain.endswith("说明书") and len(plain) <= 30:
            return True
        if re.fullmatch(r"[A-Z0-9,\-.\s()&]+", plain):
            return True
        return False

    def is_contact_line(value: str) -> bool:
        return any(
            re.search(pattern, value)
            for pattern in (
                r"^地址[：:]",
                r"^Add[:：]",
                r"^电话[：:]",
                r"^Tel[:：]",
                r"^传真[：:]",
                r"^Fax[:：]",
                r"^邮编[：:]",
                r"^P\.?C\.?[:：]",
                r"^电子邮箱[：:]",
                r"^E-?mail[:：]",
                r"^网址[：:]",
                r"^Http",
                r".+公司$",
            )
        )

    def heading_has_body(index: int) -> bool:
        body_lines = 0
        lookahead = 0
        for probe in range(index + 1, min(len(lines), index + 12)):
            lookahead += 1
            stripped_probe = lines[probe].strip()
            if not stripped_probe:
                continue
            if "[IMG " in stripped_probe:
                continue
            if is_heading(stripped_probe):
                return body_lines > 0
            if is_contact_line(stripped_probe):
                return False
            body_lines += 1
            if body_lines >= 1:
                return True
            if lookahead >= 10:
                break
        return body_lines > 0

    start_index = 0
    first_h1_candidate = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if "[IMG " in stripped:
            continue
        if is_contact_line(stripped):
            continue
        if not is_heading(stripped):
            continue
        if is_title_only_heading(stripped):
            continue
        level = int(re.match(r"^\[H([1-4])\]\s+.+$", stripped).group(1))
        if level == 1 and first_h1_candidate is None:
            first_h1_candidate = index
        if heading_has_body(index):
            start_index = index
            break
    if first_h1_candidate is not None and start_index == 0:
        # 仅在无法识别有效正文标题时，才退化到首个 H1
        return "\n".join(lines[first_h1_candidate:]).strip()
    return "\n".join(lines[start_index:]).strip()


def _trim_to_first_real_content_heading(text: str) -> str:
    lines = text.splitlines()

    def is_real_h1(value: str) -> bool:
        stripped = value.strip()
        return bool(
            re.match(r"^\[H1\]\s*[一二三四五六七八九十]+、.+$", stripped)
            or re.match(r"^\[H1\]\s*第[一二三四五六七八九十0-9]+[章节部分篇].+$", stripped)
            or re.match(r"^\[H1\]\s*[0-9]+[、\.．].+$", stripped)
            or re.search(r"^\[H1\]\s*(introduction|overview|installation|operation|maintenance|troubleshooting|specification)s?\b", stripped, flags=re.IGNORECASE)
            or re.search(r"^\[H1\].*(hydraulic\s+system|diesel\s+engine|steering\s+gear|stabilizer|thruster)", stripped, flags=re.IGNORECASE)
        )

    for index, line in enumerate(lines[:400]):
        if is_real_h1(line):
            return "\n".join(lines[index:]).strip()

    return text.strip()


def _trim_manual_tail_matter(text: str) -> str:
    lines = text.splitlines()
    tail_patterns = (
        r"本资料由.+公司.+编制",
        r"^编制[：:]",
        r"^审核[：:]",
        r"^批准[：:]",
        r"^地址[：:]",
        r"^电话[：:]",
        r"^传真[：:]",
        r"^网址[：:]",
        r"^邮编[：:]",
        r"版权所有",
    )
    cutoff = None
    threshold = int(len(lines) * 0.7)
    for index, line in enumerate(lines):
        if index < threshold:
            continue
        stripped = line.strip()
        if any(re.search(pattern, stripped) for pattern in tail_patterns):
            cutoff = index
            break
    if cutoff is None:
        return text.strip()
    return "\n".join(lines[:cutoff]).strip()


def _trim_leading_toc_sections(text: str) -> str:
    lines = text.splitlines()
    index = 0

    def is_heading(value: str) -> bool:
        return bool(re.match(r"^\[H[1-4]\]\s+.+$", value))

    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            continue
        if not is_heading(stripped):
            break
        body_count = 0
        probe = index + 1
        while probe < len(lines):
            candidate = lines[probe].strip()
            if not candidate:
                probe += 1
                continue
            if "[IMG " in candidate:
                probe += 1
                continue
            if is_heading(candidate):
                break
            body_count += 1
            if body_count >= 1:
                return "\n".join(lines[index:]).strip()
            probe += 1
        index = probe
    return "\n".join(lines[index:]).strip()


def _looks_like_toc_entry(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if re.match(r"^\[H[1-6]\]\s+", stripped):
        stripped = re.sub(r"^\[H[1-6]\]\s+", "", stripped).strip()
    compact = stripped.replace(" ", "")
    if "..." in stripped and re.search(r"\d+\s*$", stripped):
        return True
    if re.search(r"[\.。·]{8,}", stripped) and re.search(r"\d+\s*$", stripped):
        return True
    if re.match(r"^[^\n]{1,80}\.{5,}\s*\d+\s*$", stripped):
        return True
    if re.match(r"^[^\n]{1,80}…{3,}\s*\d+\s*$", stripped):
        return True
    if re.match(r"^[A-Za-z\u4e00-\u9fff（）()0-9,，、\- /]{1,80}\s+\d{1,4}$", compact):
        return True
    return False


def _drop_toc_blocks(text: str) -> str:
    lines = text.splitlines()
    kept: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            i += 1
            continue

        heading_plain = re.sub(r"^\[H[1-6]\]\s+", "", stripped).strip()
        is_toc_heading = heading_plain in {"目录", "内容", "目次", "索引", "Contents", "CONTENTS", "Index", "INDEX"}
        toc_run = 0
        probe = i + 1 if is_toc_heading else i
        while probe < len(lines):
            candidate = lines[probe].strip()
            if not candidate:
                probe += 1
                continue
            if _looks_like_toc_entry(candidate):
                toc_run += 1
                probe += 1
                continue
            break

        if is_toc_heading and toc_run >= 3:
            i = probe
            continue
        if not is_toc_heading and toc_run >= 8:
            i = probe
            continue

        kept.append(line)
        i += 1
    return "\n".join(kept).strip()


def _drop_index_tail(text: str) -> str:
    lines = text.splitlines()
    cutoff: int | None = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        plain = re.sub(r"^\[H[1-6]\]\s+", "", stripped).strip()
        if plain not in {"索引", "Index", "INDEX"}:
            continue
        score = 0
        for probe in range(idx + 1, min(len(lines), idx + 80)):
            candidate = lines[probe].strip()
            if not candidate:
                continue
            if _looks_like_toc_entry(candidate):
                score += 2
                continue
            if re.match(r"^[A-Za-z\u4e00-\u9fff]$", candidate):
                score += 1
                continue
            if re.match(r"^\[H[2-6]\]\s*[A-Za-z\u4e00-\u9fff]$", candidate):
                score += 1
                continue
        if score >= 8:
            cutoff = idx
            break
    if cutoff is None:
        return text.strip()
    return "\n".join(lines[:cutoff]).strip()


def _is_foreign_dominant_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if "[IMG " in stripped:
        return False
    plain = re.sub(r"^\[H[1-6]\]\s+", "", stripped).strip()
    swedish_markers = (
        "allmänt",
        "säkerhetsinformation",
        "denna",
        "motorn",
        "service",
        "underhåll",
        "varning",
        "för",
        "och",
        "inte",
        "använd",
        "volvo penta",
    )
    lowered = plain.lower()
    if any(marker in lowered for marker in swedish_markers):
        return True
    if len(plain) < 20:
        return False

    chinese = len(re.findall(r"[\u4e00-\u9fff]", plain))
    latin = len(re.findall(r"[A-Za-zÅÄÖåäö]", plain))
    digits = len(re.findall(r"\d", plain))
    return latin >= 12 and chinese <= 2 and (latin + digits) > chinese * 6


def _drop_foreign_language_noise(text: str) -> str:
    lines = text.splitlines()
    kept: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if kept and kept[-1] != "":
                kept.append("")
            continue
        if _is_foreign_dominant_line(stripped):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _inject_leading_context_heading(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text
    first_nonempty = next((line.strip() for line in lines if line.strip()), "")
    if re.match(r"^\[H[1-6]\]\s+", first_nonempty):
        return text
    leading_blob = "\n".join(line.strip() for line in lines[:20] if line.strip())
    if any(token in leading_blob for token in ("安全", "警告", "危险", "注意事项", "务必遵守当地的安全说明")):
        return "[H2] 安全信息\n" + text.strip()
    return text


def _drop_leading_preface_sections(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text
    preface_titles = {"欢迎", "欢迎!", "前言", "preface", "introduction"}
    first_heading_index: int | None = None
    first_heading_title = ""
    first_heading_level = 0
    next_real_heading_index: int | None = None
    next_real_heading_title = ""

    for idx, line in enumerate(lines[:120]):
        match = re.match(r"^\s*\[H([1-6])\]\s+(.+)$", line.strip())
        if not match:
            continue
        title = match.group(2).strip()
        if first_heading_index is None:
            first_heading_index = idx
            first_heading_title = title
            first_heading_level = int(match.group(1))
            continue
        next_real_heading_index = idx
        next_real_heading_title = title
        break

    if first_heading_index is None or next_real_heading_index is None:
        return text

    if first_heading_title.lower() in preface_titles and (
        "安全信息" in next_real_heading_title
        or "一般信息" in next_real_heading_title
        or "介绍" in next_real_heading_title
    ):
        return "\n".join(lines[next_real_heading_index:]).strip()

    if first_heading_level >= 3 and "安全信息" in next_real_heading_title and (
        "欢迎" in first_heading_title or "前言" in first_heading_title
    ):
        return "\n".join(lines[next_real_heading_index:]).strip()

    return text


def _drop_preface_before_first_h1(text: str) -> str:
    lines = text.splitlines()
    first_h1 = None
    headings_before_h1 = 0
    preface_blob: list[str] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if first_h1 is None:
            preface_blob.append(stripped)
        match = re.match(r"^\[H([1-4])\]\s+.+$", stripped)
        if not match:
            continue
        level = int(match.group(1))
        if level == 1:
            first_h1 = index
            break
        headings_before_h1 += 1
    if first_h1 is not None:
        preface_text = "\n".join(preface_blob)
        if headings_before_h1 >= 3:
            return "\n".join(lines[first_h1:]).strip()
        if any(
            keyword in preface_text
            for keyword in ("控制对象", "船东", "船厂", "船舶用途", "产品设计")
        ):
            return "\n".join(lines[first_h1:]).strip()
    return text.strip()


def _prefer_first_real_chapter(text: str) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines[:300]):
        stripped = line.strip()
        if re.match(r"^\[H1\]\s*[一二三四五六七八九十]+、.+$", stripped):
            return "\n".join(lines[index:]).strip()
        if re.match(r"^\[H1\]\s*第[一二三四五六七八九十0-9]+[章节部分篇].+$", stripped):
            return "\n".join(lines[index:]).strip()
    return text.strip()


def clean_text(text: str, paths: PipelinePaths, doc_id: str) -> tuple[str, list[dict[str, str]]]:
    text = re.sub(
        r"<table[^>]*>.*?</table>",
        lambda match: html_table_to_text(match.group(0)),
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text, saved_images = _extract_base64_images(text, paths, doc_id)
    text = re.sub(r"IMAGE_PLACEHOLDER\s*#?\d*", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = _normalize_manual_structure(text)
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
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    cleaned = _drop_toc_blocks(cleaned)
    anchored = _trim_to_first_real_content_heading(cleaned)
    used_anchor = anchored != cleaned
    cleaned = anchored if used_anchor else _trim_manual_front_matter(cleaned)
    if not used_anchor:
        cleaned = _trim_leading_toc_sections(cleaned)
    cleaned = _drop_foreign_language_noise(cleaned)
    cleaned = _drop_toc_blocks(cleaned)
    cleaned = _drop_index_tail(cleaned)
    cleaned = _trim_manual_tail_matter(cleaned)
    cleaned = _drop_leading_preface_sections(cleaned)
    cleaned = _inject_leading_context_heading(cleaned)
    cleaned = _rebalance_heading_levels(cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, saved_images


def clean_documents(paths: PipelinePaths) -> dict[str, Any]:
    files = _raw_documents(paths)
    if not files:
        raise RuntimeError("data/KG/raw 中没有可处理的 .md/.txt 文档")

    cleaned = []
    for source in files:
        raw_text = source.read_text(encoding="utf-8")
        doc_id = make_doc_id(source.name)
        output, saved_images = clean_text(raw_text, paths, doc_id)
        target = paths.cleaned_dir / source.name
        target.write_text(output, encoding="utf-8")
        cleaned.append(
            {
                "source": source.name,
                "raw_chars": len(raw_text),
                "cleaned_chars": len(output),
                "images": len(saved_images),
            }
        )
    return {"documents": len(cleaned), "files": cleaned}


def make_doc_id(filename: str) -> str:
    return Path(filename).stem


def extract_heading_context(text: str) -> str:
    path = extract_heading_path(text)
    return " -> ".join(path) if path else ""


def extract_heading_path(text: str) -> list[str]:
    path: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^\s*\[H([1-6])\]\s+(.+)$", line.strip())
        if not match:
            continue
        level = int(match.group(1))
        title = match.group(2).strip()
        while len(path) >= level:
            path.pop()
        path.append(f"[H{level}] {title}")
    return path


def extract_chunk_attachments(text: str) -> tuple[str, list[dict[str, str]]]:
    attachments: list[dict[str, str]] = []
    pattern = re.compile(r"^\[IMG path=(?P<path>[^\s\]]+) alt=(?P<alt>.+?)\]$", flags=re.MULTILINE)

    def replace(match: re.Match[str]) -> str:
        attachments.append(
            {
                "type": "image",
                "path": match.group("path").strip(),
                "alt": match.group("alt").strip(),
            }
        )
        return f"[图片附件] {match.group('alt').strip()}"

    cleaned = pattern.sub(replace, text)
    return cleaned.strip(), attachments


def infer_semantic_group(text: str, heading_path: list[str]) -> str:
    value = "\n".join(heading_path) + "\n" + text
    rules = [
        ("fault", ("故障", "异常", "报警", "不能", "失效", "原因", "现象", "fault", "alarm", "failure", "cause")),
        ("repair", ("维修", "排除", "更换", "调整", "检修", "处理步骤", "repair", "replace", "overhaul", "service")),
        ("diagnostic", ("检测", "诊断", "测量", "检查", "判别", "测试", "diagnos", "inspection", "test", "measurement")),
        ("safety", ("注意", "警告", "严禁", "必须", "安全", "warning", "caution", "safety", "danger")),
        ("wiring", ("接线", "端子", "通讯线", "接口", "RS-485", "输入信号", "输出信号", "wiring", "terminal", "connector", "interface")),
        ("parameter", ("参数", "规格", "电压", "电流", "温度", "压力", "防护等级", "specification", "parameter", "pressure", "temperature", "voltage", "current")),
        ("component", ("组成", "部件", "模块", "单元", "控制计算机", "显示器", "打印机", "component", "module", "valve", "pump", "cylinder", "thruster", "stabilizer")),
        ("operation", ("操作", "使用", "调试", "启动", "运行", "显示内容", "operation", "procedure", "startup", "commissioning", "manual")),
    ]
    for name, keywords in rules:
        if any(keyword in value for keyword in keywords):
            return name
    return "overview"


def infer_system_root(heading_path: list[str], text: str) -> str:
    """推断 chunk 所属系统根，避免跨系统混拼。"""
    for heading in heading_path:
        raw = re.sub(r"^\[H\d\]\s+", "", heading).strip()
        if not raw:
            continue
        if any(
            key in raw
            for key in (
                "系统",
                "主机",
                "柴油机",
                "液压",
                "舵",
                "减摇",
                "稳性",
                "推进",
                "控制",
                "Hydraulic",
                "hydraulic",
                "Engine",
                "engine",
                "Steering",
                "steering",
                "Stabilizer",
                "stabilizer",
                "Thruster",
                "thruster",
            )
        ):
            return raw[:80]

    value = "\n".join(heading_path) + "\n" + text
    system_patterns = [
        r"([\w\-（）()\u4e00-\u9fff]{2,40}系统)",
        r"([\w\-（）()\u4e00-\u9fff]{2,40}柴油机)",
        r"([\w\-（）()\u4e00-\u9fff]{2,40}液压[\w\-（）()\u4e00-\u9fff]{0,20})",
        r"([\w\-（）()\u4e00-\u9fff]{2,40}舵[\w\-（）()\u4e00-\u9fff]{0,20})",
        r"([A-Za-z0-9\- ]{3,60}(Hydraulic System|Steering System|Stabilizer|Thruster|Diesel Engine))",
        r"((Hydraulic|Steering|Stabilizer|Thruster|Diesel Engine)[A-Za-z0-9\- ]{0,40})",
    ]
    for pattern in system_patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(1).strip()[:80]

    if heading_path:
        return re.sub(r"^\[H\d\]\s+", "", heading_path[0]).strip()[:80]
    return "global"


def _heading_depth(heading_path: list[str]) -> int:
    return len(heading_path or [])


def _is_low_value_chunk(text: str, heading_path: list[str], semantic_group: str) -> bool:
    lowered = text.lower()
    heading_text = " ".join(heading_path or [])

    # 目录/封面/联系方式等结构性噪声
    noisy_tokens = (
        "目录",
        "contents",
        "table of content",
        "owner",
        "handbook",
        "preface",
        "版权",
        "版权所有",
        "地址",
        "电话",
        "传真",
        "email",
        "www.",
        "thank you",
    )
    if any(token in lowered or token in heading_text for token in noisy_tokens):
        return True

    contact_hits = 0
    for line in text.splitlines():
        stripped = line.strip().lower()
        if not stripped:
            continue
        if re.search(r"(tel\.?|fax|www\.|@|\+[0-9]{1,3}\s?[0-9]{2,}|mail)", stripped):
            contact_hits += 1
        if stripped in {"usa", "united kingdom", "netherlands", "france", "asia pacific"}:
            contact_hits += 1
    if contact_hits >= 3:
        return True

    # 仅标题、几乎无正文
    body_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not re.match(r"^\[H[1-6]\]\s+.+$", line.strip())
    ]
    if len(body_lines) <= 1 and len(text) < 220:
        return True

    # 过短且无技术词，直接丢弃
    tech_tokens = (
        "压力",
        "阀",
        "泵",
        "液压",
        "接线",
        "参数",
        "报警",
        "故障",
        "维修",
        "诊断",
        "柴油机",
        "发动机",
        "转速",
        "油压",
        "温度",
    )
    if len(text) < 160 and not any(token in text for token in tech_tokens):
        return True

    # overview 且技术信号弱，优先舍弃
    if semantic_group == "overview":
        signal = sum(1 for token in tech_tokens if token in text)
        if signal == 0 and len(text) < 700:
            return True

    return False


def _chunk_keep_score(chunk: dict[str, Any]) -> int:
    text = str(chunk.get("text", ""))
    heading_path = chunk.get("heading_path", [])
    semantic_group = str(chunk.get("semantic_group", "overview"))
    score = 30

    group_bonus = {
        "fault": 30,
        "repair": 30,
        "diagnostic": 28,
        "wiring": 25,
        "parameter": 22,
        "component": 18,
        "safety": 20,
        "operation": 16,
        "overview": 0,
    }
    score += group_bonus.get(semantic_group, 0)

    if _heading_depth(heading_path) >= 2:
        score += 8
    if _heading_depth(heading_path) >= 3:
        score += 6

    high_value_terms = (
        "故障",
        "报警",
        "维修",
        "排除",
        "诊断",
        "接线",
        "输入",
        "输出",
        "参数",
        "阈值",
        "压力",
        "温度",
        "转速",
        "油压",
        "液压",
        "柴油机",
        "发动机",
    )
    score += min(25, sum(1 for term in high_value_terms if term in text) * 3)

    if len(text) < 180:
        score -= 12
    if len(text) > 1400:
        score -= 6

    if _is_low_value_chunk(text, heading_path, semantic_group):
        score -= 40

    return max(0, min(100, score))


def _build_sections_from_headings(text: str) -> list[dict[str, Any]]:
    lines = [line.rstrip() for line in text.splitlines()]
    sections: list[dict[str, Any]] = []
    path: list[str] = []
    current_lines: list[str] = []
    current_path: list[str] = []

    warning_titles = {
        "重要事项",
        "注意事项",
        "注意",
        "警告",
        "危险",
        "小心",
        "危险!",
        "警告!",
        "注意!",
    }
    section_keywords = (
        "系统",
        "计划",
        "程序",
        "步骤",
        "检查",
        "更换",
        "保养",
        "维护",
        "润滑",
        "冷却",
        "燃油",
        "排气",
        "电气",
        "参数",
        "标贴",
        "部件",
        "元件",
        "控制",
        "操纵",
        "安装",
        "拆卸",
        "清洁",
        "概述",
        "简介",
        "故障",
        "诊断",
        "修理",
        "修复",
        "发动机",
        "变速箱",
        "仪表",
        "技术",
        "化学产品",
    )

    def heading_has_inline_body(value: str) -> bool:
        match = re.match(r"^\s*\[H([1-4])\]\s+(.+)$", value.strip())
        if not match:
            return False
        title = match.group(2).strip()
        title_core = re.sub(
            r"^[0-9一二三四五六七八九十①②③④⑤⑥⑦⑧⑨⑩()（）\s\-.．、]+",
            "",
            title,
        ).strip()
        if len(title_core) >= 20:
            return True
        return any(
            token in title_core
            for token in ("适用于", "采用", "具有", "可以", "可实现", "用于", "应", "必须", "不得")
        )

    def heading_title_core(title: str) -> str:
        return re.sub(
            r"^[0-9一二三四五六七八九十①②③④⑤⑥⑦⑧⑨⑩()（）\s\-.．、•]+",
            "",
            title.strip(),
        ).strip()

    def is_figure_or_code_heading(title: str) -> bool:
        normalized = title.strip()
        if re.match(r"^(图|表|Figure|P\d{5,}|图\s*\d+)", normalized, re.IGNORECASE):
            return True
        if re.match(r"^[A-Z][A-Z0-9/,\- ]{2,20}$", normalized):
            return True
        return False

    def is_step_like_heading(title: str) -> bool:
        normalized = title.strip()
        if re.match(r"^[0-9]+[、.．)]", normalized) and len(normalized) > 12:
            return True
        if re.match(r"^[①②③④⑤⑥⑦⑧⑨⑩]", normalized):
            return True
        return False

    def is_soft_heading(level: int, title: str, active_path: list[str]) -> bool:
        core = heading_title_core(title)
        if not core:
            return True
        if (
            level == 2
            and active_path
            and active_path[0].startswith("[H2] 安全信息")
            and "安全信息" not in core
        ):
            return True
        if core in warning_titles:
            return True
        if is_figure_or_code_heading(core):
            return True
        if is_step_like_heading(title):
            return True
        if core.startswith("图 ") or core.startswith("表 "):
            return True
        if core in {"有关发动机的信息", "个人安全设备", "保护您的眼睛", "保护您的皮肤"}:
            return True
        if level >= 4:
            return True
        if level == 3:
            has_section_keyword = any(keyword in core for keyword in section_keywords)
            if not has_section_keyword:
                return True
            if len(core) <= 4:
                return True
        return False

    def flush() -> None:
        if not current_lines:
            return
        content_lines = [line for line in current_lines if line.strip()]
        content = "\n".join(content_lines).strip()
        if not content:
            return
        body_lines = [
            line.strip()
            for line in content_lines[1:]
            if line.strip() and not re.match(r"^\s*\[H[1-4]\]\s+.+$", line.strip())
        ]
        has_image_body = any("[IMG " in line for line in content_lines[1:])
        has_inline_body = any(heading_has_inline_body(line) for line in content_lines)
        if not body_lines and not has_image_body and not has_inline_body:
            return
        sections.append(
            {
                "heading_path": list(current_path),
                "heading_context": " -> ".join(current_path) if current_path else "",
                "semantic_group": infer_semantic_group(content, current_path),
                "system_root": infer_system_root(current_path, content),
                "text": content,
            }
        )

    for line in lines:
        stripped = line.strip()
        match = re.match(r"^\s*\[H([1-4])\]\s+(.+)$", stripped)
        if match:
            level = int(match.group(1))
            title_text = match.group(2).strip()
            title = f"[H{level}] {title_text}"
            if is_soft_heading(level, title_text, path):
                if not current_lines:
                    current_lines = list(path)
                    current_path = list(path)
                current_lines.append(title)
                continue
            flush()
            while len(path) >= level:
                path.pop()
            path.append(title)
            current_path = list(path)
            current_lines = [title]
            continue
        if not current_lines and stripped:
            current_lines = list(path)
            current_path = list(path)
        if current_lines or stripped:
            current_lines.append(line)
    flush()
    return sections


def _merge_semantic_sections(
    sections: list[dict[str, Any]], chunk_size: int
) -> list[dict[str, Any]]:
    if not sections:
        return []
    merged: list[dict[str, Any]] = []
    for section in sections:
        if not merged:
            merged.append(section)
            continue
        last = merged[-1]
        same_root = (
            bool(last["heading_path"])
            and bool(section["heading_path"])
            and last["heading_path"][:1] == section["heading_path"][:1]
        )
        left_system = (last.get("system_root") or "").strip().lower()
        right_system = (section.get("system_root") or "").strip().lower()
        same_system = (
            left_system == right_system
            or not left_system
            or not right_system
            or left_system == "global"
            or right_system == "global"
        )
        same_family = _semantic_family(last["semantic_group"]) == _semantic_family(section["semantic_group"])
        combined_len = len(last["text"]) + 2 + len(section["text"])
        if same_root and same_system and same_family and combined_len <= int(chunk_size * 1.8):
            last["text"] = f"{last['text']}\n\n{section['text']}"
            last["heading_path"] = (
                last["heading_path"]
                if len(last["heading_path"]) >= len(section["heading_path"])
                else section["heading_path"]
            )
            last["heading_context"] = " -> ".join(last["heading_path"]) if last["heading_path"] else ""
            if not last.get("system_root"):
                last["system_root"] = section.get("system_root", "")
        else:
            merged.append(section)
    return merged


def _semantic_family(group: str) -> str:
    families = {
        "overview": "overview",
        "component": "overview",
        "operation": "overview",
        "repair": "maintenance",
        "diagnostic": "maintenance",
        "fault": "maintenance",
        "safety": "maintenance",
        "parameter": "spec",
        "wiring": "spec",
    }
    return families.get(group, group)


def _merge_section_pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    merged["text"] = f"{left['text']}\n\n{right['text']}"
    merged["heading_path"] = (
        left["heading_path"]
        if len(left["heading_path"]) >= len(right["heading_path"])
        else right["heading_path"]
    )
    merged["heading_context"] = (
        " -> ".join(merged["heading_path"]) if merged["heading_path"] else ""
    )
    if left["semantic_group"] == right["semantic_group"]:
        merged["semantic_group"] = left["semantic_group"]
    elif left["semantic_group"] == "overview":
        merged["semantic_group"] = right["semantic_group"]
    else:
        merged["semantic_group"] = left["semantic_group"]
    merged["system_root"] = left.get("system_root") or right.get("system_root") or "global"
    return merged


def _pack_short_semantic_sections(
    sections: list[dict[str, Any]],
    chunk_size: int,
    min_chunk_size: int,
) -> list[dict[str, Any]]:
    if not sections:
        return []

    packed: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    tiny_chunk_size = max(180, int(min_chunk_size * 0.72))
    soft_chunk_size = max(260, int(min_chunk_size * 0.82))

    def same_root(left: dict[str, Any], right: dict[str, Any]) -> bool:
        return bool(left["heading_path"]) and bool(right["heading_path"]) and left["heading_path"][:1] == right["heading_path"][:1]

    def same_system(left: dict[str, Any], right: dict[str, Any]) -> bool:
        left_system = (left.get("system_root") or "").strip().lower()
        right_system = (right.get("system_root") or "").strip().lower()
        return (
            left_system == right_system
            or not left_system
            or not right_system
            or left_system == "global"
            or right_system == "global"
        )

    def compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
        return (
            same_root(left, right)
            and same_system(left, right)
            and _semantic_family(left["semantic_group"]) == _semantic_family(right["semantic_group"])
        )

    def can_merge(left: dict[str, Any], right: dict[str, Any]) -> bool:
        return compatible(left, right) and len(left["text"]) + 2 + len(right["text"]) <= int(chunk_size * 1.65)

    def can_merge_tiny(left: dict[str, Any], right: dict[str, Any]) -> bool:
        return (
            same_root(left, right)
            and same_system(left, right)
            and len(left["text"]) + 2 + len(right["text"]) <= int(chunk_size * 1.45)
        )

    def can_merge_soft(left: dict[str, Any], right: dict[str, Any]) -> bool:
        return (
            same_root(left, right)
            and same_system(left, right)
            and len(left["text"]) + 2 + len(right["text"]) <= int(chunk_size * 1.7)
        )

    for section in sections:
        if pending is None:
            pending = section
            continue
        if len(pending["text"]) < min_chunk_size and can_merge(pending, section):
            pending = _merge_section_pair(pending, section)
            continue
        packed.append(pending)
        pending = section

    if pending is not None:
        packed.append(pending)

    changed = True
    while changed:
        changed = False
        refined: list[dict[str, Any]] = []
        for section in packed:
            if (
                refined
                and len(section["text"]) < min_chunk_size
                and can_merge(refined[-1], section)
            ):
                refined[-1] = _merge_section_pair(refined[-1], section)
                changed = True
                continue
            refined.append(section)
        packed = refined

    changed = True
    while changed:
        changed = False
        refined = []
        for section in packed:
            if (
                refined
                and len(section["text"]) < tiny_chunk_size
                and can_merge_tiny(refined[-1], section)
            ):
                refined[-1] = _merge_section_pair(refined[-1], section)
                changed = True
                continue
            refined.append(section)
        packed = refined

    changed = True
    while changed:
        changed = False
        refined = []
        for section in packed:
            if (
                refined
                and len(section["text"]) < soft_chunk_size
                and can_merge_soft(refined[-1], section)
            ):
                refined[-1] = _merge_section_pair(refined[-1], section)
                changed = True
                continue
            refined.append(section)
        packed = refined

    # Final rescue pass: aggressively attach remaining small sections to neighbors
    rescue_min = min_chunk_size
    max_rescue_len = int(chunk_size * 2.0)
    if len(packed) >= 2:
        forward: list[dict[str, Any]] = []
        idx = 0
        while idx < len(packed):
            current = packed[idx]
            if (
                len(current["text"]) < rescue_min
                and idx + 1 < len(packed)
                and same_root(current, packed[idx + 1])
                and same_system(current, packed[idx + 1])
                and len(current["text"]) + 2 + len(packed[idx + 1]["text"]) <= max_rescue_len
            ):
                merged = _merge_section_pair(current, packed[idx + 1])
                forward.append(merged)
                idx += 2
                continue
            forward.append(current)
            idx += 1
        packed = forward

    changed = True
    while changed:
        changed = False
        refined = []
        for section in packed:
            if (
                refined
                and len(section["text"]) < rescue_min
                and same_root(refined[-1], section)
                and same_system(refined[-1], section)
                and len(refined[-1]["text"]) + 2 + len(section["text"]) <= max_rescue_len
            ):
                refined[-1] = _merge_section_pair(refined[-1], section)
                changed = True
                continue
            refined.append(section)
        packed = refined

    return packed


def _split_large_semantic_section(
    section: dict[str, Any], chunk_size: int, chunk_overlap: int
) -> list[dict[str, Any]]:
    text = section["text"].strip()
    if len(text) <= chunk_size:
        return [section]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "；", "，", " "],
    )
    heading_prefix = "\n".join(section.get("heading_path", []))
    result: list[dict[str, Any]] = []
    for part in splitter.split_text(text):
        normalized = part.strip()
        if not normalized:
            continue
        if heading_prefix and not normalized.startswith(heading_prefix):
            normalized = f"{heading_prefix}\n{normalized}"
        item = dict(section)
        item["text"] = normalized
        result.append(item)
    return result


def build_semantic_chunks(
    text: str, chunk_size: int = KG_CHUNK_SIZE, chunk_overlap: int = KG_CHUNK_OVERLAP
) -> list[dict[str, Any]]:
    value = str(text or "").strip()
    if not value:
        return []
    sections = _build_sections_from_headings(value)
    if not sections:
        return []
    merged = _merge_semantic_sections(sections, chunk_size=chunk_size)
    merged = _pack_short_semantic_sections(
        merged,
        chunk_size=chunk_size,
        min_chunk_size=max(KG_MIN_CHUNK_SIZE, int(chunk_size * 0.24)),
    )
    chunks: list[dict[str, Any]] = []
    for section in merged:
        chunks.extend(
            _split_large_semantic_section(
                section,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return [chunk for chunk in chunks if len(chunk["text"].strip()) >= 20]


def split_text_by_heading_tags(
    text: str, chunk_size: int = KG_CHUNK_SIZE, chunk_overlap: int = KG_CHUNK_OVERLAP
) -> list[str]:
    return [
        item["text"]
        for item in build_semantic_chunks(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    ]


def chunk_documents(
    paths: PipelinePaths,
    chunk_size: int = KG_CHUNK_SIZE,
    chunk_overlap: int = KG_CHUNK_OVERLAP,
) -> dict[str, Any]:
    chunks = []
    doc_source_map = []
    cleaned_files = sorted(paths.cleaned_dir.glob("*.md")) + sorted(
        paths.cleaned_dir.glob("*.txt")
    )
    dropped_total = 0
    dropped_by_doc: dict[str, int] = {}
    for source in cleaned_files:
        text = source.read_text(encoding="utf-8")
        doc_id = make_doc_id(source.name)
        semantic_chunks = build_semantic_chunks(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        ranked_chunks: list[tuple[int, dict[str, Any]]] = []
        for chunk in semantic_chunks:
            fast_score = _chunk_keep_score(chunk)
            chunk["pre_prune_score"] = fast_score
            if _is_low_value_chunk(chunk.get("text", ""), chunk.get("heading_path", []), chunk.get("semantic_group", "overview")) and fast_score < 55:
                dropped_total += 1
                dropped_by_doc[doc_id] = dropped_by_doc.get(doc_id, 0) + 1
                continue
            ranked_chunks.append((fast_score, chunk))

        # 文档规模过大时按得分裁剪，避免实体爆炸（优先保留深层结构+高信息密度块）
        if len(ranked_chunks) > 320:
            keep = max(220, int(len(ranked_chunks) * 0.62))
            ranked_chunks = sorted(ranked_chunks, key=lambda item: item[0], reverse=True)[:keep]
            ranked_chunks = sorted(ranked_chunks, key=lambda item: item[1].get("chunk_index", 0))

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
                    "pre_prune_score": int(chunk.get("pre_prune_score", _chunk_keep_score(chunk))),
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

    chunk_ids = [item["chunk_id"] for item in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise RuntimeError("chunk_id 不唯一")

    write_json(paths.chunks_path, chunks)
    write_json(paths.doc_map_path, doc_source_map)
    return {
        "documents": len(doc_source_map),
        "chunks": len(chunks),
        "dropped_chunks": dropped_total,
        "dropped_by_doc": dropped_by_doc,
    }


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


def extract_score_json(text: str) -> dict[str, Any]:
    data = extract_json_from_response(text)
    score = data.get("score", 0)
    try:
        score = int(score)
    except (TypeError, ValueError):
        score = 0
    decision = str(data.get("decision", "")).strip().lower()
    if decision not in {"extract", "skip"}:
        decision = "extract" if score >= LLM_QUALITY_SCORE_THRESHOLD else "skip"
    reasons = data.get("reasons", [])
    if not isinstance(reasons, list):
        reasons = []
    return {
        "score": max(0, min(100, score)),
        "decision": decision,
        "category": str(data.get("category", "")).strip(),
        "reasons": [str(item).strip() for item in reasons if str(item).strip()],
    }


def heuristic_chunk_score(chunk: dict[str, Any]) -> dict[str, Any]:
    text = str(chunk.get("text", ""))
    heading_path = chunk.get("heading_path", [])
    semantic_group = str(chunk.get("semantic_group", "overview"))
    score = 25
    bonuses = {
        "fault": 35,
        "repair": 35,
        "diagnostic": 32,
        "wiring": 28,
        "parameter": 25,
        "component": 22,
        "safety": 25,
        "operation": 18,
        "overview": 0,
    }
    score += bonuses.get(semantic_group, 0)
    if chunk.get("attachments"):
        score += 8
    if any(keyword in text for keyword in ("故障", "报警", "维修", "接线", "参数", "诊断", "注意", "更换")):
        score += 12
    if any(keyword in text for keyword in ("概述", "简介", "公司", "产品说明书")):
        score -= 18
    if len(text) < 120:
        score -= 12
    return {
        "score": max(0, min(100, score)),
        "decision": "extract" if score >= LLM_QUALITY_SCORE_THRESHOLD else "skip",
        "category": semantic_group or (heading_path[-1] if heading_path else ""),
        "reasons": ["heuristic_fallback"],
    }


def build_quality_score_request(chunk: dict[str, Any]) -> str:
    parts = ["请判断以下 chunk 是否值得进入 KG 抽取。"]
    if chunk.get("heading_path"):
        parts.append("章节路径：\n" + "\n".join(chunk["heading_path"]))
    if chunk.get("semantic_group"):
        parts.append(f"推断系群：{chunk['semantic_group']}")
    attachments = chunk.get("attachments", [])
    if attachments:
        parts.append(
            "图片附件："
            + json.dumps(
                [
                    {"path": item.get("path", ""), "alt": item.get("alt", "")}
                    for item in attachments
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    parts.append(f"chunk文本：\n{chunk.get('text', '')}")
    return "\n\n".join(parts)


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
    apply_local_envs(paths.env_paths)
    api_key = os.environ.get(LLM_API_KEY_ENV, "").strip()
    if not api_key and not LLM_REQUIRE_API_KEY:
        return ""
    if not api_key:
        raise RuntimeError(
            f"{LLM_API_KEY_ENV} 未设置，请在环境变量、Project/.env 或 Project/rag_app/.env 中配置"
        )
    return api_key


def _resolve_api_base_url(paths: PipelinePaths) -> str:
    apply_local_envs(paths.env_paths)
    base_url = os.environ.get(LLM_BASE_URL_ENV, LLM_BASE_URL).strip()
    if not base_url:
        raise RuntimeError(
            f"{LLM_BASE_URL_ENV} 未设置，请在环境变量、Project/.env 或 Project/rag_app/.env 中配置"
        )
    return base_url


def _resolve_api_model(paths: PipelinePaths) -> str:
    apply_local_envs(paths.env_paths)
    return os.environ.get(LLM_MODEL_ENV, LLM_MODEL).strip() or LLM_MODEL


def extract_kg(
    paths: PipelinePaths,
    only_doc_id: str | None = None,
    use_context: bool = True,
    checkpoint_enabled: bool = True,
) -> dict[str, Any]:
    api_key = _resolve_api_key(paths)
    base_url = _resolve_api_base_url(paths)
    model = _resolve_api_model(paths)
    client = OpenAI(
        api_key=api_key or "ollama",
        base_url=base_url,
    )
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

            quality_payload = heuristic_chunk_score(chunk)
            if os.environ.get(LLM_MOCK_ENV, "0") != "1":
                try:
                    quality_response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": QUALITY_SCORE_PROMPT},
                            {"role": "user", "content": build_quality_score_request(chunk)},
                        ],
                        temperature=0,
                        max_tokens=LLM_QUALITY_MAX_TOKENS,
                    )
                    quality_payload = extract_score_json(
                        quality_response.choices[0].message.content or ""
                    )
                except Exception:
                    pass
            if quality_payload.get("decision") != "extract":
                done_chunk_ids.add(chunk_id)
                continue

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
    model_name = os.environ.get("BGE_MODEL_NAME", "BAAI/bge-m3").strip()
    allow_online = os.environ.get("ALLOW_ONLINE_MODEL_DOWNLOAD", "0").strip() == "1"

    def _dedupe_paths(items: list[Path]) -> list[Path]:
        seen: set[str] = set()
        ordered: list[Path] = []
        for item in items:
            key = str(item.expanduser())
            if not key or key in seen:
                continue
            seen.add(key)
            ordered.append(item.expanduser())
        return ordered

    def _resolve_snapshot_from_dir(path: Path) -> Path | None:
        if not path.exists():
            return None
        if (path / "modules.json").exists():
            return path
        snapshots_dir = path / "snapshots"
        if snapshots_dir.exists():
            snapshot_candidates = sorted(
                [child for child in snapshots_dir.iterdir() if child.is_dir()],
                key=lambda child: child.stat().st_mtime,
                reverse=True,
            )
            for candidate in snapshot_candidates:
                if (candidate / "modules.json").exists():
                    return candidate
        return None

    def _resolve_repo_snapshot(repo_id: str, cache_root: Path) -> Path | None:
        try:
            snapshot_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=str(cache_root),
                local_files_only=True,
            )
        except Exception:
            return None
        candidate = Path(snapshot_path)
        return candidate if (candidate / "modules.json").exists() else None

    candidate_paths: list[Path] = []
    if model_name:
        candidate_paths.append(Path(model_name).expanduser())

    env_cache = os.environ.get("BGE_CACHE_DIR", "").strip()
    cache_root_candidates: list[Path] = [paths.cache_dir]
    if env_cache:
        cache_root_candidates.insert(0, Path(env_cache).expanduser())
    cache_roots = _dedupe_paths(cache_root_candidates)

    model_path: Path | None = None
    resolved_from_cache: Path | None = None
    if candidate_paths and candidate_paths[0].exists():
        model_path = _resolve_snapshot_from_dir(candidate_paths[0])

    if model_path is None:
        for cache_root in cache_roots:
            if not str(cache_root):
                continue
            if cache_root.exists():
                direct = _resolve_snapshot_from_dir(cache_root)
                if direct is not None:
                    model_path = direct
                    resolved_from_cache = cache_root
                    break
            if "/" in model_name and cache_root.exists():
                snapshot = _resolve_repo_snapshot(model_name, cache_root)
                if snapshot is not None:
                    model_path = snapshot
                    resolved_from_cache = cache_root
                    break

    if model_path is None and not allow_online:
        searched = ", ".join(str(path) for path in cache_roots if str(path))
        raise RuntimeError(
            "离线加载嵌入模型失败：未在本地缓存中找到可用快照。"
            f" model={model_name}; searched_cache_roots=[{searched}]"
        )

    if model_path is None:
        primary_cache = str(cache_roots[0]) if cache_roots else None
        return SentenceTransformer(
            model_name_or_path=model_name,
            device=device,
            cache_folder=primary_cache,
            local_files_only=False,
            model_kwargs={"use_safetensors": False},
        )

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    return SentenceTransformer(
        model_name_or_path=str(model_path),
        device=device,
        cache_folder=str(resolved_from_cache) if resolved_from_cache else None,
        local_files_only=True,
        model_kwargs={"use_safetensors": False, "local_files_only": True},
        tokenizer_kwargs={"local_files_only": True},
        config_kwargs={"local_files_only": True},
    )

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

    import networkx as nx
    G = nx.Graph()
    for entity in merged_entities:
        G.add_node(entity["name"])
    for relation in clean_relations:
        G.add_edge(relation["head"], relation["tail"])

    names_to_keep = set()
    for component in nx.connected_components(G):
        if len(component) >= 5:  # 保留节点数>=5的实质性连通分量
            names_to_keep.update(component)
        else:
            merge_log.append({
                "step": "prune_isolated",
                "pruned_nodes": list(component)
            })

    # 第二道清除非主流或细碎子节点的规则
    fine_grained_keywords = {"继电器", "保险", "端子", "引脚", "接线", "二极管", "触头", "插头", "插座", "电源接口", "COM", "电阻", "电容", "螺丝", "指示灯", "开关", "按钮", "旋钮", "针脚", "线嘴", "跳线"}
    names_to_prune_fine = set()

    for node in list(names_to_keep):
        # 仅针对“主机遥控系统”相关的边缘细碎零件进行裁剪，不要误伤“监测报警系统”等别的部分
        if "遥控" not in str(node) and "CDQY2A" not in str(node):
            continue

        # 1. 如果包含敏感细碎词汇，且它不是系统的"箱"/"台"/"系统"/"柜"等主体设备
        if any(kw in str(node) for kw in fine_grained_keywords):
            if not any(safe_kw in str(node) for safe_kw in ["箱", "系统", "柜", "总成", "台", "面板", "控制模块"]):
                names_to_prune_fine.add(node)
                names_to_keep.remove(node)
                continue
        # 2. 如果度数为1，且名字太细碎或者只是补充属性（例如含有“档位”、“值”、“状态”）
        degree = G.degree[node]
        if degree == 1 and any(fw in str(node) for fw in ["值", "状态", "档位", "类型", "测试", "能力", "参数"]):
            names_to_prune_fine.add(node)
            names_to_keep.remove(node)

    if names_to_prune_fine:
        merge_log.append({
            "step": "prune_fine_grained",
            "pruned_nodes": list(names_to_prune_fine)
        })

    merged_entities = [entity for entity in merged_entities if entity["name"] in names_to_keep]
    clean_relations = [
        relation for relation in clean_relations
        if relation["head"] in names_to_keep and relation["tail"] in names_to_keep
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
    apply_local_envs(paths.env_paths)
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
                    n.all_labels = row.all_labels,
                    n.source = row.source
                """,
                rows=[
                    {
                        "entity_id": entity.get("entity_id", ""),
                        "name": entity["name"],
                        "label": entity.get("label", ""),
                        "description": entity.get("description", ""),
                        "all_labels": _serialize(entity.get("all_labels", [])),
                        "source": str(entity.get("source", "")),
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
        dump_target = paths.delivery_dir / "neo4j.dump"
        if dump_target.exists():
            dump_target.unlink()
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
