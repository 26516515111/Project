from typing import List, Dict

import requests

from .schema import Passage
from .config import SETTINGS


def extractive_answer(
    question: str, passages: List[Passage], kg_triplets: List[Dict[str, str]]
) -> str:
    lines = ["基于检索到的资料，建议如下："]
    for i, p in enumerate(passages[:3], start=1):
        snippet = p.text.replace("\n", " ")[:220]
        lines.append(f"{i}. {snippet}")
    if kg_triplets:
        lines.append("相关知识关联：")
        for t in kg_triplets[:3]:
            lines.append(
                f"- {t.get('head', '')} {t.get('rel', '')} {t.get('tail', '')}"
            )
    lines.append("请结合现场工况核验后执行。")
    return "\n".join(lines)


def _build_prompt(
    question: str, passages: List[Passage], kg_triplets: List[Dict[str, str]]
) -> str:
    ctx = []
    for i, p in enumerate(passages[:5], start=1):
        cleaned = p.text.replace("\n", " ")
        ctx.append(f"[资料{i}] {cleaned}")
    kg = []
    for t in kg_triplets[:5]:
        kg.append(f"{t.get('head', '')} {t.get('rel', '')} {t.get('tail', '')}")
    kg_text = "\n".join(kg) if kg else "无"
    context_text = "\n".join(ctx) if ctx else "无"
    return (
        "你是船舶装备故障诊断助手。根据资料与知识图谱，给出可执行的诊断与维修建议，"
        "输出结构化要点，禁止编造。\n"
        f"问题: {question}\n"
        f"资料:\n{context_text}\n"
        f"知识图谱:\n{kg_text}\n"
        "请输出: 故障判断, 可能原因, 处理步骤, 注意事项。"
    )


def _ollama_generate(prompt: str) -> str:
    url = f"{SETTINGS.ollama_base_url}/api/generate"
    payload = {
        "model": SETTINGS.llm_model,
        "prompt": prompt,
        "options": {
            "temperature": SETTINGS.llm_temperature,
            "num_predict": SETTINGS.llm_max_tokens,
        },
        "stream": False,
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()


def generate_answer(
    question: str,
    passages: List[Passage],
    kg_triplets: List[Dict[str, str]],
    use_llm: bool = True,
) -> str:
    if use_llm and SETTINGS.llm_provider == "ollama":
        prompt = _build_prompt(question, passages, kg_triplets)
        try:
            return _ollama_generate(prompt)
        except Exception:
            return extractive_answer(question, passages, kg_triplets)
    return extractive_answer(question, passages, kg_triplets)
