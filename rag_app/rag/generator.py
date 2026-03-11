from typing import List, Dict

import requests

from .schema import Passage
from .config import SETTINGS


def extractive_answer(
    question: str, passages: List[Passage], kg_triplets: List[Dict[str, str]]
) -> str:
    """生成基于检索片段的抽取式回答。

    Args:
        question: 用户问题文本。
        passages: 检索到的片段列表。
        kg_triplets: 知识图谱三元组列表。

    Returns:
        str: 抽取式答案文本。
    """
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
    """构造用于LLM的结构化提示词。

    Args:
        question: 用户问题文本。
        passages: 检索到的片段列表。
        kg_triplets: 知识图谱三元组列表。

    Returns:
        str: 提示词文本。
    """
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
        "请输出: 故障判断, 可能原因, 处理步骤, 注意事项, 每项要点不超过20字，禁止输出无关内容包括无关字符。"
    )


def _ollama_generate(prompt: str) -> str:
    """调用Ollama服务生成答案。

    Args:
        prompt: 提示词文本。

    Returns:
        str: 生成的答案文本。
    """
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
    resp = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()


def generate_answer(
    question: str,
    passages: List[Passage],
    kg_triplets: List[Dict[str, str]],
    use_llm: bool = True,
) -> str:
    """根据配置选择LLM或抽取式策略生成答案。

    Args:
        question: 用户问题文本。
        passages: 检索到的片段列表。
        kg_triplets: 知识图谱三元组列表。
        use_llm: 是否启用LLM生成。

    Returns:
        str: 生成的答案文本。
    """
    if use_llm and SETTINGS.llm_provider == "ollama":
        prompt = _build_prompt(question, passages, kg_triplets)
        try:
            result = _ollama_generate(prompt)
            print("[llm] provider=ollama status=ok")
            return result
        except Exception as exc:
            print(f"[llm] provider=ollama status=fallback error={type(exc).__name__}")
            return extractive_answer(question, passages, kg_triplets)
    if use_llm and SETTINGS.llm_provider != "ollama":
        print(f"[llm] provider={SETTINGS.llm_provider} status=disabled")
    if not use_llm:
        print("[llm] provider=none status=disabled")
    return extractive_answer(question, passages, kg_triplets)
