from typing import List, Dict
import re

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory

from .schema import Passage
from .config import SETTINGS
from .model import build_chat_llm
from .store_history import get_history


USER_PROMPT_TEMPLATE = (
    "问题: {question}\n"
    "检索资料:\n{context}\n"
    "知识图谱三元组:\n{kg_context}\n"
    "请严格按系统规则输出。"
)


def get_system_prompt_text() -> str:
    return (
        "你是船舶装备故障诊断助手。你的唯一任务是基于检索资料与知识图谱三元组输出可执行诊断方案。\n"
        "当用户输入与船舶装备故障诊断无关的信息时，只输出：与本系统内容无关，请重新输入\n"
        "\n"
        "硬性规则:\n"
        "1) 只使用输入证据推理，禁止编造，禁止使用外部信息补全关键结论。\n"
        '2) 每条结论必须包含证据编号（资料编号或三元组编号）。无证据则写"证据不足"。\n'
        "3) 优先输出最可能故障与高风险故障，并写出区分依据。\n"
        "4) 诊断与维修建议必须可执行：包含检查对象、方法、判定标准（阈值/现象）、工具与安全注意事项。\n"
        "5) 证据不足以定位到部件级时，先输出最小补充信息清单，不得强行下结论。\n"
        "6) 输出保持简洁、结构化、面向现场执行。\n"
        "\n"
        "输出格式（严格遵守）:\n"
        "- 全文按点输出，不写大段叙述。\n"
        "- 一级条目统一用'- '开头；步骤统一用数字编号。\n"
        "- 故障判断（按置信度排序）\n"
        "  - [故障名称] | 置信度: 高/中/低\n"
        "  - 依据: <证据编号列表>；关键现象: <现象>\n"
        "  - 排除点: <与相似故障的区别>\n"
        "- 诊断步骤（按优先级）\n"
        "  1. <步骤>；判定标准: <阈值或现象>；证据: <编号>\n"
        "  2. <步骤>；判定标准: <阈值或现象>；证据: <编号>\n"
        "- 维修建议\n"
        "  - 立即措施: <动作>；风险: <说明>；证据: <编号>\n"
        "  - 根因修复: <动作>；备件/工具: <清单>；证据: <编号>\n"
        "- 复测与验收\n"
        "  - <指标/工况>；合格标准: <阈值>\n"
        "- 信息缺口（如有）\n"
        "  - <缺失信息>；为何影响判断: <原因>\n"
        "\n"
        '当证据冲突时：列出冲突点与证据编号，给出保守方案，并标注"需现场复核"。'
    )


def _is_domain_related(question: str) -> bool:
    text = str(question or "").strip().lower()
    if not text:
        return False
    if any(k.lower() in text for k in SETTINGS.domain_keywords):
        return True
    return bool(re.search(r"[A-Za-z]{2,}-\d+", text))


def build_context_text(passages: List[Passage]) -> str:
    ctx = []
    for i, p in enumerate(passages[:5], start=1):
        cleaned = p.text.replace("\n", " ")
        ctx.append(f"[资料{i}] {cleaned}")
    return "\n".join(ctx) if ctx else "无"


def build_kg_text(kg_triplets: List[Dict[str, str]]) -> str:
    rows = []
    for i, t in enumerate(kg_triplets[:8], start=1):
        head = str(t.get("head", "") or "")
        rel = str(t.get("relation", "") or "")
        tail = str(t.get("tail", "") or "")
        if head or rel or tail:
            rows.append(f"[图谱{i}] ({head}, {rel}, {tail})")
    return "\n".join(rows) if rows else "无"


def render_user_prompt_text(
    question: str, passages: List[Passage], kg_triplets: List[Dict[str, str]]
) -> str:
    return USER_PROMPT_TEMPLATE.format(
        question=question,
        context=build_context_text(passages),
        kg_context=build_kg_text(kg_triplets),
    )


def extractive_answer(
    question: str, passages: List[Passage], kg_triplets: List[Dict[str, str]]
) -> str:
    """生成基于检索片段的抽取式回答。当LLM模型不可用的时候使用

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
    lines.append("请结合现场工况核验后执行。")
    return "\n".join(lines)


def _build_context(passages: List[Passage]) -> str:
    """构造用于LLM的上下文文本。"""
    return build_context_text(passages)


def _build_prompt() -> ChatPromptTemplate:
    """构造用于LLM的结构化提示词模板。"""
    system_prompt = get_system_prompt_text()
    user_prompt = USER_PROMPT_TEMPLATE
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history", optional=True),
            ("human", user_prompt),
        ]
    )


def generate_answer(
    question: str,
    passages: List[Passage],
    kg_triplets: List[Dict[str, str]],
    use_llm: bool = True,
    use_history: bool = True,
    session_id: str = "default",
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
    if not _is_domain_related(question):
        return "与本系统内容无关，请重新输入"

    if use_llm and SETTINGS.llm_provider == "ollama":
        context_text = _build_context(passages)
        kg_context_text = build_kg_text(kg_triplets)
        prompt = _build_prompt()
        llm = build_chat_llm()
        chain = prompt | llm
        try:
            if use_history and SETTINGS.use_history:
                chain = RunnableWithMessageHistory(
                    chain,
                    lambda sid: get_history(sid),
                    input_messages_key="question",
                    history_messages_key="history",
                )
                response = chain.invoke(
                    {
                        "question": question,
                        "context": context_text,
                        "kg_context": kg_context_text,
                    },
                    config={"configurable": {"session_id": session_id}},
                )
            else:
                response = chain.invoke(
                    {
                        "question": question,
                        "context": context_text,
                        "kg_context": kg_context_text,
                    }
                )
            print("[llm] provider=ollama status=ok")
            return str(getattr(response, "content", response)).strip()
        except Exception as exc:
            print(f"[llm] provider=ollama status=fallback error={type(exc).__name__}")
            return extractive_answer(question, passages, kg_triplets)
    if use_llm and SETTINGS.llm_provider != "ollama":
        print(f"[llm] provider={SETTINGS.llm_provider} status=disabled")
    if not use_llm:
        print("[llm] provider=none status=disabled")
    return extractive_answer(question, passages, kg_triplets)
