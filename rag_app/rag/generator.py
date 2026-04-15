from typing import Dict, Iterable, List, Optional
import re

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

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
        "你是船舶装备故障诊断助手。你的唯一任务是基于检索资料与知识图谱三元组回答用户问题。\n"
        "\n"
        "硬性规则:\n"
        "1) 只使用输入证据推理，禁止编造，禁止使用外部信息补全关键结论。\n"
        "2) 输出保持简洁、结构化、口语化，输出关键要点，省略输出用于推理的资料展示，面向现场执行。"
        "\n"
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
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_api_base: Optional[str] = None,
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
    # if not _is_domain_related(question):
    #     return "与本系统内容无关，请重新输入"

    if use_llm:
        context_text = _build_context(passages)
        kg_context_text = build_kg_text(kg_triplets)
        prompt = _build_prompt()
        llm = build_chat_llm(
            provider=llm_provider,
            model_name=llm_model,
            base_url=llm_api_base,
            api_key=llm_api_key,
        )
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
    if not use_llm:
        print("[llm] provider=none status=disabled")
    return extractive_answer(question, passages, kg_triplets)


def stream_answer(
    question: str,
    passages: List[Passage],
    kg_triplets: List[Dict[str, str]],
    use_llm: bool = True,
    use_history: bool = True,
    session_id: str = "default",
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_api_base: Optional[str] = None,
) -> Iterable[str]:
    if use_llm:
        context_text = _build_context(passages)
        kg_context_text = build_kg_text(kg_triplets)
        prompt = _build_prompt()
        llm = build_chat_llm(
            provider=llm_provider,
            model_name=llm_model,
            base_url=llm_api_base,
            api_key=llm_api_key,
        )
        chain = prompt | llm
        stream_input = {
            "question": question,
            "context": context_text,
            "kg_context": kg_context_text,
        }
        history = None
        if use_history and SETTINGS.use_history:
            history = get_history(session_id)
            stream_input["history"] = list(history.messages)
        pieces: List[str] = []
        try:
            for chunk in chain.stream(stream_input):
                token = str(getattr(chunk, "content", chunk) or "")
                if not token:
                    continue
                pieces.append(token)
                yield token
            if history is not None:
                final_text = "".join(pieces).strip()
                if final_text:
                    history.add_message(HumanMessage(content=question))
                    history.add_message(AIMessage(content=final_text))
            print("[llm] provider=ollama status=stream_ok")
            return
        except Exception as exc:
            print(
                f"[llm] provider=ollama status=stream_fallback error={type(exc).__name__}"
            )
    else:
        print("[llm] provider=none status=disabled")
    yield extractive_answer(question, passages, kg_triplets)
