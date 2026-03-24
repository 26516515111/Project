from typing import List, Dict, Tuple

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_models import ChatOllama

from .schema import Passage
from .config import SETTINGS
from .store_history import get_history


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
    if kg_triplets:
        lines.append("相关知识关联：")
        for t in kg_triplets[:3]:
            lines.append(
                f"- {t.get('head', '')} {t.get('rel', '')} {t.get('tail', '')}"
            )
    lines.append("请结合现场工况核验后执行。")
    return "\n".join(lines)


def _build_context(
    passages: List[Passage], kg_triplets: List[Dict[str, str]]
) -> Tuple[str, str]:
    """构造用于LLM的上下文与知识图谱文本。"""
    ctx = []
    for i, p in enumerate(passages[:5], start=1):
        cleaned = p.text.replace("\n", " ")
        ctx.append(f"[资料{i}] {cleaned}")
    kg = []
    for t in kg_triplets[:5]:
        kg.append(f"{t.get('head', '')} {t.get('rel', '')} {t.get('tail', '')}")
    context_text = "\n".join(ctx) if ctx else "无"
    kg_text = "\n".join(kg) if kg else "无"
    return context_text, kg_text


def _build_prompt() -> ChatPromptTemplate:
    """构造用于LLM的结构化提示词模板。"""
    system_prompt = (
        "你是船舶装备故障诊断助手。根据资料与知识图谱，给出可执行的诊断与维修建议，"
        "输出结构化要点，禁止编造。"
    )
    user_prompt = (
        "问题: {question}\n"
        "资料:\n{context}\n"
        "知识图谱:\n{kg_text}\n"
        "请输出: 故障判断, 可能原因, 处理步骤, 注意事项，禁止输出无关内容包括无关字符。"
    )
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("history"),
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
    if use_llm and SETTINGS.llm_provider == "ollama":
        context_text, kg_text = _build_context(passages, kg_triplets)
        prompt = _build_prompt()
        llm = ChatOllama(
            model=SETTINGS.llm_model,
            base_url=SETTINGS.ollama_base_url,
            temperature=SETTINGS.llm_temperature,
            model_kwargs={"num_predict": SETTINGS.llm_max_tokens},
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
                        "kg_text": kg_text,
                    },
                    config={"configurable": {"session_id": session_id}},
                )
            else:
                response = chain.invoke(
                    {"question": question, "context": context_text, "kg_text": kg_text}
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
