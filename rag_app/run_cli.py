from langchain_community.chat_models import ChatOpenAI

from rag.pipeline import RagPipeline
from rag.schema import QueryRequest
from rag.store_history import get_history
from rag.generator import get_system_prompt_text, render_user_prompt_text
from rag.model import build_chat_llm


def _print_dialogue_messages(
    session_id: str, question: str, passages, kg_triplets
) -> None:
    print("\nDialogue Messages:")
    print(f"[System] {get_system_prompt_text()}")
    print(f"[Human] {render_user_prompt_text(question, passages, kg_triplets)}")
    history = get_history(session_id)
    ai_text = ""
    messages = list(history.messages)
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "ai":
            ai_text = str(getattr(msg, "content", "") or "").strip()
            break
    if not ai_text:
        ai_text = "(no ai message recorded)"
    print(f"[AI] {ai_text}")


def main():
    """CLI入口，读取用户问题并输出回答。

    Args:
        None

    Returns:
        None
    """
    pipeline = RagPipeline()
    print("Ship Equipment Fault RAG CLI")
    session_id = "cli"
    while True:
        q = input("\nQuestion (empty to exit): ").strip()
        if not q:
            break
        req = QueryRequest(question=q, session_id=session_id)
        ans = pipeline.query(req)
        _print_dialogue_messages(session_id, q, ans.citations, ans.kg_triplets)


if __name__ == "__main__":
    """脚本入口。

    Args:
        None

    Returns:
        None
    """
    model = build_chat_llm()
    print(model.invoke("Hello, world!"))
    main()


