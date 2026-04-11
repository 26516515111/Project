import argparse
from typing import List

from rag.pipeline import RagPipeline
from rag.schema import QueryRequest
from rag.store_history import get_history
from rag.generator import get_system_prompt_text, render_user_prompt_text
from rag.config import SETTINGS


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


def _print_decompose_info(questions: List[str], use_decompose: bool) -> None:
    print("\nDecomposition:")
    print(f"enabled: {use_decompose}")
    print(f"method: {SETTINGS.decomposer_method}")
    if len(questions) <= 1:
        print("sub-questions: (none)")
        return
    print("sub-questions:")
    for idx, sub_q in enumerate(questions[1:], start=1):
        print(f"  {idx}. {sub_q}")


def _print_answer_details(ans, title: str = "Answer") -> None:
    print(f"\n{title}:")
    print(ans.answer)

    print("\nMeta:")
    for k, v in ans.meta.items():
        print(f"- {k}: {v}")

    print("\nRetrieved Chunks:")
    if not ans.retrieved_chunks:
        print("(none)")
    else:
        for idx, p in enumerate(ans.retrieved_chunks, start=1):
            print(f"{idx}. [{p.source}] {p.doc_id} (score={p.score:.4f})")

    print("\nCitations Used In Prompt:")
    if not ans.citations:
        print("(none)")
    else:
        for idx, p in enumerate(ans.citations, start=1):
            print(f"{idx}. [{p.source}] {p.doc_id} (score={p.score:.4f})")

    print("\nKG Triplets:")
    if not ans.kg_triplets:
        print("(none)")
    else:
        for idx, t in enumerate(ans.kg_triplets, start=1):
            print(f"{idx}. {t}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ship Equipment Fault RAG CLI")
    parser.add_argument(
        "--ab-decompose",
        action="store_true",
        help="Run A/B comparison for decomposition on the same question",
    )
    parser.add_argument(
        "--ab-parent",
        action="store_true",
        help="Run A/B comparison for parent retriever on the same question",
    )
    return parser.parse_args()


def _run_single_query(pipeline: RagPipeline, req: QueryRequest) -> None:
    prepared = pipeline._prepare_request(req)
    _print_decompose_info(
        questions=prepared.get("questions", [req.question]),
        use_decompose=bool(prepared.get("use_decompose", False)),
    )
    ans = pipeline.query(req)
    _print_answer_details(ans)
    _print_dialogue_messages(
        req.session_id or "cli", req.question, ans.citations, ans.kg_triplets
    )


def _run_ab_query(pipeline: RagPipeline, req_base: QueryRequest) -> None:
    print("\n===== A/B: Decompose OFF =====")
    req_off = req_base.model_copy(update={"enable_decompose": False})
    prepared_off = pipeline._prepare_request(req_off)
    _print_decompose_info(
        questions=prepared_off.get("questions", [req_off.question]),
        use_decompose=bool(prepared_off.get("use_decompose", False)),
    )
    ans_off = pipeline.query(req_off)
    _print_answer_details(ans_off, title="Answer (Decompose OFF)")

    print("\n===== A/B: Decompose ON =====")
    req_on = req_base.model_copy(update={"enable_decompose": True})
    prepared_on = pipeline._prepare_request(req_on)
    _print_decompose_info(
        questions=prepared_on.get("questions", [req_on.question]),
        use_decompose=bool(prepared_on.get("use_decompose", False)),
    )
    ans_on = pipeline.query(req_on)
    _print_answer_details(ans_on, title="Answer (Decompose ON)")

    print("\n===== A/B Summary =====")
    print(f"OFF retrieved_chunks: {len(ans_off.retrieved_chunks)}")
    print(f"ON  retrieved_chunks: {len(ans_on.retrieved_chunks)}")
    print(f"OFF citations: {len(ans_off.citations)}")
    print(f"ON  citations: {len(ans_on.citations)}")


def _run_ab_parent_query(pipeline: RagPipeline, req_base: QueryRequest) -> None:
    print("\n===== A/B: Parent Retriever OFF =====")
    req_off = req_base.model_copy(update={"enable_parent_retriever": False})
    prepared_off = pipeline._prepare_request(req_off)
    print(f"parent_retriever: {prepared_off.get('use_parent_retriever')}")
    ans_off = pipeline.query(req_off)
    _print_answer_details(ans_off, title="Answer (Parent OFF)")

    print("\n===== A/B: Parent Retriever ON =====")
    req_on = req_base.model_copy(update={"enable_parent_retriever": True})
    prepared_on = pipeline._prepare_request(req_on)
    print(f"parent_retriever: {prepared_on.get('use_parent_retriever')}")
    ans_on = pipeline.query(req_on)
    _print_answer_details(ans_on, title="Answer (Parent ON)")

    print("\n===== A/B Summary =====")
    print(f"OFF retrieved_chunks: {len(ans_off.retrieved_chunks)}")
    print(f"ON  retrieved_chunks: {len(ans_on.retrieved_chunks)}")
    print(f"OFF citations: {len(ans_off.citations)}")
    print(f"ON  citations: {len(ans_on.citations)}")
    print(f"OFF parent_source_doc_ids: {ans_off.meta.get('parent_source_doc_ids', '')}")
    print(f"ON  parent_source_doc_ids: {ans_on.meta.get('parent_source_doc_ids', '')}")


def main():
    """CLI入口，读取用户问题并输出回答。

    Args:
        None

    Returns:
        None
    """
    args = _parse_args()
    pipeline = RagPipeline()
    print("Ship Equipment Fault RAG CLI")
    print(
        f"Config: decompose={SETTINGS.use_query_decomposition}, "
        f"decomposer_method={SETTINGS.decomposer_method}, "
        f"reranker={SETTINGS.use_reranker}"
    )
    session_id = "cli"
    while True:
        q = input("\nQuestion (empty to exit): ").strip()
        if not q:
            break
        req = QueryRequest(question=q, session_id=session_id)
        if args.ab_parent:
            _run_ab_parent_query(pipeline, req)
        elif args.ab_decompose:
            _run_ab_query(pipeline, req)
        else:
            _run_single_query(pipeline, req)


if __name__ == "__main__":
    """脚本入口。

    Args:
        None

    Returns:
        None
    """
    main()
