import json
from rag.pipeline import RagPipeline
from rag.schema import QueryRequest


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
        print("\nAnswer:\n")
        print(ans.answer)
        print("\nCitations:")
        for p in ans.citations:
            print(f"- {p.source} ({p.score:.3f})")
        print("\nJSON:\n")
        print(json.dumps(ans.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    """脚本入口。

    Args:
        None

    Returns:
        None
    """
    main()
