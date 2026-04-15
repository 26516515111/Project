import json
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List


def _setup_import_path() -> None:
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_setup_import_path()

from rag.pipeline import RagPipeline
from rag.schema import QueryRequest


def _load_eval_data(eval_file: Path) -> List[Dict[str, Any]]:
    if not eval_file.exists():
        raise FileNotFoundError(f"未找到测评文件: {eval_file}")

    with eval_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("eval.json 格式错误，期望为列表(list)")

    return data


def _flatten_qa_items(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    qa_items: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        questions = item.get("questions")
        if isinstance(questions, list):
            for q_item in questions:
                if isinstance(q_item, dict):
                    qa_items.append(q_item)
            continue

        if "question" in item or "queston" in item:
            qa_items.append(item)
    return qa_items


def _extract_question(item: Dict[str, Any]) -> str:
    q = item.get("queston")
    if q is None:
        q = item.get("question")
    if q is None:
        return ""
    return str(q).strip()


def build_eval_finish(
    output_name: str = "eval_finish.json",
    use_kg: bool = True,
    use_llm: bool = True,
    use_history: bool = False,
    disable_decompose: bool = False,
    disable_parent_retriever: bool = False,
    disable_retrieval_optimization: bool = False,
) -> None:
    current_dir = Path(__file__).resolve().parent
    eval_dir = current_dir / "eval_data"
    eval_file = eval_dir / "eval.json"
    output_file = eval_dir / output_name

    data = _load_eval_data(eval_file)
    qa_items = _flatten_qa_items(data)
    pipeline = RagPipeline()

    results: List[Dict[str, Any]] = []
    total = len(qa_items)

    for idx, item in enumerate(qa_items, start=1):
        question = _extract_question(item)
        # Historical datasets used `group_truth`; keep backward compatibility.
        ground_truth = item.get("ground_truth")
        if ground_truth is None:
            ground_truth = item.get("group_truth", "")

        print(f"[{idx}/{total}] 正在处理问题: {question[:80]}")

        if not question:
            results.append(
                {
                    "queston": "",
                    "context": [],
                    "answear": "",
                    "ground_truth": ground_truth,
                }
            )
            continue

        req = QueryRequest(
            question=question,
            use_kg=use_kg,
            use_llm=use_llm,
            use_history=use_history,
            session_id=f"eval_{idx}",
            enable_decompose=(False if disable_decompose else None),
            enable_parent_retriever=(False if disable_parent_retriever else None),
            enable_retrieval_optimization=(
                False if disable_retrieval_optimization else None
            ),
        )
        answer = pipeline.query(req)

        context = [p.text for p in answer.citations]

        results.append(
            {
                "queston": question,
                "context": context,
                "answear": answer.answer,
                "ground_truth": ground_truth,
            }
        )

    eval_dir.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"已生成: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成RAG评测数据集(eval_finish*.json)")
    parser.add_argument(
        "--output-name",
        default="eval_finish.json",
        help="输出文件名，默认 eval_finish.json",
    )
    parser.add_argument(
        "--disable-decompose",
        action="store_true",
        help="关闭问题分解",
    )
    parser.add_argument(
        "--disable-parent-retriever",
        action="store_true",
        help="关闭父文档检索路由",
    )
    parser.add_argument(
        "--disable-retrieval-optimization",
        action="store_true",
        help="关闭检索优化开关",
    )
    parser.add_argument(
        "--no-kg",
        action="store_true",
        help="关闭KG增强",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="关闭LLM生成",
    )
    parser.add_argument(
        "--use-history",
        action="store_true",
        help="启用历史对话",
    )
    args = parser.parse_args()
    build_eval_finish(
        output_name=args.output_name,
        use_kg=not args.no_kg,
        use_llm=not args.no_llm,
        use_history=args.use_history,
        disable_decompose=args.disable_decompose,
        disable_parent_retriever=args.disable_parent_retriever,
        disable_retrieval_optimization=args.disable_retrieval_optimization,
    )
