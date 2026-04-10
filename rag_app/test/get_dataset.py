import json
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


def build_eval_finish() -> None:
    current_dir = Path(__file__).resolve().parent
    eval_dir = current_dir / "eval_data"
    eval_file = eval_dir / "eval.json"
    output_file = eval_dir / "eval_finish.json"

    data = _load_eval_data(eval_file)
    qa_items = _flatten_qa_items(data)
    pipeline = RagPipeline()

    results: List[Dict[str, Any]] = []
    total = len(qa_items)

    for idx, item in enumerate(qa_items, start=1):
        question = _extract_question(item)
        ground_truth = item.get("ground_truth", "")

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
            use_kg=True,
            use_llm=True,
            use_history=False,
            session_id=f"eval_{idx}",
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
    build_eval_finish()
