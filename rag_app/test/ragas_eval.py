import json
import os
import re
import sys
import traceback
import asyncio
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig


def _setup_import_path() -> None:
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_setup_import_path()


_IMG_TAG_RE = re.compile(r"\[IMG\s+[^\]]*\]", re.IGNORECASE)
_MD_IMG_RE = re.compile(r"!\[[^\]]*\]\([^\)]*\)", re.IGNORECASE)
_LINE_WITH_LOCAL_IMG_RE = re.compile(
    r"^\s*(?:\[图片附件\]\s*)?(?:[A-Za-z]:\\|\.?\.?/|/)?[^\n]*\.(?:png|jpg|jpeg|gif|webp|bmp|svg)\s*$",
    re.IGNORECASE,
)


def _clean_text(text: str) -> str:
    cleaned = _IMG_TAG_RE.sub(" ", text)
    cleaned = _MD_IMG_RE.sub(" ", cleaned)
    lines = cleaned.splitlines()
    lines = [line for line in lines if not _LINE_WITH_LOCAL_IMG_RE.match(line)]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _load_eval_finish(eval_finish_file: Path) -> List[Dict[str, Any]]:
    if not eval_finish_file.exists():
        raise FileNotFoundError(f"未找到评测文件: {eval_finish_file}")

    with eval_finish_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("eval_finish.json 格式错误，期望为列表(list)")

    return data


def _to_ragas_dataset(data: List[Dict[str, Any]]) -> Dataset:
    questions: List[str] = []
    answers: List[str] = []
    contexts: List[List[str]] = []
    ground_truths: List[str] = []

    for item in data:
        question = _clean_text(str(item.get("queston") or item.get("question") or ""))
        answer = _clean_text(str(item.get("answear") or item.get("answer") or ""))
        context = item.get("context", [])
        if isinstance(context, str):
            context = [context]
        elif not isinstance(context, list):
            context = []
        context = [_clean_text(str(c)) for c in context]
        context = [c for c in context if c]
        ground_truth = _clean_text(str(item.get("ground_truth") or ""))

        questions.append(question)
        answers.append(answer)
        contexts.append(context)
        ground_truths.append(ground_truth)

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(
            f"缺少环境变量 {name}。请先配置后再执行，例如:\n"
            f'PowerShell: $env:{name}="<your_key>"\n'
            f"CMD: set {name}=<your_key>"
        )
    return value


def _get_int_env(name: str, default: int, min_value: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as e:
        raise ValueError(f"环境变量 {name} 必须是整数，当前值: {raw}") from e
    if value < min_value:
        raise ValueError(f"环境变量 {name} 必须 >= {min_value}，当前值: {value}")
    return value


def run_ragas_eval() -> None:
    current_dir = Path(__file__).resolve().parent
    eval_finish_file = current_dir / "eval_data" / "eval_finish.json"
    output_excel = current_dir / "eval_data" / "ragas_eval.xlsx"

    data = _load_eval_finish(eval_finish_file)
    dataset = _to_ragas_dataset(data)

    api_key = _require_env("DASHSCOPE_API_KEY")
    timeout_sec = _get_int_env("RAGAS_TIMEOUT_SEC", 600)
    max_retries = _get_int_env("RAGAS_MAX_RETRIES", 5)
    max_workers = _get_int_env("RAGAS_MAX_WORKERS", 2)
    run_config = RunConfig(
        timeout=timeout_sec,
        max_retries=max_retries,
        max_workers=max_workers,
        max_wait=30,
    )
    print(
        f"评测配置: timeout={timeout_sec}s, retries={max_retries}, workers={max_workers}"
    )

    llm = ChatTongyi(model="qwen2.5-vl-72b-instruct", api_key=api_key)
    embedding = DashScopeEmbeddings(
        model="text-embedding-v3", dashscope_api_key=api_key
    )
    vllm = LangchainLLMWrapper(llm, run_config=run_config)
    vembedding = LangchainEmbeddingsWrapper(embedding)

    try:
        result = evaluate(
            dataset=dataset,
            llm=vllm,
            embeddings=vembedding,
            metrics=[
                answer_relevancy,
                faithfulness,
                context_recall,
                context_precision,
            ],
            run_config=run_config,
            raise_exceptions=True,
        )
    except Exception as e:
        print(f"评测失败: {type(e).__name__}: {e}")
        if isinstance(e, (TimeoutError, asyncio.TimeoutError)):
            print(
                "超时建议: 降低并发或增加超时时间。可设置环境变量 "
                "RAGAS_MAX_WORKERS=2 或 RAGAS_TIMEOUT_SEC=600 后重试。"
            )
        print("详细堆栈如下:")
        print(traceback.format_exc())
        raise

    df = result.to_pandas()
    metric_columns = [
        "answer_relevancy",
        "faithfulness",
        "context_recall",
        "context_precision",
    ]
    failed_counts = {
        col: int(df[col].isna().sum()) for col in metric_columns if col in df.columns
    }
    failed_counts = {k: v for k, v in failed_counts.items() if v > 0}
    if failed_counts:
        print(f"警告: 部分指标为空，可能存在样本级失败: {failed_counts}")

    df.to_excel(output_excel, index=False)
    avg_row: Dict[str, Any] = {col: "" for col in df.columns}
    avg_row["question"] = "平均值"
    for col in metric_columns:
        if col in df.columns:
            avg_row[col] = df[col].astype(float).mean()

    df_with_avg = df.copy()
    df_with_avg.loc[len(df_with_avg)] = avg_row
    df_with_avg.to_excel(output_excel, index=False)
    print(f"评测完成，结果已保存: {output_excel}")


if __name__ == "__main__":
    run_ragas_eval()
