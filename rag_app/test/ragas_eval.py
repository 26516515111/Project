import json
import argparse
import os
import re
import sys
import traceback
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
_ESCAPED_PUNCT_RE = re.compile(r"\\(?=[()\[\]{}])")


def _clean_text(text: str) -> str:
    cleaned = _IMG_TAG_RE.sub(" ", text)
    cleaned = _MD_IMG_RE.sub(" ", cleaned)
    cleaned = _ESCAPED_PUNCT_RE.sub("", cleaned)
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


def _get_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"环境变量 {name} 必须是布尔值(true/false/1/0/yes/no/on/off)，当前值: {raw}"
    )


def _build_tongyi_llm_candidates(
    api_key: str,
    model: str = "qwen3-30b-a3b",
    enable_thinking: bool = False,
) -> List[Tuple[str, ChatTongyi]]:
    """构造多种 ChatTongyi 参数注入方式，兼容不同 SDK 版本。"""
    base_kwargs: Dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "streaming": False,
    }
    variants: List[Tuple[str, Dict[str, Any]]] = [
        (
            "model_kwargs_enable_thinking",
            {**base_kwargs, "model_kwargs": {"enable_thinking": enable_thinking}},
        ),
        (
            "model_kwargs_parameters_enable_thinking",
            {
                **base_kwargs,
                "model_kwargs": {"parameters": {"enable_thinking": enable_thinking}},
            },
        ),
        (
            "direct_enable_thinking",
            {**base_kwargs, "enable_thinking": enable_thinking},
        ),
        (
            "extra_body_enable_thinking",
            {**base_kwargs, "extra_body": {"enable_thinking": enable_thinking}},
        ),
        (
            "base_no_thinking_field",
            {**base_kwargs},
        ),
    ]

    candidates: List[Tuple[str, ChatTongyi]] = []
    for mode, kwargs in variants:
        try:
            candidates.append((mode, ChatTongyi(**kwargs)))
        except TypeError:
            continue

    if not candidates:
        raise RuntimeError(
            "无法构造 ChatTongyi，请升级 langchain-community，或检查 ChatTongyi 参数签名。"
        )
    return candidates


def _is_enable_thinking_non_streaming_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "invalidparameter" in text
        and "enable_thinking" in text
        and ("non-streaming" in text or "non streaming" in text)
    )


def _is_invalid_parameter_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "invalidparameter" in text and "400" in text


def _run_eval(
    dataset: Dataset,
    vllm: LangchainLLMWrapper,
    vembedding: LangchainEmbeddingsWrapper,
    run_config: RunConfig,
    raise_exceptions: bool,
):
    return evaluate(
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
        raise_exceptions=raise_exceptions,
    )


def run_ragas_eval(
    eval_finish_name: str = "eval_finish.json", output_name: str = "ragas_eval.xlsx"
) -> None:
    current_dir = Path(__file__).resolve().parent
    eval_finish_file = current_dir / "eval_data" / eval_finish_name
    output_excel = current_dir / "eval_data" / output_name

    data = _load_eval_finish(eval_finish_file)
    dataset = _to_ragas_dataset(data)

    api_key = _require_env("DASHSCOPE_API_KEY")
    timeout_sec = _get_int_env("RAGAS_TIMEOUT_SEC", 600)
    max_retries = _get_int_env("RAGAS_MAX_RETRIES", 5)
    max_workers = _get_int_env("RAGAS_MAX_WORKERS", 2)
    raise_exceptions = _get_bool_env("RAGAS_RAISE_EXCEPTIONS", False)
    run_config = RunConfig(
        timeout=timeout_sec,
        max_retries=max_retries,
        max_workers=max_workers,
        max_wait=30,
    )
    print(
        "评测配置: "
        f"timeout={timeout_sec}s, retries={max_retries}, workers={max_workers}, "
        f"raise_exceptions={raise_exceptions}"
    )

    requested_thinking = _get_bool_env("RAGAS_ENABLE_THINKING", False)
    if requested_thinking:
        print("警告: 当前评测使用非流式调用，已强制 enable_thinking=False 以避免 DashScope 400。")
    forced_thinking = False
    llm_candidates = _build_tongyi_llm_candidates(
        api_key=api_key,
        enable_thinking=forced_thinking,
    )
    print(
        f"LLM配置: model=qwen3-30b-a3b, streaming=False, enable_thinking={forced_thinking}, "
        f"candidate_count={len(llm_candidates)}"
    )
    embedding = DashScopeEmbeddings(
        model="text-embedding-v3", dashscope_api_key=api_key
    )
    vembedding = LangchainEmbeddingsWrapper(embedding)

    result = None
    last_error: Optional[Exception] = None
    tried_modes: List[str] = []
    for llm_mode, llm in llm_candidates:
        tried_modes.append(llm_mode)
        vllm = LangchainLLMWrapper(llm, run_config=run_config)
        print(f"评测尝试: ctor_mode={llm_mode}")
        try:
            result = _run_eval(
                dataset=dataset,
                vllm=vllm,
                vembedding=vembedding,
                run_config=run_config,
                raise_exceptions=raise_exceptions,
            )
            break
        except Exception as e:
            last_error = e
            if _is_enable_thinking_non_streaming_error(e):
                print(f"命中 enable_thinking 非流式限制，切换参数模式重试: {llm_mode}")
                continue
            if _is_invalid_parameter_error(e):
                print(f"命中 InvalidParameter(400)，切换参数模式重试: {llm_mode}")
                continue
            print(f"评测失败: {type(e).__name__}: {e}")
            if isinstance(e, (TimeoutError, asyncio.TimeoutError)):
                print(
                    "超时建议: 降低并发或增加超时时间。可设置环境变量 "
                    "RAGAS_MAX_WORKERS=2 或 RAGAS_TIMEOUT_SEC=600 后重试。"
                )
            print("详细堆栈如下:")
            print(traceback.format_exc())
            raise

    if result is None:
        print(
            "评测失败: 所有 Tongyi 参数模式均尝试失败。"
            f"已尝试: {', '.join(tried_modes)}"
        )
        if last_error is not None:
            print(f"最后一次错误: {type(last_error).__name__}: {last_error}")
            print("详细堆栈如下:")
            print(traceback.format_exc())
            raise last_error
        raise RuntimeError("评测失败，未获得有效评测结果。")

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
    parser = argparse.ArgumentParser(description="执行RAGAS评测并导出Excel")
    parser.add_argument(
        "--input",
        default="eval_finish.json",
        help="输入评测json文件名（位于 test/eval_data 下）",
    )
    parser.add_argument(
        "--output",
        default="ragas_eval.xlsx",
        help="输出Excel文件名（位于 test/eval_data 下）",
    )
    args = parser.parse_args()
    run_ragas_eval(eval_finish_name=args.input, output_name=args.output)
