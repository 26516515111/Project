#!/usr/bin/env python3
"""
LangGraph 版 KG 流水线编排入口（包装现有 step1 ~ step7 脚本）

用法：
  python Project/rag_app/run_langgraph_pipeline.py
  python Project/rag_app/run_langgraph_pipeline.py --from 4
  python Project/rag_app/run_langgraph_pipeline.py --to 6
  python Project/rag_app/run_langgraph_pipeline.py --only 7
  python Project/rag_app/run_langgraph_pipeline.py --skip-neo4j
  python Project/rag_app/run_langgraph_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

SHIP_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = SHIP_ROOT / "scripts"

STEPS = [
    (1, "step1_clean.py", "语料清洗"),
    (2, "step2_chunk.py", "文本切块"),
    (3, "step3_extract_kg.py", "KG 三元组抽取"),
    (4, "step4_merge.py", "实体合并"),
    (5, "step5_neo4j.py", "Neo4j 导入与导出"),
    (6, "step6_visualize.py", "图谱可视化"),
    (7, "step7_publish_project.py", "发布交付物并推送远程仓库"),
]


class PipelineState(TypedDict, total=False):
    start: int
    end: int
    only: int | None
    skip_neo4j: bool
    dry_run: bool
    execution_mode: str
    selected_steps: list[dict[str, Any]]
    current_index: int
    failed: bool
    failed_step: int | None
    logs: list[str]


def load_env_file(env_path: Path) -> dict[str, str]:
    env_vars: dict[str, str] = {}
    if not env_path.exists():
        return env_vars
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env_vars[key.strip()] = value.strip().strip('"').strip("'")
    return env_vars


def build_selected_steps(start: int, end: int, only: int | None, skip_neo4j: bool) -> list[dict[str, Any]]:
    if only is not None:
        selected = [s for s in STEPS if s[0] == only]
    else:
        selected = [s for s in STEPS if start <= s[0] <= end]

    if skip_neo4j:
        selected = [s for s in selected if s[0] != 5]

    return [{"step_no": s[0], "script": s[1], "desc": s[2]} for s in selected]


def pick_python_executable() -> str:
    venv_python = SHIP_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


@contextmanager
def temporary_env(extra_env: dict[str, str]):
    old_values: dict[str, str | None] = {}
    for k, v in extra_env.items():
        old_values[k] = os.environ.get(k)
        os.environ[k] = v
    try:
        yield
    finally:
        for k, old_v in old_values.items():
            if old_v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old_v


def execute_step_inline(script_path: Path) -> int:
    module_name = f"pipeline_inline_{script_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载脚本: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "main"):
        raise RuntimeError(f"脚本未定义 main(): {script_path.name}")

    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    try:
        sys.argv = [str(script_path)]
        with temporary_env(load_env_file(SHIP_ROOT / ".env") | {"HF_ENDPOINT": os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")}):
            os.chdir(SHIP_ROOT)
            ret = module.main()
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)

    if ret is None:
        return 0
    if isinstance(ret, int):
        return ret
    return 0


def execute_step_subprocess(script_path: Path) -> int:
    python_exe = pick_python_executable()
    child_env = os.environ.copy()
    child_env.update(load_env_file(SHIP_ROOT / ".env"))
    child_env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    cmd = [python_exe, str(script_path)]
    ret = subprocess.run(cmd, cwd=str(SHIP_ROOT), env=child_env)
    return ret.returncode


def planner_node(state: PipelineState) -> PipelineState:
    selected_steps = build_selected_steps(
        start=state["start"],
        end=state["end"],
        only=state.get("only"),
        skip_neo4j=state.get("skip_neo4j", False),
    )

    logs = list(state.get("logs", []))
    if not selected_steps:
        logs.append("没有可执行步骤，请检查参数")
        return {
            "selected_steps": [],
            "current_index": 0,
            "failed": True,
            "failed_step": None,
            "logs": logs,
        }

    logs.append("将执行以下步骤：")
    logs.append(f"执行模式: {state.get('execution_mode', 'inline')}")
    for s in selected_steps:
        logs.append(f"  - Step {s['step_no']}: {s['desc']} ({s['script']})")

    return {
        "selected_steps": selected_steps,
        "current_index": 0,
        "failed": False,
        "failed_step": None,
        "logs": logs,
    }


def execute_step_node(state: PipelineState) -> PipelineState:
    logs = list(state.get("logs", []))
    idx = state.get("current_index", 0)
    selected_steps = state.get("selected_steps", [])

    if idx >= len(selected_steps):
        return {"logs": logs}

    step = selected_steps[idx]
    step_no = int(step["step_no"])
    script = str(step["script"])
    desc = str(step["desc"])

    script_path = SCRIPTS_DIR / script
    if not script_path.exists():
        logs.append(f"✗ Step {step_no} 文件不存在: {script_path}")
        return {
            "failed": True,
            "failed_step": step_no,
            "logs": logs,
        }

    logs.append("=" * 60)
    logs.append(f"Step {step_no}: {desc}")
    logs.append(f"执行: {script_path}")
    logs.append("=" * 60)

    if state.get("dry_run", False):
        logs.append(f"[dry-run] 跳过执行 Step {step_no}")
        return {
            "current_index": idx + 1,
            "logs": logs,
        }

    t0 = time.time()
    mode = state.get("execution_mode", "inline")
    try:
        if mode == "inline":
            ret_code = execute_step_inline(script_path)
        else:
            ret_code = execute_step_subprocess(script_path)
    except Exception as e:
        logs.append(f"✗ Step {step_no} 异常: {e}")
        return {
            "failed": True,
            "failed_step": step_no,
            "logs": logs,
        }
    elapsed = time.time() - t0

    if ret_code != 0:
        logs.append(f"✗ Step {step_no} 失败，退出码 {ret_code}，耗时 {elapsed:.1f}s")
        return {
            "failed": True,
            "failed_step": step_no,
            "logs": logs,
        }

    logs.append(f"✓ Step {step_no} 完成，耗时 {elapsed:.1f}s")
    return {
        "current_index": idx + 1,
        "logs": logs,
    }


def route_after_planner(state: PipelineState) -> str:
    if state.get("failed", False):
        return "fail"
    return "execute"


def route_after_execute(state: PipelineState) -> str:
    if state.get("failed", False):
        return "fail"
    if state.get("current_index", 0) >= len(state.get("selected_steps", [])):
        return "done"
    return "execute"


def fail_node(state: PipelineState) -> PipelineState:
    logs = list(state.get("logs", []))
    failed_step = state.get("failed_step")
    if failed_step is None:
        logs.append("流水线失败")
    else:
        logs.append(f"流水线中断于 Step {failed_step}")
    return {"logs": logs}


def done_node(state: PipelineState) -> PipelineState:
    logs = list(state.get("logs", []))
    logs.append("流水线执行完成")
    return {"logs": logs}


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("planner", planner_node)
    graph.add_node("execute", execute_step_node)
    graph.add_node("fail", fail_node)
    graph.add_node("done", done_node)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "execute": "execute",
            "fail": "fail",
        },
    )
    graph.add_conditional_edges(
        "execute",
        route_after_execute,
        {
            "execute": "execute",
            "done": "done",
            "fail": "fail",
        },
    )
    graph.add_edge("fail", END)
    graph.add_edge("done", END)
    return graph.compile()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LangGraph 编排 KG 流水线")
    parser.add_argument("--from", dest="start", type=int, default=1, help="从第几步开始")
    parser.add_argument("--to", dest="end", type=int, default=7, help="运行到第几步结束")
    parser.add_argument("--only", type=int, default=None, help="只运行某一步")
    parser.add_argument("--skip-neo4j", action="store_true", help="跳过 step5")
    parser.add_argument("--dry-run", action="store_true", help="仅打印计划，不实际执行")
    parser.add_argument(
        "--execution-mode",
        choices=["inline", "subprocess"],
        default="inline",
        help="执行模式：inline=进程内调用 main，subprocess=子进程执行脚本",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not (1 <= args.start <= 7 and 1 <= args.end <= 7):
        raise SystemExit("参数错误: --from/--to 必须在 1~7")
    if args.start > args.end and args.only is None:
        raise SystemExit("参数错误: --from 不能大于 --to")
    if args.only is not None and not (1 <= args.only <= 7):
        raise SystemExit("参数错误: --only 必须在 1~7")


def main() -> int:
    args = parse_args()
    validate_args(args)

    app = build_graph()
    final_state = app.invoke(
        {
            "start": args.start,
            "end": args.end,
            "only": args.only,
            "skip_neo4j": args.skip_neo4j,
            "dry_run": args.dry_run,
            "execution_mode": args.execution_mode,
            "logs": [],
        }
    )

    for line in final_state.get("logs", []):
        print(line)

    if final_state.get("failed", False):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
