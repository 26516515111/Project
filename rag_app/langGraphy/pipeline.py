#!/usr/bin/env python3
"""LangGraph KG pipeline orchestrator.

Wraps step1~step7 scripts into a LangGraph state machine and offers both
CLI and programmatic interfaces.
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

from .config import (
    DEFAULT_END,
    DEFAULT_EXECUTION_MODE,
    DEFAULT_HF_ENDPOINT,
    DEFAULT_START,
    NODES,
    NODE_ORDER,
    SCRIPTS_DIR,
    SHIP_ROOT,
)


class PipelineState(TypedDict, total=False):
    """State container for pipeline execution."""

    start: int
    end: int
    only: int | None
    skip_neo4j: bool
    dry_run: bool
    execution_mode: str
    selected_nodes: list[dict[str, Any]]
    current_node: str | None
    failed: bool
    failed_step: int | None
    logs: list[str]


def load_env_file(env_path: Path) -> dict[str, str]:
    """Load key=value pairs from .env file into a dict."""

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


def build_selected_nodes(
    start: int, end: int, only: int | None, skip_neo4j: bool
) -> list[dict[str, Any]]:
    """Build the ordered node list to execute based on CLI args."""

    if not NODES:
        return []

    clamped_start = max(1, start)
    clamped_end = min(len(NODES), end)
    if only is not None:
        if not (1 <= only <= len(NODES)):
            return []
        selected = [NODES[only - 1]]
    else:
        selected = NODES[clamped_start - 1 : clamped_end]

    if skip_neo4j:
        selected = [n for n in selected if n[0] != "neo4j"]

    return [{"node_id": n[0], "script": n[1], "desc": n[2]} for n in selected]


def pick_python_executable() -> str:
    """Pick venv python if present, otherwise current interpreter."""

    venv_python = SHIP_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


@contextmanager
def temporary_env(extra_env: dict[str, str]):
    """Temporarily inject environment variables for script execution."""

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
    """Execute a step by importing and calling its main() in-process."""

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
        with temporary_env(
            load_env_file(SHIP_ROOT / ".env")
            | {"HF_ENDPOINT": os.environ.get("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)}
        ):
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
    """Execute a step as a subprocess using python interpreter."""

    python_exe = pick_python_executable()
    child_env = os.environ.copy()
    child_env.update(load_env_file(SHIP_ROOT / ".env"))
    child_env.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
    cmd = [python_exe, str(script_path)]
    ret = subprocess.run(cmd, cwd=str(SHIP_ROOT), env=child_env)
    return ret.returncode


def planner_node(state: PipelineState) -> PipelineState:
    """Plan which nodes will be executed and seed the state."""

    selected_nodes = build_selected_nodes(
        start=state["start"],
        end=state["end"],
        only=state.get("only"),
        skip_neo4j=state.get("skip_neo4j", False),
    )

    logs = list(state.get("logs", []))
    if not selected_nodes:
        logs.append("没有可执行节点，请检查参数")
        return {
            "selected_nodes": [],
            "current_node": None,
            "failed": True,
            "failed_step": None,
            "logs": logs,
        }

    logs.append("将执行以下节点：")
    logs.append(f"执行模式: {state.get('execution_mode', DEFAULT_EXECUTION_MODE)}")
    for n in selected_nodes:
        logs.append(f"  - Node {n['node_id']}: {n['desc']} ({n['script']})")

    first_node = selected_nodes[0]["node_id"] if selected_nodes else None
    return {
        "selected_nodes": selected_nodes,
        "current_node": str(first_node) if first_node is not None else None,
        "failed": False,
        "failed_step": None,
        "logs": logs,
    }


def _get_node_meta(
    selected_nodes: list[dict[str, Any]], node_id: str
) -> dict[str, Any] | None:
    """Find node metadata in the selected nodes list."""

    for item in selected_nodes:
        if str(item["node_id"]) == node_id:
            return item
    return None


def _next_node_id(selected_nodes: list[dict[str, Any]], node_id: str) -> str | None:
    """Return the next node id in the execution plan."""

    ordered = [str(item["node_id"]) for item in selected_nodes]
    if node_id not in ordered:
        return None
    idx = ordered.index(node_id)
    if idx + 1 >= len(ordered):
        return None
    return ordered[idx + 1]


def execute_graph_node(state: PipelineState, node_id: str) -> PipelineState:
    """Execute a single graph node and return updated state."""

    logs = list(state.get("logs", []))
    selected_nodes = state.get("selected_nodes", [])

    node = _get_node_meta(selected_nodes, node_id)
    if node is None:
        logs.append(f"✗ Node {node_id} 未在计划中，无法执行")
        return {
            "failed": True,
            "failed_step": None,
            "logs": logs,
        }

    script = str(node["script"])
    desc = str(node["desc"])

    script_path = SCRIPTS_DIR / script
    if not script_path.exists():
        logs.append(f"✗ Node {node_id} 文件不存在: {script_path}")
        return {
            "failed": True,
            "failed_step": None,
            "logs": logs,
        }

    logs.append("=" * 60)
    logs.append(f"Node {node_id}: {desc}")
    logs.append(f"执行: {script_path}")
    logs.append("=" * 60)

    if state.get("dry_run", False):
        logs.append(f"[dry-run] 跳过执行 Node {node_id}")
        return {
            "current_node": _next_node_id(selected_nodes, node_id),
            "logs": logs,
        }

    t0 = time.time()
    mode = state.get("execution_mode", DEFAULT_EXECUTION_MODE)
    try:
        if mode == "inline":
            ret_code = execute_step_inline(script_path)
        else:
            ret_code = execute_step_subprocess(script_path)
    except Exception as e:
        logs.append(f"✗ Node {node_id} 异常: {e}")
        return {
            "failed": True,
            "failed_step": None,
            "logs": logs,
        }
    elapsed = time.time() - t0

    if ret_code != 0:
        logs.append(f"✗ Node {node_id} 失败，退出码 {ret_code}，耗时 {elapsed:.1f}s")
        return {
            "failed": True,
            "failed_step": None,
            "logs": logs,
        }

    logs.append(f"✓ Node {node_id} 完成，耗时 {elapsed:.1f}s")
    return {
        "current_node": _next_node_id(selected_nodes, node_id),
        "logs": logs,
    }


def route_after_planner(state: PipelineState) -> str:
    """Route to next node after planning."""

    if state.get("failed", False):
        return "fail"
    return "route"


def route_after_execute(state: PipelineState) -> str:
    """Route to step node, done, or fail based on current state."""

    if state.get("failed", False):
        return "fail"
    if state.get("current_node") is None:
        return "done"
    return f"node_{state['current_node']}"


def fail_node(state: PipelineState) -> PipelineState:
    """Finalize state when the pipeline fails."""

    logs = list(state.get("logs", []))
    failed_step = state.get("failed_step")
    if failed_step is None:
        logs.append("流水线失败")
    else:
        logs.append(f"流水线中断于 Step {failed_step}")
    return {"logs": logs}


def done_node(state: PipelineState) -> PipelineState:
    """Finalize state when the pipeline completes."""

    logs = list(state.get("logs", []))
    logs.append("流水线执行完成")
    return {"logs": logs}


def _graph_node_factory(node_id: str):
    """Create a graph node callable for a specific node id."""

    def _node(state: PipelineState) -> PipelineState:
        return execute_graph_node(state, node_id)

    return _node


def build_graph():
    """Build and compile the LangGraph state machine."""

    builder = StateGraph(PipelineState)
    builder.add_node("planner", planner_node)
    builder.add_node("route", lambda state: state)
    builder.add_node("fail", fail_node)
    builder.add_node("done", done_node)

    for node_id, _, _ in NODES:
        builder.add_node(f"node_{node_id}", _graph_node_factory(node_id))

    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "route": "route",
            "fail": "fail",
        },
    )
    builder.add_conditional_edges(
        "route",
        route_after_execute,
        {
            **{f"node_{node_id}": f"node_{node_id}" for node_id, _, _ in NODES},
            "done": "done",
            "fail": "fail",
        },
    )
    for node_id, _, _ in NODES:
        builder.add_edge(f"node_{node_id}", "route")

    builder.add_edge("fail", END)
    builder.add_edge("done", END)
    return builder.compile()


def run_pipeline(
    *,
    start: int = DEFAULT_START,
    end: int = DEFAULT_END,
    only: int | None = None,
    skip_neo4j: bool = False,
    dry_run: bool = False,
    execution_mode: str = DEFAULT_EXECUTION_MODE,
) -> PipelineState:
    """Run the pipeline with the provided parameters and return final state."""
    _ = NODE_ORDER
    app = build_graph()
    return app.invoke(
        {
            "start": start,
            "end": end,
            "only": only,
            "skip_neo4j": skip_neo4j,
            "dry_run": dry_run,
            "execution_mode": execution_mode,
            "logs": [],
        }
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="LangGraph 编排 KG 流水线")
    parser.add_argument(
        "--from", dest="start", type=int, default=DEFAULT_START, help="从第几步开始"
    )
    parser.add_argument(
        "--to", dest="end", type=int, default=DEFAULT_END, help="运行到第几步结束"
    )
    parser.add_argument("--only", type=int, default=None, help="只运行某一步")
    parser.add_argument("--skip-neo4j", action="store_true", help="跳过 step5")
    parser.add_argument("--dry-run", action="store_true", help="仅打印计划，不实际执行")
    parser.add_argument(
        "--execution-mode",
        choices=["inline", "subprocess"],
        default=DEFAULT_EXECUTION_MODE,
        help="执行模式：inline=进程内调用 main，subprocess=子进程执行脚本",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments to ensure they are in allowed ranges."""

    if not NODES:
        return
    if not (1 <= args.start <= len(NODES) and 1 <= args.end <= len(NODES)):
        raise SystemExit("参数错误: --from/--to 必须在 1~7")
    if args.start > args.end and args.only is None:
        raise SystemExit("参数错误: --from 不能大于 --to")
    if args.only is not None and not (1 <= args.only <= len(NODES)):
        raise SystemExit("参数错误: --only 必须在 1~7")


def main() -> int:
    """CLI entrypoint for executing the pipeline."""

    args = parse_args()
    validate_args(args)

    final_state = run_pipeline(
        start=args.start,
        end=args.end,
        only=args.only,
        skip_neo4j=args.skip_neo4j,
        dry_run=args.dry_run,
        execution_mode=args.execution_mode,
    )

    for line in final_state.get("logs", []):
        print(line)

    if final_state.get("failed", False):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
