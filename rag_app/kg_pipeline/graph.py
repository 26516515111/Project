from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .paths import PipelinePaths
from .workflows import (
    run_chunk_workflow,
    run_clean_workflow,
    run_extract_workflow,
    run_merge_workflow,
    run_neo4j_workflow,
    run_visualize_workflow,
)

STEP_SEQUENCE = [1, 2, 3, 4, 5, 6]


class PipelineState(TypedDict, total=False):
    paths: PipelinePaths
    start: int
    end: int
    only: int | None
    skip_neo4j: bool
    dry_run: bool
    only_doc_id: str | None
    use_context: bool
    checkpoint_enabled: bool
    neo4j_import: bool
    neo4j_dump: bool
    visualize_top_n: int
    visualize_label: str | None
    selected_steps: list[int]
    logs: list[str]
    outputs: dict[str, Any]
    failed: bool
    failed_step: int | None


def _select_steps(start: int, end: int, only: int | None, skip_neo4j: bool) -> list[int]:
    if only is not None:
        steps = [step for step in STEP_SEQUENCE if step == only]
    else:
        steps = [step for step in STEP_SEQUENCE if start <= step <= end]
    if skip_neo4j:
        steps = [step for step in steps if step != 5]
    return steps


def planner_node(state: PipelineState) -> PipelineState:
    selected_steps = _select_steps(
        start=state["start"],
        end=state["end"],
        only=state.get("only"),
        skip_neo4j=state.get("skip_neo4j", False),
    )
    logs = list(state.get("logs", []))
    if not selected_steps:
        logs.append("没有可执行步骤，请检查参数")
        return {"selected_steps": [], "logs": logs, "failed": True}
    logs.append(f"执行步骤: {selected_steps}")
    return {"selected_steps": selected_steps, "logs": logs, "outputs": {}, "failed": False, "failed_step": None}


def _run_step(state: PipelineState, step_no: int, description: str, action) -> PipelineState:
    logs = list(state.get("logs", []))
    outputs = dict(state.get("outputs", {}))
    if state.get("failed", False):
        return {"logs": logs, "outputs": outputs}
    if step_no not in state.get("selected_steps", []):
        logs.append(f"跳过 Step {step_no}: {description}")
        return {"logs": logs, "outputs": outputs}
    if state.get("dry_run", False):
        logs.append(f"[dry-run] Step {step_no}: {description}")
        return {"logs": logs, "outputs": outputs}
    try:
        outputs[f"step_{step_no}"] = action(state["paths"], state)
        logs.append(f"完成 Step {step_no}: {description}")
        return {"logs": logs, "outputs": outputs}
    except Exception as exc:
        logs.append(f"失败 Step {step_no}: {description} -> {exc}")
        return {"logs": logs, "outputs": outputs, "failed": True, "failed_step": step_no}


def step1_node(state: PipelineState) -> PipelineState:
    return _run_step(state, 1, "语料清洗", lambda paths, _: run_clean_workflow(paths))


def step2_node(state: PipelineState) -> PipelineState:
    return _run_step(state, 2, "文本切块", lambda paths, _: run_chunk_workflow(paths))


def step3_node(state: PipelineState) -> PipelineState:
    return _run_step(
        state,
        3,
        "KG 三元组抽取",
        lambda paths, st: run_extract_workflow(
            paths,
            only_doc_id=st.get("only_doc_id"),
            use_context=st.get("use_context", True),
            checkpoint_enabled=st.get("checkpoint_enabled", True),
        ),
    )


def step4_node(state: PipelineState) -> PipelineState:
    return _run_step(state, 4, "实体合并", lambda paths, _: run_merge_workflow(paths))


def step5_node(state: PipelineState) -> PipelineState:
    return _run_step(
        state,
        5,
        "Neo4j 导入与导出",
        lambda paths, st: run_neo4j_workflow(
            paths,
            import_to_neo4j=st.get("neo4j_import", True),
            export_dump=st.get("neo4j_dump", True),
        ),
    )


def step6_node(state: PipelineState) -> PipelineState:
    return _run_step(
        state,
        6,
        "图谱可视化",
        lambda paths, st: run_visualize_workflow(
            paths,
            top_n=st.get("visualize_top_n", 300),
            filter_label=st.get("visualize_label"),
        ),
    )


def route_after_node(state: PipelineState) -> str:
    return "fail" if state.get("failed", False) else "next"


def fail_node(state: PipelineState) -> PipelineState:
    logs = list(state.get("logs", []))
    if state.get("failed_step") is not None:
        logs.append(f"流水线中断于 Step {state['failed_step']}")
    else:
        logs.append("流水线失败")
    return {"logs": logs}


def done_node(state: PipelineState) -> PipelineState:
    logs = list(state.get("logs", []))
    logs.append("流水线执行完成")
    return {"logs": logs}


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("planner", planner_node)
    graph.add_node("step1", step1_node)
    graph.add_node("step2", step2_node)
    graph.add_node("step3", step3_node)
    graph.add_node("step4", step4_node)
    graph.add_node("step5", step5_node)
    graph.add_node("step6", step6_node)
    graph.add_node("fail", fail_node)
    graph.add_node("done", done_node)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", route_after_node, {"next": "step1", "fail": "fail"})
    graph.add_conditional_edges("step1", route_after_node, {"next": "step2", "fail": "fail"})
    graph.add_conditional_edges("step2", route_after_node, {"next": "step3", "fail": "fail"})
    graph.add_conditional_edges("step3", route_after_node, {"next": "step4", "fail": "fail"})
    graph.add_conditional_edges("step4", route_after_node, {"next": "step5", "fail": "fail"})
    graph.add_conditional_edges("step5", route_after_node, {"next": "step6", "fail": "fail"})
    graph.add_conditional_edges("step6", route_after_node, {"next": "done", "fail": "fail"})
    graph.add_edge("done", END)
    graph.add_edge("fail", END)
    return graph.compile()
