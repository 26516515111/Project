#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from kg_pipeline.graph import build_graph
from kg_pipeline.paths import PipelinePaths
from kg_pipeline.pipeline_logging import PipelineLogger, build_run_log_path
from kg_pipeline.runtime_state import (
    archive_current_kg,
    clear_derived_outputs,
    current_raw_files,
    load_kg_state,
    merge_completed_steps,
    next_incomplete_step,
    save_kg_state,
)
from kg_pipeline.steps import migrate_legacy_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project 内部 KG LangGraph 流水线")
    parser.add_argument(
        "--mode",
        choices=["overwrite", "incremental"],
        default="incremental",
        help="KG 运行模式：覆盖或增量",
    )
    parser.add_argument(
        "--kg-name",
        type=str,
        default=None,
        help="当前 KG 名称，用于状态文件和覆盖模式备份名",
    )
    parser.add_argument(
        "--ingest",
        dest="ingest_path",
        type=str,
        default=None,
        help="增量模式：一次性输入文件路径，自动完成 step1~4",
    )
    parser.add_argument(
        "--from", dest="start", type=int, default=1, help="从第几步开始"
    )
    parser.add_argument(
        "--to", dest="end", type=int, default=6, help="运行到第几步结束"
    )
    parser.add_argument("--only", type=int, default=None, help="只运行某一步")
    parser.add_argument("--skip-neo4j", action="store_true", help="跳过 step5")
    parser.add_argument("--dry-run", action="store_true", help="仅打印计划，不实际执行")
    parser.add_argument(
        "--only-doc",
        dest="only_doc_id",
        type=str,
        default=None,
        help="step3 仅处理指定 doc_id",
    )
    parser.add_argument(
        "--no-context", action="store_true", help="step3 关闭上下文注入"
    )
    parser.add_argument(
        "--no-checkpoint", action="store_true", help="step3 不使用 checkpoint"
    )
    parser.add_argument(
        "--no-neo4j-import", action="store_true", help="step5 仅生成 CSV，不导入 Neo4j"
    )
    parser.add_argument(
        "--no-neo4j-dump", action="store_true", help="step5 导入后不导出 dump"
    )
    parser.add_argument(
        "--visualize-top", type=int, default=300, help="step6 最多展示节点数"
    )
    parser.add_argument(
        "--visualize-label", type=str, default=None, help="step6 仅展示指定标签"
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.ingest_path:
        return
    if not (1 <= args.start <= 6 and 1 <= args.end <= 6):
        raise SystemExit("参数错误: --from/--to 必须在 1~6")
    if args.start > args.end and args.only is None:
        raise SystemExit("参数错误: --from 不能大于 --to")
    if args.only is not None and not (1 <= args.only <= 6):
        raise SystemExit("参数错误: --only 必须在 1~6")


def main() -> int:
    args = parse_args()
    validate_args(args)

    paths = PipelinePaths.discover()
    paths.ensure_dirs()
    log_path = build_run_log_path(paths.logs_dir)
    logger = PipelineLogger(log_path)
    migration = migrate_legacy_files(paths)

    try:
        logger.info(f"日志文件: {log_path}")
        logger.info("启动 KG LangGraph 流水线")
        if migration["moved"] or migration["removed_duplicates"]:
            logger.info(
                "已迁移旧文件: %s，清理重复文件: %s"
                % (len(migration["moved"]), len(migration["removed_duplicates"]))
            )

        if args.ingest_path:
            source_path = Path(args.ingest_path).expanduser().resolve()
            if not source_path.exists():
                raise SystemExit(f"增量文件不存在: {source_path}")
            if source_path.is_dir():
                raise SystemExit("增量模式暂不支持目录，请提供单个文件路径")
            target_path = paths.raw_dir / source_path.name
            shutil.copy2(source_path, target_path)
            logger.info(f"增量导入文件: {source_path} -> {target_path}")
            args.start = 1
            args.end = 4
            args.only = None
            args.only_doc_id = source_path.stem

        kg_state = load_kg_state(paths)
        raw_files = current_raw_files(paths)
        current_name = str(args.kg_name or kg_state.get("kg_name") or "KG").strip() or "KG"

        if args.mode == "overwrite":
            archive_source_name = str(kg_state.get("kg_name") or current_name).strip() or "KG"
            if args.dry_run:
                logger.info(
                    "覆盖模式预演 | kg_name=%s | raw_files=%s | archive_name=%s"
                    % (current_name, len(raw_files), archive_source_name)
                )
            else:
                archive_path = archive_current_kg(paths, archive_source_name)
                logger.info(f"覆盖模式备份完成: {archive_path}")
                clear_derived_outputs(paths)
                kg_state = {
                    "kg_name": current_name,
                    "used_files": raw_files,
                    "completed_steps": [],
                    "last_archive": str(archive_path),
                }
                save_kg_state(paths, kg_state)
                logger.info(
                    "覆盖模式初始化完成 | kg_name=%s | raw_files=%s"
                    % (current_name, len(raw_files))
                )
        else:
            used_files = set(kg_state.get("used_files", []))
            new_files = sorted(set(raw_files) - used_files)
            next_step = next_incomplete_step(kg_state.get("completed_steps", []))
            logger.info(
                "增量模式检查 | kg_name=%s | used_files=%s | raw_files=%s | new_files=%s | completed_steps=%s"
                % (
                    current_name,
                    len(used_files),
                    len(raw_files),
                    len(new_files),
                    kg_state.get("completed_steps", []),
                )
            )
            if new_files:
                if args.dry_run:
                    logger.info(f"增量模式预演发现新文件: {new_files}")
                else:
                    clear_derived_outputs(paths)
                    kg_state["used_files"] = sorted(set(kg_state.get("used_files", [])) | set(new_files))
                    kg_state["kg_name"] = current_name
                    kg_state["completed_steps"] = []
                    save_kg_state(paths, kg_state)
                logger.info(f"增量模式发现新文件: {new_files}")
                args.start = 1
                args.end = 6
            elif next_step is not None and args.only is None:
                args.start = next_step
                args.end = 6
                logger.info(f"增量模式续跑未完成步骤: Step {next_step} -> 6")
            elif next_step is None and args.only is None:
                logger.info("增量模式无新文件，且流水线已完成，无需执行")
                return 0

        app = build_graph()
        state = app.invoke(
            {
                "paths": paths,
                "start": args.start,
                "end": args.end,
                "only": args.only,
                "skip_neo4j": args.skip_neo4j,
                "dry_run": args.dry_run,
                "only_doc_id": Path(args.only_doc_id).stem if args.only_doc_id else None,
                "use_context": not args.no_context,
                "checkpoint_enabled": not args.no_checkpoint,
                "neo4j_import": not args.no_neo4j_import,
                "neo4j_dump": not args.no_neo4j_dump,
                "visualize_top_n": args.visualize_top,
                "visualize_label": args.visualize_label,
                "logs": [],
                "logger": logger,
            }
        )
        selected_steps = state.get("selected_steps", [])
        failed_step = state.get("failed_step")
        if not args.dry_run:
            updated_state = load_kg_state(paths)
            updated_state["kg_name"] = current_name
            updated_state["used_files"] = current_raw_files(paths)
            updated_state["completed_steps"] = merge_completed_steps(
                updated_state.get("completed_steps", []),
                selected_steps,
                failed_step,
            )
            save_kg_state(paths, updated_state)
            logger.info(
                "状态文件已更新 | kg_name=%s | used_files=%s | completed_steps=%s"
                % (
                    updated_state.get("kg_name", ""),
                    len(updated_state.get("used_files", [])),
                    updated_state.get("completed_steps", []),
                )
            )
        exit_code = 1 if state.get("failed", False) else 0
        logger.info(f"退出码: {exit_code}")
        return exit_code
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
