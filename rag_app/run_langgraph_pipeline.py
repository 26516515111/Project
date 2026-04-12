#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from kg_pipeline.graph import build_graph
from kg_pipeline.paths import PipelinePaths
from kg_pipeline.pipeline_logging import PipelineLogger, build_run_log_path
from kg_pipeline.runtime_state import (
    current_raw_files,
    load_kg_state,
    merge_completed_steps,
    save_kg_state,
)
from kg_pipeline.steps import migrate_legacy_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project 内部 KG LangGraph 流水线")
    parser.add_argument(
        "--kg-name",
        type=str,
        default=None,
        help="当前 KG 名称，用于状态文件",
    )
    parser.add_argument(
        "--from", dest="start", type=int, default=1, help="从第几步开始"
    )
    parser.add_argument(
        "--to", dest="end", type=int, default=7, help="运行到第几步结束"
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
    parser.add_argument(
        "--release-kg",
        dest="release_kg_names",
        action="append",
        default=[],
        help="step7 发布时包含的 KG 名称，可重复传入；默认发布当前 kg-name",
    )
    parser.add_argument(
        "--release-name",
        dest="release_output_name",
        type=str,
        default="release",
        help="step7 发布包名称",
    )
    parser.add_argument(
        "--no-release-import-backups",
        action="store_true",
        help="step7 发布前不自动导入 KG/backups 下的 tar.gz",
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

    current_name = PipelinePaths.normalize_kg_name(args.kg_name or "default")
    paths = PipelinePaths.discover(current_name)
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

        kg_state = load_kg_state(paths)
        raw_files = current_raw_files(paths)
        logger.info(
            "标准模式执行 | kg_name=%s | raw_files=%s | selected=%s | release_kgs=%s"
            % (
                current_name,
                len(raw_files),
                args.only if args.only else f"{args.start}-{args.end}",
                args.release_kg_names or [current_name],
            )
        )

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
                "release_kg_names": [
                    PipelinePaths.normalize_kg_name(name)
                    for name in (args.release_kg_names or [current_name])
                ],
                "release_output_name": args.release_output_name,
                "release_import_backups": not args.no_release_import_backups,
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
