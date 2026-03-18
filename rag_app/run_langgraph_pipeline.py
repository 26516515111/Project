#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from kg_pipeline.graph import build_graph
from kg_pipeline.paths import PipelinePaths
from kg_pipeline.steps import migrate_legacy_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project 内部 KG LangGraph 流水线")
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
    migration = migrate_legacy_files(paths)

    if args.ingest_path:
        source_path = Path(args.ingest_path).expanduser().resolve()
        if not source_path.exists():
            raise SystemExit(f"增量文件不存在: {source_path}")
        if source_path.is_dir():
            raise SystemExit("增量模式暂不支持目录，请提供单个文件路径")
        target_path = paths.raw_dir / source_path.name
        shutil.copy2(source_path, target_path)
        args.start = 1
        args.end = 4
        args.only = None
        args.only_doc_id = source_path.stem

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
        }
    )

    if migration["moved"] or migration["removed_duplicates"]:
        print(
            f"已迁移旧文件: {len(migration['moved'])}，清理重复文件: {len(migration['removed_duplicates'])}"
        )
    for line in state.get("logs", []):
        print(line)
    return 1 if state.get("failed", False) else 0


if __name__ == "__main__":
    raise SystemExit(main())
