from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from .config import PipelineSettings
from .runner import run_pipeline
from .storage import load_run_summary
from .sync import sync_data


def _parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone data pipeline for eq_prediction.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the data pipeline.")
    run_parser.add_argument("--source", choices=["local", "db", "all"], default="all")
    run_parser.add_argument("--fetch-new", action="store_true")
    run_parser.add_argument(
        "--timeseries",
        action="store_true",
        help="Add cyclical time and rolling-window features for time-series workflows.",
    )
    run_parser.add_argument(
        "--enrich-details",
        action="store_true",
        help="Fetch USGS detail URLs to fill missing nst, dmin, and gap values.",
    )
    run_parser.add_argument("--no-save", action="store_true")
    run_parser.add_argument("--fetch-start", help="Optional ISO datetime for USGS fetch start.")
    run_parser.add_argument("--fetch-end", help="Optional ISO datetime for USGS fetch end.")

    subparsers.add_parser("status", help="Print the latest run summary.")
    sync_parser = subparsers.add_parser("sync", help="Synchronize data CSV snapshots with PostgreSQL.")
    sync_parser.add_argument("--direction", choices=["auto", "push", "pull"], default="auto")
    sync_parser.add_argument("--force", action="store_true", help="Resolve conflicts in the selected push/pull direction.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "status":
        try:
            print(json.dumps(load_run_summary(), indent=2))
            return 0
        except FileNotFoundError as exc:
            print(exc)
            return 1

    if args.command == "run":
        result = run_pipeline(
            source=args.source,
            fetch_new=args.fetch_new,
            timeseries=args.timeseries,
            enrich_details=args.enrich_details,
            save=not args.no_save,
            settings=PipelineSettings.from_env(),
            fetch_start=_parse_datetime(args.fetch_start),
            fetch_end=_parse_datetime(args.fetch_end),
        )
        print(f"status={result.status}")
        print(f"rows_loaded={result.rows_loaded}")
        print(f"rows_clean={result.rows_clean}")
        print(f"rows_model_ready={result.rows_model_ready}")
        print(f"rows_rejected={result.rows_rejected}")
        if result.errors:
            print("errors:")
            for error in result.errors:
                print(f"- {error}")
        if result.output_paths:
            print("outputs:")
            for key, path in result.output_paths.items():
                print(f"- {key}: {path}")
        return 0 if result.status != "failed" else 1

    if args.command == "sync":
        result = sync_data(PipelineSettings.from_env(), direction=args.direction, force=args.force)
        print(f"status={result.status}")
        for item in result.files:
            print(f"- {item.path}: {item.action} ({item.table})")
        for error in result.errors:
            print(f"warning: {error}")
        return 0 if result.status == "completed" else 1

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
