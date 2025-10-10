"""CLI entry point for the consolidated ACM pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

from .acm_pipeline_core import PipelineRunner, load_config, __version__


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("acm_pipeline", description="ACM pipeline orchestrator")
    parser.add_argument("--config", type=Path, help="Optional configuration file (YAML/JSON).")
    parser.add_argument("--artifacts", type=Path, default=Path("acm_artifacts"), help="Artifacts directory root.")
    parser.add_argument("--equip", required=True, help="Equipment identifier.")
    parser.add_argument("--t0", help="ISO timestamp for start of window.")
    parser.add_argument("--t1", help="ISO timestamp for end of window.")
    parser.add_argument("--csv", required=True, type=Path, help="CSV source for data ingestion.")
    parser.add_argument("--mode", choices=["train", "score"], required=True, help="Pipeline mode.")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    runner = PipelineRunner(args.equip, Path(args.artifacts), config)
    t0 = _parse_time(args.t0)
    t1 = _parse_time(args.t1)

    if args.mode == "train":
        result = runner.train(args.csv, t0=t0, t1=t1)
        print(f"[ACM] Training run {result.run_id} complete — features stored at {result.features_path}")
    elif args.mode == "score":
        result = runner.score(args.csv, t0=t0, t1=t1)
        print(f"[ACM] Scoring run {result.run_id} complete — scores stored at {result.scores_path}")
    else:  # pragma: no cover
        parser.error(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
