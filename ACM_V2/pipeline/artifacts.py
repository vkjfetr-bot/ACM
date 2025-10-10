"""Artifact management utilities."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from ..acm_observe import write_payloads  # reuse consolidated payload builder
from ..acm_observe import write_run_health, write_run_summary


@dataclass
class ArtifactPaths:
    root: Path
    equip_root: Path
    run_dir: Path
    model_dir: Path
    images_dir: Path
    run_id: str


@dataclass
class ArtifactManager:
    """Handle per-run artifact layout and persistence."""

    art_root: Path
    equip: str
    run_type: str
    run_paths: Optional[ArtifactPaths] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.art_root.mkdir(parents=True, exist_ok=True)
        equip_root = self.art_root / self.equip
        equip_root.mkdir(parents=True, exist_ok=True)
        model_dir = equip_root / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        self._equip_root = equip_root
        self._model_dir = model_dir

    def start_run(self, timestamp: Optional[datetime] = None) -> ArtifactPaths:
        ts = (timestamp or datetime.utcnow()).strftime("%Y%m%d_%H%M%S")
        run_id = f"{self.run_type}_{ts}"
        run_dir = self._equip_root / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        images_dir = run_dir / "imgs"
        images_dir.mkdir(parents=True, exist_ok=True)
        paths = ArtifactPaths(
            root=self.art_root,
            equip_root=self._equip_root,
            run_dir=run_dir,
            model_dir=self._model_dir,
            images_dir=images_dir,
            run_id=run_id,
        )
        self.run_paths = paths
        return paths

    def write_json(self, data: Dict[str, Any], name: str) -> Path:
        path = self._ensure_run_dir() / name
        path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")
        return path

    def write_table(self, df: pd.DataFrame, name: str, *, prefer_parquet: bool = True) -> Path:
        df = df.copy()
        path = self._ensure_run_dir() / name
        if prefer_parquet and name.endswith(".parquet"):
            try:
                df.to_parquet(path, index=True)
                return path
            except Exception:
                # Fallback to CSV but keep name consistent for future SQL export.
                csv_path = path.with_suffix(".csv")
                df.to_csv(csv_path, index=True)
                return csv_path
        elif name.endswith(".csv"):
            df.to_csv(path, index=True)
        else:
            df.to_csv(path, index=True)
        return path

    def write_payloads(self) -> None:
        """Regenerate payload JSON files in equip root."""
        write_payloads(str(self._equip_root))

    def emit_run_summary(self, row: Dict[str, Any]) -> None:
        """Append run summary CSV & health snapshot."""
        write_run_summary(str(self._equip_root), row)
        write_run_health(str(self._equip_root), row)
        # Placeholder for SQL persistence of run summary.
        # TODO: Replace with usp_WriteRunSummary call when SQL wiring is enabled.

    def mark_success(self, extra: Optional[Dict[str, Any]] = None) -> None:
        self.metadata["status"] = "ok"
        if extra:
            self.metadata.update(extra)
        self._write_metadata()

    def mark_failure(self, err: Exception) -> None:
        self.metadata["status"] = "error"
        self.metadata["error"] = str(err)
        self._write_metadata()

    def _ensure_run_dir(self) -> Path:
        if not self.run_paths:
            raise RuntimeError("Run directory not initialised. Call start_run() first.")
        return self.run_paths.run_dir

    def _write_metadata(self) -> None:
        if not self.run_paths:
            return
        meta = {**self.metadata, "run_id": self.run_paths.run_id, "equip": self.equip, "run_type": self.run_type}
        (self.run_paths.run_dir / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def latest_model_path(self, name: str) -> Path:
        return self._model_dir / name

    def archive_model(self, source: Path, name: str) -> Path:
        target = self._model_dir / name
        if source != target:
            target.write_bytes(source.read_bytes())
        return target

    def list_runs(self) -> Iterable[Path]:
        return sorted(self._equip_root.glob("run_*"), key=os.path.getmtime, reverse=True)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    return str(obj)
