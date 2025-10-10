"""Configuration loading utilities for the ACM pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

from . import default_config_path


DEFAULT_CONFIG: Dict[str, Any] = {
    "sampling": {"period": "10s"},
    "features": {"window": "60s", "step": "10s", "spectral_bands": [[0.0, 0.1], [0.1, 0.3], [0.3, 0.5]]},
    "pca": {"variance": 0.95},
    "segmentation": {"model": "rbf", "min_duration_s": 60},
    "clustering": {"algo": "kmeans_auto", "k_range": [2, 8], "min_cluster_minutes": 3},
    "hmm": {
        "min_state_seconds": 60,
        "allowed_transitions": [["Idle", "Start"], ["Start", "Run"], ["Run", "Stop"], ["Stop", "Idle"]],
    },
    "detectors": {"per_regime": "iforest", "fusion_weights": [0.5, 0.3, 0.2]},
    "thresholds": {"method": "EVT", "alpha": 0.001, "fallback_quantile": 0.995},
    "eventization": {"min_len": "30s", "merge_gap": "120s"},
    "drift": {"psi_regime_warn": 0.25, "psi_regime_alert": 0.5, "adwin_delta": 0.002},
    "report": {"key_tags": []},
}


@dataclass
class PipelineConfig:
    """Simple wrapper over the configuration dictionary."""

    raw: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[Path] = None

    def get(self, dotted: str, default: Optional[Any] = None) -> Any:
        node = self.raw
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def __getitem__(self, item: str) -> Any:
        return self.raw[item]


def load_config(path: Optional[Path] = None) -> PipelineConfig:
    """Load configuration from YAML/JSON or fall back to defaults."""
    cfg_path = path or default_config_path()
    data: Dict[str, Any]
    if cfg_path.exists():
        text = cfg_path.read_text(encoding="utf-8")
        if yaml is not None:
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
    else:
        data = {}
    merged = DEFAULT_CONFIG.copy()
    merged.update(data)
    return PipelineConfig(raw=merged, source_path=cfg_path if cfg_path.exists() else None)
