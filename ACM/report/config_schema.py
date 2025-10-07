from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class PlotCfg:
    dpi: int = 150
    width: int = 900
    height: int = 300


@dataclass
class Config:
    sampling_rate: str = "1min"
    time_format: str = "%Y-%m-%d %H:%M:%S"
    top_n_tags: int = 6
    colors: Dict[str, str] = field(default_factory=lambda: {
        "score": "#60a5fa",
        "threshold": "#34d399",
        "anom": "#ef4444",
    })
    anomaly_thresholds: Dict[str, float] = field(default_factory=lambda: {"fused_p": 0.95})
    dq_rules: Dict[str, Any] = field(default_factory=dict)
    plots: PlotCfg = field(default_factory=PlotCfg)


def load_config(path: str | None) -> Config:
    if not path:
        return Config()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = Config()
    for k, v in (data or {}).items():
        if k == "plots" and isinstance(v, dict):
            cfg.plots = PlotCfg(**{**cfg.plots.__dict__, **v})
        elif hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg

