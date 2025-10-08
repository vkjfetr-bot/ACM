"""Schemas — Pydantic models for configs and outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    csv: Path
    equip: str
    out_dir: Path
    resample: str = Field(default="1min")
    use_h1: bool = True
    use_h2: bool = True
    use_h3: bool = True


class ScoreRow(BaseModel):
    ts: str
    score: float
    h1: Optional[float] = None
    h2: Optional[float] = None
    h3: Optional[float] = None
    regime: Optional[int] = None


class Event(BaseModel):
    id: int
    start: str
    end: str
    duration: float
    peak: float
    top_tags: List[str] = []

