"""Split a training CSV into multiple score-ready segments without modifying the source."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


TIME_COLUMNS = [
    "ts",
    "timestamp",
    "time",
    "datetime",
    "Ts",
    "TS",
    "Timestamp",
    "Time",
    "Datetime",
]


def ensure_time_index(df: pd.DataFrame) -> pd.DataFrame:
    for col in TIME_COLUMNS:
        if col in df.columns:
            ts = pd.to_datetime(df[col], errors="coerce", utc=False)
            if ts.notna().any():
                df = df.loc[ts.notna()].copy()
                df.index = ts.loc[ts.notna()]
                df.index.name = str(col)
                df = df.drop(columns=[col])
                return df.sort_index()
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    # fall back to numeric index
    df = df.copy()
    df.index = pd.RangeIndex(len(df))
    df.index.name = "Index"
    return df


def split_frame(df: pd.DataFrame, splits: int) -> List[pd.DataFrame]:
    if splits <= 1:
        return [df]
    n = len(df)
    size = max(1, n // splits)
    frames: List[pd.DataFrame] = []
    for i in range(splits):
        start = i * size
        end = (i + 1) * size if i < splits - 1 else n
        chunk = df.iloc[start:end]
        if not chunk.empty:
            frames.append(chunk)
    return frames


def main() -> None:
    parser = argparse.ArgumentParser("split_train")
    parser.add_argument("--csv", required=True, help="Path to the training CSV (read-only)")
    parser.add_argument("--splits", type=int, required=True, help="Number of splits to create")
    parser.add_argument("--out", required=True, help="Output directory for generated split CSVs")
    args = parser.parse_args()

    src = Path(args.csv)
    if not src.exists():
        raise FileNotFoundError(f"Source CSV not found: {src}")

    df = pd.read_csv(src)
    df = ensure_time_index(df)
    frames = split_frame(df, args.splits)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, chunk in enumerate(frames, start=1):
        fname = out_dir / f"{src.stem}_split_{i}.csv"
        chunk.to_csv(fname, index_label="Ts")
        print(f"[SPLIT] Wrote {len(chunk)} rows -> {fname}")


if __name__ == "__main__":
    main()