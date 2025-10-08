"""acmnxt report --art-dir --equip

Default: Build native ACMnxt report with real charts.
Option --legacy: Call existing ACM/next/build_report.py tool.
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from acmnxt.report.native import build_native_report


def main() -> int:
    ap = argparse.ArgumentParser(description="ACMnxt reporting")
    ap.add_argument("--art-dir", required=True, help="Artifacts root directory")
    ap.add_argument("--equip", required=True, help="Equipment name")
    ap.add_argument("--legacy", action="store_true", help="Use legacy ACM builder instead of native report")
    args = ap.parse_args()
    print(f"[report] art_dir={args.art_dir} equip={args.equip}")
    if not args.legacy:
        out = build_native_report(args.art_dir, args.equip)
        print(f"[report] Wrote {out}")
        return 0
    equip = args.equip
    equip_s = "".join(ch if ch.isalnum() else "_" for ch in equip)
    root = Path(args.art_dir)
    equip_dir = root / equip_s
    scores = str(equip_dir / "scores.csv")
    events = str(equip_dir / "events.jsonl")
    builder = Path("ACM") / "next" / "build_report.py"
    cmd = [
        os.sys.executable,
        str(builder),
        "build-report",
        "--equip",
        equip,
        "--art-dir",
        str(root),
        "--scores",
        scores,
        "--events-json",
        events,
    ]
    print(f"[report] invoking legacy: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        return proc.returncode
    print(proc.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
