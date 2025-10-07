import os
import json
import tempfile
import pandas as pd
import numpy as np


def _make_scores(tmp: str) -> str:
    n = 400
    t = pd.date_range("2023-01-01", periods=n, freq="T")
    df = pd.DataFrame({
        "Ts": t,
        "TagA": np.sin(np.linspace(0, 12, n)) + 0.1 * np.random.RandomState(0).randn(n),
        "TagB": np.cos(np.linspace(0, 8, n)) + 0.1 * np.random.RandomState(1).randn(n),
        "FusedScore": np.clip(np.random.RandomState(2).randn(n) * 0.3 + 1.0, 0, 3),
        "H1_Forecast": np.random.RandomState(3).rand(n),
        "H2_Recon": np.random.RandomState(4).rand(n),
        "H3_Contrast": np.random.RandomState(5).rand(n),
    })
    p = os.path.join(tmp, "scores.csv")
    df.to_csv(p, index=False)
    return p


def _make_events(tmp: str) -> str:
    rows = [
        {"event_id": 1, "t0": 120, "t1": 160},
        {"event_id": 2, "t0": 260, "t1": 300},
    ]
    p = os.path.join(tmp, "events.jsonl")
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def test_build_report_smoke():
    import subprocess, sys
    here = os.path.dirname(os.path.dirname(__file__))
    tmp = tempfile.mkdtemp()
    scores = _make_scores(tmp)
    events = _make_events(tmp)

    out_dir = os.path.join(tmp, "artifacts")
    os.makedirs(out_dir, exist_ok=True)

    cmd = [sys.executable, os.path.join(here, "ACM", "next", "build_report.py"),
           "build-report", "--equip", "FD FAN", "--art-dir", out_dir,
           "--scores", scores, "--events-json", events]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    equip_dir = os.path.join(out_dir, "FD_FAN")
    assert os.path.exists(os.path.join(equip_dir, "report.html"))
    imgs = os.listdir(os.path.join(equip_dir, "images"))
    assert any(n.endswith(".png") for n in imgs)

