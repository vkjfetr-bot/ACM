# ACM Reporting Guide

There are two reporting paths available. The new, integrated path is recommended for day‑to‑day use; the older experimental path remains available for advanced visuals.

Sections:
- New integrated report (ACM/report)
- Experimental next report (ACM/next)
- Outputs and tips

## New Integrated Report (recommended)

Command-line

```bash
python -m ACM.report.cli build \
  --train "ACM/Dummy Data/FD FAN TRAINING DATA.csv" \
  --test  "ACM/Dummy Data/FD FAN TEST DATA.csv" \
  --equip "FD FAN" \
  --out   reports \
  --config report_config.yaml
```

PowerShell wrapper

```powershell
.\run_report.ps1 -TrainCsv "ACM\Dummy Data\FD FAN TRAINING DATA.csv" `
                 -TestCsv  "ACM\Dummy Data\FD FAN TEST DATA.csv" `
                 -Equip    "FD FAN" `
                 -Out      ".\reports"
```

Outputs
- HTML: `reports/ACM_Report_<equip>_<yyyymmdd_HHMM>.html`
- PNGs: `reports/assets/*.png`
- Embedded JSON: `<script type="application/json" id="report-json">…</script>` inside the HTML

Notes
- Uses only two CSV inputs (train/test); synthesizes a simple fused score if missing.
- Focused on clean visuals and layman summary; does not alter your existing ACM pipeline.

## Experimental Next Report (optional, advanced)

The experimental path remains available for artifact-driven reports and additional visuals.

```bash
python ACM/next/build_report.py build-report \
  --equip "FD FAN" \
  --art-dir "C:\\Users\\<you>\\Documents\\CPCL\\ACM\\acm_artifacts" \
  --top-tags 9 \
  --scores path\\to\\scores.parquet \
  --events-json path\\to\\events.jsonl \
  --contrib path\\to\\contrib.jsonl \
  --attn path\\to\\attn.npz
```

Outputs
- `<art-dir>/<equip>/report.html`, `snapshot.csv`, `images/*.png`

Tips
- Use `--no-matrix`, `--no-attention`, or `--no-latent` to skip heavy plots.
- Tolerates NaNs/Infs and produces placeholders when inputs are missing.
