ACMnxt — Asset Condition Monitoring (Next‑Gen)

Overview
- Modular, fast, explainable pipeline for 15‑minute scoring loops.
- Outputs into `acmnxt/artifacts/<equip>/`: `scores.csv`, `events.jsonl`, `report_native.html`, and PNGs.

Run (PowerShell)
- Train on training CSV, score/report on test CSV:
  ./acmnxt/scripts/run_acmnxt.ps1 -Csv "<TRAIN.csv>" -ScoreCsv "<TEST.csv>" -Equip "<EQUIP>" -OutDir "acmnxt/artifacts" -Mode full -Fast
- Score/report only (artifacts already exist):
  ./acmnxt/scripts/run_acmnxt.ps1 -Csv "<TEST.csv>" -Equip "<EQUIP>" -OutDir "acmnxt/artifacts" -Mode score

Report contents
- Health Overview: fused timeline with severity bands; Recent Health (48h, smoothed) + anomaly markers; regime ribbon under timelines.
- Severity Occupancy: % time Low/Medium/High.
- Operating Regimes: distribution, transitions; per‑regime fused boxplot and severity bars.
- Tag Trends: overlay + matrix of individual tag mini‑trends.
- Events: table (Start/End, Duration, Peak, Severity, Regime@Start, Switches, Top Tags) + larger panels with regime spans.

