ACMnxt — Detailed Implementation Tasklist (v0.1.0 baseline)

Milestones
- M1: Scaffold + Data IO + DQ + heatmap (DONE)
- M2: H1/H2, fused, events; score-only path; train on TRAIN, report on TEST (DONE)
- M3: Native report (timeline, recent health, severity, regimes, events, tag matrix) (DONE)
- M4: Clean-up + docs (THIS RELEASE)

Visualization
- [x] Native timeline with severity bands; recent health zoom with smoothing + anomaly markers
- [x] Event panels (larger, fused overlay, regime spans) + Event table enrichments
- [x] Tag Trends matrix; regime ribbon under timelines; per‑regime summaries

Report
- [x] Native HTML + PNGs under artifacts/<equip>/

CLI & Scripts
- [x] train / score / report; PowerShell run_acmnxt.ps1 with -ScoreCsv and -Fast

Docs
- [x] Concise README; heavy docs later

