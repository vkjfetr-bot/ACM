# ACM Next — Reporting Guide

The next-gen reporting pipeline is a standalone path that does not modify your existing solution. It generates a self-contained HTML file and images directly from artifacts (scores, events, optional contributions and attention).

Quick start

1) Prepare artifacts:
   - `scores.csv|parquet` with numeric columns (e.g., tag values, `FusedScore`, optional H1/H2/H3 components).
   - Optional `events.jsonl` with fields like `{event_id, t0, t1}` (index-based).
   - Optional `contrib.jsonl` lines: `{window_id, contrib:[{tag, score}, ...]}`.
   - Optional `attn.npz` with arrays `temporal` and/or `spatial`.

2) Run the CLI:

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
- `<art-dir>/<equip>/report.html`
- `<art-dir>/<equip>/snapshot.csv`
- `<art-dir>/<equip>/snippet_table.html`
- `<art-dir>/<equip>/images/*.png`:
  - `*_trend_*.png`, `anomaly_matrix.png`, `threshold_trace.png`
  - `event_*.png`, `waterfall_*.png`, `tag_health.png`
  - `contrib_*.png`, `temporal_attention.png`, `spatial_attention.png`, `latent_space.png`

Tips
- Use `--no-matrix`, `--no-attention`, or `--no-latent` to skip heavy plots.
- The renderer tolerates NaNs/Infs and will produce placeholders if inputs are missing.
- Keep file sizes in check by reducing tags via `--top-tags`.

