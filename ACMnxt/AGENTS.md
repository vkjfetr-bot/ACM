ACMnxt Agent Guidelines (Phase 1)

- Python 3.11+. Type hints encouraged, not blocking. Do not enforce mypy in Phase 1.
- No GitHub Actions or CI. Local `make` targets only (optional).
- Inputs are a single training CSV (FD FAN) and we scale later.
- Reuse existing ACM report and chart code where it saves time.
- Prefer vectorized numpy/pandas; avoid DataFrame.apply in hot paths.
- All timestamps in UTC internally; log local IST alongside as needed.
- Aim for clear, concise docstrings; keep modules readable. Heavy docs later.
- Keep functions small and composable; avoid hidden state.
- Write artifacts under a provided `--out-dir` (default near current layout).

Scope: Applies to all files under `acmnxt/`.
