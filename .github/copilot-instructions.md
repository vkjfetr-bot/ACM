# ACM Copilot Guardrails (current)

- **Mission**: keep the ACM pipeline healthy (CSV/SQL ingest, detector fusion, analytics outputs under `artifacts/{EQUIP}/run_*`). Primary entrypoint: `python -m core.acm_main --equip FD_FAN` (`run_pipeline()` in `core/acm_main.py`).
- **Modes**: file-mode reads `data/*.csv`; SQL-mode uses `configs/sql_connection.ini` via `core/sql_client.SQLClient`. File-mode must stay working before SQL-path changes ship.
- **Batch mode**: use `scripts/run_batch_mode.ps1` (params: `-Equipment`, `-NumBatches`, `-StartBatch`) to run sequential batches; the script sets `ACM_BATCH_MODE`/`ACM_BATCH_NUM` env vars—don’t override in code.
- **No emojis ever** in code, comments, tests, or generated content.

## Core contracts
- **Config**: `utils/config_dict.ConfigDict` loads cascading `configs/config_table.csv` (global `*` rows overridden by equipment rows). Keep dot-path access intact (e.g., `cfg['fusion']['weights']['omr_z']`).
- **Output manager**: all CSV/PNG/SQL writes go through `core/output_manager.OutputManager`; respect `ALLOWED_TABLES` and batched transactions.
- **Time policy**: timestamps are local-naive; do not reintroduce UTC conversions. Use `_to_naive*` helpers in `core/output_manager.py` and `core/acm_main.py`.
- **Detectors/analytics**: detectors live in `core/` (`omr.py`, `correlation.py`, `outliers.py`, plus PCA/IForest/GMM/AR1). Fusion in `core/fuse.py`; regimes in `core/regimes.py`; drift in `core/drift.py`; episodes via `core/episode_culprits_writer`.
- **Performance**: `core/fast_features.py` supports pandas + optional Polars; keep API backward compatible and tested (`tests/test_fast_features.py`).
- **Rust bridge**: optional accelerator in `rust_bridge/`; Python path remains primary.

## Workflows
- Rapid run: `python -m core.acm_main --equip GAS_TURBINE`.
- SQL smoke: `python scripts/sql/verify_acm_connection.py` (needs SQL Server creds).
- File-mode helper: `scripts/run_file_mode.ps1` wraps a baseline local run.

## Testing
- Targeted suites: `pytest tests/test_fast_features.py`, `pytest tests/test_dual_write.py`, `pytest tests/test_progress_tracking.py`. Respect existing skips/markers (Polars, SQL Server).

## Data & artifacts
- Sample CSVs under `data/` have datetime index + sensor columns; keep names aligned with `configs/config_table.csv`.
- `artifacts/` and `backups/` stay gitignored; do not add tracked artifact folders.

## Documentation policy
- README is authoritative. Do **not** auto-create new docs/changelogs for routine changes; only update docs when explicitly requested.

## Style & safety
- Python 3.11, ~100-char lines, vectorized pandas/NumPy. Use existing lint/type tooling (`ruff`, `mypy`) when touching shared modules.
- Never commit credentials; `configs/sql_connection.ini` stays local. Prefer env-var fallbacks when describing new settings.

## Source control hygiene
- Use branches for all non-trivial work: `feature/<topic>` or `fix/<topic>`; avoid pushing directly to `main`.
- Keep branches focused and short-lived; prefer small, frequent merges over large drops.
- Rebase onto `main` before opening/merging PRs to keep history clean and reduce conflicts.
- Write clear, imperative commits (e.g., “Add batch-mode env guards”); squash noisy fixups before merge.
- Run relevant tests before merging; do not merge with failing checks.
- Never commit artifacts/logs/secrets; respect `.gitignore`.
- Prefer review for all changes; merge only when checks are green and approvals are in. If self-merging is allowed, still require passing checks.

## Discoverability quick-links
- **System overview**: `docs/ACM_SYSTEM_OVERVIEW.md`; analytics flow: `docs/Analytics Backbone.md`.
- **Coldstart**: `docs/COLDSTART_MODE.md`; **OMR**: `docs/OMR_DETECTOR.md`; **Batch SQL audit**: `docs/BATCH_MODE_SQL_AUDIT.md`.
- **Schema reference**: `docs/sql/SQL_SCHEMA_REFERENCE.md`; **Grafana**: `grafana_dashboards/README.md` + dashboards under `grafana_dashboards/*.json`.
- **Run helpers**: `scripts/run_batch_mode.ps1`, `scripts/run_file_mode.ps1`, `scripts/sql/verify_acm_connection.py`.
- **Core modules**: entry `core/acm_main.py`; writes `core/output_manager.py`; detectors/fusion/regimes/drift under `core/` (omr.py, correlation.py, outliers.py, fuse.py, regimes.py, drift.py); episodes `core/episode_culprits_writer.py`; run metadata `core/run_metadata_writer.py`; config loader `utils/config_dict.py`.
- **Search tips**: `rg "ALLOWED_TABLES" core`, `rg "run_pipeline" core`, `rg "ACM_" scripts docs core`, `rg --files -g "*.sql" scripts/sql`.

## Config sync discipline
- When `configs/config_table.csv` changes, run `python scripts/sql/populate_acm_config.py` to sync `ACM_Config` in SQL.
- Keep `ConfigDict` dotted-path semantics intact so the populate script remains compatible.
