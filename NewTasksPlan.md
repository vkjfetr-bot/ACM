## New Tasks Implementation Plan

This document turns the high-level issues captured in `NewTasks.md` into a concrete implementation roadmap. It also incorporates observations from the existing SQL scripts so we can align future work with the batch runner, helpers, and validation scripts that rely on the same health / forecast stack.

### TIME-01 — Harmonize timestamp handling inside the enhanced RUL/forecast stack
1. Inventory every `pd.to_datetime` (plus any `tz_localize`/`tz_convert`) call in `enhanced_forecasting.py`, `rul_estimator.py`, and `enhanced_rul_estimator.py`. Document the locations in a checklist.
2. Implement a shared utility (e.g., `core.utils.timestamps.ensure_local_naive_index(df, cols)`) so we can call it right after loading CSV/SQL data from these modules. This helper should:
   - Drop `utc=True`
   - Always return a timezone-naive `DatetimeIndex`
   - Optionally log when invalid timestamps are coerced
3. Update the entry points so that any health timeline, score frame, or detector history is normalized immediately after loading and before being passed to any model.
4. Run existing scripts (especially those in `core/enhanced_forecasting_deprecated.py`) against sanitized data to confirm there are no regressions or unexpected timezone-sensitive comparisons.
5. Propagate documentation (code comments or docstrings) that explain the new policy so future contributors know the stack prefers naive timestamps.

### TIME-02 — Align SQL timestamp policy with the rest of ACM
1. Review `enhanced_forecasting_sql.py`/`rul_estimator._load_*` to ensure DB reads no longer force UTC conversion.
2. Add a module-level note stating: “We expect timestamps to be naive local times, mirroring `OutputManager` / `ACM_Config` conventions.”
3. Introduce regression tests (even simple sanity checks) that assert the SQL loader returns naive datetimes; e.g., `assert df.index.tz is None`.
4. Update any downstream code that currently assumes UTC normalized timestamps (for example, gap detection or window trimming) so it uses local naive times consistently.

### STACK-01 — Choose the canonical forecast/RUL stack in configuration
1. Define a new config flag such as `forecast.mode` or `rul.mode` in `configs/config_table.csv` (with documentation in `docs/SQL_MODE_CONFIGURATION.md`).
2. Centralize orchestration inside `acm_main.py`: replace ad-hoc calls with one “strategy selector” that reads the config flag and invokes either the simple stack or the enhanced stack.
3. Ensure the non-selected stack remains importable for tests/tooling but is never run during a default pipeline execution.
4. Update `docs/FORECASTING_VALIDATION_REPORT.md` and any README references to describe the chosen “truth” stack and how to switch modes.

### STACK-02 — Deduplicate shared RUL logic
1. Extract common helpers (`_norm_cdf`, baseline config defaults, window builders, etc.) into a new module such as `core/rul_common.py`.
2. Refactor both `rul_estimator.py` and `enhanced_rul_estimator.py` to import this module for shared math.
3. Ensure the helper module exposes testing hooks so we can easily cover the shared logic (e.g., verifying `compute_failure_probability` works the same way for both stacks).
4. Add comments describing why this logic is shared so future contributors don’t accidentally diverge.

### INTEG-01 — Wire enhanced forecasting into the main pipeline
1. Implement an orchestration helper (e.g., `acm_main.run_enhanced_forecasting(equip_id, run_ctx, sql_client, output_manager)`).
2. Call this helper after health/score data is persisted but before `OutputManager` finalizes the run so the enhanced tables share the same run context.
3. Guard execution with a config flag (e.g., `forecasting.enhanced_enabled`) and provide clear log messages via `Console`.
4. Ensure the helper returns dataframes that `OutputManager` already knows how to persist (reuse the same `ef_sql_map` from the current SQL logic).

### INTEG-02 — Provide a single entrypoint for RUL estimation
1. Create one public function (e.g., `core.rul.estimator.compute_rul(health_df, cfg)`) that internally dispatches to either the simple or enhanced estimator depending on the configuration from STACK-01.
2. Update all call-sites (`acm_main`, `enhanced_forecasting`, batch scripts, tests) to use this entrypoint rather than referencing specific modules.
3. Document the inputs/outputs so that tests and future refactors have a single contract to verify.

### LOG-01 — Standardize logging across forecasting and RUL modules
1. Replace `print` statements or ad-hoc logging inside `enhanced_forecasting*` and `rul_*` modules with the shared `Console` logger used by the rest of ACM.
2. Ensure each log entry includes context (equip ID/run ID) and that the forecasting/RUL stages emit metrics that can be folded into the `TIMER` section in `acm_main`.
3. Add a debug log when the shared orchestration discussed in STACK-01/INTEG-01 chooses which stack to run.
4. Consider piggy-backing on the new `_finalize_run` logging (from scripts) so any stage that fails reports enough context for triage.

## Script Review & Additional Implementation Notes
1. **`scripts/sql_batch_runner.py`**
   - This script drives repeated SQL runs via `python -m core.acm_main`; any change to the new orchestration or config flags must be exposed via the runner’s CLI (e.g., allow overriding `forecast.mode` or `forecasting.enhanced_enabled` per equipment).
   - It also parses historian metadata (`_log_historian_overview`), so TIME-01/TIME-02 changes should be communicated in the runner’s comments/documentation so operators know the timestamp policy changed.
   - Action: add a checklist entry to ensure `sql_batch_runner` passes config overrides into the ACM run (and maybe exposes the new mode flag via `--forecast-mode`).

2. **`scripts/validate_sql_only_mode.py` and `scripts/sql/test_sql_mode_loading.py`**
   - These scripts validate SQL-only behavior and should include tests for the new RUL/forecast orchestration. When the shared entrypoint is ready, update their assertions to verify the chosen stack runs (and that `Console` logging reflects the same signals).
   - Document any new config path (`forecast.mode`) in these scripts so they can exercise both simple and enhanced modes in validation runs.

3. **`scripts/sql/test_dual_write_config.py`**
   - Since the plan centralizes output writing, review this script to make sure it still exercises both CSV and SQL paths after we reroute everything through the shared entrypoints.
   - Task: annotate the script with pointers to the new `OutputManager` call sequence, ensuring the new orchestration writes to the caches the same way as before.

4. **`scripts/sql/patches/*.sql` (including the new `2025-11-19` migrations)**
   - These patches adjust schema elements that the plan relies on (RunLog RunID type, etc.). Keep them aligned with any new metadata emitted from the shared orchestration (e.g., ensure `RunLog` still receives the same `WindowStart/WindowEnd` values).
   - Action: add a note to the implementation plan to re-verify these migrations once the stack decision is finalized so downstream patch scripts emit compatible column names/types.

## Next Steps
- Iterate on each task in priority order (TIME tasks first, since they underpin downstream integration).
- After each module-level change, rerun the relevant `scripts/sql_*` validators so we catch integration regressions early.
- Keep this plan updated with status/comments in `NewTasksPlan.md` as/when tasks progress.
