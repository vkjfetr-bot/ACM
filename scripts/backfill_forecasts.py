"""Backfill missing health and failure forecasts for historical runs.

Usage (PowerShell):
  python scripts/backfill_forecasts.py --equip FD_FAN --dry-run
  python scripts/backfill_forecasts.py --equip FD_FAN --max 50

Logic:
  1. Connect using existing sql_connection.ini via core.sql_client.SqlClient
  2. Find RunIDs for the given equipment that have score entries in ACM_Scores_Wide
     but no corresponding rows in ACM_HealthForecast_TS (anti-join).
  3. For each missing RunID (ordered by earliest score timestamp):
       - Determine anchor timestamp (max Timestamp in ACM_Scores_Wide for that RunID)
       - Invoke run_and_persist_enhanced_forecasting to generate health/failure forecasts.
       - Count rows written per table; log summary.
  4. Stops when --max processed or none remaining.

Constraints:
  - Only runs after coldstart threshold (>= 200 score rows) are processed.
  - Skips runs that already have any health forecast rows.
  - Keeps file-mode intact by not modifying existing OutputManager contracts.

Exit codes:
  0 on success; non-zero on fatal connection or execution error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from core.sql_client import SQLClient
    from core.sql_protocol import SqlClient  # Protocol for typing
    from core.forecasting import run_and_persist_enhanced_forecasting
    from core.output_manager import OutputManager
    from utils.config_dict import ConfigDict
    from utils.logger import Console
except Exception as e:  # pragma: no cover
    print(f"[BACKFILL] Failed to import ACM modules after path injection: {e}", file=sys.stderr)
    sys.exit(2)


def _load_config() -> ConfigDict:
    cfg_path = Path("configs/config_table.csv")
    if not cfg_path.exists():
        Console.warn("[BACKFILL] config_table.csv missing; proceeding with empty config")
        return ConfigDict({}, root="*")
    return ConfigDict.from_csv(cfg_path)


def _get_sql_client() -> SqlClient:
    # SQLClient implements SqlClient Protocol
    return SQLClient.from_ini("acm").connect()


def _query_df(sql: SqlClient, tsql: str, params: Optional[Tuple]=None) -> pd.DataFrame:
    cur = sql.cursor()
    try:
        if params:
            cur.execute(tsql, params)
        else:
            cur.execute(tsql)
        rows_raw = cur.fetchall()
        cols = [c[0] for c in cur.description]
        rows = [tuple(r) for r in rows_raw]
        return pd.DataFrame(rows, columns=cols)
    finally:
        cur.close()

def _scalar(sql: SqlClient, tsql: str, params: Optional[Tuple]=None) -> Optional[int]:
    cur = sql.cursor()
    try:
        if params:
            cur.execute(tsql, params)
        else:
            cur.execute(tsql)
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        cur.close()

def _fetch_missing_run_ids(sql: SqlClient, equip: str) -> pd.DataFrame:
    query = f"""
    WITH Runs AS (
        SELECT r.RunID, r.EquipID, r.CreatedAt
        FROM dbo.ACM_Runs r
        JOIN dbo.Equipment e ON r.EquipID = e.EquipID
        WHERE e.EquipCode = ?
    ), ScoreCounts AS (
        SELECT RunID, COUNT(*) AS ScoreRows, MIN(Timestamp) AS MinTs, MAX(Timestamp) AS MaxTs
        FROM dbo.ACM_Scores_Wide sw
        GROUP BY RunID
    ), ForecastPresence AS (
        SELECT DISTINCT RunID FROM dbo.ACM_HealthForecast_TS
    )
    SELECT r.RunID, r.EquipID, sc.ScoreRows, sc.MinTs, sc.MaxTs, r.CreatedAt
    FROM Runs r
    JOIN ScoreCounts sc ON r.RunID = sc.RunID
    LEFT JOIN ForecastPresence fp ON r.RunID = fp.RunID
    WHERE fp.RunID IS NULL AND sc.ScoreRows >= 200 -- coldstart guard
    ORDER BY sc.MinTs ASC;
    """
    return _query_df(sql, query, params=(equip,))


def _process_run(sql: SqlClient, cfg: ConfigDict, run_row: pd.Series, equip: str) -> Tuple[str, int]:
    run_id = str(run_row.RunID)
    equip_id = int(run_row.EquipID)
    anchor_ts = pd.to_datetime(run_row.MaxTs).to_pydatetime()

    # Minimal OutputManager stub (SQL-only mode)
    om = OutputManager(
        run_id=run_id,
        equip=equip,
        equip_id=equip_id,
        output_root=Path("artifacts") / equip / "backfill",
        sql_client=sql,
        sql_only_mode=True,
    )
    om.ensure_dirs()

    result = run_and_persist_enhanced_forecasting(
        sql_client=sql,
        equip_id=equip_id,
        run_id=run_id,
        config=cfg,
        output_manager=om,
        tables_dir=om.run_dir / "tables",
        equip=equip,
        current_batch_time=anchor_ts,
        sensor_data=None,
    )

    # Count rows written for health & failure forecast tables
    health_rows = _scalar(sql, "SELECT COUNT(*) FROM dbo.ACM_HealthForecast_TS WHERE RunID = ?", params=(run_id,)) or 0
    failure_rows = _scalar(sql, "SELECT COUNT(*) FROM dbo.ACM_FailureForecast_TS WHERE RunID = ?", params=(run_id,)) or 0
    Console.info(f"[BACKFILL] Run {run_id} -> health={health_rows}, failure={failure_rows}")
    return run_id, health_rows + failure_rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill missing enhanced health/failure forecasts")
    ap.add_argument("--equip", required=True, help="Equipment code (e.g., FD_FAN)")
    ap.add_argument("--max", type=int, default=25, help="Max runs to process")
    ap.add_argument("--dry-run", action="store_true", help="List missing runs only")
    args = ap.parse_args()

    try:
        sql = _get_sql_client()
    except Exception as e:
        print(f"[BACKFILL] SQL connection failed: {e}", file=sys.stderr)
        return 3

    cfg = _load_config()

    missing_df = _fetch_missing_run_ids(sql, args.equip)
    if missing_df.empty:
        Console.info("[BACKFILL] No missing forecast runs found.")
        return 0

    Console.info(f"[BACKFILL] Missing runs: {len(missing_df)} (will process up to {args.max})")
    if args.dry_run:
        Console.info(missing_df[['RunID','ScoreRows','MinTs','MaxTs']].to_string())
        return 0

    processed = 0
    for _, row in missing_df.iterrows():
        if processed >= args.max:
            break
        try:
            _, total_rows = _process_run(sql, cfg, row, args.equip)
            processed += 1
        except Exception as run_e:
            Console.warn(f"[BACKFILL] Failed run {row.RunID}: {run_e}")
            continue

    Console.info(f"[BACKFILL] Completed {processed} runs.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
