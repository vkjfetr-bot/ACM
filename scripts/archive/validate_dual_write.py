from __future__ import annotations

import argparse
from typing import Optional

from core.output_manager import ALLOWED_TABLES
from core.sql_client import SQLClient
from core.observability import Console


def _table_exists(client: SQLClient, table_name: str) -> bool:
    cur = client.cursor()
    try:
        cur.execute("SELECT OBJECT_ID(?, 'U')", (f"dbo.{table_name}",))
        row = cur.fetchone()
        return bool(row and row[0])
    finally:
        cur.close()


def _count_rows(client: SQLClient, table_name: str, run_id: str, equip_id: int) -> Optional[int]:
    cur = client.cursor()
    try:
        cur.execute(
            f"SELECT COUNT(*) FROM dbo.[{table_name}] WHERE RunID = ? AND EquipID = ?",
            (run_id, equip_id)
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0
    except Exception as exc:
        Console.warn(f"Failed to count rows for {table_name}: {exc}", component="DUAL-WRITE")
        return None
    finally:
        cur.close()


def _latest_run_id(client: SQLClient, equip_id: int) -> Optional[str]:
    cur = client.cursor()
    try:
        cur.execute(
            "SELECT TOP 1 RunID FROM dbo.ACM_Run_Stats WHERE EquipID = ? ORDER BY StartTime DESC",
            (equip_id,)
        )
        row = cur.fetchone()
        return str(row[0]) if row and row[0] else None
    finally:
        cur.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dual-write SQL coverage for a run.")
    parser.add_argument("--equip", type=int, required=True, help="EquipID to validate")
    parser.add_argument("--run-id", type=str, help="Optional RunID to validate (defaults to latest RunStats entry)")
    args = parser.parse_args()

    client = SQLClient.from_ini("acm")
    client.connect()

    run_id = args.run_id or _latest_run_id(client, args.equip)
    if not run_id:
        Console.error(f"No run found for EquipID={args.equip}", component="DUAL-WRITE")
        return

    Console.info(f"Validating dual-write for RunID={run_id} EquipID={args.equip}", component="DUAL-WRITE")
    total_tables = 0
    rows_written = 0
    missing_tables = []

    for table_name in sorted(ALLOWED_TABLES):
        total_tables += 1
        if not _table_exists(client, table_name):
            Console.warn(f"Missing table {table_name}", component="DUAL-WRITE")
            missing_tables.append(table_name)
            continue

        count = _count_rows(client, table_name, run_id, args.equip)
        if count is None:
            continue
        rows_written += count
        status = "OK" if count > 0 else "EMPTY"
        Console.info(f"{table_name}: {count} rows ({status})", component="DUAL-WRITE")

    Console.info(f"Tables checked: {total_tables}, Total rows: {rows_written}", component="DUAL-WRITE")
    if missing_tables:
        Console.warn(f"Missing tables: {', '.join(missing_tables)}", component="DUAL-WRITE")


if __name__ == "__main__":
    main()
