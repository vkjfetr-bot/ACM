# ACM Installer Bundle

This folder contains a self-contained SQL installer generated from the current ACM database.

Contents (in `install/sql/`):
- `00_create_database.sql` – create `ACM` DB if missing.
- `10_tables.sql` – all 87 tables (matching `docs/sql/COMPREHENSIVE_SCHEMA_REFERENCE.md`).
- `15_unique_constraints.sql` – unique constraints.
- `20_foreign_keys.sql` – 43 foreign keys.
- `30_indexes.sql` – non-PK indexes.
- `40_views.sql` – 14 views (`CREATE OR ALTER`).
- `50_procedures.sql` – 23 stored procedures (`CREATE OR ALTER`).

Entry point:
```bash
python install/install_acm.py --ini-section acm
# Or override connection info:
python install/install_acm.py --server localhost\\B19CL3PCQLSERVER --database ACM --trusted-connection
```

Regenerating scripts from the live DB (optional):
```bash
python install/generate_install_scripts.py --ini-section acm
```

Notes:
- Uses `configs/sql_connection.ini` by default; CLI flags override.
- Scripts are idempotent (`IF NOT EXISTS` for tables/FKs/indexes, `CREATE OR ALTER` for views/SPs). Running again is safe.
