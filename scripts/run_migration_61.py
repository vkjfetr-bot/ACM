#!/usr/bin/env python
"""Execute migration 61_adaptive_config_v10.sql"""
import pyodbc
from pathlib import Path
import sys

def run_migration():
    # Connection string
    conn_str = 'Driver={ODBC Driver 18 for SQL Server};Server=localhost;Database=ACM;Trusted_Connection=yes;TrustServerCertificate=yes'
    
    try:
        conn = pyodbc.connect(conn_str, autocommit=False)
        print('✓ Connected to ACM database')
        
        # Read migration script
        migration_path = Path('scripts/sql/migrations/61_adaptive_config_v10.sql')
        migration_sql = migration_path.read_text(encoding='utf-8')
        
        cursor = conn.cursor()
        
        # Split by GO and execute each batch
        batches = [b.strip() for b in migration_sql.split('GO') if b.strip() and not b.strip().startswith('--')]
        
        print(f'Executing {len(batches)} SQL batches...\n')
        
        for i, batch in enumerate(batches, 1):
            try:
                cursor.execute(batch)
                conn.commit()
                print(f'✓ [{i}/{len(batches)}] Batch executed')
            except Exception as e:
                conn.rollback()
                print(f'✗ [{i}/{len(batches)}] Batch failed: {str(e)[:150]}')
                raise
        
        cursor.close()
        conn.close()
        print('\n✓ Migration 61_adaptive_config completed successfully')
        return True
        
    except Exception as e:
        print(f'\n✗ Migration failed: {e}')
        return False

if __name__ == '__main__':
    success = run_migration()
    sys.exit(0 if success else 1)
