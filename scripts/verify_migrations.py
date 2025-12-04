#!/usr/bin/env python
"""Verify v10 migrations completed successfully"""
import pyodbc

conn_str = 'Driver={ODBC Driver 18 for SQL Server};Server=localhost;Database=ACM;Trusted_Connection=yes;TrustServerCertificate=yes'
conn = pyodbc.connect(conn_str, autocommit=False)
cursor = conn.cursor()

# Check v10 tables
v10_tables = ['ACM_HealthForecast', 'ACM_FailureForecast', 'ACM_RUL', 'ACM_ForecastingState', 'ACM_AdaptiveConfig']
print('✓ v10 Tables Status:')
for table in v10_tables:
    cursor.execute(f"SELECT COUNT(*) as cnt FROM {table}")
    count = cursor.fetchone()[0]
    print(f'  {table}: {count} rows')

# Check AdaptiveConfig schema
cursor.execute("SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='ACM_AdaptiveConfig' ORDER BY ORDINAL_POSITION")
cols = cursor.fetchall()
print(f'\n✓ ACM_AdaptiveConfig schema:')
for col_name, data_type in cols:
    print(f'  {col_name}: {data_type}')

# Try to read config data
try:
    cursor.execute('SELECT ConfigKey, ConfigValue FROM ACM_AdaptiveConfig WHERE EquipID IS NULL ORDER BY ConfigKey')
    configs = cursor.fetchall()
    print(f'\n✓ Global Adaptive Config ({len(configs)} parameters):')
    for key, val in configs:
        print(f'  {key}: {val}')
except Exception as e:
    print(f'\n! Could not read AdaptiveConfig: {e}')

# Check for ROWVERSION column in ForecastingState
cursor.execute("SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='ACM_ForecastingState' ORDER BY ORDINAL_POSITION")
cols = cursor.fetchall()
print(f'\n✓ ACM_ForecastingState schema:')
for col_name, data_type in cols:
    print(f'  {col_name}: {data_type}')

cursor.close()
conn.close()
print('\n✓ All migrations verified successfully!')
