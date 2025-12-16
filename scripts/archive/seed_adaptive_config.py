#!/usr/bin/env python
"""Seed global adaptive config parameters"""
import pyodbc

conn_str = 'Driver={ODBC Driver 18 for SQL Server};Server=localhost;Database=ACM;Trusted_Connection=yes;TrustServerCertificate=yes'
conn = pyodbc.connect(conn_str, autocommit=False)
cursor = conn.cursor()

# Define global config parameters with research-backed bounds
params = [
    ('alpha', 0.3, 0.05, 0.95, 'Hyndman & Athanasopoulos (2018) - Exponential smoothing level'),
    ('beta', 0.1, 0.01, 0.30, 'Hyndman & Athanasopoulos (2018) - Exponential smoothing trend'),
    ('training_window_hours', 168.0, 72.0, 720.0, 'NIST SP 1225 - 3-30 day training window'),
    ('failure_threshold', 70.0, 40.0, 80.0, 'ISO 13381-1:2015 - Health index threshold'),
    ('confidence_min', 0.80, 0.50, 0.95, 'Agresti & Coull (1998) - Statistical confidence'),
    ('max_forecast_hours', 168.0, 168.0, 720.0, 'Industry standard - 7-30 day horizon'),
    ('monte_carlo_simulations', 1000.0, 500.0, 5000.0, 'Saxena et al. (2008) - RUL simulation count'),
    ('blend_tau_hours', 12.0, 6.0, 48.0, 'Expert tuning - Warm-start alpha blending'),
    ('auto_tune_data_threshold', 10000.0, 5000.0, 50000.0, 'Expert tuning - Auto-tuning trigger'),
]

print('Seeding global adaptive config parameters...\n')

try:
    # Clear existing global configs
    cursor.execute('DELETE FROM dbo.ACM_AdaptiveConfig WHERE EquipID IS NULL')
    print('✓ Cleared existing global configs')
    
    # Insert parameters
    for key, val, min_bound, max_bound, ref in params:
        cursor.execute('''
            INSERT INTO dbo.ACM_AdaptiveConfig 
            (EquipID, ConfigKey, ConfigValue, MinBound, MaxBound, IsLearned, Source, ResearchReference)
            VALUES (NULL, ?, ?, ?, ?, 0, 'global_default', ?)
        ''', (key, val, min_bound, max_bound, ref))
        print(f'✓ {key}: {val} [{min_bound}, {max_bound}]')
    
    conn.commit()
    print(f'\n✓ Seeded {len(params)} global parameters')
    
    # Verify
    cursor.execute('SELECT COUNT(*) as cnt FROM ACM_AdaptiveConfig WHERE EquipID IS NULL')
    count = cursor.fetchone()[0]
    print(f'✓ Verified: {count} parameters in database')
    
except Exception as e:
    conn.rollback()
    print(f'✗ Failed: {e}')
finally:
    cursor.close()
    conn.close()
