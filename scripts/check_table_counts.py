"""Check row counts in key ACM SQL tables."""
from core.sql_client import SQLClient
from utils.config_dict import ConfigDict

cfg = ConfigDict({
    'sql_connection': {
        'server': 'localhost\\B19CL3PCQLSERVER',
        'database': 'ACM',
        'trusted_connection': True
    }
})

sql = SQLClient(cfg)
sql.connect()
cur = sql.cursor()

tables = [
    'ACM_Scores_Wide',
    'ACM_Scores_Long', 
    'ACM_HealthTimeline',
    'ACM_Episodes',
    'ACM_Runs',
    'ModelRegistry',
    'ACM_RegimeTimeline',
    'ACM_ContributionCurrent',
    'ACM_DriftSeries'
]

print('\n' + '='*60)
print('SQL TABLE ROW COUNTS')
print('='*60 + '\n')

for table in tables:
    try:
        cur.execute(f"SELECT COUNT(*) FROM dbo.[{table}]")
        count = cur.fetchone()[0]
        print(f'{table:30s}: {count:>10,} rows')
    except Exception as e:
        print(f'{table:30s}: ERROR - {str(e)[:50]}')

sql.close()
