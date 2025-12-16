import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

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

cur.execute('SELECT TOP 5 RunID, EquipID, StartedAt, TrainRowCount, ScoreRowCount FROM ACM_Runs ORDER BY StartedAt DESC')
print('Recent runs:')
for row in cur.fetchall():
    print(f'  RunID={row[0][:30]}... | EquipID={row[1]} | {row[2]} | Train={row[3]} Score={row[4]}')

sql.close()
