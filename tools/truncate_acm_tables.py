import pyodbc
import sys

def main():
    conn = pyodbc.connect('DRIVER={ODBC Driver 18 for SQL Server};SERVER=localhost\\B19CL3PCQLSERVER;DATABASE=ACM;Trusted_Connection=yes;TrustServerCertificate=yes;')
    cur = conn.cursor()
    cur.fast_executemany = True
    cur.execute("SELECT name FROM sys.tables WHERE name LIKE 'ACM_%' ORDER BY name")
    tables = [row[0] for row in cur.fetchall()]
    print(f"Found {len(tables)} ACM tables")
    failures = []
    for t in tables:
        try:
            print(f"Deleting from {t}...")
            cur.execute(f"DELETE FROM dbo.[{t}]")
            conn.commit()
        except Exception as e:
            print(f"Failed on {t}: {e}")
            failures.append((t, str(e)))
    cur.close()
    conn.close()
    if failures:
        print("Failures:")
        for t, e in failures:
            print(f" - {t}: {e}")
        sys.exit(1)
    print("Done. Deleted across tables.")

if __name__ == '__main__':
    main()
