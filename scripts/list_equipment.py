#!/usr/bin/env python
"""List equipment with data tables in ACM database."""

from core.sql_client import SQLClient

def main():
    c = SQLClient.from_ini("acm").connect()
    
    # Find equipment data tables
    cur = c.cursor()
    cur.execute("""
        SELECT TABLE_NAME 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_NAME LIKE '%_Data' 
        ORDER BY TABLE_NAME
    """)
    data_tables = [r[0] for r in cur.fetchall()]
    cur.close()
    
    print("=" * 60)
    print("EQUIPMENT DATA TABLES IN ACM DATABASE")
    print("=" * 60)
    
    for i, table in enumerate(data_tables, 1):
        # Get row count
        try:
            cur2 = c.cursor()
            cur2.execute(f"SELECT COUNT(*) FROM [{table}]")
            count = cur2.fetchone()[0]
            cur2.close()
        except:
            count = "ERROR"
        
        # Get date range
        try:
            cur3 = c.cursor()
            cur3.execute(f"SELECT MIN(EntryDateTime), MAX(EntryDateTime) FROM [{table}]")
            row = cur3.fetchone()
            date_range = f"{row[0]} to {row[1]}" if row[0] else "No data"
            cur3.close()
        except:
            date_range = "N/A"
        
        print(f"\n{i}. {table}")
        print(f"   Rows: {count:,}" if isinstance(count, int) else f"   Rows: {count}")
        print(f"   Date Range: {date_range}")
    
    print(f"\n{'=' * 60}")
    print(f"TOTAL: {len(data_tables)} equipment data tables")
    print("=" * 60)
    
    # Also show Equipment table
    print("\nEQUIPMENT TABLE ENTRIES:")
    print("-" * 40)
    try:
        cur4 = c.cursor()
        cur4.execute("SELECT EquipID, EquipCode, EquipName FROM Equipment ORDER BY EquipID")
        for row in cur4.fetchall():
            print(f"  ID={row[0]}: {row[1]} ({row[2]})")
        cur4.close()
    except Exception as e:
        print(f"  Error: {e}")
    
    c.close()

if __name__ == "__main__":
    main()
