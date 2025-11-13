# SQL Backend Audit: output_manager.py

Auditing this file for SQL backend readiness reveals several critical issues and opportunities for improvement:

---

## 🔴 **CRITICAL ISSUES**

### 1. **DELETE Before INSERT Pattern (Lines 645-665) - REMOVED BUT COMMENTED**
```python
# NOTE: DELETE before INSERT removed - unnecessary since RunID is unique per run
# Each pipeline execution generates a new RunID, so no duplicate data exists
```

**Issue**: The commented-out upsert logic suggests historical uncertainty about data freshness guarantees. While the comment claims "RunID is unique per run," this needs verification:

- **Risk**: If ACM reruns with the same RunID (e.g., manual retry, scheduler bug), duplicate rows will accumulate
- **Recommendation**: Either:
  1. Add `UNIQUE` constraint on `(RunID, EquipID, Timestamp)` in SQL schema
  2. Restore DELETE logic with transaction safety
  3. Add deduplication at query time via `ROW_NUMBER()` window functions

---

### 2. **SQL Health Check Caching (Lines 438-465)**
```python
self._sql_health_cache: Tuple[float, bool] = (0.0, False)
self._sql_health_cache_duration = sql_health_cache_seconds
```

**Issues**:
- Cache doesn't detect connection loss between checks
- No exponential backoff on repeated failures
- Silent failures accumulate in `stats['sql_failures']` without alerting

**Recommendation**:
```python
def _check_sql_health(self) -> bool:
    """Check SQL with circuit breaker pattern."""
    now = time.time()
    last_check, last_result = self._sql_health_cache
    
    # Exponential backoff on repeated failures
    if not last_result:
        backoff = min(300, self._sql_health_cache_duration * (2 ** self.stats['sql_failures']))
        if now - last_check < backoff:
            return False
    
    # Standard cache check
    if now - last_check < self._sql_health_cache_duration:
        return last_result
    
    # Perform health check with transaction test
    try:
        cur = self.sql_client.cursor()
        cur.execute("BEGIN TRANSACTION; SELECT 1; COMMIT;")
        cur.fetchone()
        self._sql_health_cache = (now, True)
        self.stats['sql_failures'] = 0  # Reset on success
        return True
    except Exception as e:
        self._sql_health_cache = (now, False)
        self.stats['sql_failures'] += 1
        Console.error(f"[OUTPUT] SQL health check failed (attempt {self.stats['sql_failures']}): {e}")
        return False
    finally:
        try:
            if 'cur' in locals():
                cur.close()
        except:
            pass
```

---

### 3. **Batched Transaction Commit Logic (Lines 353-404)**
```python
if hasattr(self.sql_client, "commit"):
    self.sql_client.commit()
elif hasattr(self.sql_client, "conn") and hasattr(self.sql_client.conn, "commit"):
    if not getattr(self.sql_client.conn, "autocommit", True):
        self.sql_client.conn.commit()
```

**Issues**:
- Assumes `autocommit=True` is default (SQL Server default is `autocommit=False` for pyodbc)
- No verification that commit actually succeeded
- Rollback on error may fail silently

**Recommendation**:
```python
def _commit_transaction(self):
    """Commit with verification."""
    try:
        # Try direct commit first
        if hasattr(self.sql_client, "commit"):
            self.sql_client.commit()
            return
        
        # Fallback to conn.commit
        if hasattr(self.sql_client, "conn"):
            conn = self.sql_client.conn
            # Force explicit commit regardless of autocommit mode
            if hasattr(conn, "commit"):
                conn.commit()
                return
        
        # Verify commit succeeded by checking @@TRANCOUNT
        cur = self.sql_client.cursor()
        cur.execute("SELECT @@TRANCOUNT")
        trancount = cur.fetchone()[0]
        if trancount > 0:
            raise RuntimeError(f"Transaction not committed (@@TRANCOUNT={trancount})")
    except Exception as e:
        Console.error(f"[OUTPUT] Commit failed: {e}")
        raise
```

---

## ⚠️ **MAJOR CONCERNS**

### 4. **Float Overflow Handling (Lines 670-690)**
```python
# Replace extreme float values BEFORE replacing NaN
for col in df_clean.columns:
    if df_clean[col].dtype in [np.float64, np.float32]:
        extreme_mask = valid_mask & (df_clean[col].abs() > 1e100)
        if extreme_mask.any():
            df_clean.loc[extreme_mask, col] = None
```

**Issues**:
- Magic number `1e100` not documented (SQL Server `FLOAT` max is ~1.79e308)
- Replaces extremes with `NULL` but doesn't track which rows were affected
- May hide data quality issues in source data

**Recommendation**:
1. Add config parameter for `extreme_float_threshold` (default 1e100)
2. Log affected rows with context:
   ```python
   if extreme_mask.any():
       extreme_rows = df_clean[extreme_mask][[col]].head(5)
       Console.warn(f"[SQL] Replaced {extreme_mask.sum()} extreme values in {table_name}.{col}: {extreme_rows.to_dict()}")
   ```
3. Consider writing extreme values to audit table instead of dropping

---

### 5. **Schema Mismatch Handling (Lines 630-645)**
```python
# only insert columns that actually exist in the table
columns = [c for c in df.columns if c in table_cols]
```

**Issues**:
- Silently drops columns not in SQL schema
- No warning if critical columns are missing (e.g., `RunID`, `EquipID`)
- Could lead to incomplete data without operator awareness

**Recommendation**:
```python
# Validate critical columns before filtering
CRITICAL_COLUMNS = {'RunID', 'EquipID'}
missing_critical = CRITICAL_COLUMNS - set(df.columns)
if missing_critical:
    raise ValueError(f"[SQL] Missing critical columns for {table_name}: {missing_critical}")

# Filter to schema columns with warning for dropped columns
columns = [c for c in df.columns if c in table_cols]
dropped = set(df.columns) - set(columns)
if dropped:
    Console.warn(f"[SQL] Dropping {len(dropped)} columns not in {table_name} schema: {dropped}")
```

---

### 6. **SQL Injection Risk (Lines 618-642)**
```python
insert_sql = f"INSERT INTO dbo.[{table_name}] ({cols_str}) VALUES ({placeholders})"
```

**Issues**:
- `table_name` comes from `ALLOWED_TABLES` set (Lines 21-35) - **GOOD**
- Column names from DataFrame are bracketed but not validated - **CONCERN**
- Could inject malicious column names if DataFrame is compromised upstream

**Recommendation**:
```python
# Validate column names match SQL identifier rules
import re
SQL_IDENTIFIER_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

for col in columns:
    if not SQL_IDENTIFIER_PATTERN.match(col):
        raise ValueError(f"[SQL] Invalid column name for {table_name}: {col}")

# Proceed with bracketed names
cols_str = ", ".join(f"[{c}]" for c in columns)
```

---

## 📊 **OPTIMIZATION OPPORTUNITIES**

### 7. **Bulk Insert Performance (Lines 692-710)**
```python
for i in range(0, len(records), self.batch_size):
    batch = records[i:i+self.batch_size]
    try:
        cur.executemany(insert_sql, batch)
```

**Current**: Using `executemany()` with default batch size of 5000

**Recommendations**:
1. **Use Table-Valued Parameters (TVP)** for SQL Server (10-100x faster):
   ```python
   # Create TVP type in SQL Server
   CREATE TYPE dbo.ACM_Scores_TVP AS TABLE (
       Timestamp DATETIME2,
       RunID VARCHAR(50),
       EquipID INT,
       DetectorType VARCHAR(50),
       ZScore FLOAT
   );
   
   # Python code
   from pyodbc import TVP
   tvp_data = TVP(table_type_name='dbo.ACM_Scores_TVP', rows=records)
   cur.execute("EXEC dbo.usp_BulkInsert_Scores @data=?", tvp_data)
   ```

2. **Use BCP (Bulk Copy Program)** for very large datasets (>100K rows):
   ```python
   # Write to temp CSV, then BULK INSERT
   temp_csv = f"/tmp/{table_name}_{uuid.uuid4()}.csv"
   df_clean.to_csv(temp_csv, index=False, header=False)
   cur.execute(f"""
       BULK INSERT dbo.[{table_name}]
       FROM '{temp_csv}'
       WITH (FORMAT = 'CSV', FIRSTROW = 1)
   """)
   os.remove(temp_csv)
   ```

3. **Tune `fast_executemany`** batch size per table:
   ```python
   # Different tables have different optimal batch sizes
   BATCH_SIZES = {
       'ACM_Scores_Long': 10000,  # Narrow schema, small rows
       'ACM_Scores_Wide': 1000,   # Wide schema, large rows
       'ACM_Episodes': 500        # Complex validation, smaller batches
   }
   batch_size = BATCH_SIZES.get(table_name, self.batch_size)
   ```

---

### 8. **Connection Pooling (Constructor, Lines 291-336)**
```python
def __init__(self, sql_client=None, ...):
    self.sql_client = sql_client
```

**Current**: Single connection passed in from caller

**Recommendation**: Implement connection pooling for parallel writes:
```python
from queue import Queue
import threading

class OutputManager:
    def __init__(self, sql_client=None, pool_size: int = 5, ...):
        self.sql_client = sql_client
        self._connection_pool = Queue(maxsize=pool_size)
        
        # Pre-populate pool
        if sql_client:
            for _ in range(pool_size):
                conn = sql_client.clone_connection()  # Implement this
                self._connection_pool.put(conn)
    
    @contextmanager
    def _get_connection(self):
        """Get connection from pool with timeout."""
        conn = self._connection_pool.get(timeout=30)
        try:
            yield conn
        finally:
            self._connection_pool.put(conn)
    
    def _bulk_insert_sql(self, table_name, df):
        """Use pooled connection."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            # ... existing logic
```

---

### 9. **Table Schema Caching (Lines 599-607)**
```python
self._table_exists_cache: Dict[str, bool] = {}
self._table_columns_cache: Dict[str, set] = {}
self._table_insertable_cache: Dict[str, set] = {}
```

**Issues**:
- Caches never invalidate (stale schema if tables altered mid-run)
- No TTL or max size limits

**Recommendation**:
```python
from collections import OrderedDict
from time import time

class LRUCache:
    """LRU cache with TTL."""
    def __init__(self, max_size=100, ttl_seconds=300):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl_seconds
    
    def get(self, key):
        if key not in self.cache:
            return None
        value, timestamp = self.cache[key]
        if time() - timestamp > self.ttl:
            del self.cache[key]
            return None
        self.cache.move_to_end(key)
        return value
    
    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        self.cache[key] = (value, time())

# Use in constructor
self._table_schema_cache = LRUCache(max_size=50, ttl_seconds=300)
```

---

### 10. **Timestamp Handling (Lines 145-187)**
```python
def _to_naive(ts) -> Optional[pd.Timestamp]:
    """Convert to timezone-naive local timestamp or None."""
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return None
    try:
        result = pd.to_datetime(ts, errors='coerce')
        # Strip timezone info if present
        if hasattr(result, 'tz') and result.tz is not None:
            return result.tz_localize(None)
        return result
    except Exception:
        return None
```

**Issues**:
- **"Local time policy"** comment (Line 134) contradicts SQL best practices
- SQL Server `DATETIME2` should store UTC, not local time
- Timezone stripping loses critical context for global deployments

**Recommendation**:
```python
def _to_utc_naive(ts) -> Optional[pd.Timestamp]:
    """Convert to timezone-naive UTC timestamp for SQL Server.
    
    SQL Server best practice: Store all timestamps as UTC in DATETIME2,
    convert to local time at query/display time.
    """
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return None
    try:
        result = pd.to_datetime(ts, errors='coerce', utc=True)
        if result is pd.NaT:
            return None
        # Convert to UTC then strip tz (SQL Server doesn't support tz-aware)
        if hasattr(result, 'tz') and result.tz is not None:
            result = result.tz_convert('UTC').tz_localize(None)
        return result
    except Exception:
        return None
```

**Schema Change Required**:
```sql
-- Add computed column for local time display
ALTER TABLE dbo.ACM_HealthTimeline 
ADD TimestampLocal AS DATEADD(HOUR, DATEDIFF(HOUR, GETUTCDATE(), GETDATE()), Timestamp);

-- Create index on UTC column for temporal queries
CREATE INDEX IX_HealthTimeline_Timestamp ON dbo.ACM_HealthTimeline(Timestamp);
```

---

## 🟢 **STRENGTHS**

1. **Whitelisted Tables** (Lines 21-35): Prevents SQL injection via table names
2. **Batched Transactions** (Lines 353-404): Reduces commit overhead
3. **Schema Repair with Audit Trail** (Lines 554-596): `OUT-17` pattern is excellent
4. **Float Cleaning** (Lines 670-690): Handles `NaN`/`Inf` before SQL write
5. **Connection Health Checks** (Lines 438-465): Prevents blind writes to dead connections

---

## 📋 **ACTION ITEMS**

### Immediate (Pre-Deployment)
1. ✅ Add `UNIQUE` constraint on `(RunID, EquipID, Timestamp)` in all time-series tables
2. ✅ Restore DELETE-before-INSERT with transaction safety OR document RunID uniqueness guarantee
3. ✅ Validate SQL Server autocommit mode in deployment environment
4. ✅ Add circuit breaker pattern to `_check_sql_health()`

### High Priority (Week 1)
5. ✅ Implement connection pooling for parallel writes
6. ✅ Add TVP support for bulk inserts (10-100x faster)
7. ✅ Fix timestamp handling to use UTC storage
8. ✅ Add LRU cache with TTL for schema metadata

### Medium Priority (Month 1)
9. ✅ Add SQL query performance monitoring (execution time, rows affected)
10. ✅ Implement audit table for dropped extreme values
11. ✅ Add retry logic with exponential backoff for transient SQL errors
12. ✅ Create stored procedures for common bulk operations

---

## 🎯 **SQL Backend Readiness Score: 6.5/10**

**Blocking Issues**: 2 (autocommit assumption, timestamp tz handling)  
**Critical Issues**: 4 (health check, schema validation, float handling, commit verification)  
**Optimization Gaps**: 4 (connection pooling, TVP, caching, indexing)

**Recommendation**: Address blocking issues before production deployment. This file is **70% SQL-ready** but needs hardening for enterprise SQL Server environments.