"""
ACM Config History Writer

Writes configuration change audit log to ACM_ConfigHistory table for:
- Configuration change tracking
- Auto-tuning transparency
- Compliance and auditing
- Rollback capability

Called whenever ConfigDict.update_param() is invoked.
"""

from typing import Any, Optional
from datetime import datetime, timezone
import json
from core.observability import Console, Heartbeat


def _ensure_table(sql_client) -> None:
    """Create ACM_ConfigHistory if missing (best-effort, no raise)."""
    try:
        cur = sql_client.cursor()
        cur.execute(
            """
            IF NOT EXISTS (
                SELECT 1 FROM sys.objects WHERE object_id = OBJECT_ID(N'dbo.ACM_ConfigHistory') AND type = N'U'
            )
            BEGIN
                CREATE TABLE dbo.ACM_ConfigHistory (
                    ID BIGINT IDENTITY(1,1) PRIMARY KEY,
                    Timestamp DATETIME2(3) NOT NULL CONSTRAINT DF_ACM_ConfigHistory_Timestamp DEFAULT SYSUTCDATETIME(),
                    EquipID INT NOT NULL,
                    ParameterPath NVARCHAR(256) NOT NULL,
                    OldValue NVARCHAR(MAX) NULL,
                    NewValue NVARCHAR(MAX) NULL,
                    ChangedBy NVARCHAR(64) NULL,
                    ChangeReason NVARCHAR(256) NULL,
                    RunID NVARCHAR(64) NULL
                );
            END
            """
        )
        try:
            sql_client.conn.commit()
        except Exception:
            pass
    except Exception as e:
        try:
            Console.warn(f"Failed to ensure ACM_ConfigHistory table: {e}", component="CONFIG_HIST")
        except Exception:
            pass


def write_config_change(
    sql_client,
    equip_id: int,
    parameter_path: str,
    old_value: Any,
    new_value: Any,
    changed_by: str = "SYSTEM",
    change_reason: str = "",
    run_id: Optional[str] = None
) -> bool:
    """
    Write config change record to ACM_ConfigHistory table.
    
    Args:
        sql_client: SQL connection client
        equip_id: Equipment ID
        parameter_path: Dot-separated parameter path (e.g., "thresholds.q")
        old_value: Previous value (will be JSON-encoded if complex type)
        new_value: New value (will be JSON-encoded if complex type)
        changed_by: Who/what made the change (default: SYSTEM)
        change_reason: Human-readable reason for change
        run_id: Optional RunID that triggered this change
    
    Returns:
        bool: True if write succeeded, False otherwise
    """
    
    if sql_client is None:
        Console.warn("No SQL client provided, skipping ACM_ConfigHistory write", component="CONFIG_HIST")
        return False
    _ensure_table(sql_client)
    
    try:
        # Serialize complex values to JSON
        def serialize_value(val):
            if val is None:
                return None
            if isinstance(val, (dict, list)):
                return json.dumps(val, sort_keys=True)
            return str(val)
        
        old_value_str = serialize_value(old_value)
        new_value_str = serialize_value(new_value)
        
        # Skip if values are identical (no actual change)
        if old_value_str == new_value_str:
            Console.info(f"Skipping write - no change detected for {parameter_path}", component="CONFIG_HIST")
            return True
        
        # Build insert statement
        insert_sql = """
        INSERT INTO dbo.ACM_ConfigHistory (
            Timestamp, EquipID, ParameterPath, OldValue, NewValue, 
            ChangedBy, ChangeReason, RunID
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        # Prepare record
        record = (
            datetime.now().replace(tzinfo=None),  # SQL datetime2 requires naive UTC
            int(equip_id),
            str(parameter_path),
            old_value_str,
            new_value_str,
            str(changed_by),
            str(change_reason) if change_reason else None,
            str(run_id) if run_id else None
        )
        
        # Execute insert
        with sql_client.cursor() as cur:
            cur.execute(insert_sql, record)
        
        # Commit
        sql_client.conn.commit()
        
        Console.info(f"Logged config change: {parameter_path} = {new_value} (reason: {change_reason})", component="CONFIG_HIST")
        return True
        
    except Exception as e:
        Console.error(f"Failed to write ACM_ConfigHistory: {e}", component="CONFIG_HIST")
        try:
            sql_client.conn.rollback()
        except:
            pass
        return False


def write_config_changes_bulk(
    sql_client,
    equip_id: int,
    changes: list,
    changed_by: str = "SYSTEM",
    run_id: Optional[str] = None
) -> bool:
    """
    Write multiple config changes in a single transaction.
    
    Args:
        sql_client: SQL connection client
        equip_id: Equipment ID
        changes: List of dicts with keys: parameter_path, old_value, new_value, change_reason
        changed_by: Who/what made the changes
        run_id: Optional RunID that triggered these changes
    
    Returns:
        bool: True if all writes succeeded, False otherwise
    """
    
    if sql_client is None:
        Console.warn("No SQL client provided, skipping bulk write", component="CONFIG_HIST")
        return False
    _ensure_table(sql_client)
    
    if not changes:
        return True
    
    try:
        # Serialize complex values to JSON
        def serialize_value(val):
            if val is None:
                return None
            if isinstance(val, (dict, list)):
                return json.dumps(val, sort_keys=True)
            return str(val)
        
        # Build records list
        records = []
        timestamp = datetime.now().replace(tzinfo=None)
        
        for change in changes:
            parameter_path = change["parameter_path"]
            old_value_str = serialize_value(change.get("old_value"))
            new_value_str = serialize_value(change.get("new_value"))
            change_reason = change.get("change_reason", "")
            
            # Skip if no actual change
            if old_value_str == new_value_str:
                continue
            
            records.append((
                timestamp,
                int(equip_id),
                str(parameter_path),
                old_value_str,
                new_value_str,
                str(changed_by),
                str(change_reason) if change_reason else None,
                str(run_id) if run_id else None
            ))
        
        if not records:
            Console.info("No actual changes to write (all values unchanged)", component="CONFIG_HIST")
            return True
        
        # Build bulk insert statement
        insert_sql = """
        INSERT INTO dbo.ACM_ConfigHistory (
            Timestamp, EquipID, ParameterPath, OldValue, NewValue, 
            ChangedBy, ChangeReason, RunID
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        # Execute bulk insert
        with sql_client.cursor() as cur:
            cur.fast_executemany = True
            cur.executemany(insert_sql, records)
        
        # Commit
        sql_client.conn.commit()
        
        Console.info(f"Logged {len(records)} config changes for RunID={run_id}", component="CONFIG_HIST")
        return True
        
    except Exception as e:
        Console.error(f"Failed to write bulk ACM_ConfigHistory: {e}", component="CONFIG_HIST")
        try:
            sql_client.conn.rollback()
        except:
            pass
        return False


def log_auto_tune_changes(
    sql_client,
    equip_id: int,
    tuning_actions: list,
    run_id: str,
    trigger_refit: bool = False
) -> bool:
    """
    Convenience function to log auto-tuning parameter changes.
    Task 9: Extended to support PendingApply flag and refit signaling.
    
    Args:
        sql_client: SQL connection client
        equip_id: Equipment ID
        tuning_actions: List of tuning action strings (e.g., "clip_z: 12.0->14.4")
        run_id: RunID that triggered the tuning
        trigger_refit: If True, creates refit request to apply changes next run
    
    Returns:
        bool: True if write succeeded, False otherwise
    """
    
    if not tuning_actions:
        return True
    
    # Parse tuning actions into structured changes
    changes = []
    for action in tuning_actions:
        try:
            # Parse format: "parameter: old_value->new_value"
            parts = action.split(":")
            if len(parts) != 2:
                continue
            
            param_name = parts[0].strip()
            value_change = parts[1].strip()
            
            # Parse old->new values
            if "->" in value_change:
                old_val, new_val = value_change.split("->")
                old_val = old_val.strip()
                new_val = new_val.strip()
                
                # Convert to appropriate type (float for most tuning params)
                try:
                    old_val = float(old_val)
                    new_val = float(new_val)
                except:
                    pass  # Keep as strings if not numeric
                
                changes.append({
                    "parameter_path": param_name,
                    "old_value": old_val,
                    "new_value": new_val,
                    "change_reason": "Auto-tuning based on quality assessment"
                })
        except Exception as e:
            Console.warn(f"Failed to parse tuning action '{action}': {e}", component="CONFIG_HIST")
    
    if not changes:
        return True
    
    success = write_config_changes_bulk(
        sql_client=sql_client,
        equip_id=equip_id,
        changes=changes,
        changed_by="AUTO_TUNE",
        run_id=run_id
    )
    
    # Task 9: If trigger_refit=True, create refit request to apply config changes
    if success and trigger_refit:
        try:
            with sql_client.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO [dbo].[ACM_RefitRequests]
                        (EquipID, Reason, Acknowledged)
                    VALUES
                        (?, ?, 0)
                    """,
                    (equip_id, f"Auto-tune config changes: {', '.join([c['parameter_path'] for c in changes])}")
                )
            Console.info("Refit request created to apply config changes next run", component="AUTO-TUNE")
        except Exception as e:
            Console.warn(f"Failed to create refit request: {e}", component="AUTO-TUNE")
    
    return success

