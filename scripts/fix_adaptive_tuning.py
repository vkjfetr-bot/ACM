#!/usr/bin/env python3
"""
Remove dead adaptive tuning code.

Since param_adjustments is always empty (MHAL was removed), the entire
if param_adjustments: block never executes.
"""

def main():
    filepath = "core/acm_main.py"
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    original_length = len(content)
    
    # Find and replace the dead adaptive tuning section
    old_section = '''        # ===== ADAPTIVE PARAMETER TUNING: Continuous Self-Monitoring =====
        # Check model health and auto-adjust parameters if needed
        Console.info("Checking model health...", component="ADAPTIVE")
        param_adjustments = []
        
        # NOTE: MHAL condition number / NaN checks removed v9.1.0 - detector deprecated
        
        # ACM-CSV-03: Write parameter adjustments (file-mode: CSV, SQL-mode: ACM_ConfigHistory)
        if param_adjustments:
            Console.info(f"Writing {len(param_adjustments)} parameter adjustment(s) to config...", component="ADAPTIVE")
            
            # File mode: Update config_table.csv
            if not SQL_MODE:
                config_table_path = Path("configs/config_table.csv")
                
                if config_table_path.exists():
                    try:
                        df_config = pd.read_csv(config_table_path)
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        for adj in param_adjustments:
                            param_parts = adj['param'].split('.')
                            category = param_parts[0]
                            param_path = '.'.join(param_parts[1:])
                            
                            # Try to parse EquipID from equip string
                            try:
                                equip_id_int = int(equip)
                            except ValueError:
                                equip_id_int = 0
                            
                            # Check if row exists
                            mask = (df_config["EquipID"] == equip_id_int) & \\
                                   (df_config["Category"] == category) & \\
                                   (df_config["ParamPath"] == param_path)
                            
                            if mask.any():
                                # Update existing
                                df_config.loc[mask, "ParamValue"] = adj['new']
                                df_config.loc[mask, "UpdatedDateTime"] = timestamp
                                df_config.loc[mask, "UpdatedBy"] = "ADAPTIVE_TUNING"
                                df_config.loc[mask, "ChangeReason"] = adj['reason']
                                Console.info(f"  Updated {adj['param']}: {adj['old']} -> {adj['new']}")
                            else:
                                # Insert new
                                new_row = {
                                    "EquipID": equip_id_int,
                                    "Category": category,
                                    "ParamPath": param_path,
                                    "ParamValue": adj['new'],
                                    "ValueType": "float",
                                    "UpdatedDateTime": timestamp,
                                    "UpdatedBy": "ADAPTIVE_TUNING",
                                    "ChangeReason": adj['reason']
                                }
                                df_config = pd.concat([df_config, pd.DataFrame([new_row])], ignore_index=True)
                                Console.info(f"  Inserted {adj['param']}: {adj['new']}")
                        
                        df_config.to_csv(config_table_path, index=False)
                        Console.info(f"Config updated: {config_table_path}", component="ADAPTIVE")
                        Console.info(f"Rerun ACM to apply new parameters (current run continues with old params)", component="ADAPTIVE")
                        
                    except Exception as e:
                        Console.error(f"Failed to update config CSV: {e}", component="ADAPTIVE",
                                     equip=equip, error_type=type(e).__name__, error=str(e)[:200])
                else:
                    Console.warn(f"Config table not found at {config_table_path}, skipping parameter updates", component="ADAPTIVE",
                                 equip=equip, config_path=str(config_table_path))
            
            # SQL mode: Write to ACM_ConfigHistory via config_history_writer
            elif sql_client and SQL_MODE:
                try:
                    from core.config_history_writer import write_config_changes_bulk
                    
                    # Transform param_adjustments to config_history format
                    changes = [
                        {
                            "parameter_path": adj['param'],
                            "old_value": adj['old'],
                            "new_value": adj['new'],
                            "change_reason": adj['reason']
                        }
                        for adj in param_adjustments
                    ]
                    
                    success = write_config_changes_bulk(
                        sql_client=sql_client,
                        equip_id=int(equip_id),
                        changes=changes,
                        changed_by="ADAPTIVE_TUNING",
                        run_id=run_id
                    )
                    
                    if success:
                        Console.info(f"Config changes written to SQL:ACM_ConfigHistory", component="ADAPTIVE")
                        Console.info(f"Rerun ACM to apply new parameters (current run continues with old params)", component="ADAPTIVE")
                    else:
                        Console.warn(f"Failed to write config changes to SQL", component="ADAPTIVE",
                                     equip=equip, run_id=run_id)
                        
                except Exception as e:
                    Console.error(f"Failed to update config SQL: {e}", component="ADAPTIVE",
                                 equip=equip, run_id=run_id, error_type=type(e).__name__, error=str(e)[:200])
        else:
            Console.info("All model parameters within healthy ranges", component="ADAPTIVE")'''
    
    new_section = '''        # ===== ADAPTIVE PARAMETER TUNING: Model health check =====
        # NOTE: Adaptive tuning logic removed in v9.1.0 (MHAL detector deprecated)
        Console.info("All model parameters within healthy ranges", component="ADAPTIVE")'''
    
    if old_section not in content:
        print("ERROR: Could not find adaptive tuning section")
        print("First 500 chars of section we're looking for:")
        print(old_section[:500])
        return
    
    content = content.replace(old_section, new_section)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    new_length = len(content)
    print(f"SUCCESS: Simplified adaptive tuning section")
    print(f"Original: {original_length} chars, New: {new_length} chars")
    print(f"Removed: {original_length - new_length} chars")

if __name__ == "__main__":
    main()
