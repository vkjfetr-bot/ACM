#!/usr/bin/env python
"""Add _run_comprehensive_analytics helper function to acm_main.py"""
import re

HELPER_CODE = '''
@dataclass
class AnalyticsResult:
    """Result of comprehensive analytics generation."""
    table_count: int
    forecast_success: bool
    rul_p50: Optional[float] = None
    rul_p10: Optional[float] = None
    rul_p90: Optional[float] = None
    tables_written: List[str] = field(default_factory=list)
    error: Optional[str] = None


def _run_comprehensive_analytics(
    frame: pd.DataFrame,
    episodes: pd.DataFrame,
    run_dir: Path,
    output_manager: "OutputManager",
    regime_model: Optional[Any],
    fusion_weights_used: Optional[Dict[str, float]],
    sensor_context: Optional[Dict[str, Any]],
    equip_id: int,
    run_id: str,
    equip: str,
    cfg: Dict[str, Any],
    T: "Timer",
    SQL_MODE: bool,
) -> AnalyticsResult:
    """
    Run comprehensive analytics and forecasting.
    
    Generates all analytics tables and runs the unified forecast engine.
    Falls back to basic table generation if comprehensive analytics fail.
    
    Args:
        frame: Score frame with detector outputs
        episodes: Detected anomaly episodes
        run_dir: Run output directory
        output_manager: Output manager for writes
        regime_model: Trained regime model (optional)
        fusion_weights_used: Tuned fusion weights dict
        sensor_context: Sensor context for analytics
        equip_id: Equipment ID
        run_id: Current run ID
        equip: Equipment name
        cfg: Configuration dictionary
        T: Timer instance
        SQL_MODE: Whether SQL mode is enabled
        
    Returns:
        AnalyticsResult with table_count and forecast results
    """
    from core.observability import Console
    from core.forecast_engine import ForecastEngine
    from core.metrics import record_rul, record_active_defects
    from utils.version import ACM_VERSION
    
    tables_dir = run_dir / "tables"
    table_count = 0
    
    # Create tables directory if writing files
    if not SQL_MODE:
        tables_dir.mkdir(exist_ok=True)
    
    with T.section("outputs.comprehensive_analytics"):
        Console.info("Generating comprehensive analytics tables...", component="ANALYTICS")
        
        # TUNE-FIX: Inject tuned weights into cfg for ACM_FusionQualityReport
        if fusion_weights_used:
            if 'fusion' not in cfg:
                cfg['fusion'] = {}
            cfg['fusion']['weights'] = dict(fusion_weights_used)
        
        try:
            # Generate all 23+ analytical tables
            output_manager.generate_all_analytics_tables(
                scores_df=frame,
                episodes_df=episodes,
                cfg=cfg,
                tables_dir=tables_dir,
                sensor_context=sensor_context
            )
            Console.info("Successfully generated all comprehensive analytics tables", component="ANALYTICS")
            table_count = 23
        except Exception as e:
            Console.error(
                f"Error generating comprehensive analytics: {str(e)}",
                component="ANALYTICS",
                equip=equip,
                run_id=run_id,
                error_type=type(e).__name__,
                error=str(e)[:500]
            )
            Console.info("Falling back to basic table generation...", component="ANALYTICS")
            
            # Fallback: Health timeline
            table_count = _generate_fallback_health_timeline(
                frame=frame,
                output_manager=output_manager,
                run_id=run_id,
                equip_id=equip_id,
                equip=equip,
                cfg=cfg,
            )
            
            # Fallback: Regime timeline
            table_count += _generate_fallback_regime_timeline(
                frame=frame,
                output_manager=output_manager,
                regime_model=regime_model,
                run_id=run_id,
                equip_id=equip_id,
            )
    
    Console.info(
        f"Generated {table_count} analytics tables via OutputManager",
        component="OUTPUTS"
    )
    Console.info(
        f"Analytics tables written to SQL (mode=sql, dir={tables_dir})",
        component="OUTPUTS",
        table_count=table_count
    )
    
    # Run forecasting
    forecast_result = _run_forecast_engine(
        output_manager=output_manager,
        equip_id=equip_id,
        run_id=run_id,
        equip=equip,
        cfg=cfg,
        T=T,
    )
    
    return AnalyticsResult(
        table_count=table_count,
        forecast_success=forecast_result.get("success", False),
        rul_p50=forecast_result.get("rul_p50"),
        rul_p10=forecast_result.get("rul_p10"),
        rul_p90=forecast_result.get("rul_p90"),
        tables_written=forecast_result.get("tables_written", []),
        error=forecast_result.get("error"),
    )


def _generate_fallback_health_timeline(
    frame: pd.DataFrame,
    output_manager: "OutputManager",
    run_id: str,
    equip_id: int,
    equip: str,
    cfg: Dict[str, Any],
) -> int:
    """Generate fallback health timeline when comprehensive analytics fail."""
    from core.observability import Console
    
    if 'fused' not in frame.columns:
        return 0
    
    # Calculate raw health index using softer sigmoid formula
    z_threshold = cfg.get('health', {}).get('z_threshold', 5.0)
    steepness = cfg.get('health', {}).get('steepness', 1.5)
    abs_z = np.abs(frame['fused'])
    normalized = (abs_z - z_threshold / 2) / (z_threshold / 4)
    sigmoid = 1 / (1 + np.exp(-normalized * steepness))
    raw_health = np.clip(100.0 * (1 - sigmoid), 0.0, 100.0)
    
    # Apply exponential smoothing
    alpha = cfg.get('health', {}).get('smoothing_alpha', 0.3)
    smoothed_health = pd.Series(raw_health).ewm(alpha=alpha, adjust=False).mean()
    
    # Data quality flags
    health_change = smoothed_health.diff().abs()
    max_change_per_period = cfg.get('health', {}).get('max_change_per_period', 20.0)
    quality_flag = np.where(health_change > max_change_per_period, 'VOLATILE', 'NORMAL')
    quality_flag[0] = 'NORMAL'
    
    # Check extreme FusedZ values
    extreme_z_threshold = cfg.get('health', {}).get('extreme_z_threshold', 10.0)
    quality_flag = np.where(np.abs(frame['fused']) > extreme_z_threshold, 'EXTREME_ANOMALY', quality_flag)
    
    health_df = pd.DataFrame({
        'Timestamp': frame.index,
        'HealthIndex': smoothed_health,
        'RawHealthIndex': raw_health,
        'HealthZone': pd.cut(
            smoothed_health,
            bins=[-1, 30, 50, 70, 85, 101],
            labels=['CRITICAL', 'ALERT', 'WATCH', 'CAUTION', 'GOOD']
        ),
        'FusedZ': frame['fused'],
        'QualityFlag': quality_flag,
        'RunID': run_id,
        'EquipID': equip_id
    })
    
    # Log quality issues
    volatile_count = (quality_flag == 'VOLATILE').sum()
    extreme_count = (quality_flag == 'EXTREME_ANOMALY').sum()
    if volatile_count > 0:
        Console.warn(
            f"{volatile_count} volatile health transitions detected (>{max_change_per_period}% change)",
            component="HEALTH",
            equip=equip,
            volatile_count=volatile_count,
            threshold=max_change_per_period
        )
    if extreme_count > 0:
        Console.warn(
            f"{extreme_count} extreme anomaly scores detected (|Z| > {extreme_z_threshold})",
            component="HEALTH",
            equip=equip,
            extreme_count=extreme_count,
            z_threshold=extreme_z_threshold
        )
    
    output_manager.write_dataframe(
        health_df,
        "health_timeline",
        sql_table="ACM_HealthTimeline",
        add_created_at=True
    )
    return 1


def _generate_fallback_regime_timeline(
    frame: pd.DataFrame,
    output_manager: "OutputManager",
    regime_model: Optional[Any],
    run_id: str,
    equip_id: int,
) -> int:
    """Generate fallback regime timeline when comprehensive analytics fail."""
    if 'regime_label' not in frame.columns:
        return 0
    
    regime_df = pd.DataFrame({
        'Timestamp': frame.index,
        'RegimeLabel': frame['regime_label'].astype(int),
        'EquipID': equip_id,
        'RunID': run_id
    })
    
    # Add RegimeState based on regime_model
    if regime_model and hasattr(regime_model, 'health_labels'):
        regime_df['RegimeState'] = regime_df['RegimeLabel'].map(
            lambda x: regime_model.health_labels.get(int(x), 'unknown')
        )
    else:
        regime_df['RegimeState'] = 'unknown'
    
    output_manager.write_dataframe(
        regime_df,
        "regime_timeline",
        sql_table="ACM_RegimeTimeline",
        add_created_at=True
    )
    return 1


def _run_forecast_engine(
    output_manager: "OutputManager",
    equip_id: int,
    run_id: str,
    equip: str,
    cfg: Dict[str, Any],
    T: "Timer",
) -> Dict[str, Any]:
    """Run the unified forecast engine and record metrics."""
    from core.observability import Console
    from core.forecast_engine import ForecastEngine
    from core.metrics import record_rul, record_active_defects
    from utils.version import ACM_VERSION
    
    with T.section("outputs.forecasting"):
        try:
            Console.info(f"Running unified forecasting engine (v{ACM_VERSION})", component="FORECAST")
            forecast_engine = ForecastEngine(
                sql_client=getattr(output_manager, "sql_client", None),
                output_manager=output_manager,
                equip_id=equip_id,
                run_id=str(run_id) if run_id else None,
                config=cfg
            )
            
            forecast_results = forecast_engine.run_forecast()
            
            if forecast_results.get('success'):
                Console.info(
                    f"[FORECAST] RUL P50={forecast_results['rul_p50']:.1f}h, "
                    f"P10={forecast_results['rul_p10']:.1f}h, P90={forecast_results['rul_p90']:.1f}h"
                )
                Console.info(f"Top sensors: {forecast_results['top_sensors']}", component="FORECAST")
                Console.info(f"Wrote tables: {', '.join(forecast_results['tables_written'])}", component="FORECAST")
                
                # Record RUL metrics for Prometheus
                try:
                    record_rul(
                        equip,
                        rul_hours=float(forecast_results['rul_p50']),
                        p10=float(forecast_results['rul_p10']),
                        p50=float(forecast_results['rul_p50']),
                        p90=float(forecast_results['rul_p90'])
                    )
                    if 'active_defects' in forecast_results:
                        record_active_defects(equip, int(forecast_results['active_defects']))
                except Exception as metric_err:
                    Console.debug(f"Failed to record forecast metrics: {metric_err}", component="OTEL")
            else:
                Console.warn(
                    f"Forecast failed: {forecast_results.get('error', 'Unknown error')}",
                    component="FORECAST",
                    equip=equip,
                    run_id=run_id,
                    data_quality=forecast_results.get('data_quality')
                )
            
            return forecast_results
            
        except Exception as e:
            Console.error(
                f"Unified forecasting engine failed: {e}",
                component="FORECAST",
                equip=equip,
                run_id=run_id,
                error_type=type(e).__name__,
                error=str(e)[:500]
            )
            Console.error(
                "RUL estimation skipped - ForecastEngine must be fixed",
                component="FORECAST",
                equip=equip,
                run_id=run_id
            )
            return {"success": False, "error": str(e)}

'''

def main():
    filepath = "core/acm_main.py"
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find the insertion point - after _finalize_sql_and_record_metrics function
    pattern = r"(def _finalize_sql_and_record_metrics\([\s\S]*?pass\n)"
    match = re.search(pattern, content)
    
    if not match:
        print("ERROR: Could not find _finalize_sql_and_record_metrics function")
        return
    
    insert_pos = match.end()
    
    # Check if helper already exists
    if "def _run_comprehensive_analytics" in content:
        print("WARNING: _run_comprehensive_analytics already exists")
        return
    
    # Insert the helper
    new_content = content[:insert_pos] + HELPER_CODE + content[insert_pos:]
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print("SUCCESS: Added AnalyticsResult dataclass and analytics helper functions")

if __name__ == "__main__":
    main()
