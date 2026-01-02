# core/acm.py
"""
ACM v11 Entry Point

Single entry point that routes to ONLINE or OFFLINE pipelines based on mode.

Usage:
    python -m core.acm --equip FD_FAN --mode auto
    python -m core.acm --equip FD_FAN --mode online
    python -m core.acm --equip FD_FAN --mode offline

Modes:
    online  - Scoring only, requires existing model. Fast. (default for production)
    offline - Full discovery + model training. Slow.
    auto    - Check if model exists: if yes -> online, if no -> offline

The old entry point (python -m core.acm_main) still works but is deprecated.
"""
import argparse
import sys
from typing import Optional


def main() -> int:
    """Main entry point for ACM."""
    ap = argparse.ArgumentParser(
        prog="python -m core.acm",
        description="ACM v11 - Automated Condition Monitoring",
        epilog="""
Modes:
  online   Scoring only using existing model (fast, for production)
  offline  Full regime discovery and model training (slow)
  auto     Auto-detect: run online if model exists, else offline

Examples:
  python -m core.acm --equip FD_FAN --mode auto
  python -m core.acm --equip FD_FAN --mode offline --start-time 2023-01-01T00:00:00
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--equip", required=True, help="Equipment name (e.g., FD_FAN)")
    ap.add_argument("--mode", choices=["online", "offline", "auto"], default="auto",
                    help="Pipeline mode: online (scoring), offline (discovery), auto (detect)")
    ap.add_argument("--start-time", help="Start time (ISO format)")
    ap.add_argument("--end-time", help="End time (ISO format)")
    ap.add_argument("--config", help="Config file path")
    ap.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    
    args = ap.parse_args()
    
    # Determine effective mode
    effective_mode = args.mode
    if effective_mode == "auto":
        effective_mode = _detect_mode(args.equip)
    
    # Route to appropriate pipeline
    if effective_mode == "online":
        return _run_online(args)
    else:
        return _run_offline(args)


def _detect_mode(equipment: str) -> str:
    """Check if active model exists for equipment. Returns 'online' or 'offline'."""
    try:
        from core.model_persistence import load_model_from_sql
        from core.sql_client import SQLClient
        from utils.config_dict import ConfigDict
        
        cfg = ConfigDict().as_dict()
        sql_client = SQLClient(cfg)
        
        # Try to load model from registry
        model_data = load_model_from_sql(sql_client, equipment)
        
        if model_data is not None and model_data.get("manifest"):
            print(f"[ACM] Active model found for {equipment} -> ONLINE mode")
            return "online"
        else:
            print(f"[ACM] No active model for {equipment} -> OFFLINE mode")
            return "offline"
            
    except Exception as e:
        print(f"[ACM] Model check failed ({e}) -> OFFLINE mode")
        return "offline"


def _run_online(args: argparse.Namespace) -> int:
    """Run ONLINE pipeline (scoring only)."""
    print(f"[ACM] Running ONLINE pipeline for {args.equip}")
    
    # Build command for acm_main with --mode online
    cmd_args = [
        "--equip", args.equip,
        "--mode", "online",
    ]
    if args.start_time:
        cmd_args.extend(["--start-time", args.start_time])
    if args.end_time:
        cmd_args.extend(["--end-time", args.end_time])
    if args.config:
        cmd_args.extend(["--config", args.config])
    if args.log_level:
        cmd_args.extend(["--log-level", args.log_level])
    
    # Import and run acm_main directly (no subprocess)
    sys.argv = ["acm_main"] + cmd_args
    
    try:
        from core import acm_main
        acm_main.main()
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1
    except Exception as e:
        print(f"[ACM] ONLINE pipeline failed: {e}")
        # Fallback to OFFLINE
        print(f"[ACM] Falling back to OFFLINE mode...")
        return _run_offline(args)


def _run_offline(args: argparse.Namespace) -> int:
    """Run OFFLINE pipeline (full discovery)."""
    print(f"[ACM] Running OFFLINE pipeline for {args.equip}")
    
    # Build command for acm_main with --mode offline
    cmd_args = [
        "--equip", args.equip,
        "--mode", "offline",
    ]
    if args.start_time:
        cmd_args.extend(["--start-time", args.start_time])
    if args.end_time:
        cmd_args.extend(["--end-time", args.end_time])
    if args.config:
        cmd_args.extend(["--config", args.config])
    if args.log_level:
        cmd_args.extend(["--log-level", args.log_level])
    
    sys.argv = ["acm_main"] + cmd_args
    
    try:
        from core import acm_main
        acm_main.main()
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1
    except Exception as e:
        print(f"[ACM] OFFLINE pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
