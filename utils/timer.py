# utils/timer.py
"""Performance timing utilities for ACM.

Provides section-based timing with automatic summary at exit.
Integrates with the enhanced logging system.
"""
from __future__ import annotations
import atexit
import time
import os
import functools
import json
from typing import Any, Callable, Dict

# Import Console for consistent logging
try:
    from utils.logger import Console
except Exception as exc:
    raise SystemExit(f"FATAL: Cannot import utils.logger.Console: {exc}") from exc

class Timer:
    """Lightweight stage timer. Usage:
       T = Timer();  with T.section("load_data"): ...;  T.log("custom", extra="info")
    
    Environment variables:
        ACM_TIMINGS: Enable/disable timing output (default: 1)
        LOG_FORMAT: Output format (text|json) - inherited from logger
    """
    def __init__(self, enable: bool | None = None):
        self.enable = bool(int(os.getenv("ACM_TIMINGS", "1"))) if enable is None else enable
        self.sections: Dict[str, float] = {}
        self.totals: Dict[str, float] = {}
        self._stack: list[str] = []
        self._t0 = time.perf_counter()
        self._json_mode = os.getenv("LOG_FORMAT", "text").lower() == "json"
        atexit.register(self._print_summary)

    def section(self, name: str):
        if not self.enable:
            return _NullContext()
        self.sections[name] = time.perf_counter()
        self._stack.append(name)
        return _Close(self, name)

    def end(self, name: str):
        if not self.enable: 
            return 0.0
        t1 = time.perf_counter()
        t0 = self.sections.pop(name, t1)
        dur = t1 - t0
        self.totals[name] = self.totals.get(name, 0.0) + dur
        
        if self._json_mode:
            Console.info(json.dumps({
                "timer": name,
                "duration_s": round(dur, 3),
                "event": "section_end"
            }))
        else:
            Console.info(f"[TIMER] {name:<20} {dur:7.3f}s")
        return dur

    def wrap(self, name: str):
        """Decorator: @T.wrap('features') on a function."""
        def deco(fn: Callable[..., Any]):
            @functools.wraps(fn)
            def inner(*a, **k):
                with self.section(name):
                    return fn(*a, **k)
            return inner
        return deco

    def log(self, name: str, **kv: Any):
        if not self.enable: 
            return
        
        if self._json_mode:
            data = {"timer": name, "event": "log"}
            data.update(kv)
            Console.info(json.dumps(data))
        else:
            extra = " ".join(f"{k}={v}" for k, v in kv.items())
            Console.info(f"[TIMER] {name:<20} {extra}".rstrip())

    def _print_summary(self):
        if not self.enable: 
            return
        total = time.perf_counter() - self._t0
        
        if self._json_mode:
            # JSON summary
            sections = [
                {"name": k, "duration_s": round(v, 3), "percent": round((v / total) * 100.0, 1)}
                for k, v in sorted(self.totals.items(), key=lambda x: -x[1])
            ]
            Console.info(json.dumps({
                "event": "timer_summary",
                "total_duration_s": round(total, 3),
                "sections": sections
            }))
        else:
            # Text summary
            if self.totals:
                Console.info("[TIMER] -- Summary ------------------------------")
                for k, v in sorted(self.totals.items(), key=lambda x: -x[1]):
                    pct = (v / total) * 100.0 if total > 0 else 0.0
                    Console.info(f"[TIMER] {k:<20} {v:7.3f}s ({pct:5.1f}%)")
            Console.info(f"[TIMER] total_run            {total:7.3f}s")

class _Close:
    def __init__(self, T: Timer, name: str): self.T, self.name = T, name
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): self.T.end(self.name)

class _NullContext:
    def __enter__(self): return self
    def __exit__(self, *a): return False
