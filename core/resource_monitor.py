"""
Resource Monitor for ACM - Tracks CPU, memory, and timing metrics.

Usage:
    from core.resource_monitor import ResourceMonitor, R

    # Global singleton (like Timer T)
    R = ResourceMonitor()

    # Context manager for sections
    with R.section("detector.fit.gmm"):
        gmm_detector.fit(train)

    # Get metrics
    metrics = R.get_metrics()
    R.write_to_sql(sql_client, equip_id, run_id)

    # Reset for next run
    R.reset()
"""

import os
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

import pandas as pd


@dataclass
class SectionMetrics:
    """Metrics captured for a single code section."""
    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration_s: float = 0.0
    
    # Memory metrics (bytes)
    mem_start_rss: int = 0
    mem_end_rss: int = 0
    mem_peak_rss: int = 0
    mem_delta_mb: float = 0.0
    
    # CPU metrics
    cpu_percent_avg: float = 0.0
    cpu_samples: List[float] = field(default_factory=list)
    
    # Thread count
    thread_count: int = 0
    
    # Nested depth (for hierarchy)
    depth: int = 0
    
    def finalize(self):
        """Calculate derived metrics after section completes."""
        self.duration_s = self.end_time - self.start_time
        self.mem_delta_mb = (self.mem_end_rss - self.mem_start_rss) / (1024 * 1024)
        if self.cpu_samples:
            self.cpu_percent_avg = sum(self.cpu_samples) / len(self.cpu_samples)


class ResourceMonitor:
    """
    Lightweight resource monitor for ACM pipeline sections.
    
    Tracks:
    - Wall-clock time (always)
    - Memory RSS delta (if psutil available)
    - CPU usage samples (if psutil available, background thread)
    - Peak memory per section
    
    Design goals:
    - Minimal overhead (<1% runtime impact)
    - Non-blocking CPU sampling via background thread
    - Compatible with existing Timer (T) pattern
    - SQL output for Grafana dashboards
    """
    
    def __init__(self, sample_interval: float = 0.5, enabled: bool = True):
        """
        Initialize resource monitor.
        
        Args:
            sample_interval: CPU sampling interval in seconds (default 0.5s)
            enabled: Whether monitoring is active (can disable for production)
        """
        self.enabled = enabled and PSUTIL_AVAILABLE
        self.sample_interval = sample_interval
        self._sections: Dict[str, SectionMetrics] = {}
        self._section_order: List[str] = []
        self._active_stack: List[str] = []
        self._lock = threading.Lock()
        
        # Background CPU sampler
        self._sampling = False
        self._sampler_thread: Optional[threading.Thread] = None
        self._current_section: Optional[str] = None
        
        # Process handle
        self._process = psutil.Process() if PSUTIL_AVAILABLE else None
        
        # Run-level aggregates
        self._run_start_time: float = 0.0
        self._run_start_mem: int = 0
        self._peak_mem_rss: int = 0
    
    def start_run(self):
        """Mark the start of an ACM run for aggregate metrics."""
        if not self.enabled:
            return
        self._run_start_time = time.perf_counter()
        self._run_start_mem = self._get_memory_rss()
        self._peak_mem_rss = self._run_start_mem
    
    def _get_memory_rss(self) -> int:
        """Get current RSS memory in bytes."""
        if self._process:
            try:
                return self._process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return 0
        return 0
    
    def _get_cpu_percent(self) -> float:
        """Get current CPU percent (0-100 per core)."""
        if self._process:
            try:
                return self._process.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return 0.0
        return 0.0
    
    def _start_cpu_sampling(self, section_name: str):
        """Start background CPU sampling for a section."""
        if not self.enabled:
            return
        
        self._current_section = section_name
        self._sampling = True
        
        def sampler():
            # Prime the CPU percent counter
            self._get_cpu_percent()
            while self._sampling and self._current_section == section_name:
                time.sleep(self.sample_interval)
                if self._sampling and self._current_section == section_name:
                    cpu = self._get_cpu_percent()
                    mem = self._get_memory_rss()
                    with self._lock:
                        if section_name in self._sections:
                            self._sections[section_name].cpu_samples.append(cpu)
                            if mem > self._sections[section_name].mem_peak_rss:
                                self._sections[section_name].mem_peak_rss = mem
                            if mem > self._peak_mem_rss:
                                self._peak_mem_rss = mem
        
        self._sampler_thread = threading.Thread(target=sampler, daemon=True)
        self._sampler_thread.start()
    
    def _stop_cpu_sampling(self):
        """Stop background CPU sampling."""
        self._sampling = False
        self._current_section = None
        # Don't wait for thread - it's daemon and will stop on next iteration
    
    @contextmanager
    def section(self, name: str):
        """
        Context manager to monitor a code section.
        
        Args:
            name: Dotted section name (e.g., "detector.fit.gmm")
        
        Yields:
            SectionMetrics object (can be ignored)
        """
        if not self.enabled:
            yield None
            return
        
        depth = len(self._active_stack)
        self._active_stack.append(name)
        
        metrics = SectionMetrics(name=name, depth=depth)
        metrics.start_time = time.perf_counter()
        metrics.mem_start_rss = self._get_memory_rss()
        metrics.mem_peak_rss = metrics.mem_start_rss
        metrics.thread_count = threading.active_count()
        
        with self._lock:
            self._sections[name] = metrics
            if name not in self._section_order:
                self._section_order.append(name)
        
        # Start CPU sampling for longer sections
        self._start_cpu_sampling(name)
        
        try:
            yield metrics
        finally:
            self._stop_cpu_sampling()
            
            metrics.end_time = time.perf_counter()
            metrics.mem_end_rss = self._get_memory_rss()
            if metrics.mem_end_rss > metrics.mem_peak_rss:
                metrics.mem_peak_rss = metrics.mem_end_rss
            metrics.finalize()
            
            self._active_stack.pop()
    
    def record(self, name: str, duration_s: float, mem_delta_mb: float = 0.0):
        """
        Manually record metrics for a section (for integration with Timer).
        
        Args:
            name: Section name
            duration_s: Duration in seconds
            mem_delta_mb: Memory change in MB (optional)
        """
        if not self.enabled:
            return
        
        metrics = SectionMetrics(
            name=name,
            duration_s=duration_s,
            mem_delta_mb=mem_delta_mb
        )
        
        with self._lock:
            self._sections[name] = metrics
            if name not in self._section_order:
                self._section_order.append(name)
    
    def get_metrics(self) -> List[Dict[str, Any]]:
        """Get all section metrics as list of dicts."""
        with self._lock:
            result = []
            for name in self._section_order:
                if name in self._sections:
                    m = self._sections[name]
                    result.append({
                        "section": m.name,
                        "duration_s": round(m.duration_s, 4),
                        "mem_start_mb": round(m.mem_start_rss / (1024*1024), 2),
                        "mem_end_mb": round(m.mem_end_rss / (1024*1024), 2),
                        "mem_peak_mb": round(m.mem_peak_rss / (1024*1024), 2),
                        "mem_delta_mb": round(m.mem_delta_mb, 2),
                        "cpu_avg_pct": round(m.cpu_percent_avg, 1),
                        "cpu_samples": len(m.cpu_samples),
                        "depth": m.depth,
                        "threads": m.thread_count
                    })
            return result
    
    def get_summary(self) -> Dict[str, Any]:
        """Get run-level summary metrics."""
        if not self.enabled:
            return {}
        
        current_mem = self._get_memory_rss()
        return {
            "total_duration_s": time.perf_counter() - self._run_start_time,
            "mem_start_mb": round(self._run_start_mem / (1024*1024), 2),
            "mem_current_mb": round(current_mem / (1024*1024), 2),
            "mem_peak_mb": round(self._peak_mem_rss / (1024*1024), 2),
            "mem_delta_mb": round((current_mem - self._run_start_mem) / (1024*1024), 2),
            "section_count": len(self._sections),
            "psutil_available": PSUTIL_AVAILABLE
        }
    
    def print_summary(self, top_n: int = 20):
        """Print a summary of resource usage to console."""
        from utils.logger import Console
        
        metrics = self.get_metrics()
        summary = self.get_summary()
        
        Console.info("[RESOURCE] === Resource Usage Summary ===")
        Console.info(f"[RESOURCE] Total duration: {summary.get('total_duration_s', 0):.2f}s")
        Console.info(f"[RESOURCE] Memory: start={summary.get('mem_start_mb', 0):.0f}MB, "
                    f"peak={summary.get('mem_peak_mb', 0):.0f}MB, "
                    f"delta={summary.get('mem_delta_mb', 0):+.0f}MB")
        
        # Sort by duration and show top N
        sorted_metrics = sorted(metrics, key=lambda x: x['duration_s'], reverse=True)[:top_n]
        
        Console.info(f"[RESOURCE] Top {min(top_n, len(sorted_metrics))} sections by duration:")
        for m in sorted_metrics:
            indent = "  " * m['depth']
            Console.info(f"[RESOURCE] {indent}{m['section']}: {m['duration_s']:.3f}s, "
                        f"mem={m['mem_delta_mb']:+.1f}MB, cpu={m['cpu_avg_pct']:.0f}%")
    
    def to_dataframe(self, equip_id: int, run_id: str) -> pd.DataFrame:
        """Convert metrics to DataFrame for SQL storage."""
        metrics = self.get_metrics()
        if not metrics:
            return pd.DataFrame()
        
        df = pd.DataFrame(metrics)
        df["EquipID"] = equip_id
        df["RunID"] = run_id
        df["CreatedAt"] = datetime.now()
        
        # Rename for SQL table
        df = df.rename(columns={
            "section": "SectionName",
            "duration_s": "DurationSeconds",
            "mem_start_mb": "MemStartMB",
            "mem_end_mb": "MemEndMB", 
            "mem_peak_mb": "MemPeakMB",
            "mem_delta_mb": "MemDeltaMB",
            "cpu_avg_pct": "CpuAvgPct",
            "cpu_samples": "CpuSampleCount",
            "depth": "SectionDepth",
            "threads": "ThreadCount"
        })
        
        return df
    
    def write_to_sql(self, output_manager, run_id: str, equip_id: int) -> bool:
        """
        Write metrics to SQL via output_manager.
        
        Args:
            output_manager: ACM OutputManager instance
            run_id: Current run ID
            equip_id: Equipment ID
            
        Returns:
            True if successful
        """
        if not self.enabled:
            return False
        
        try:
            df = self.to_dataframe(equip_id, run_id)
            if df.empty:
                return False
            
            output_manager.write_dataframe(
                df,
                path=None,  # SQL only
                sql_table="ACM_ResourceMetrics",
                add_created_at=False  # Already added
            )
            return True
        except Exception as e:
            from utils.logger import Console
            Console.warn(f"[RESOURCE] Failed to write metrics to SQL: {e}")
            return False
    
    def reset(self):
        """Reset all metrics for next run."""
        with self._lock:
            self._sections.clear()
            self._section_order.clear()
            self._active_stack.clear()
        self._run_start_time = 0.0
        self._run_start_mem = 0
        self._peak_mem_rss = 0


# Global singleton instance
R = ResourceMonitor()


def get_system_info() -> Dict[str, Any]:
    """Get static system information for context."""
    info = {
        "python_version": os.sys.version.split()[0],
        "platform": os.sys.platform,
        "psutil_available": PSUTIL_AVAILABLE
    }
    
    if PSUTIL_AVAILABLE:
        info.update({
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 1),
            "available_memory_gb": round(psutil.virtual_memory().available / (1024**3), 1)
        })
    
    return info
