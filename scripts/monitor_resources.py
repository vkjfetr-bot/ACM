"""
Resource Monitor for ACM Batch Processing
Tracks CPU, Memory, and Disk I/O in real-time.
"""

import psutil
import time
import csv
import sys
from datetime import datetime
from pathlib import Path

OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(exist_ok=True)


def get_python_processes():
    """Find all Python processes related to ACM."""
    acm_procs = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
        try:
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'acm' in cmdline.lower() or 'batch_runner' in cmdline.lower():
                    acm_procs.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return acm_procs


def monitor_resources(duration_seconds=600, interval=5):
    """Monitor system resources and ACM processes."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"resource_monitor_{timestamp}.csv"
    
    print(f"[MONITOR] Starting resource monitoring")
    print(f"[MONITOR] Duration: {duration_seconds}s, Interval: {interval}s")
    print(f"[MONITOR] Output: {csv_path}")
    print("-" * 80)
    
    # Headers
    headers = [
        'timestamp', 'elapsed_s', 
        'system_cpu_pct', 'system_mem_pct', 'system_mem_used_gb', 'system_mem_avail_gb',
        'disk_read_mb', 'disk_write_mb', 'disk_read_rate_mbps', 'disk_write_rate_mbps',
        'acm_proc_count', 'acm_cpu_pct', 'acm_mem_mb'
    ]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        start_time = time.time()
        prev_disk = psutil.disk_io_counters()
        prev_time = start_time
        
        while (time.time() - start_time) < duration_seconds:
            now = datetime.now()
            elapsed = time.time() - start_time
            
            # System metrics
            cpu_pct = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            
            # Disk I/O
            curr_disk = psutil.disk_io_counters()
            curr_time = time.time()
            time_delta = curr_time - prev_time
            
            disk_read_mb = curr_disk.read_bytes / (1024 * 1024)
            disk_write_mb = curr_disk.write_bytes / (1024 * 1024)
            
            read_rate = (curr_disk.read_bytes - prev_disk.read_bytes) / (1024 * 1024) / time_delta if time_delta > 0 else 0
            write_rate = (curr_disk.write_bytes - prev_disk.write_bytes) / (1024 * 1024) / time_delta if time_delta > 0 else 0
            
            prev_disk = curr_disk
            prev_time = curr_time
            
            # ACM process metrics
            acm_procs = get_python_processes()
            acm_count = len(acm_procs)
            acm_cpu = sum(p.cpu_percent() for p in acm_procs) if acm_procs else 0
            acm_mem = sum(p.memory_info().rss for p in acm_procs) / (1024 * 1024) if acm_procs else 0
            
            row = {
                'timestamp': now.isoformat(),
                'elapsed_s': round(elapsed, 1),
                'system_cpu_pct': round(cpu_pct, 1),
                'system_mem_pct': round(mem.percent, 1),
                'system_mem_used_gb': round(mem.used / (1024**3), 2),
                'system_mem_avail_gb': round(mem.available / (1024**3), 2),
                'disk_read_mb': round(disk_read_mb, 1),
                'disk_write_mb': round(disk_write_mb, 1),
                'disk_read_rate_mbps': round(read_rate, 2),
                'disk_write_rate_mbps': round(write_rate, 2),
                'acm_proc_count': acm_count,
                'acm_cpu_pct': round(acm_cpu, 1),
                'acm_mem_mb': round(acm_mem, 1)
            }
            
            writer.writerow(row)
            f.flush()
            
            # Console output
            print(f"[{now.strftime('%H:%M:%S')}] CPU={cpu_pct:5.1f}% | "
                  f"Mem={mem.percent:5.1f}% ({mem.used/(1024**3):.1f}GB) | "
                  f"Disk R/W={read_rate:.1f}/{write_rate:.1f} MB/s | "
                  f"ACM procs={acm_count}, CPU={acm_cpu:.0f}%, Mem={acm_mem:.0f}MB")
            
            time.sleep(interval)
    
    print("-" * 80)
    print(f"[MONITOR] Complete. Data saved to: {csv_path}")
    return csv_path


if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    monitor_resources(duration, interval)
