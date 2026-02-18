#!/usr/bin/env python3
"""
Health Check Script - Monitor application health
Run periodically via cron or scheduler
"""

import subprocess
import time
import torch
import json
from pathlib import Path
from analytics import analytics
import psutil
import socket

def check_health():
    """Perform health check"""
    
    print("🏥 Starting health check...")
    start_time = time.time()
    
    # Initialize status
    is_healthy = True
    model_loaded = False
    database_ok = False
    disk_space_mb = 0
    
    # 1. Check model
    print("  ✓ Checking model...")
    try:
        model_path = Path("model_weights.pth")
        if model_path.exists():
            # Try to load model
            weights = torch.load(model_path, map_location="cpu")
            model_loaded = True
            print("    ✓ Model loaded successfully")
        else:
            print("    ⚠️ Model file not found")
            is_healthy = False
    except Exception as e:
        print(f"    ❌ Error loading model: {e}")
        is_healthy = False
    
    # 2. Check database
    print("  ✓ Checking database...")
    try:
        db_path = Path("data/analytics/analytics.db")
        if db_path.exists():
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sessions")
            count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM classifications")
            classifications = cursor.fetchone()[0]
            conn.close()
            database_ok = True
            print(f"    ✓ Database OK ({count} sessions, {classifications} classifications)")
        else:
            print("    ⚠️ Database not found")
    except Exception as e:
        print(f"    ❌ Database error: {e}")
        is_healthy = False
    
    # 3. Check disk space
    print("  ✓ Checking disk space...")
    try:
        disk = psutil.disk_usage('/')
        disk_space_mb = disk.free / (1024*1024)
        if disk.percent > 90:
            print(f"    ⚠️ Low disk space: {disk.percent}%")
            is_healthy = False
        else:
            print(f"    ✓ Disk space OK: {disk_space_mb:.0f} MB available")
    except Exception as e:
        print(f"    ❌ Disk check error: {e}")
    
    # 4. Check data directories
    print("  ✓ Checking data directories...")
    required_dirs = [
        "data/user_submissions",
        "data/bristol_stool_dataset",
        "data/analytics"
    ]
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"    ✓ {dir_path}")
        else:
            print(f"    ⚠️ Missing: {dir_path}")
            is_healthy = False
    
    # Calculate response time
    response_time_ms = (time.time() - start_time) * 1000
    
    # Log health check
    status = "healthy" if is_healthy else "degraded"
    
    try:
        analytics.log_health_check(
            status=status,
            model_loaded=model_loaded,
            database_ok=database_ok,
            disk_space_mb=disk_space_mb,
            response_time_ms=response_time_ms,
            details=f"Model: {model_loaded}, DB: {database_ok}, Disk: {disk_space_mb:.0f}MB"
        )
        print(f"\n✅ Health check complete: {status.upper()}")
        print(f"   Response time: {response_time_ms:.0f}ms")
    except Exception as e:
        print(f"❌ Failed to log health check: {e}")
    
    return is_healthy

def run_continuous_monitoring(interval_seconds=300):
    """
    Run continuous health monitoring
    
    Args:
        interval_seconds: Interval between checks (default 5 minutes)
    """
    print(f"🔄 Starting continuous monitoring (check every {interval_seconds}s)...")
    
    try:
        while True:
            check_health()
            print(f"\n⏳ Next check in {interval_seconds}s...\n")
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("\n✋ Monitoring stopped")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        # Run continuous monitoring
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 300
        run_continuous_monitoring(interval)
    else:
        # Run single health check
        is_healthy = check_health()
        sys.exit(0 if is_healthy else 1)
