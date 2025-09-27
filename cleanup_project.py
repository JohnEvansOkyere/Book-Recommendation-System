#!/usr/bin/env python3
"""
Project Cleanup Script
====================

This script helps maintain project organization by:
- Moving files to appropriate directories
- Cleaning up temporary files
- Archiving old logs and reports

Usage: python cleanup_project.py
"""

import os
import shutil
import glob
from datetime import datetime, timedelta
from pathlib import Path

def cleanup_temp_files():
    """Remove temporary files"""
    temp_patterns = ['*.tmp', '*.temp', '*~', '*.bak']
    for pattern in temp_patterns:
        for file in glob.glob(pattern):
            os.remove(file)
            print(f"Removed: {file}")

def archive_old_logs():
    """Archive logs older than 30 days"""
    log_dir = Path('logs')
    if not log_dir.exists():
        return
    
    cutoff_date = datetime.now() - timedelta(days=30)
    
    for log_file in log_dir.rglob('*.log'):
        if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff_date:
            archive_dir = Path('backup/logs')
            archive_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(log_file), str(archive_dir / log_file.name))
            print(f"Archived: {log_file}")

def organize_new_files():
    """Organize any new files that might have been created"""
    # Move new PNG files to visualizations
    for png_file in glob.glob('*.png'):
        dest = Path('images/visualizations') / png_file
        Path('images/visualizations').mkdir(parents=True, exist_ok=True)
        shutil.move(png_file, str(dest))
        print(f"Moved: {png_file} → images/visualizations/")
    
    # Move new report files
    for report_file in glob.glob('*_report_*.txt'):
        dest = Path('reports/performance') / report_file
        Path('reports/performance').mkdir(parents=True, exist_ok=True)
        shutil.move(report_file, str(dest))
        print(f"Moved: {report_file} → reports/performance/")

if __name__ == "__main__":
    print("🧹 Starting project cleanup...")
    cleanup_temp_files()
    archive_old_logs()
    organize_new_files()
    print("✅ Cleanup completed!")