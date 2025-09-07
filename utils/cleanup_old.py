#!/usr/bin/env python
"""
Clean up old JTBD reports
"""
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import argparse

def cleanup_reports(days_to_keep=7, dry_run=True):
    """Remove reports older than specified days"""
    
    reports_dir = Path("reports")
    if not reports_dir.exists():
        print("No reports directory found")
        return
    
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    
    removed_count = 0
    kept_count = 0
    
    for date_dir in reports_dir.iterdir():
        if not date_dir.is_dir():
            continue
            
        try:
            # Parse date from folder name (YYYY-MM-DD format)
            dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d")
            
            if dir_date < cutoff_date:
                if dry_run:
                    print(f"Would remove: {date_dir}")
                else:
                    shutil.rmtree(date_dir)
                    print(f"Removed: {date_dir}")
                removed_count += 1
            else:
                kept_count += 1
                
        except ValueError:
            # Skip directories that don't match date format
            print(f"Skipping non-date directory: {date_dir.name}")
    
    print(f"\nSummary:")
    print(f"  Kept: {kept_count} days")
    print(f"  {'Would remove' if dry_run else 'Removed'}: {removed_count} days")
    
    if dry_run:
        print("\nThis was a dry run. Use --execute to actually delete.")

def main():
    parser = argparse.ArgumentParser(description='Clean up old JTBD reports')
    parser.add_argument('--days', type=int, default=7, 
                        help='Keep reports from last N days (default: 7)')
    parser.add_argument('--execute', action='store_true',
                        help='Actually delete files (default: dry run)')
    
    args = parser.parse_args()
    
    cleanup_reports(days_to_keep=args.days, dry_run=not args.execute)

if __name__ == "__main__":
    main()