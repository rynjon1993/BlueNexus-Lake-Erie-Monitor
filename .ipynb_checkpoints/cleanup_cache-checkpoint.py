"""
One-time cleanup script for 03_realtime_monitor/cache/
Run this ONCE to remove orphaned temp .nc files from failed downloads.

Usage:
    cd E:\BlueNexus\03_realtime_monitor
    python cleanup_cache.py
"""

from pathlib import Path

cache_dir = Path(__file__).parent / "cache"

if not cache_dir.exists():
    print("No cache directory found. Nothing to clean.")
    exit()

# Remove orphaned temp .nc files
temp_files = list(cache_dir.glob("temp_*.nc"))
print(f"Found {len(temp_files)} orphaned temp files")

for f in temp_files:
    try:
        f.unlink()
        print(f"  Deleted: {f.name}")
    except Exception as e:
        print(f"  Could not delete {f.name}: {e}")

print(f"\nDone. Cleaned {len(temp_files)} files.")
print("You can delete this script after running it.")