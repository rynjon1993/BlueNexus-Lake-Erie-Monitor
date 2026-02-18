"""
Lake Erie Historical Archive Builder
Bulk-fetches SST and chlorophyll data from NOAA GLERL ERDDAP into the
local SQLite database used by the Streamlit dashboard.

Dataset coverage (verified against ERDDAP metadata):
  SST (GLSEA_ACSPO_GCS):    2006-12-11 to present
  CHL (GR_Daily_CHL_test):   2012-03    to 2023-12

USAGE:
    python build_archive.py --quick      # 2023-2025 bloom season
    python build_archive.py --full       # Full historical archive
    python build_archive.py --resume     # Resume interrupted build
"""

import sys
import time
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import data_fetcher
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The GLSEA_ACSPO_GCS dataset starts on 2006-12-11.  Requesting months
# before this date always returns an error.  Starting from January 2007
# avoids 12 guaranteed-failure months at the beginning of the build.
SST_START_YEAR = 2007

# GR_Daily_CHL_test covers 2012-03 through 2023-12.
CHL_START_YEAR = 2012
CHL_END_YEAR = 2023

PROGRESS_FILE = Path(__file__).parent / "cache" / "build_progress.json"
LOG_FILE = Path(__file__).parent / "logs" / "archive_build.log"

LOG_FILE.parent.mkdir(exist_ok=True)
PROGRESS_FILE.parent.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Logging & progress
# ---------------------------------------------------------------------------

def log_message(message, level="INFO"):
    """Log to both console and file (UTF-8 safe on Windows)."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] [{level}] {message}"
    try:
        print(entry)
    except UnicodeEncodeError:
        print(entry.encode("ascii", errors="replace").decode("ascii"))
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(entry + "\n")
    except OSError:
        pass


def save_progress(data):
    """Persist progress state so the build can be resumed."""
    try:
        with open(PROGRESS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        log_message(f"Warning: Could not save progress: {e}", "WARNING")


def load_progress():
    """Load previously saved progress, or return None."""
    try:
        if PROGRESS_FILE.exists():
            with open(PROGRESS_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Config verification
# ---------------------------------------------------------------------------

def verify_config():
    """Ensure config.yaml has all keys the archive builder depends on."""
    log_message("Verifying configuration...")

    config = data_fetcher.CONFIG
    required_keys = {
        "bbox": ["lat_min", "lat_max", "lon_min", "lon_max"],
        "erddap": [
            "base_url", "sst_dataset", "chl_dataset",
            "retry_attempts", "retry_delay", "request_delay",
        ],
        "thresholds": ["western_basin_lon"],
    }

    missing = []
    for section, keys in required_keys.items():
        if section not in config:
            missing.append(f"Missing section: {section}")
        else:
            for key in keys:
                if key not in config[section]:
                    missing.append(f"Missing key: {section}.{key}")

    if missing:
        log_message("Configuration errors found:", "ERROR")
        for err in missing:
            log_message(f"  - {err}", "ERROR")
        return False

    if config["erddap"]["sst_dataset"] != "GLSEA_ACSPO_GCS":
        log_message(
            f"Wrong SST dataset ID: {config['erddap']['sst_dataset']} "
            f"(expected GLSEA_ACSPO_GCS)",
            "ERROR",
        )
        return False

    log_message("Configuration verified.")
    return True


def check_erddap_connectivity():
    """Verify the ERDDAP server is reachable before starting a long build.

    Returns True if the server responds, False otherwise.
    """
    erddap = data_fetcher.CONFIG["erddap"]
    url = f"{erddap['base_url']}/info/index.json"
    connect_timeout = erddap.get("connect_timeout", 30)

    log_message("Checking ERDDAP server connectivity...")
    try:
        session = data_fetcher._get_session()
        resp = session.get(url, timeout=(connect_timeout, 60))
        resp.raise_for_status()
        log_message("  ERDDAP server is reachable.")
        return True
    except Exception as e:
        log_message(f"  Cannot reach ERDDAP server: {e}", "ERROR")
        log_message("  Check your internet connection and try again.", "ERROR")
        return False


# ---------------------------------------------------------------------------
# Month-level fetch
# ---------------------------------------------------------------------------

def fetch_and_store_month(year, month):
    """
    Download and store SST + chlorophyll data for a single calendar month.

    Returns:
        dict with year, month, success flags, and record counts.
    """
    start_date = f"{year:04d}-{month:02d}-01"

    if month == 12:
        end_date_obj = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date_obj = datetime(year, month + 1, 1) - timedelta(days=1)

    # CRITICAL FIX: Cap end_date to yesterday if we're in the current month
    # This prevents requesting future dates that don't exist on ERDDAP yet
    now = datetime.now()
    if end_date_obj >= now:
        # Use yesterday's date to ensure data is available
        end_date_obj = now - timedelta(days=1)
        log_message(
            f"  Current month detected: capping end_date to {end_date_obj.strftime('%Y-%m-%d')} "
            f"(yesterday) to avoid requesting future data"
        )

    end_date = end_date_obj.strftime("%Y-%m-%d")

    result = {
        "year": year,
        "month": month,
        "sst_success": False,
        "chl_success": False,
        "sst_count": 0,
        "chl_count": 0,
    }

    config = data_fetcher.CONFIG
    erddap = config["erddap"]

    # ---- SST ----
    log_message(f"=== {year}-{month:02d}: SST ===")
    try:
        ds_sst = data_fetcher.download_erddap_data(
            erddap["sst_dataset"], "sst", start_date, end_date
        )
        if ds_sst is not None:
            sst_stats = data_fetcher.calculate_statistics(ds_sst, "sst")
            if sst_stats:
                data_fetcher.store_data("sst", sst_stats)
                data_fetcher.store_spatial_data("sst", ds_sst, "sst")
                result["sst_success"] = True
                result["sst_count"] = len(sst_stats)
                log_message(f"  Stored {len(sst_stats)} SST records")
            ds_sst.close()
    except Exception as e:
        log_message(f"  SST error: {e}", "ERROR")

    time.sleep(erddap["request_delay"])

    # ---- Chlorophyll (only within known dataset coverage) ----
    if CHL_START_YEAR <= year <= CHL_END_YEAR:
        log_message(f"=== {year}-{month:02d}: Chlorophyll ===")
        try:
            chl_variable = erddap.get("chl_variable", "chl")
            ds_chl = data_fetcher.download_erddap_data(
                erddap["chl_dataset"], chl_variable, start_date, end_date
            )
            if ds_chl is not None:
                chl_stats = data_fetcher.calculate_statistics(ds_chl, chl_variable)
                if chl_stats:
                    data_fetcher.store_data("chl", chl_stats)
                    data_fetcher.store_spatial_data("chl", ds_chl, chl_variable)
                    result["chl_success"] = True
                    result["chl_count"] = len(chl_stats)
                    log_message(f"  Stored {len(chl_stats)} Chl records")
                ds_chl.close()
        except Exception as e:
            log_message(f"  Chl error: {e}", "ERROR")

        time.sleep(erddap["request_delay"])

    # Clean up any orphaned temp files after each month
    data_fetcher.cleanup_temp_files()

    return result


# ---------------------------------------------------------------------------
# Build modes
# ---------------------------------------------------------------------------

def _generate_months(start_year, end_year, end_month=12):
    """Yield (year, month) tuples within a date range."""
    for year in range(start_year, end_year + 1):
        last_month = end_month if year == end_year else 12
        for month in range(1, last_month + 1):
            yield (year, month)


def build_quick():
    """Fetch 2023-2025 bloom season (May-Oct) — about 20 minutes."""
    log_message("=" * 70)
    log_message("BUILDING QUICK ARCHIVE (2023-2025 BLOOM SEASON)")
    log_message("=" * 70)

    data_fetcher.init_database()

    bloom_months = [5, 6, 7, 8, 9, 10]
    now = datetime.now()
    years = [2023, 2024, 2025]

    all_months = []
    for y in years:
        for m in bloom_months:
            # Skip future months
            if y > now.year or (y == now.year and m > now.month):
                continue
            all_months.append((y, m))

    total = len(all_months)
    success = 0

    for idx, (year, month) in enumerate(all_months, 1):
        log_message(f"\n[{idx}/{total}] Processing {year}-{month:02d}")
        result = fetch_and_store_month(year, month)
        if result["sst_success"] or result["chl_success"]:
            success += 1

    # Calculate bloom risk for ALL archived dates
    log_message("Calculating bloom risk scores for full archive...")
    data_fetcher.calculate_bloom_risk(all_dates=True)

    log_message("")
    log_message("=" * 70)
    log_message(f"BUILD COMPLETE: {success}/{total} months successful")
    log_message("=" * 70)


def build_full(start_year=None, end_year=None, resume=False):
    """
    Build the complete historical archive.

    Args:
        start_year: Defaults to SST_START_YEAR (2007).
        end_year:   Defaults to current year.
        resume:     If True, skip already-completed months.
    """
    if start_year is None:
        start_year = SST_START_YEAR
    if end_year is None:
        end_year = datetime.now().year

    # Don't request months that haven't happened yet
    now = datetime.now()
    end_month = now.month if end_year == now.year else 12

    log_message("=" * 70)
    log_message("BUILDING FULL HISTORICAL ARCHIVE")
    log_message("=" * 70)
    log_message(f"Date range: {start_year}-01 to {end_year}-{end_month:02d}")
    log_message(f"Resume mode: {resume}")
    log_message("=" * 70)

    data_fetcher.init_database()

    # Resume support
    completed = set()
    if resume:
        progress = load_progress()
        if progress:
            completed = set(tuple(m) for m in progress.get("completed", []))
            log_message(f"Resuming: {len(completed)} months already completed")

    all_months = [
        (y, m) for y, m in _generate_months(start_year, end_year, end_month)
        if (y, m) not in completed
    ]

    total = len(all_months)
    log_message(f"Months to fetch: {total}")

    if total == 0:
        log_message("Archive already complete!")
        return

    est_hours = total * 2.0 / 60
    log_message(f"Estimated time: {est_hours:.1f} hours")
    log_message("")
    log_message("Starting build... (Press Ctrl+C to pause)")
    log_message("=" * 70)

    success = 0
    failed = []
    consecutive_failures = 0
    base_request_delay = data_fetcher.CONFIG.get("erddap", {}).get("request_delay", 5)

    try:
        for idx, (year, month) in enumerate(all_months, 1):
            log_message(f"\n[{idx}/{total}] Processing {year}-{month:02d}")

            # Adaptive cooldown: if the server is struggling, slow down
            if consecutive_failures >= 5:
                cooldown = min(300, 30 * consecutive_failures)
                log_message(
                    f"  {consecutive_failures} consecutive failures detected. "
                    f"Cooling down for {cooldown}s to let the server recover..."
                )
                time.sleep(cooldown)
            elif consecutive_failures >= 2:
                cooldown = 15 * consecutive_failures
                log_message(f"  Extended delay: {cooldown}s (server may be throttling)")
                time.sleep(cooldown)

            try:
                result = fetch_and_store_month(year, month)

                if result["sst_success"] or result["chl_success"]:
                    success += 1
                    consecutive_failures = 0  # Reset on success
                    completed.add((year, month))
                    save_progress({
                        "completed": [list(m) for m in completed],
                        "last_update": datetime.now().isoformat(),
                        "total_success": success,
                    })
                else:
                    consecutive_failures += 1
                    failed.append((year, month))

                if idx % 10 == 0:
                    elapsed_pct = idx / total * 100
                    log_message(
                        f"\n=== CHECKPOINT: {success}/{idx} successful "
                        f"({elapsed_pct:.0f}% complete) ===\n"
                    )

            except Exception as e:
                log_message(f"Month failed: {e}", "ERROR")
                consecutive_failures += 1
                failed.append((year, month))

            # Standard inter-request delay
            time.sleep(base_request_delay)

    except KeyboardInterrupt:
        log_message("")
        log_message("=" * 70)
        log_message("BUILD PAUSED (Ctrl+C)")
        log_message(f"Progress: {success} months completed so far")
        log_message("Resume with: python build_archive.py --resume")
        log_message("=" * 70)
        return

    # Post-build: calculate bloom risk for ALL dates in a single pass
    log_message("Calculating bloom risk scores for full archive...")
    data_fetcher.calculate_bloom_risk(all_dates=True)

    log_message("")
    log_message("=" * 70)
    log_message("BUILD COMPLETE!")
    log_message("=" * 70)
    log_message(f"Successful: {success}/{total}")
    log_message(f"Failed: {len(failed)}/{total}")

    if failed and len(failed) <= 30:
        log_message("")
        log_message("Failed months (can retry with --resume):")
        for year, month in sorted(failed):
            log_message(f"  {year}-{month:02d}")

    if failed:
        log_message("")
        log_message("Tip: Run 'python build_archive.py --resume' to retry failed months.")

    log_message("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build Lake Erie historical data archive"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick build: 2023-2025 bloom season",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Full build: 2007-present, all months",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume an interrupted full build",
    )
    parser.add_argument(
        "--start-year", type=int, default=SST_START_YEAR,
        help=f"Start year for full build (default: {SST_START_YEAR})",
    )
    parser.add_argument("--end-year", type=int, help="End year for full build")

    args = parser.parse_args()

    if not verify_config():
        log_message("BUILD FAILED -- fix config.yaml and retry.", "ERROR")
        sys.exit(1)

    if not check_erddap_connectivity():
        sys.exit(1)

    if args.quick:
        build_quick()
    elif args.full or args.resume:
        build_full(args.start_year, args.end_year, resume=args.resume)
    else:
        print()
        print("Lake Erie Historical Archive Builder")
        print("=" * 70)
        print()
        print("Build Options:")
        print(f"  1. Quick (2023-2025 bloom season)")
        print(f"  2. Full  ({SST_START_YEAR}-present, all months)")
        print( "  3. Resume interrupted build")
        print()
        choice = input("Select [1-3]: ").strip()

        if choice == "1":
            build_quick()
        elif choice == "2":
            build_full()
        elif choice == "3":
            build_full(resume=True)
        else:
            print("Invalid choice.")
            sys.exit(1)


if __name__ == "__main__":
    main()
