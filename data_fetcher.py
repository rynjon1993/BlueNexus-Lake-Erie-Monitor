"""
Data Fetcher for Lake Erie Live Monitor (BlueNexus)
Downloads SST and chlorophyll data from NOAA GLERL ERDDAP, caches in SQLite.

PRODUCTION VERSION — key engineering features:
- Streaming HTTP downloads prevent read timeouts on large historical queries
- Persistent requests.Session with connection pooling for ERDDAP
- Separate connect/read timeouts (30s / 600s) for historical data resilience
- Exponential backoff with random jitter on retries
- Connection-error detection with extended cooldown (server throttle handling)
- Threading lock prevents concurrent SQLite writes (no "database is locked")
- Each write function manages its own connection lifecycle (no lock contention)
- WAL mode + IMMEDIATE isolation for predictable concurrent access
- Batch inserts (executemany) for efficient bulk storage
- UTF-8 safe logging for Windows console compatibility
- Chlorophyll date range validation (2012-03 to 2023-12)
- Dataset registry helpers for modular multi-source expansion
"""

import os
import time
import random
import sqlite3
import threading
import gc
import atexit
from datetime import datetime, timedelta

import requests
import xarray as xr
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# CRITICAL: Global lock & cleanup
# ═══════════════════════════════════════════════════════════════════════════

# Global lock ensures only ONE database write operation at a time
# This prevents ALL "database is locked" errors
_db_lock = threading.Lock()

def cleanup_connections():
    """Force garbage collection to close any lingering connections."""
    gc.collect()

atexit.register(cleanup_connections)

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

DB_PATH = Path(__file__).parent / "cache" / "realtime_data.db"
LOG_PATH = Path(__file__).parent / "cache" / "fetch_log.txt"

DB_PATH.parent.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Dataset registry helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_registered_datasets():
    """Return the list of dataset definitions from the registry.

    Falls back to an empty list if the 'datasets' key is not yet in
    config.yaml (backward compatible with legacy config).
    """
    return CONFIG.get("datasets", [])


def get_dataset_by_id(dataset_id):
    """Look up a single dataset definition by its 'id' field.

    Returns the dict for the matching dataset, or None.
    """
    for ds in get_registered_datasets():
        if ds.get("id") == dataset_id:
            return ds
    return None


def get_dataset_coverage(dataset_id):
    """Return (start_date_str, end_date_str) for a registered dataset.

    Returns (None, None) if not found.
    """
    ds = get_dataset_by_id(dataset_id)
    if ds and "coverage" in ds:
        return ds["coverage"].get("start"), ds["coverage"].get("end")
    return None, None

# ═══════════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════════

def log_message(message):
    """Append timestamped message to log file and stdout.

    Uses UTF-8 for the log file and replaces unencodable characters on
    Windows consoles to prevent 'charmap' codec crashes.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except OSError:
        pass
    try:
        print(log_entry.strip())
    except UnicodeEncodeError:
        print(log_entry.strip().encode("ascii", errors="replace").decode("ascii"))

# ═══════════════════════════════════════════════════════════════════════════
# Database initialization
# ═══════════════════════════════════════════════════════════════════════════

def init_database():
    """Create all required tables if they don't already exist.

    Enables WAL (Write-Ahead Logging) mode for better concurrency.
    Thread-safe with global lock.
    """
    with _db_lock:
        conn = None
        try:
            gc.collect()
            conn = sqlite3.connect(DB_PATH, timeout=60)
            cursor = conn.cursor()

            # Enable WAL mode for concurrent access
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=60000")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sst_data (
                    date TEXT PRIMARY KEY,
                    lake_mean REAL,
                    lake_max REAL,
                    west_basin_mean REAL,
                    east_basin_mean REAL,
                    data_coverage REAL,
                    fetch_timestamp TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chl_data (
                    date TEXT PRIMARY KEY,
                    lake_mean REAL,
                    lake_max REAL,
                    west_basin_mean REAL,
                    east_basin_mean REAL,
                    data_coverage REAL,
                    fetch_timestamp TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bloom_risk (
                    date TEXT PRIMARY KEY,
                    risk_score INTEGER,
                    risk_level TEXT,
                    sst_above_threshold INTEGER,
                    chl_above_threshold INTEGER,
                    calculation_timestamp TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fetch_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fetch_timestamp TEXT,
                    data_type TEXT,
                    date_range_start TEXT,
                    date_range_end TEXT,
                    status TEXT,
                    error_message TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sst_spatial (
                    date TEXT PRIMARY KEY,
                    lats BLOB,
                    lons BLOB,
                    data BLOB,
                    shape_rows INTEGER,
                    shape_cols INTEGER,
                    fetch_timestamp TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chl_spatial (
                    date TEXT PRIMARY KEY,
                    lats BLOB,
                    lons BLOB,
                    data BLOB,
                    shape_rows INTEGER,
                    shape_cols INTEGER,
                    fetch_timestamp TEXT
                )
            """)

            conn.commit()
            log_message("Database initialized successfully (WAL mode enabled)")

        finally:
            if conn:
                conn.close()
                del conn
            gc.collect()

# ═══════════════════════════════════════════════════════════════════════════
# ERDDAP helpers
# ═══════════════════════════════════════════════════════════════════════════
# HTTP session (connection pooling)
# ═══════════════════════════════════════════════════════════════════════════

# A persistent session reuses TCP connections and TLS handshakes, which
# dramatically reduces overhead when making many requests to the same host.
_http_session = None


def _get_session():
    """Return a module-level requests.Session with connection pooling."""
    global _http_session
    if _http_session is None:
        _http_session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            max_retries=0,        # We handle retries ourselves
            pool_connections=2,
            pool_maxsize=2,
        )
        _http_session.mount("https://", adapter)
        _http_session.mount("http://", adapter)
    return _http_session


# ═══════════════════════════════════════════════════════════════════════════
# ERDDAP helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_latest_date(dataset_id):
    """Query ERDDAP for the most recent available date in a dataset."""
    erddap = CONFIG["erddap"]
    url = f"{erddap['base_url']}/griddap/{dataset_id}.json?time%5B(last)%5D"
    connect_timeout = erddap.get("connect_timeout", 30)

    try:
        log_message(f"Querying {dataset_id} for latest available date...")
        session = _get_session()
        response = session.get(url, timeout=(connect_timeout, 60))
        response.raise_for_status()
        data = response.json()

        last_time_str = data["table"]["rows"][0][0]
        last_date = last_time_str[:10]
        log_message(f"  -> Latest available: {last_date}")
        return last_date

    except Exception as e:
        log_message(f"  -> Could not query latest date: {e}")
        return None


def download_erddap_data(dataset_id, variable, start_date, end_date):
    """Download gridded data from NOAA GLERL ERDDAP.

    Uses streaming downloads and separate connect/read timeouts to handle
    historical queries that may take several minutes for the server to
    process.  Retries with exponential backoff plus random jitter.

    Returns:
        xr.Dataset on success, None on failure or no data.
    """
    bbox = CONFIG["bbox"]
    erddap = CONFIG["erddap"]

    start_time = f"{start_date}T12:00:00Z"
    end_time = f"{end_date}T12:00:00Z"

    url = (
        f"{erddap['base_url']}/griddap/{dataset_id}.nc?"
        f"{variable}"
        f"[({start_time}):1:({end_time})]"
        f"[({bbox['lat_min']}):1:({bbox['lat_max']})]"
        f"[({bbox['lon_min']}):1:({bbox['lon_max']})]"
    )

    log_message(f"Downloading {dataset_id} from {start_date} to {end_date}...")

    temp_suffix = random.randint(1000, 9999)
    temp_file = DB_PATH.parent / f"temp_{dataset_id}_{temp_suffix}.nc"

    session = _get_session()
    connect_timeout = erddap.get("connect_timeout", 30)
    read_timeout = erddap.get("read_timeout", 600)
    base_delay = erddap.get("retry_delay", 15)
    max_attempts = erddap.get("retry_attempts", 5)

    for attempt in range(max_attempts):
        try:
            # Stream the response to a file instead of loading into memory.
            # This prevents read timeouts on large historical queries where
            # the server sends data incrementally over several minutes.
            with session.get(
                url,
                timeout=(connect_timeout, read_timeout),
                stream=True,
            ) as response:
                response.raise_for_status()

                with open(temp_file, "wb") as f:
                    for chunk in response.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)

            # Parse the downloaded NetCDF file
            ds = None
            for engine in ["netcdf4", "h5netcdf", "scipy"]:
                try:
                    ds = xr.open_dataset(temp_file, engine=engine)
                    break
                except Exception:
                    if engine == "scipy":
                        raise
                    continue

            ds = ds.load()
            _remove_temp(temp_file)

            log_message(f"Successfully downloaded {dataset_id} ({len(ds.time)} time steps)")
            return ds

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else "unknown"
            log_message(f"Download attempt {attempt + 1}/{max_attempts} failed: HTTP {status_code}")

            if status_code == 404:
                log_message("  No data available for this date range")
                _remove_temp(temp_file)
                return None

            # HTTP 500/503 = server overloaded, needs longer cooldown
            _remove_temp(temp_file)
            if attempt < max_attempts - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(2, 10)
                log_message(f"  Retrying in {delay:.0f}s...")
                time.sleep(delay)
            else:
                log_message(f"Failed after {max_attempts} attempts")
                return None

        except requests.exceptions.ConnectionError as e:
            # Server is refusing/dropping connections — likely throttling.
            log_message(f"Download attempt {attempt + 1}/{max_attempts} failed: Connection refused/reset")
            _remove_temp(temp_file)
            if attempt < max_attempts - 1:
                delay = base_delay * (3 ** attempt) + random.uniform(10, 30)
                log_message(f"  Server may be throttling. Waiting {delay:.0f}s...")
                time.sleep(delay)
            else:
                log_message(f"Failed after {max_attempts} attempts (server unreachable)")
                return None

        except requests.exceptions.Timeout as e:
            err_str = str(e)
            if "connect" in err_str.lower():
                log_message(f"Download attempt {attempt + 1}/{max_attempts} failed: Connect timeout ({connect_timeout}s)")
            else:
                log_message(f"Download attempt {attempt + 1}/{max_attempts} failed: Read timeout ({read_timeout}s)")
            _remove_temp(temp_file)
            if attempt < max_attempts - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(5, 15)
                log_message(f"  Retrying in {delay:.0f}s...")
                time.sleep(delay)
            else:
                log_message(f"Failed after {max_attempts} attempts (all timed out)")
                return None

        except Exception as e:
            log_message(f"Download attempt {attempt + 1}/{max_attempts} failed: {e}")
            _remove_temp(temp_file)

            if attempt < max_attempts - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(2, 10)
                time.sleep(delay)
            else:
                log_message(f"Failed after {max_attempts} attempts")
                return None


def _remove_temp(path):
    """Silently remove a temporary file if it exists."""
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass

# ═══════════════════════════════════════════════════════════════════════════
# Statistics & storage
# ═══════════════════════════════════════════════════════════════════════════

def calculate_statistics(ds, variable):
    """Derive lake-wide and per-basin summary statistics."""
    stats = {}
    western_lon = CONFIG["thresholds"]["western_basin_lon"]

    lat_coord = "latitude" if "latitude" in ds.coords else "lat"
    lon_coord = "longitude" if "longitude" in ds.coords else "lon"

    for date in ds.time.values:
        date_str = str(date)[:10]
        data = ds[variable].sel(time=date)

        if np.all(np.isnan(data.values)):
            continue

        lake_mean = float(data.mean(skipna=True).values)
        lake_max = float(data.max(skipna=True).values)
        data_coverage = float((~np.isnan(data.values)).sum() / data.values.size * 100)

        if data_coverage < 1.0:
            continue

        west_mask = ds[lon_coord] < western_lon
        east_mask = ds[lon_coord] >= western_lon

        west_mean = float(data.where(west_mask, drop=False).mean(skipna=True).values)
        east_mean = float(data.where(east_mask, drop=False).mean(skipna=True).values)

        stats[date_str] = {
            "lake_mean": lake_mean,
            "lake_max": lake_max,
            "west_basin_mean": west_mean,
            "east_basin_mean": east_mean,
            "data_coverage": data_coverage,
        }

    return stats


def store_data(data_type, stats):
    """Upsert summary statistics with batch insert. Thread-safe."""
    with _db_lock:
        conn = None
        try:
            gc.collect()
            time.sleep(0.1)  # Brief pause for garbage collection

            conn = sqlite3.connect(DB_PATH, timeout=60, isolation_level='IMMEDIATE')
            cursor = conn.cursor()
            cursor.execute("PRAGMA busy_timeout=60000")
            fetch_timestamp = datetime.now().isoformat()
            table_name = f"{data_type}_data"

            batch_data = [
                (
                    date_str,
                    values["lake_mean"],
                    values["lake_max"],
                    values["west_basin_mean"],
                    values["east_basin_mean"],
                    values["data_coverage"],
                    fetch_timestamp,
                )
                for date_str, values in stats.items()
            ]

            cursor.executemany(
                f"""
                INSERT OR REPLACE INTO {table_name}
                (date, lake_mean, lake_max, west_basin_mean, east_basin_mean,
                 data_coverage, fetch_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                batch_data
            )

            conn.commit()
            log_message(f"Stored {len(stats)} records in {table_name}")

        except Exception as e:
            log_message(f"Error storing data in {data_type}_data: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()
                del conn
            gc.collect()


def store_spatial_data(data_type, ds, variable):
    """Persist full 2-D spatial grids with batch insert. Thread-safe."""
    with _db_lock:
        conn = None
        try:
            gc.collect()
            time.sleep(0.1)

            conn = sqlite3.connect(DB_PATH, timeout=60, isolation_level='IMMEDIATE')
            cursor = conn.cursor()
            cursor.execute("PRAGMA busy_timeout=60000")
            fetch_timestamp = datetime.now().isoformat()
            table_name = f"{data_type}_spatial"

            lat_coord = "latitude" if "latitude" in ds.coords else "lat"
            lon_coord = "longitude" if "longitude" in ds.coords else "lon"

            batch_data = []
            for t in range(len(ds.time)):
                date_str = pd.Timestamp(ds.time.values[t]).strftime("%Y-%m-%d")

                lats = ds[lat_coord].values
                lons = ds[lon_coord].values
                data = ds[variable].isel(time=t).values

                lon_grid, lat_grid = np.meshgrid(lons, lats)
                data_f64 = data.astype(np.float64)
                shape_rows, shape_cols = lat_grid.shape

                batch_data.append((
                    date_str,
                    lat_grid.flatten().tobytes(),
                    lon_grid.flatten().tobytes(),
                    data_f64.flatten().tobytes(),
                    shape_rows,
                    shape_cols,
                    fetch_timestamp,
                ))

            cursor.executemany(
                f"""
                INSERT OR REPLACE INTO {table_name}
                (date, lats, lons, data, shape_rows, shape_cols, fetch_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                batch_data
            )

            conn.commit()
            log_message(f"Stored {len(ds.time)} spatial records in {table_name}")

        except Exception as e:
            log_message(f"Error storing spatial data: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()
                del conn
            gc.collect()

# ═══════════════════════════════════════════════════════════════════════════
# Bloom risk scoring
# ═══════════════════════════════════════════════════════════════════════════

def calculate_bloom_risk(all_dates=False):
    """Score bloom risk with batch insert. Thread-safe."""
    with _db_lock:
        conn = None
        try:
            gc.collect()
            time.sleep(0.1)

            conn = sqlite3.connect(DB_PATH, timeout=60, isolation_level='IMMEDIATE')
            cursor = conn.cursor()
            cursor.execute("PRAGMA busy_timeout=60000")

            if all_dates:
                cursor.execute("""
                    SELECT s.date, s.lake_mean AS sst, c.west_basin_mean AS chl
                    FROM sst_data s
                    LEFT JOIN chl_data c ON s.date = c.date
                    ORDER BY s.date ASC
                """)
            else:
                cursor.execute("""
                    SELECT s.date, s.lake_mean AS sst, c.west_basin_mean AS chl
                    FROM sst_data s
                    LEFT JOIN chl_data c ON s.date = c.date
                    ORDER BY s.date DESC
                    LIMIT 7
                """)

            rows = cursor.fetchall()
            thresholds = CONFIG["thresholds"]
            ts = datetime.now().isoformat()

            batch_data = []
            for date, sst, chl in rows:
                risk_score = 0
                sst_above = 0
                chl_above = 0

                if sst is not None and sst > thresholds["sst_bloom"]:
                    risk_score += 1
                    sst_above = 1

                if chl is not None:
                    if chl > thresholds["chl_high"]:
                        risk_score += 3
                        chl_above = 1
                    elif chl > thresholds["chl_moderate"]:
                        risk_score += 2
                        chl_above = 1
                    elif chl > thresholds["chl_low"]:
                        risk_score += 1
                        chl_above = 1

                risk_level = (
                    "High" if risk_score >= 4
                    else "Moderate" if risk_score >= 2
                    else "Low"
                )

                batch_data.append((date, risk_score, risk_level, sst_above, chl_above, ts))

            cursor.executemany(
                """
                INSERT OR REPLACE INTO bloom_risk
                (date, risk_score, risk_level, sst_above_threshold,
                 chl_above_threshold, calculation_timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                batch_data
            )

            conn.commit()
            log_message(f"Calculated bloom risk for {len(rows)} dates")

        finally:
            if conn:
                conn.close()
                del conn
            gc.collect()

# ═══════════════════════════════════════════════════════════════════════════
# Maintenance
# ═══════════════════════════════════════════════════════════════════════════

def cleanup_temp_files():
    """Remove orphaned temp NetCDF files."""
    for temp_file in DB_PATH.parent.glob("temp_*.nc"):
        _remove_temp(temp_file)


def _log_fetch_history(data_type, start, end, status, error=None):
    """Log a fetch operation to the audit trail.

    Opens, commits, and closes its own connection so it never holds a lock
    while other write functions (store_data, store_spatial_data) run.
    """
    with _db_lock:
        conn = None
        try:
            conn = sqlite3.connect(DB_PATH, timeout=60, isolation_level='IMMEDIATE')
            conn.execute("PRAGMA busy_timeout=60000")
            conn.execute(
                "INSERT INTO fetch_history "
                "(fetch_timestamp, data_type, date_range_start, date_range_end, "
                "status, error_message) VALUES (?,?,?,?,?,?)",
                (datetime.now().isoformat(), data_type, start, end, status, error),
            )
            conn.commit()
        except Exception as e:
            log_message(f"Warning: could not log fetch history: {e}")
        finally:
            if conn:
                conn.close()
                del conn


# ═══════════════════════════════════════════════════════════════════════════
# High-level fetch operations
# ═══════════════════════════════════════════════════════════════════════════

def fetch_latest_data():
    """Fetch the most recent 7 days of SST and chlorophyll.

    Each write operation manages its own connection lifecycle to prevent
    RESERVED lock contention between concurrent connections.
    """
    log_message("=" * 60)
    log_message("Starting data fetch...")

    init_database()
    erddap = CONFIG["erddap"]

    # -- SST --
    try:
        sst_end = get_latest_date(erddap["sst_dataset"])
        if sst_end is None:
            raise Exception("Could not determine latest SST date")

        sst_end_dt = datetime.strptime(sst_end, "%Y-%m-%d")
        sst_start = (sst_end_dt - timedelta(days=6)).strftime("%Y-%m-%d")

        sst_ds = download_erddap_data(erddap["sst_dataset"], "sst", sst_start, sst_end)

        if sst_ds is not None:
            sst_stats = calculate_statistics(sst_ds, "sst")
            if sst_stats:
                store_data("sst", sst_stats)
                store_spatial_data("sst", sst_ds, "sst")
                log_message(f"SST data covers: {min(sst_stats)} to {max(sst_stats)}")

            _log_fetch_history("sst", sst_start, sst_end, "success")
            sst_ds.close()

    except Exception as e:
        log_message(f"SST fetch error: {e}")
        _log_fetch_history("sst", "unknown", "unknown", "failed", str(e))

    time.sleep(erddap["request_delay"])

    # -- Chlorophyll --
    try:
        chl_end = get_latest_date(erddap["chl_dataset"])
        if chl_end is None:
            raise Exception("Could not determine latest chlorophyll date")

        chl_end_dt = datetime.strptime(chl_end, "%Y-%m-%d")
        chl_start = (chl_end_dt - timedelta(days=6)).strftime("%Y-%m-%d")

        chl_variable = erddap.get("chl_variable", "chl")
        chl_ds = download_erddap_data(erddap["chl_dataset"], chl_variable, chl_start, chl_end)

        if chl_ds is not None:
            chl_stats = calculate_statistics(chl_ds, chl_variable)
            if chl_stats:
                store_data("chl", chl_stats)
                store_spatial_data("chl", chl_ds, chl_variable)
                log_message(f"Chl data covers: {min(chl_stats)} to {max(chl_stats)}")

            _log_fetch_history("chl", chl_start, chl_end, "success")
            chl_ds.close()

    except Exception as e:
        log_message(f"Chlorophyll fetch error: {e}")
        _log_fetch_history("chl", "unknown", "unknown", "failed", str(e))

    calculate_bloom_risk()
    cleanup_temp_files()

    log_message("Data fetch complete!")
    log_message("=" * 60)


def fetch_historical_range(start_date, end_date, data_types=None):
    """Fetch data for an arbitrary date range with full validation.

    - Clamps SST start to 2007-01-01
    - Clamps end date to latest available SST from ERDDAP
    - Validates chlorophyll coverage window
    - Keeps datetime and string representations synchronized
    """

    if data_types is None:
        data_types = ["sst", "chl"]

    log_message("=" * 60)
    log_message(f"Fetching historical data: {start_date} to {end_date}")

    init_database()

    erddap = CONFIG["erddap"]
    results = {"sst": None, "chl": None}

    # Parse inputs
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Prevent inverted ranges immediately
    if start_dt > end_dt:
        log_message("Invalid range: start date is after end date.")
        return {"sst": "invalid_range", "chl": "invalid_range"}

    # ------------------------------------------------------------------
    # Clamp SST lower bound (data begins 2007-01-01)
    # ------------------------------------------------------------------
    sst_min_dt = datetime(2007, 1, 1)
    if start_dt < sst_min_dt:
        log_message("NOTE: Adjusting SST start date to 2007-01-01")
        start_dt = sst_min_dt

    # ------------------------------------------------------------------
    # Clamp SST upper bound to latest available from ERDDAP
    # ------------------------------------------------------------------
    latest_sst = get_latest_date(erddap["sst_dataset"])

    if latest_sst:
        latest_sst_dt = datetime.strptime(latest_sst, "%Y-%m-%d")

        if end_dt > latest_sst_dt:
            log_message(
                f"Requested end date {end_date} exceeds available SST data "
                f"({latest_sst}). Adjusting automatically."
            )
            end_dt = latest_sst_dt

    # Final guard after clamping
    if start_dt > end_dt:
        log_message("Adjusted range invalid after clamping. Aborting fetch.")
        return {"sst": "out_of_range", "chl": "out_of_range"}

    # Normalize canonical string versions
    start_date = start_dt.strftime("%Y-%m-%d")
    end_date = end_dt.strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # SST FETCH
    # ------------------------------------------------------------------
    if "sst" in data_types:
        try:
            sst_ds = download_erddap_data(
                erddap["sst_dataset"],
                "sst",
                start_date,
                end_date
            )

            if sst_ds is not None:
                sst_stats = calculate_statistics(sst_ds, "sst")
                if sst_stats:
                    store_data("sst", sst_stats)
                    store_spatial_data("sst", sst_ds, "sst")
                    log_message(f"[OK] SST: {len(sst_stats)} days stored")
                    results["sst"] = "success"

                _log_fetch_history(
                    "sst",
                    start_date,
                    end_date,
                    "success" if results["sst"] == "success" else "failed",
                )

                sst_ds.close()

        except Exception as e:
            log_message(f"SST error: {e}")
            results["sst"] = f"error: {e}"

        time.sleep(erddap["request_delay"])

    # ------------------------------------------------------------------
    # CHLOROPHYLL FETCH
    # ------------------------------------------------------------------
    if "chl" in data_types:
        chl_min_dt = datetime(2012, 3, 1)
        chl_max_dt = datetime(2023, 12, 31)

        if end_dt < chl_min_dt or start_dt > chl_max_dt:
            log_message("NOTE: Chlorophyll only available 2012-03 to 2023-12. Skipping.")
            results["chl"] = "out_of_range"

            _log_fetch_history(
                "chl",
                start_date,
                end_date,
                "skipped",
                "Date range outside chlorophyll coverage"
            )
        else:
            adjusted_start_dt = max(start_dt, chl_min_dt)
            adjusted_end_dt = min(end_dt, chl_max_dt)

            adjusted_start = adjusted_start_dt.strftime("%Y-%m-%d")
            adjusted_end = adjusted_end_dt.strftime("%Y-%m-%d")

            if adjusted_start != start_date or adjusted_end != end_date:
                log_message(
                    f"NOTE: Adjusting chlorophyll range: "
                    f"{adjusted_start} to {adjusted_end}"
                )

            try:
                chl_variable = erddap.get("chl_variable", "chl")
                chl_ds = download_erddap_data(
                    erddap["chl_dataset"],
                    chl_variable,
                    adjusted_start,
                    adjusted_end
                )

                if chl_ds is not None:
                    chl_stats = calculate_statistics(chl_ds, chl_variable)
                    if chl_stats:
                        store_data("chl", chl_stats)
                        store_spatial_data("chl", chl_ds, chl_variable)
                        log_message(f"[OK] Chlorophyll: {len(chl_stats)} days stored")
                        results["chl"] = "success"

                    _log_fetch_history(
                        "chl",
                        adjusted_start,
                        adjusted_end,
                        "success" if results["chl"] == "success" else "failed",
                    )

                    chl_ds.close()

            except Exception as e:
                log_message(f"Chlorophyll error: {e}")
                results["chl"] = f"error: {e}"

    # ------------------------------------------------------------------
    # Bloom risk recalculation
    # ------------------------------------------------------------------
    if results["sst"] == "success" or results["chl"] == "success":
        log_message("Calculating bloom risk for full archive...")
        calculate_bloom_risk(all_dates=True)

    return results

def fetch_demo_dataset():
    """Load August 2023 harmful algal bloom event."""
    log_message("=" * 60)
    log_message("Fetching August 2023 Bloom Event (Demo Dataset)")
    log_message("=" * 60)
    results = fetch_historical_range("2023-08-10", "2023-08-20", ["sst", "chl"])
    log_message("Demo dataset fetch complete!")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        fetch_demo_dataset()
    else:
        fetch_latest_data()
