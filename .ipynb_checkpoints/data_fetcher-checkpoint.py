"""
Data Fetcher for Lake Erie Live Monitor
Downloads latest SST and chlorophyll data from NOAA GLERL ERDDAP
Caches results in SQLite database

Data Sources:
  SST: GLSEA_ACSPO_GCS (2006-present, daily, ~1 day processing lag)
  Chlorophyll: LE_CHL_NRT (2021-present, near-real-time per-pass)
"""

import os
import time
import sqlite3
from datetime import datetime, timedelta
import requests
import xarray as xr
import numpy as np
import yaml
from pathlib import Path

# Load configuration
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)

# Database path
DB_PATH = Path(__file__).parent / "cache" / "realtime_data.db"
LOG_PATH = Path(__file__).parent / "cache" / "fetch_log.txt"

# Ensure cache directory exists
DB_PATH.parent.mkdir(exist_ok=True)


def log_message(message):
    """Append message to log file with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    
    with open(LOG_PATH, 'a') as f:
        f.write(log_entry)
    
    print(log_entry.strip())


def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # SST data table
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
    
    # Chlorophyll data table
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
    
    # Bloom risk scores table
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
    
    # Fetch history table
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
    
    conn.commit()
    conn.close()
    log_message("Database initialized successfully")


def get_latest_date(dataset_id):
    """
    Query ERDDAP server to find the actual latest available date for a dataset.
    
    CRITICAL: ERDDAP returns 404 if you request ANY date beyond what's available,
    even if the start date is valid. We must check first.
    
    Uses the ERDDAP convention: time[(last)] returns the most recent time value.
    
    Args:
        dataset_id: ERDDAP dataset identifier
        
    Returns:
        str: Latest available date as 'YYYY-MM-DD', or None if query fails
    """
    erddap = CONFIG['erddap']
    # ERDDAP special syntax: (last) returns the final time step
    url = f"{erddap['base_url']}/griddap/{dataset_id}.json?time%5B(last)%5D"
    
    try:
        log_message(f"Querying {dataset_id} for latest available date...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Response format: {"table":{"columnNames":["time"],"rows":[["2026-02-08T12:00:00Z"]]}}
        last_time_str = data['table']['rows'][0][0]
        last_date = last_time_str[:10]  # Extract YYYY-MM-DD
        
        log_message(f"  -> Latest available: {last_date}")
        return last_date
        
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response else "unknown"
        log_message(f"  -> Could not query latest date for {dataset_id}: HTTP {status_code}")
        return None
    except Exception as e:
        log_message(f"  -> Could not query latest date for {dataset_id}: {str(e)}")
        return None


def download_erddap_data(dataset_id, variable, start_date, end_date):
    """
    Download data from NOAA GLERL ERDDAP server
    
    Args:
        dataset_id: ERDDAP dataset identifier
        variable: Variable name to download
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        xarray.Dataset or None if download fails
    """
    bbox = CONFIG['bbox']
    erddap = CONFIG['erddap']
    
    # Time format depends on dataset
    # GLSEA_ACSPO_GCS: daily composites at T12:00:00Z
    # LE_CHL_NRT: per-pass data, use T00:00:00Z to T23:59:59Z range
    if dataset_id == erddap['sst_dataset']:
        start_time = f"{start_date}T12:00:00Z"
        end_time = f"{end_date}T12:00:00Z"
    else:  # Chlorophyll NRT — use full day range to capture all passes
        start_time = f"{start_date}T00:00:00Z"
        end_time = f"{end_date}T23:59:59Z"
    
    # Build ERDDAP URL
    url = (
        f"{erddap['base_url']}/griddap/{dataset_id}.nc?"
        f"{variable}"
        f"[({start_time}):1:({end_time})]"
        f"[({bbox['lat_min']}):1:({bbox['lat_max']})]"
        f"[({bbox['lon_min']}):1:({bbox['lon_max']})]"
    )
    
    log_message(f"Downloading {dataset_id} from {start_date} to {end_date}...")
    
    # Use unique temp filename to avoid conflicts
    import random
    temp_suffix = random.randint(1000, 9999)
    temp_file = DB_PATH.parent / f"temp_{dataset_id}_{temp_suffix}.nc"
    
    for attempt in range(erddap['retry_attempts']):
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            
            # Save to temporary file
            with open(temp_file, 'wb') as f:
                f.write(response.content)
            
            # Load with xarray
            ds = xr.open_dataset(temp_file)
            
            # IMPORTANT: Load data into memory before closing file
            ds = ds.load()
            
            # Now we can safely delete the temp file
            try:
                temp_file.unlink()
            except:
                pass  # Ignore if deletion fails
            
            log_message(f"Successfully downloaded {dataset_id} ({len(ds.time)} time steps)")
            return ds
            
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else "unknown"
            log_message(f"Download attempt {attempt + 1} failed: HTTP {status_code}")
            
            # 404 = no data for this date range — don't retry
            if status_code == 404:
                log_message(f"  No data available for {dataset_id} in range {start_date} to {end_date}")
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                except:
                    pass
                return None
            
            # Clean up temp file on failure
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except:
                pass
            
            if attempt < erddap['retry_attempts'] - 1:
                time.sleep(erddap['retry_delay'])
            else:
                log_message(f"Failed to download {dataset_id} after {erddap['retry_attempts']} attempts")
                return None
                
        except Exception as e:
            log_message(f"Download attempt {attempt + 1} failed: {str(e)}")
            
            # Clean up temp file on failure
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except:
                pass
            
            if attempt < erddap['retry_attempts'] - 1:
                time.sleep(erddap['retry_delay'])
            else:
                log_message(f"Failed to download {dataset_id} after {erddap['retry_attempts']} attempts")
                return None


def calculate_statistics(ds, variable):
    """
    Calculate lake-wide and basin-specific statistics.
    
    Handles both daily composites (1 time step per day) and NRT per-pass data
    (multiple time steps per day). For NRT, the last pass per day wins.
    
    Args:
        ds: xarray.Dataset
        variable: Variable name
    
    Returns:
        dict with statistics for each date
    """
    stats = {}
    western_lon = CONFIG['thresholds']['western_basin_lon']
    
    # Get coordinate names (ERDDAP uses 'latitude'/'longitude' or 'lat'/'lon')
    lat_coord = 'latitude' if 'latitude' in ds.coords else 'lat'
    lon_coord = 'longitude' if 'longitude' in ds.coords else 'lon'
    
    for date in ds.time.values:
        date_str = str(date)[:10]  # YYYY-MM-DD (handles both daily and per-pass)
        data = ds[variable].sel(time=date)
        
        # Skip if all NaN (common for winter chlorophyll)
        if np.all(np.isnan(data.values)):
            continue
        
        # Lake-wide statistics
        lake_mean = float(data.mean(skipna=True).values)
        lake_max = float(data.max(skipna=True).values)
        data_coverage = float((~np.isnan(data.values)).sum() / data.values.size * 100)
        
        # Skip if coverage is negligibly small (< 1%)
        if data_coverage < 1.0:
            continue
        
        # Basin-specific statistics
        west_mask = ds[lon_coord] < western_lon
        east_mask = ds[lon_coord] >= western_lon
        
        west_data = data.where(west_mask, drop=False)
        east_data = data.where(east_mask, drop=False)
        
        west_mean = float(west_data.mean(skipna=True).values)
        east_mean = float(east_data.mean(skipna=True).values)
        
        # For NRT data with multiple passes per day, last pass overwrites
        stats[date_str] = {
            'lake_mean': lake_mean,
            'lake_max': lake_max,
            'west_basin_mean': west_mean,
            'east_basin_mean': east_mean,
            'data_coverage': data_coverage
        }
    
    return stats


def store_data(data_type, stats):
    """
    Store calculated statistics in database
    
    Args:
        data_type: 'sst' or 'chl'
        stats: Dictionary of statistics by date
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        cursor = conn.cursor()
        fetch_timestamp = datetime.now().isoformat()
        
        table_name = f"{data_type}_data"
        
        for date_str, values in stats.items():
            cursor.execute(f"""
                INSERT OR REPLACE INTO {table_name}
                (date, lake_mean, lake_max, west_basin_mean, east_basin_mean, 
                 data_coverage, fetch_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                date_str,
                values['lake_mean'],
                values['lake_max'],
                values['west_basin_mean'],
                values['east_basin_mean'],
                values['data_coverage'],
                fetch_timestamp
            ))
        
        conn.commit()
        log_message(f"Stored {len(stats)} records in {table_name}")
    
    except Exception as e:
        log_message(f"Error storing data in {data_type}_data: {str(e)}")
        if conn:
            conn.rollback()
    
    finally:
        if conn:
            conn.close()


def calculate_bloom_risk():
    """
    Calculate bloom risk scores based on latest SST and chlorophyll data
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get latest 7 days of data
    cursor.execute("""
        SELECT s.date, s.lake_mean as sst, c.west_basin_mean as chl
        FROM sst_data s
        LEFT JOIN chl_data c ON s.date = c.date
        ORDER BY s.date DESC
        LIMIT 7
    """)
    
    rows = cursor.fetchall()
    thresholds = CONFIG['thresholds']
    calculation_timestamp = datetime.now().isoformat()
    
    for date, sst, chl in rows:
        risk_score = 0
        sst_above = 0
        chl_above = 0
        
        # SST contribution
        if sst is not None and sst > thresholds['sst_bloom']:
            risk_score += 1
            sst_above = 1
        
        # Chlorophyll contribution
        if chl is not None:
            if chl > thresholds['chl_high']:
                risk_score += 3
                chl_above = 1
            elif chl > thresholds['chl_moderate']:
                risk_score += 2
                chl_above = 1
            elif chl > thresholds['chl_low']:
                risk_score += 1
                chl_above = 1
        
        # Determine risk level
        if risk_score >= 4:
            risk_level = "High"
        elif risk_score >= 2:
            risk_level = "Moderate"
        else:
            risk_level = "Low"
        
        # Store in database
        cursor.execute("""
            INSERT OR REPLACE INTO bloom_risk
            (date, risk_score, risk_level, sst_above_threshold, 
             chl_above_threshold, calculation_timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (date, risk_score, risk_level, sst_above, chl_above, calculation_timestamp))
    
    conn.commit()
    conn.close()
    log_message(f"Calculated bloom risk for {len(rows)} dates")


def cleanup_old_data():
    """Remove data older than retention period (relative to newest data, not wall clock).
    
    Uses the most recent data date as reference instead of datetime.now()
    to prevent historical/demo data from being wiped.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Use the most recent data date as the reference point, NOT datetime.now()
    cursor.execute("SELECT MAX(date) FROM sst_data")
    row = cursor.fetchone()
    
    if row[0] is None:
        conn.close()
        return
    
    max_date = datetime.strptime(row[0], "%Y-%m-%d")
    cutoff_date = (max_date - timedelta(days=CONFIG['retention_days'])).strftime("%Y-%m-%d")
    
    total_deleted = 0
    for table in ['sst_data', 'chl_data', 'bloom_risk']:
        cursor.execute(f"DELETE FROM {table} WHERE date < ?", (cutoff_date,))
        total_deleted += cursor.rowcount
    
    conn.commit()
    conn.close()
    
    if total_deleted > 0:
        log_message(f"Cleaned up {total_deleted} old records (before {cutoff_date})")


def fetch_latest_data():
    """
    Main function - discovers latest available dates from ERDDAP, then downloads.
    
    Strategy:
    1. Query each dataset for its actual latest available date via time[(last)]
    2. Request (latest_date - 6 days) through latest_date (7 days inclusive)
    3. This avoids 404 errors from requesting beyond what exists
    """
    log_message("=" * 60)
    log_message("Starting data fetch...")
    
    # Initialize database if needed
    init_database()
    
    erddap = CONFIG['erddap']
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # ---- SST ----
    try:
        # Step 1: Discover latest available date from ERDDAP
        sst_end = get_latest_date(erddap['sst_dataset'])
        
        if sst_end is None:
            raise Exception("Could not determine latest SST date from ERDDAP server")
        
        sst_end_dt = datetime.strptime(sst_end, "%Y-%m-%d")
        sst_start = (sst_end_dt - timedelta(days=6)).strftime("%Y-%m-%d")  # 7 days inclusive
        
        # Step 2: Download data for the verified date range
        sst_ds = download_erddap_data(
            erddap['sst_dataset'],
            'sst',
            sst_start,
            sst_end
        )
        
        if sst_ds is not None:
            sst_stats = calculate_statistics(sst_ds, 'sst')
            if sst_stats:
                store_data('sst', sst_stats)
                log_message(f"SST data covers: {min(sst_stats.keys())} to {max(sst_stats.keys())}")
            else:
                log_message("SST download succeeded but no valid data points found")
            
            cursor.execute("""
                INSERT INTO fetch_history 
                (fetch_timestamp, data_type, date_range_start, date_range_end, status, error_message)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (datetime.now().isoformat(), 'sst', sst_start, sst_end, 'success', None))
            
            sst_ds.close()
        else:
            cursor.execute("""
                INSERT INTO fetch_history 
                (fetch_timestamp, data_type, date_range_start, date_range_end, status, error_message)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (datetime.now().isoformat(), 'sst', sst_start, sst_end, 'failed', 'Download returned no data'))
    
    except Exception as e:
        log_message(f"SST fetch error: {str(e)}")
        cursor.execute("""
            INSERT INTO fetch_history 
            (fetch_timestamp, data_type, date_range_start, date_range_end, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (datetime.now().isoformat(), 'sst', 'unknown', 'unknown', 'failed', str(e)))
    
    # Wait between requests (server throttling)
    time.sleep(erddap['request_delay'])
    
    # ---- Chlorophyll ----
    try:
        # Step 1: Discover latest available date from ERDDAP
        chl_end = get_latest_date(erddap['chl_dataset'])
        
        if chl_end is None:
            raise Exception(
                "Could not determine latest chlorophyll date from ERDDAP. "
                "This is normal in winter — satellite cannot see through ice/clouds."
            )
        
        chl_end_dt = datetime.strptime(chl_end, "%Y-%m-%d")
        chl_start = (chl_end_dt - timedelta(days=6)).strftime("%Y-%m-%d")
        
        # Step 2: Download data for the verified date range
        chl_ds = download_erddap_data(
            erddap['chl_dataset'],
            'Chlorophyll',
            chl_start,
            chl_end
        )
        
        if chl_ds is not None:
            chl_stats = calculate_statistics(chl_ds, 'Chlorophyll')
            if chl_stats:
                store_data('chl', chl_stats)
                log_message(f"Chlorophyll data covers: {min(chl_stats.keys())} to {max(chl_stats.keys())}")
            else:
                log_message("Chlorophyll download succeeded but no valid data (likely winter/cloud cover)")
            
            cursor.execute("""
                INSERT INTO fetch_history 
                (fetch_timestamp, data_type, date_range_start, date_range_end, status, error_message)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (datetime.now().isoformat(), 'chl', chl_start, chl_end, 'success', None))
            
            chl_ds.close()
        else:
            cursor.execute("""
                INSERT INTO fetch_history 
                (fetch_timestamp, data_type, date_range_start, date_range_end, status, error_message)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (datetime.now().isoformat(), 'chl', chl_start, chl_end, 'failed', 'Download returned no data'))
    
    except Exception as e:
        log_message(f"Chlorophyll fetch error: {str(e)}")
        cursor.execute("""
            INSERT INTO fetch_history 
            (fetch_timestamp, data_type, date_range_start, date_range_end, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (datetime.now().isoformat(), 'chl', 'unknown', 'unknown', 'failed', str(e)))
    
    conn.commit()
    conn.close()
    
    # Calculate bloom risk scores
    calculate_bloom_risk()
    
    # Cleanup old data
    cleanup_old_data()
    
    # Cleanup orphaned temp .nc files from failed downloads
    for temp_file in DB_PATH.parent.glob("temp_*.nc"):
        try:
            temp_file.unlink()
        except:
            pass
    
    log_message("Data fetch complete!")
    log_message("=" * 60)


if __name__ == "__main__":
    # Run standalone for testing
    fetch_latest_data()