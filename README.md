# 🌊 Lake Erie Monitor — BlueNexus

**Satellite-derived SST & chlorophyll monitoring · NOAA GLERL ERDDAP**

A production-grade environmental monitoring dashboard tracking harmful algal bloom (HAB) conditions in Lake Erie using 18+ years of satellite remote sensing data. Built as Lab 03 of the BlueNexus oceanographic research workstation series.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Why This Exists

Lake Erie — the shallowest and warmest of the Great Lakes — has experienced increasingly severe harmful algal blooms since the mid-2000s. In 2014, a bloom forced Toledo, Ohio to issue a "Do Not Drink" advisory for 500,000 residents. This dashboard tracks two satellite-derived measurements that together predict bloom risk: **sea surface temperature (SST)** and **chlorophyll-a concentration**.

## Quick Start

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/bluenexus-lake-erie-monitor.git
cd bluenexus-lake-erie-monitor
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py
```

The dashboard loads with demo data. Use the sidebar controls to fetch live data from NOAA ERDDAP.

## Data Sources

| Dataset                 | ERDDAP ID           | Variable | Coverage     | Resolution    |
| ----------------------- | ------------------- | -------- | ------------ | ------------- |
| Sea Surface Temperature | `GLSEA_ACSPO_GCS`   | `sst`    | 2007–present | Daily, ~1.3km |
| Chlorophyll-a           | `GR_Daily_CHL_test` | `chl`    | 2012–2023    | Daily, ~1.3km |

All data is sourced from [NOAA GLERL](https://coastwatch.glerl.noaa.gov/erddap/) via their ERDDAP server.

## Dashboard Features

### Overview Tab

- **Current readings**: Lake-wide SST, peak SST, west basin SST, chlorophyll-a
- **Historical anomaly context**: Compares current readings to the long-term day-of-year average ("Current SST is 3.2°C below the historical average for this day of year")
- **Dynamic insight callouts**: Season-appropriate commentary on lake conditions
- **7-day trend direction**: Visual indicator when SST is rising or falling
- **7-day summary**: Running averages with bloom threshold tracking
- **Bloom risk score**: 0–6 composite scoring system

### Trends & Analysis Tab

- **Time range selector**: Last 30 days, 90 days, or Full Archive with range slider
- **Basin comparison**: West vs East basin SST overlay
- **Seasonal heatmap**: Year × month grid of average SST or chlorophyll showing multi-year patterns at a glance
- **Year-over-year comparison**: Overlay up to 8 years of SST data with season presets (Full Year, Bloom Season, Summer Peak, Shoulder)
- **Bloom Season Report Card**: Annual June–October severity comparison with peak SST, days above bloom threshold, and chlorophyll statistics across all years
- **Data export**: Download full SST and CHL archives as CSV

### Spatial Explorer Tab

- **Interactive Leaflet maps**: SST and chlorophyll grids for any archived date
- **Date navigation**: Year/month/day selectors with bloom season quick-jump
- **Color-coded legends**: Temperature and chlorophyll threshold indicators

### Field Guide Tab

- Complete reference for every measurement, threshold, and dashboard feature
- Written for both limnologists and first-time users

### System Status Tab

- **Pipeline health**: Record counts, date spans, spatial map count
- **Data coverage heatmap**: Year × month grid showing observation days per month
- **Recent fetch log**: Timestamped download history with error tracking
- **Registered datasets**: Shows all datasets defined in the dataset registry
- **Active configuration**: Current ERDDAP endpoints, thresholds, schedule

## Building the Archive

The dashboard works with any amount of data — from a single day's demo to 18 years of daily records.

```bash
# Load demo data (instant)
# Use the "Load Demo Dataset" button in the sidebar

# Quick build: 2023-2025 bloom season (~20 minutes)
python build_archive.py --quick

# Full build: 2007-present (~4-8 hours)
python build_archive.py --full

# Resume an interrupted build
python build_archive.py --resume

# Custom range
python build_archive.py --full --start-year 2015 --end-year 2020
```

The archive builder can also be run from the dashboard sidebar under "Archive Builder."

## Production Engineering

This isn't a prototype — it's built for reliability:

1. **Streaming HTTP downloads** — Data written to disk in 64KB chunks as it arrives, preventing read timeouts on large queries
2. **Persistent connection pooling** — `requests.Session` with `HTTPAdapter` reuses TCP/TLS handshakes
3. **Separate connect/read timeouts** — 30s connect, 600s (10 min) read, accommodating historical cold-storage queries
4. **Error classification** — HTTP 404 (no data), 500/503 (server overload), ConnectionError (throttling), and Timeout each get tailored retry strategies
5. **Exponential backoff with jitter** — Random delays prevent synchronized retry storms
6. **Adaptive cooldown** — Detects consecutive failures and progressively increases delay (up to 5 min)
7. **ERDDAP connectivity pre-check** — Verifies server reachability before multi-hour builds
8. **Read-only SQLite connections** — All dashboard queries use `mode=ro` to prevent write lock contention
9. **Threading lock** — All write operations use `_db_lock` for safe concurrent access
10. **WAL mode** — SQLite Write-Ahead Logging for concurrent read/write
11. **UTF-8 safe logging** — Console output sanitized for Windows compatibility
12. **Month-by-month chunking** — Memory-safe bulk downloads with per-month resume
13. **Auto-refresh** — Dashboard clears cache and reloads after data operations

## Dataset Registry (Modularity Foundation)

The `config.yaml` includes a `datasets` section that defines data sources as structured objects. This is the foundation for scaling BlueNexus beyond Lake Erie.

```yaml
datasets:
  - id: sst
    name: "Sea Surface Temperature"
    source: erddap
    erddap_id: "GLSEA_ACSPO_GCS"
    variable: "sst"
    unit: "°C"
    coverage:
      start: "2006-12-11"
      end: "present"
    color: "#ff6b6b"
    thresholds:
      bloom_favorable: 20.0
    region: "Lake Erie"
```

To add a new dataset, add an entry following the template in `config.yaml`. Registry helper functions in `data_fetcher.py` provide programmatic lookup.

## Project Files

| File               | Lines | Purpose                                                  |
| ------------------ | ----- | -------------------------------------------------------- |
| `app.py`           | 2,267 | Streamlit dashboard (5 tabs + sidebar controls)          |
| `data_fetcher.py`  | 875   | ERDDAP download engine, SQLite storage, dataset registry |
| `build_archive.py` | 496   | CLI bulk archive builder with adaptive cooldown          |
| `alert_engine.py`  | 231   | Bloom risk scoring engine (read-only)                    |
| `utils.py`         | 267   | Folium map rendering, formatting helpers                 |
| `config.yaml`      | 149   | All configuration + dataset registry                     |
| `requirements.txt` | —     | Python dependencies                                      |
| `DEPLOY.md`        | —     | GitHub + Streamlit Cloud deployment guide                |

## Requirements

- Python 3.9+
- See `requirements.txt` for all dependencies

## Deployment

See [DEPLOY.md](DEPLOY.md) for step-by-step instructions to push to GitHub and deploy to Streamlit Community Cloud for free.

## License

MIT License. Data sourced from NOAA GLERL (public domain).

---

_BlueNexus Lab 03 · Project Blue Nexus · Lake Erie Environmental Monitoring_

MIT License. Data sourced from NOAA GLERL (public domain).

---

_BlueNexus Lab 03 · Project Blue Nexus · Lake Erie Environmental Monitoring_
