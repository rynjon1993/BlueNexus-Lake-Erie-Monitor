"""
Lake Erie Live Monitor — BlueNexus Dashboard
Real-time and historical monitoring of SST and harmful algal bloom risk
via NOAA GLERL ERDDAP satellite data.

USAGE:
    streamlit run app.py

TABS:
    Overview          -> Current conditions, anomaly context, 7-day trends
    Trends & Analysis -> Time series, seasonal heatmap, year-over-year,
                         bloom report card, CSV export
    Spatial Explorer  -> Interactive SST/CHL maps for any archived date
    Field Guide       -> Reference docs for all measurements
    System Status     -> Pipeline health, coverage heatmap, dataset registry

ARCHITECTURE:
    app.py (this file)  -> Streamlit UI / dashboard
    data_fetcher.py     -> ERDDAP data pipeline + SQLite caching
    alert_engine.py     -> Bloom risk scoring
    utils.py            -> Plotting helpers (Folium maps, formatting)
    config.yaml         -> Configuration (datasets, thresholds, schedule)
    build_archive.py    -> Bulk historical data builder (CLI)

BlueNexus Lab 03 · Project Blue Nexus
"""

import streamlit as st
from streamlit_folium import st_folium
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from calendar import monthrange
import yaml
import time
import gc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import custom modules
from data_fetcher import (
    fetch_latest_data, init_database, DB_PATH,
    fetch_historical_range, fetch_demo_dataset
)
from alert_engine import (
    get_current_risk, get_risk_trend, get_alert_message,
    get_risk_color, get_risk_emoji, get_recent_conditions
)
from utils import (
    create_interactive_map, format_date, format_metric,
    get_data_freshness_status, calculate_time_until_next_update,
    plot_time_series
)

# ──────────────────────────────────────────────────────────────
# Page config & theme
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Lake Erie Monitor — BlueNexus",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)

# ──────────────────────────────────────────────────────────────
# Custom CSS — Deep ocean science aesthetic
# ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Import distinctive fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&family=Playfair+Display:wght@700;800&display=swap');

    /* ── Root variables ── */
    :root {
        --ocean-deep: #0a1628;
        --ocean-mid: #132744;
        --ocean-surface: #1b3a5c;
        --foam: #e8f4f8;
        --cyan-glow: #00d4ff;
        --teal-accent: #0ea5a0;
        --warm-coral: #ff6b6b;
        --amber-warn: #ffb347;
        --sea-green: #2dd4a8;
        --pale-blue: #a8d8ea;
        --card-bg: rgba(19, 39, 68, 0.45);
        --card-border: rgba(0, 212, 255, 0.12);
        --text-primary: #e8f4f8;
        --text-secondary: #8ba4b8;
    }

    /* ── Global typography ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 { font-family: 'Playfair Display', serif !important; }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 16px 20px;
        transition: border-color 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(0, 212, 255, 0.35);
    }
    [data-testid="stMetricLabel"] {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--text-secondary) !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.7rem !important;
        font-weight: 500 !important;
        color: var(--foam) !important;
    }
    [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 2px solid rgba(0, 212, 255, 0.1);
        padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        font-size: 0.88rem;
        letter-spacing: 0.02em;
        padding: 12px 24px;
        border-radius: 8px 8px 0 0;
        color: var(--text-secondary);
        border-bottom: 3px solid transparent;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        color: var(--cyan-glow) !important;
        border-bottom: 3px solid var(--cyan-glow) !important;
        background: rgba(0, 212, 255, 0.06);
    }

    /* ── Sidebar polish ── */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(0, 212, 255, 0.08);
    }
    section[data-testid="stSidebar"] [data-testid="stMetric"] {
        padding: 10px 14px;
    }
    section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
        font-size: 1.3rem !important;
    }

    /* ── Custom components ── */
    .nexus-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 16px;
        backdrop-filter: blur(8px);
    }
    .nexus-card-header {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-secondary);
        margin-bottom: 8px;
    }

    /* ── Risk badge ── */
    .risk-badge {
        text-align: center;
        padding: 18px 14px;
        border-radius: 14px;
        margin-bottom: 20px;
    }
    .risk-badge.low {
        background: linear-gradient(135deg, rgba(45, 212, 168, 0.12), rgba(45, 212, 168, 0.04));
        border: 1px solid rgba(45, 212, 168, 0.3);
    }
    .risk-badge.moderate {
        background: linear-gradient(135deg, rgba(255, 179, 71, 0.15), rgba(255, 179, 71, 0.04));
        border: 1px solid rgba(255, 179, 71, 0.35);
    }
    .risk-badge.high {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.18), rgba(255, 107, 107, 0.04));
        border: 1px solid rgba(255, 107, 107, 0.4);
    }
    .risk-badge.unknown {
        background: rgba(139, 164, 184, 0.08);
        border: 1px solid rgba(139, 164, 184, 0.2);
    }
    .risk-badge .risk-icon { font-size: 1.8rem; }
    .risk-badge .risk-level {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-top: 4px;
    }
    .risk-badge .risk-score {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: var(--text-secondary);
        margin-top: 2px;
    }

    /* ── Header brand ── */
    .brand-header {
        display: flex;
        align-items: baseline;
        gap: 12px;
        margin-bottom: 2px;
    }
    .brand-header h1 {
        font-size: 1.9rem !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.2 !important;
    }
    .brand-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 500;
        letter-spacing: 0.1em;
        color: var(--cyan-glow);
        background: rgba(0, 212, 255, 0.08);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 4px;
        padding: 2px 8px;
        text-transform: uppercase;
    }
    .brand-subtitle {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.82rem;
        color: var(--text-secondary);
        margin-top: 4px;
        margin-bottom: 20px;
    }

    /* ── Section headers ── */
    .section-label {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--teal-accent);
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid rgba(14, 165, 160, 0.2);
    }

    /* ── Divider ── */
    .nexus-divider {
        height: 1px;
        background: linear-gradient(90deg,
            transparent,
            rgba(0, 212, 255, 0.15) 20%,
            rgba(0, 212, 255, 0.15) 80%,
            transparent);
        margin: 24px 0;
        border: none;
    }

    /* ── Status dot ── */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-dot.online { background: var(--sea-green); box-shadow: 0 0 6px var(--sea-green); }
    .status-dot.stale { background: var(--amber-warn); }
    .status-dot.offline { background: var(--warm-coral); }

    /* ── Button polish ── */
    .stButton > button {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
    }

    /* ── Plotly containers ── */
    .js-plotly-plot { border-radius: 12px; overflow: hidden; }

    /* ── Hide Streamlit branding ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* ── Responsive ── */
    @media (max-width: 768px) {
        .brand-header h1 { font-size: 1.4rem !important; }
        [data-testid="stMetricValue"] { font-size: 1.3rem !important; }
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_latest_data():
    """Load latest data using read-only connection to prevent write conflicts."""
    with sqlite3.connect(f'file:{DB_PATH}?mode=ro', uri=True) as conn:
        sst_df = pd.read_sql_query(
            "SELECT date, lake_mean, lake_max, west_basin_mean, east_basin_mean, "
            "data_coverage, fetch_timestamp FROM sst_data ORDER BY date DESC LIMIT 7", conn)
        chl_df = pd.read_sql_query(
            "SELECT date, lake_mean, lake_max, west_basin_mean, east_basin_mean, "
            "data_coverage, fetch_timestamp FROM chl_data ORDER BY date DESC LIMIT 7", conn)
        risk_df = pd.read_sql_query(
            "SELECT date, risk_score, risk_level FROM bloom_risk ORDER BY date DESC LIMIT 7", conn)
        history_df = pd.read_sql_query(
            "SELECT fetch_timestamp, data_type, date_range_start, date_range_end, "
            "status, error_message FROM fetch_history ORDER BY fetch_timestamp DESC LIMIT 20", conn)
    return sst_df, chl_df, risk_df, history_df


@st.cache_data(ttl=300)
def load_extended_data(days=90):
    """Load extended data using read-only connection to prevent write conflicts."""
    with sqlite3.connect(f'file:{DB_PATH}?mode=ro', uri=True) as conn:
        sst_df = pd.read_sql_query(
            f"SELECT date, lake_mean, lake_max, west_basin_mean, east_basin_mean "
            f"FROM sst_data ORDER BY date DESC LIMIT {days}", conn)
        chl_df = pd.read_sql_query(
            f"SELECT date, lake_mean, lake_max, west_basin_mean, east_basin_mean "
            f"FROM chl_data ORDER BY date DESC LIMIT {days}", conn)
        risk_df = pd.read_sql_query(
            f"SELECT date, risk_score, risk_level FROM bloom_risk ORDER BY date DESC LIMIT {days}", conn)
    return sst_df, chl_df, risk_df


@st.cache_data(ttl=300)
def load_full_sst_series():
    """Load full SST series using read-only connection to prevent write conflicts."""
    with sqlite3.connect(f'file:{DB_PATH}?mode=ro', uri=True) as conn:
        df = pd.read_sql_query(
            "SELECT date, lake_mean, lake_max, west_basin_mean FROM sst_data ORDER BY date ASC", conn)
    return df


@st.cache_data(ttl=300)
def load_full_chl_series():
    """Load full chlorophyll series for export and analysis."""
    with sqlite3.connect(f'file:{DB_PATH}?mode=ro', uri=True) as conn:
        df = pd.read_sql_query(
            "SELECT date, lake_mean, lake_max, west_basin_mean, east_basin_mean "
            "FROM chl_data ORDER BY date ASC", conn)
    return df


@st.cache_data(ttl=300)
def load_historical_averages():
    """Calculate day-of-year averages across the entire archive for anomaly detection.

    Returns a dict mapping day-of-year (1-366) to average lake_mean SST.
    """
    with sqlite3.connect(f'file:{DB_PATH}?mode=ro', uri=True) as conn:
        df = pd.read_sql_query(
            "SELECT date, lake_mean FROM sst_data WHERE lake_mean IS NOT NULL", conn)
    if len(df) < 30:
        return {}
    df['doy'] = pd.to_datetime(df['date']).dt.dayofyear
    return df.groupby('doy')['lake_mean'].mean().to_dict()


@st.cache_data(ttl=300)
def load_monthly_matrix():
    """Build year x month matrices for SST and CHL seasonal heatmaps.

    Returns two DataFrames pivoted with years as rows, months 1-12 as columns.
    """
    with sqlite3.connect(f'file:{DB_PATH}?mode=ro', uri=True) as conn:
        sst_df = pd.read_sql_query(
            "SELECT date, lake_mean FROM sst_data WHERE lake_mean IS NOT NULL", conn)
        chl_df = pd.read_sql_query(
            "SELECT date, west_basin_mean FROM chl_data WHERE west_basin_mean IS NOT NULL", conn)

    result = {}
    for label, df, col in [('sst', sst_df, 'lake_mean'), ('chl', chl_df, 'west_basin_mean')]:
        if len(df) > 0:
            df['dt'] = pd.to_datetime(df['date'])
            df['year'] = df['dt'].dt.year
            df['month'] = df['dt'].dt.month
            pivot = df.groupby(['year', 'month'])[col].mean().unstack(fill_value=float('nan'))
            result[label] = pivot
        else:
            result[label] = pd.DataFrame()
    return result


@st.cache_data(ttl=300)
def load_bloom_season_stats():
    """Calculate annual bloom season (Jun-Oct) statistics across the archive.

    Returns a DataFrame with one row per year and columns:
    year, peak_sst, avg_sst, days_above_20, peak_chl, avg_chl, data_days
    """
    with sqlite3.connect(f'file:{DB_PATH}?mode=ro', uri=True) as conn:
        sst_df = pd.read_sql_query(
            "SELECT date, lake_mean, lake_max, west_basin_mean FROM sst_data "
            "WHERE lake_mean IS NOT NULL", conn)
        chl_df = pd.read_sql_query(
            "SELECT date, west_basin_mean FROM chl_data "
            "WHERE west_basin_mean IS NOT NULL", conn)

    if len(sst_df) < 30:
        return pd.DataFrame()

    sst_df['dt'] = pd.to_datetime(sst_df['date'])
    sst_df['year'] = sst_df['dt'].dt.year
    sst_df['month'] = sst_df['dt'].dt.month

    # Bloom season = June through October
    bloom_sst = sst_df[(sst_df['month'] >= 6) & (sst_df['month'] <= 10)]

    if len(bloom_sst) < 10:
        return pd.DataFrame()

    bloom_threshold = CONFIG['thresholds']['sst_bloom']

    # Aggregate by year
    years = sorted(bloom_sst['year'].unique())
    rows = []
    for year in years:
        ys = bloom_sst[bloom_sst['year'] == year]
        if len(ys) < 5:
            continue

        row = {
            'year': year,
            'peak_sst': round(ys['lake_max'].max(), 1),
            'avg_sst': round(ys['lake_mean'].mean(), 1),
            'days_above_20': int((ys['lake_mean'] >= bloom_threshold).sum()),
            'data_days': len(ys),
        }

        # CHL stats for this year's bloom season
        if len(chl_df) > 0:
            chl_df_dt = pd.to_datetime(chl_df['date'])
            chl_bloom = chl_df[
                (chl_df_dt.dt.year == year) &
                (chl_df_dt.dt.month >= 6) &
                (chl_df_dt.dt.month <= 10)
            ]
            if len(chl_bloom) > 0:
                row['peak_chl'] = round(chl_bloom['west_basin_mean'].max(), 1)
                row['avg_chl'] = round(chl_bloom['west_basin_mean'].mean(), 1)
            else:
                row['peak_chl'] = None
                row['avg_chl'] = None
        else:
            row['peak_chl'] = None
            row['avg_chl'] = None

        rows.append(row)

    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def load_data_coverage():
    """Calculate data coverage: how many days have data per month/year.

    Returns a DataFrame pivoted with years as rows, months 1-12 as columns,
    containing the count of observation days.
    """
    with sqlite3.connect(f'file:{DB_PATH}?mode=ro', uri=True) as conn:
        df = pd.read_sql_query(
            "SELECT date FROM sst_data WHERE lake_mean IS NOT NULL", conn)
    if len(df) == 0:
        return pd.DataFrame()

    df['dt'] = pd.to_datetime(df['date'])
    df['year'] = df['dt'].dt.year
    df['month'] = df['dt'].dt.month
    pivot = df.groupby(['year', 'month']).size().unstack(fill_value=0)
    return pivot


def get_available_dates():
    """Get available dates for spatial data using read-only connection."""
    try:
        with sqlite3.connect(f'file:{DB_PATH}?mode=ro', uri=True) as conn:
            dates = [r[0] for r in conn.execute(
                "SELECT DISTINCT date FROM sst_spatial ORDER BY date DESC").fetchall()]
        return dates
    except Exception:
        return []


def get_date_range_stats():
    """Return record counts and date ranges for SST and CHL tables,
    plus a ``unique_days`` count (distinct calendar dates across both
    tables) used by the header badge."""
    stats = {'sst': {'count': 0, 'min': None, 'max': None},
             'chl': {'count': 0, 'min': None, 'max': None},
             'unique_days': 0}
    try:
        conn = sqlite3.connect(f'file:{DB_PATH}?mode=ro', uri=True)
        for dtype in ['sst', 'chl']:
            row = conn.execute(
                f"SELECT COUNT(*), MIN(date), MAX(date) FROM {dtype}_data").fetchone()
            stats[dtype] = {'count': row[0], 'min': row[1], 'max': row[2]}
        # Unique calendar dates across both tables (no double-counting)
        row = conn.execute(
            "SELECT COUNT(*) FROM ("
            "  SELECT date FROM sst_data UNION SELECT date FROM chl_data"
            ")").fetchone()
        stats['unique_days'] = row[0] if row else 0
        conn.close()
    except Exception:
        pass
    return stats


def group_dates_by_month(dates):
    grouped = {}
    for d in dates:
        try:
            dt = datetime.strptime(d, "%Y-%m-%d")
            grouped.setdefault(dt.year, {}).setdefault(dt.month, []).append(d)
        except ValueError:
            continue
    return grouped


def _decode_spatial_blob(blob, expected_elements):
    """Decode a binary BLOB into a 1-D numpy array, auto-detecting float64 vs float32.

    Historical data may have been stored as the xarray dataset's native dtype
    (often float32 for chlorophyll), while newer data is explicitly cast to
    float64 by data_fetcher.py.  This function tries float64 first (8 bytes
    per element); if the element count doesn't match ``expected_elements`` it
    falls back to float32 (4 bytes per element).
    """
    arr64 = np.frombuffer(blob, dtype=np.float64)
    if arr64.size == expected_elements:
        return arr64
    # Try float32
    arr32 = np.frombuffer(blob, dtype=np.float32)
    if arr32.size == expected_elements:
        return arr32.astype(np.float64)   # upcast for uniform downstream math
    # Neither matched — return whichever is closest so the caller can
    # decide how to handle it (the reshape will raise a clear ValueError).
    return arr64


def _load_spatial_grid(row):
    """Unpack a spatial DB row into (data, lats, lons) 2-D arrays.

    Handles both float64 and float32 BLOBs, and also the case where
    lats/lons were stored as 1-D coordinate vectors (unique values)
    instead of full meshgrids.

    Returns (data, lats, lons) as 2-D float64 arrays, or raises ValueError.
    """
    blob_lats, blob_lons, blob_data, shape_rows, shape_cols = row
    expected = shape_rows * shape_cols

    # --- Data array ---
    data = _decode_spatial_blob(blob_data, expected).reshape(shape_rows, shape_cols)

    # --- Lat / lon arrays ---
    raw_lats = _decode_spatial_blob(blob_lats, expected)
    raw_lons = _decode_spatial_blob(blob_lons, expected)

    if raw_lats.size == expected:
        # Full meshgrid was stored
        lats = raw_lats.reshape(shape_rows, shape_cols)
        lons = raw_lons.reshape(shape_rows, shape_cols)
    else:
        # 1-D coordinate vectors — broadcast to full grid
        n_lats = raw_lats.size
        n_lons = raw_lons.size
        if n_lats * n_lons == expected:
            lats = np.broadcast_to(
                raw_lats.reshape(n_lats, 1), (n_lats, n_lons)).copy()
            lons = np.broadcast_to(
                raw_lons.reshape(1, n_lons), (n_lats, n_lons)).copy()
        else:
            raise ValueError(
                f"Cannot reconstruct coordinate grid: lats={n_lats}, "
                f"lons={n_lons}, expected=({shape_rows},{shape_cols})")

    return data, lats, lons


# ──────────────────────────────────────────────────────────────
# Plotly chart builders (dark ocean theme)
# ──────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(10,22,40,0.6)',
    font=dict(family='DM Sans, sans-serif', color='#a8d8ea', size=12),
    margin=dict(l=50, r=30, t=50, b=50),
    xaxis=dict(gridcolor='rgba(0,212,255,0.06)', zerolinecolor='rgba(0,212,255,0.06)'),
    yaxis=dict(gridcolor='rgba(0,212,255,0.06)', zerolinecolor='rgba(0,212,255,0.06)'),
    legend=dict(bgcolor='rgba(0,0,0,0)', borderwidth=0),
    hoverlabel=dict(bgcolor='#132744', bordercolor='#00d4ff',
                    font=dict(family='JetBrains Mono, monospace', size=11, color='#e8f4f8')),
)


def build_trends_chart(sst_df, chl_df, title="Lake Erie Temperature & Chlorophyll"):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    dates = list(reversed(sst_df['date'].tolist()))
    sst_vals = list(reversed(sst_df['lake_mean'].tolist()))

    fig.add_trace(go.Scatter(
        x=dates, y=sst_vals, name='SST (Lake-wide)',
        line=dict(color='#ff6b6b', width=2.5),
        marker=dict(size=5, color='#ff6b6b'),
        mode='lines+markers',
        hovertemplate='%{x|%b %d, %Y}<br><b>%{y:.1f}°C</b><extra>SST</extra>'
    ), secondary_y=False)

    fig.add_hline(y=CONFIG['thresholds']['sst_bloom'], line_dash="dot",
                  line_color="rgba(255,107,107,0.4)", line_width=1,
                  annotation_text=f"Bloom threshold ({CONFIG['thresholds']['sst_bloom']}°C)",
                  annotation_font_color="rgba(255,107,107,0.6)",
                  annotation_font_size=10, secondary_y=False)

    if len(chl_df) > 0:
        chl_dict = dict(zip(chl_df['date'], chl_df['west_basin_mean']))
        chl_dates = [d for d in dates if d in chl_dict and chl_dict[d] is not None]
        chl_vals = [chl_dict[d] for d in chl_dates]
        if chl_dates:
            fig.add_trace(go.Bar(
                x=chl_dates, y=chl_vals, name='Chlorophyll (West Basin)',
                marker=dict(color='rgba(45, 212, 168, 0.5)',
                            line=dict(color='#2dd4a8', width=1)),
                hovertemplate='%{x|%b %d, %Y}<br><b>%{y:.1f} mg/m³</b><extra>Chl-a</extra>'
            ), secondary_y=True)

    layout_overrides = {**PLOTLY_LAYOUT,
        'title': dict(text=title, font=dict(size=15, color='#e8f4f8')),
        'height': 380,
        'legend': dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0,
                        bgcolor='rgba(0,0,0,0)', borderwidth=0),
        'barmode': 'overlay',
    }
    fig.update_layout(**layout_overrides)
    fig.update_yaxes(title_text='SST (°C)', title_font_color='#ff6b6b',
                     tickfont_color='#ff6b6b', secondary_y=False)
    fig.update_yaxes(title_text='Chlorophyll (mg/m³)', title_font_color='#2dd4a8',
                     tickfont_color='#2dd4a8', secondary_y=True)
    fig.update_xaxes(tickformat='%b %d', tickangle=-30)
    return fig


def build_risk_chart(risk_df):
    dates = list(reversed(risk_df['date'].tolist()))
    scores = list(reversed(risk_df['risk_score'].tolist()))
    colors = ['#ff6b6b' if s >= 4 else '#ffb347' if s >= 2 else '#2dd4a8' for s in scores]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates, y=scores, marker=dict(color=colors, line=dict(width=0)),
        hovertemplate='%{x|%b %d, %Y}<br>Risk Score: <b>%{y}</b>/6<extra></extra>'
    ))
    fig.add_hline(y=4, line_dash="dot", line_color="rgba(255,107,107,0.35)", line_width=1,
                  annotation_text="High", annotation_font_color="rgba(255,107,107,0.5)",
                  annotation_font_size=10)
    fig.add_hline(y=2, line_dash="dot", line_color="rgba(255,179,71,0.35)", line_width=1,
                  annotation_text="Moderate", annotation_font_color="rgba(255,179,71,0.5)",
                  annotation_font_size=10)

    risk_layout = {**PLOTLY_LAYOUT,
        'title': dict(text='Bloom Risk Score', font=dict(size=14, color='#e8f4f8')),
        'height': 240,
        'yaxis': dict(range=[0, 6.5], dtick=1, gridcolor='rgba(0,212,255,0.06)'),
        'showlegend': False,
    }
    fig.update_layout(**risk_layout)
    fig.update_xaxes(tickformat='%b %d', tickangle=-30)
    return fig


def build_long_term_chart(sst_full):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sst_full['date'], y=sst_full['lake_mean'], name='Lake-wide SST',
        line=dict(color='#ff6b6b', width=1.2),
        fill='tozeroy', fillcolor='rgba(255,107,107,0.06)',
        hovertemplate='%{x|%b %d, %Y}<br><b>%{y:.1f}°C</b><extra></extra>'
    ))
    if 'west_basin_mean' in sst_full.columns:
        fig.add_trace(go.Scatter(
            x=sst_full['date'], y=sst_full['west_basin_mean'], name='West Basin SST',
            line=dict(color='#00d4ff', width=1, dash='dot'),
            hovertemplate='%{x|%b %d, %Y}<br><b>%{y:.1f}°C</b><extra>West Basin</extra>'
        ))
    fig.add_hline(y=CONFIG['thresholds']['sst_bloom'], line_dash="dot",
                  line_color="rgba(255,107,107,0.3)", line_width=1)
    lt_layout = {**PLOTLY_LAYOUT,
        'title': dict(text='SST Archive — Full Dataset', font=dict(size=15, color='#e8f4f8')),
        'height': 350,
        'legend': dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0,
                       bgcolor='rgba(0,0,0,0)', borderwidth=0),
        'xaxis': dict(
            gridcolor='rgba(0,212,255,0.06)',
            rangeslider=dict(visible=True, bgcolor='rgba(10,22,40,0.4)',
                             bordercolor='rgba(0,212,255,0.1)', thickness=0.06),
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="All"),
                ],
                bgcolor='rgba(19,39,68,0.8)', activecolor='rgba(0,212,255,0.2)',
                bordercolor='rgba(0,212,255,0.15)',
                font=dict(size=11, color='#a8d8ea'),
            ),
        ),
    }
    fig.update_layout(**lt_layout)
    return fig


def build_sparkline(values, color='#00d4ff', height=40):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=values, mode='lines', line=dict(color=color, width=1.5),
        fill='tozeroy',
        fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)',
        hoverinfo='skip',
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0), height=height,
        xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False,
    )
    return fig


def build_seasonal_heatmap(matrix_df, title, colorscale, unit):
    """Build a year x month heatmap from a pivoted DataFrame."""
    MONTH_ABBR = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    years = list(matrix_df.index)
    months = list(range(1, 13))
    z_data = []
    for y in years:
        row = []
        for m in months:
            row.append(matrix_df.loc[y, m] if m in matrix_df.columns else float('nan'))
        z_data.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=MONTH_ABBR,
        y=[str(y) for y in years],
        colorscale=colorscale,
        hoverongaps=False,
        hovertemplate='%{y} %{x}<br><b>%{z:.1f} ' + unit + '</b><extra></extra>',
        colorbar=dict(
            title=dict(text=unit, font=dict(color='#a8d8ea', size=11)),
            tickfont=dict(color='#a8d8ea', size=10),
            bgcolor='rgba(0,0,0,0)',
        ),
    ))

    hm_layout = {**PLOTLY_LAYOUT,
        'title': dict(text=title, font=dict(size=14, color='#e8f4f8')),
        'height': max(200, len(years) * 28 + 100),
        'xaxis': dict(side='top', tickangle=0, gridcolor='rgba(0,0,0,0)'),
        'yaxis': dict(autorange='reversed', gridcolor='rgba(0,0,0,0)'),
    }
    fig.update_layout(**hm_layout)
    return fig


def build_yoy_chart(sst_full, selected_years, selected_month_range):
    """Build a year-over-year SST overlay chart.

    Args:
        sst_full: Full SST DataFrame with 'date' and 'lake_mean' columns.
        selected_years: List of years to compare.
        selected_month_range: Tuple of (start_month, end_month) 1-indexed.
    """
    YEAR_COLORS = [
        '#ff6b6b', '#00d4ff', '#2dd4a8', '#ffb347', '#c792ea',
        '#82aaff', '#f78c6c', '#89ddff', '#ffcb6b', '#c3e88d',
    ]

    fig = go.Figure()
    m_start, m_end = selected_month_range

    for i, year in enumerate(sorted(selected_years)):
        mask = (
            (pd.to_datetime(sst_full['date']).dt.year == year) &
            (pd.to_datetime(sst_full['date']).dt.month >= m_start) &
            (pd.to_datetime(sst_full['date']).dt.month <= m_end)
        )
        subset = sst_full[mask].copy()
        if len(subset) == 0:
            continue

        # Use day-of-year as x-axis for alignment
        subset['doy'] = pd.to_datetime(subset['date']).dt.dayofyear
        color = YEAR_COLORS[i % len(YEAR_COLORS)]

        fig.add_trace(go.Scatter(
            x=subset['doy'], y=subset['lake_mean'],
            name=str(year),
            line=dict(color=color, width=2),
            mode='lines',
            hovertemplate=f'{year}<br>Day %{{x}}<br><b>%{{y:.1f}}°C</b><extra></extra>',
        ))

    fig.add_hline(y=CONFIG['thresholds']['sst_bloom'], line_dash="dot",
                  line_color="rgba(255,107,107,0.3)", line_width=1,
                  annotation_text="Bloom threshold",
                  annotation_font_color="rgba(255,107,107,0.5)",
                  annotation_font_size=10)

    # Convert DOY ticks to month labels
    import calendar
    month_starts = []
    for m in range(m_start, m_end + 1):
        doy = (datetime(2020, m, 1) - datetime(2020, 1, 1)).days + 1  # leap year as reference
        month_starts.append((doy, calendar.month_abbr[m]))

    yoy_layout = {**PLOTLY_LAYOUT,
        'title': dict(text='Year-over-Year SST Comparison', font=dict(size=15, color='#e8f4f8')),
        'height': 400,
        'legend': dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0,
                       bgcolor='rgba(0,0,0,0)', borderwidth=0),
        'xaxis': dict(
            tickvals=[ms[0] for ms in month_starts],
            ticktext=[ms[1] for ms in month_starts],
            gridcolor='rgba(0,212,255,0.06)',
        ),
        'yaxis_title': 'SST (°C)',
    }
    fig.update_layout(**yoy_layout)
    return fig


def build_bloom_report_chart(bloom_df):
    """Build a multi-metric annual bloom severity comparison chart."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=('Peak SST by Year', 'Days Above Bloom Threshold'),
    )

    years_str = [str(y) for y in bloom_df['year']]
    bloom_threshold = CONFIG['thresholds']['sst_bloom']

    # Peak SST bars
    colors_sst = [
        '#ff6b6b' if v >= 25 else '#ffb347' if v >= 20 else '#2dd4a8'
        for v in bloom_df['peak_sst']
    ]
    fig.add_trace(go.Bar(
        x=years_str, y=bloom_df['peak_sst'],
        marker=dict(color=colors_sst, line=dict(width=0)),
        hovertemplate='%{x}<br>Peak SST: <b>%{y:.1f}°C</b><extra></extra>',
        showlegend=False,
    ), row=1, col=1)

    fig.add_hline(y=bloom_threshold, line_dash="dot",
                  line_color="rgba(255,107,107,0.4)", line_width=1,
                  row=1, col=1)

    # Days above threshold bars
    colors_days = [
        '#ff6b6b' if d >= 90 else '#ffb347' if d >= 60 else '#2dd4a8'
        for d in bloom_df['days_above_20']
    ]
    fig.add_trace(go.Bar(
        x=years_str, y=bloom_df['days_above_20'],
        marker=dict(color=colors_days, line=dict(width=0)),
        hovertemplate='%{x}<br>Days ≥' + str(bloom_threshold) + '°C: <b>%{y}</b><extra></extra>',
        showlegend=False,
    ), row=2, col=1)

    report_layout = {**PLOTLY_LAYOUT,
        'height': 420,
    }
    fig.update_layout(**report_layout)
    fig.update_annotations(font_size=12, font_color='#a8d8ea')
    fig.update_yaxes(title_text='°C', row=1, col=1)
    fig.update_yaxes(title_text='Days', row=2, col=1)
    return fig


def build_coverage_heatmap(coverage_df):
    """Build a data coverage heatmap (year x month, count of observation days)."""
    MONTH_ABBR = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    years = list(coverage_df.index)
    z_data = []
    for y in years:
        row = []
        for m in range(1, 13):
            row.append(int(coverage_df.loc[y, m]) if m in coverage_df.columns else 0)
        z_data.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=MONTH_ABBR,
        y=[str(y) for y in years],
        colorscale=[
            [0, '#0a1628'],
            [0.01, '#132744'],
            [0.3, '#1a5276'],
            [0.6, '#00d4ff'],
            [0.85, '#2dd4a8'],
            [1.0, '#ffffff'],
        ],
        hoverongaps=False,
        hovertemplate='%{y} %{x}<br><b>%{z} days</b><extra></extra>',
        colorbar=dict(
            title=dict(text='Days', font=dict(color='#a8d8ea', size=11)),
            tickfont=dict(color='#a8d8ea', size=10),
            bgcolor='rgba(0,0,0,0)',
        ),
        zmin=0,
        zmax=31,
    ))

    cov_layout = {**PLOTLY_LAYOUT,
        'title': dict(text='Data Coverage — Observation Days per Month',
                       font=dict(size=14, color='#e8f4f8')),
        'height': max(200, len(years) * 28 + 100),
        'xaxis': dict(side='top', tickangle=0, gridcolor='rgba(0,0,0,0)'),
        'yaxis': dict(autorange='reversed', gridcolor='rgba(0,0,0,0)'),
    }
    fig.update_layout(**cov_layout)
    return fig


# ──────────────────────────────────────────────────────────────
# Main dashboard
# ──────────────────────────────────────────────────────────────

def main():
    init_database()

    try:
        sst_df, chl_df, risk_df, history_df = load_latest_data()
        data_loaded = len(sst_df) > 0 or len(chl_df) > 0
    except Exception as e:
        st.error(f"Error loading data: {e}")
        data_loaded = False
        sst_df = chl_df = risk_df = history_df = pd.DataFrame()

    date_stats = get_date_range_stats()
    available_dates = get_available_dates()
    risk_data = get_current_risk()

    # ── Header ──
    col_h1, col_h2 = st.columns([4, 1])
    with col_h1:
        st.markdown("""
        <div class="brand-header">
            <h1>🌊 Lake Erie Monitor</h1>
            <span class="brand-tag">BlueNexus</span>
        </div>
        <div class="brand-subtitle">
            Satellite-derived SST &amp; chlorophyll monitoring · NOAA GLERL ERDDAP
        </div>
        """, unsafe_allow_html=True)
    with col_h2:
        if date_stats['sst']['count'] > 0:
            st.metric("Archive", f"{date_stats['unique_days']} days")

    st.markdown('<div class="nexus-divider"></div>', unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        if risk_data['date']:
            level_lower = risk_data['risk_level'].lower()
            emoji = get_risk_emoji(risk_data['risk_level'])
            st.markdown(f"""
            <div class="risk-badge {level_lower}">
                <div class="risk-icon">{emoji}</div>
                <div class="risk-level">{risk_data['risk_level']} Risk</div>
                <div class="risk-score">Score {risk_data['risk_score']} / 6</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">Current Readings</div>',
                    unsafe_allow_html=True)

        if len(sst_df) > 0:
            st.metric("Lake-Wide SST",
                      format_metric(sst_df.iloc[0]['lake_mean'], 1, "°C"))
            if len(chl_df) > 0 and chl_df.iloc[0]['west_basin_mean'] is not None:
                st.metric("West Basin Chl-a",
                          format_metric(chl_df.iloc[0]['west_basin_mean'], 1, "mg/m³"))
            else:
                st.metric("West Basin Chl-a", "—")

            if len(sst_df) >= 3:
                spark = list(reversed(sst_df['lake_mean'].dropna().tolist()))
                st.plotly_chart(build_sparkline(spark, '#ff6b6b'),
                                width='stretch', config={'displayModeBar': False})

            threshold = CONFIG['thresholds']['sst_bloom']
            days_warm = sum(1 for i in range(min(7, len(sst_df)))
                           if sst_df.iloc[i]['lake_mean'] is not None
                           and sst_df.iloc[i]['lake_mean'] > threshold)
            st.metric(f"Days >{threshold}°C", f"{days_warm} / 7")

        st.markdown('<div class="nexus-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Data Controls</div>',
                    unsafe_allow_html=True)

        # Refresh from ERDDAP - always visible, fetches latest 7 days
        col1, col2 = st.columns([0.92, 0.08])
        with col1:
            refresh_btn = st.button("🔄  Refresh from ERDDAP", width='stretch', key="refresh_btn")
        with col2:
            st.markdown("""
            <div style="padding-top: 4px;">
                <span title="Fetches the latest 7 days of SST and chlorophyll data from NOAA ERDDAP and updates the database. Does not delete any existing data."
                      style="cursor: help; color: var(--text-secondary);">ℹ️</span>
            </div>
            """, unsafe_allow_html=True)

        if refresh_btn:
            with st.spinner("Fetching latest satellite data..."):
                try:
                    # Clear cache BEFORE fetching to release any database connections
                    st.cache_data.clear()
                    gc.collect()
                    time.sleep(0.3)
                    fetch_latest_data()
                    st.success("✅ Data updated!")
                    time.sleep(0.8)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Fetch error: {e}")

        # Demo dataset button - ONLY visible when database is empty
        if not data_loaded:
            col1, col2 = st.columns([0.92, 0.08])
            with col1:
                demo_btn = st.button("📥  Load Demo Dataset", width='stretch', type="primary", key="demo_btn")
            with col2:
                st.markdown("""
                <div style="padding-top: 4px;">
                    <span title="Loads August 2023 harmful algal bloom event data (10 days) as a demonstration. Perfect for first-time users to see what the dashboard looks like with real data."
                          style="cursor: help; color: var(--text-secondary);">ℹ️</span>
                </div>
                """, unsafe_allow_html=True)

            if demo_btn:
                with st.spinner("Loading Aug 2023 bloom event..."):
                    try:
                        # Clear cache BEFORE fetching
                        st.cache_data.clear()
                        gc.collect()
                        time.sleep(0.3)
                        fetch_demo_dataset()
                        st.success("✅ Demo data loaded!")
                        time.sleep(0.8)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

        # Archive Builder - expandable section for manual archive building
        with st.expander("🗄️  Archive Builder", expanded=not data_loaded):
            st.markdown("""
            <div style="font-size: 0.82rem; color: var(--text-secondary); margin-bottom: 12px;">
                Build historical archives. Data coverage: SST (2007-present), Chlorophyll (2012-2023).
            </div>
            """, unsafe_allow_html=True)

            archive_mode = st.radio(
                "Select archive mode:",
                ["Quick (2023-2025)", "Full (2007-present)", "Custom Date Range"],
                key="archive_mode",
                index=0,
                help="Quick: bloom season data | Full: Complete historical archive | Custom: Specific date range"
            )

            # Yesterday is the latest date ERDDAP reliably serves
            yesterday = datetime.now().date() - timedelta(days=1)

            # ── Mode-specific UI ──
            if archive_mode == "Custom Date Range":
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input(
                        "Start Date",
                        value=datetime(2023, 5, 1),
                        min_value=datetime(2007, 1, 1),
                        max_value=yesterday,
                        key="archive_start",
                        help="SST data available from 2007-01-01"
                    )
                with col2:
                    end_date = st.date_input(
                        "End Date",
                        value=yesterday,
                        min_value=datetime(2007, 1, 1),
                        max_value=yesterday,
                        key="archive_end",
                        help="Satellite data is typically available through yesterday"
                    )

                if start_date > end_date:
                    st.warning("Start date must be before end date.")
                else:
                    _months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
                    _est_min = max(1, round(_months * 1.5))
                    st.info(f"Will fetch: **{start_date}** to **{end_date}** (~{_months} months, ~{_est_min} min)")

                    chl_start_d = datetime(2012, 3, 1).date()
                    chl_end_d = datetime(2023, 12, 31).date()
                    if start_date > chl_end_d or end_date < chl_start_d:
                        st.info("Chlorophyll not available for this range (CHL covers 2012-03 to 2023-12). SST only.")
                    elif start_date < chl_start_d or end_date > chl_end_d:
                        st.info(f"Chlorophyll will be fetched for: {max(start_date, chl_start_d)} to {min(end_date, chl_end_d)}")

            elif archive_mode == "Full (2007-present)":
                _full_months = (yesterday.year - 2007) * 12 + yesterday.month
                _full_est_hr = max(1, round(_full_months * 2.5 / 60, 1))
                st.info(f"Will fetch: **2007-01** to **{yesterday}** (~{_full_months} months, ~{_full_est_hr} hours)")
                st.caption("Tip: For large builds, `python build_archive.py --full` from the command line supports pause/resume with Ctrl+C.")

            elif archive_mode == "Quick (2023-2025)":
                st.info("Fetches bloom season months (May-Oct) for 2023-2025.")

            # ── Build button ──
            col1, col2 = st.columns([0.92, 0.08])
            with col1:
                build_btn = st.button("Build Archive", width='stretch', type="primary", key="build_btn")
            with col2:
                st.markdown("""
                <div style="padding-top: 4px;">
                    <span title="Downloads satellite data month-by-month from NOAA ERDDAP. Large ranges may take a while. Progress is shown during the build."
                          style="cursor: help; color: var(--text-secondary);">ℹ️</span>
                </div>
                """, unsafe_allow_html=True)

            if build_btn:
                try:
                    st.cache_data.clear()
                    now = datetime.now()

                    # ── Build month list based on mode ──
                    month_list = []

                    if archive_mode == "Quick (2023-2025)":
                        bloom_months = [5, 6, 7, 8, 9, 10]
                        for y in [2023, 2024, 2025]:
                            for m in bloom_months:
                                if (y < now.year) or (y == now.year and m <= now.month):
                                    month_list.append((y, m))

                    elif archive_mode == "Full (2007-present)":
                        y, m = 2007, 1
                        while (y, m) <= (yesterday.year, yesterday.month):
                            month_list.append((y, m))
                            m += 1
                            if m > 12:
                                m = 1
                                y += 1

                    elif archive_mode == "Custom Date Range":
                        if start_date > end_date:
                            st.error("Invalid date range.")
                            month_list = []
                        else:
                            y, m = start_date.year, start_date.month
                            while (y, m) <= (end_date.year, end_date.month):
                                month_list.append((y, m))
                                m += 1
                                if m > 12:
                                    m = 1
                                    y += 1

                    if not month_list:
                        st.warning("No months to fetch.")
                    else:
                        # ── Month-by-month fetch loop ──
                        total = len(month_list)
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        elapsed_container = st.empty()

                        processed = 0
                        failed_months = []
                        sst_total = 0
                        chl_total = 0
                        build_start = time.time()
                        request_delay = CONFIG.get("erddap", {}).get("request_delay", 5)
                        consecutive_failures = 0

                        # Determine date boundaries for custom mode
                        custom_start = start_date if archive_mode == "Custom Date Range" else None
                        custom_end = end_date if archive_mode == "Custom Date Range" else None

                        for y, m in month_list:
                            elapsed_sec = time.time() - build_start
                            if processed > 0:
                                avg_per_month = elapsed_sec / processed
                                remaining = avg_per_month * (total - processed)
                                eta_str = f" | ~{int(remaining // 60)}m {int(remaining % 60)}s remaining"
                            else:
                                eta_str = ""

                            # Adaptive cooldown: slow down when server is struggling
                            if consecutive_failures >= 5:
                                cooldown = min(300, 30 * consecutive_failures)
                                status_text.text(f"Server cooldown ({cooldown}s)... {consecutive_failures} consecutive failures")
                                time.sleep(cooldown)
                            elif consecutive_failures >= 2:
                                cooldown = 15 * consecutive_failures
                                status_text.text(f"Extended delay ({cooldown}s)...")
                                time.sleep(cooldown)

                            status_text.text(f"Fetching {y}-{m:02d}... ({processed + 1}/{total}){eta_str}")

                            try:
                                # Month start
                                if custom_start and (y, m) == (custom_start.year, custom_start.month):
                                    m_start = custom_start.strftime("%Y-%m-%d")
                                else:
                                    m_start = f"{y}-{m:02d}-01"

                                # Month end
                                if custom_end and (y, m) == (custom_end.year, custom_end.month):
                                    m_end_dt = custom_end
                                elif m == 12:
                                    m_end_dt = (datetime(y + 1, 1, 1) - timedelta(days=1)).date()
                                else:
                                    m_end_dt = (datetime(y, m + 1, 1) - timedelta(days=1)).date()

                                # Cap to yesterday
                                if m_end_dt >= now.date():
                                    m_end_dt = yesterday
                                m_end = m_end_dt.strftime("%Y-%m-%d")

                                result = fetch_historical_range(m_start, m_end)

                                # Track success for adaptive cooldown
                                month_ok = False
                                if result.get("sst") == "success":
                                    sst_total += 1
                                    month_ok = True
                                if result.get("chl") == "success":
                                    chl_total += 1
                                    month_ok = True

                                if month_ok:
                                    consecutive_failures = 0
                                else:
                                    consecutive_failures += 1

                            except Exception as month_err:
                                failed_months.append(f"{y}-{m:02d}: {month_err}")
                                consecutive_failures += 1

                            processed += 1
                            progress_bar.progress(processed / total)
                            time.sleep(request_delay)

                        progress_bar.empty()
                        status_text.empty()
                        elapsed_container.empty()

                        # ── Summary ──
                        total_elapsed = time.time() - build_start
                        elapsed_str = f"{int(total_elapsed // 60)}m {int(total_elapsed % 60)}s"

                        if failed_months:
                            st.warning(
                                f"Archive build done in {elapsed_str} with "
                                f"{len(failed_months)}/{total} months that had issues. "
                                f"SST: {sst_total} months, CHL: {chl_total} months. Check logs for details."
                            )
                        else:
                            st.success(
                                f"Archive complete in {elapsed_str}! "
                                f"Processed {total} months. SST: {sst_total}, CHL: {chl_total}."
                            )

                        # Clear all cached data so the dashboard loads fresh results
                        st.cache_data.clear()
                        time.sleep(1.0)
                        st.rerun()

                except Exception as e:
                    st.error(f"Archive build error: {e}")

        # Reset All Data - dangerous operation with confirmation
        if data_loaded:
            st.markdown('<div class="nexus-divider"></div>', unsafe_allow_html=True)

            with st.expander("⚠️  Reset All Data", expanded=False):
                st.markdown("""
                <div style="font-size: 0.82rem; color: var(--warm-coral); margin-bottom: 12px;">
                    <strong>⚠️ WARNING:</strong> This will delete ALL archived data and reset the database.
                    Use this only if experiencing data issues.
                </div>
                """, unsafe_allow_html=True)

                # Two-step confirmation
                confirm = st.checkbox("I understand this will delete all data", key="reset_confirm")

                if confirm:
                    col1, col2 = st.columns([0.92, 0.08])
                    with col1:
                        reset_btn = st.button("🗑️  Reset Database", type="secondary", width='stretch', key="reset_btn")
                    with col2:
                        st.markdown("""
                        <div style="padding-top: 4px;">
                            <span title="Permanently deletes all archived data including SST, chlorophyll, bloom risk scores, and spatial maps. This action cannot be undone. The database will be completely reset to empty state."
                                  style="cursor: help; color: var(--warm-coral);">ℹ️</span>
                        </div>
                        """, unsafe_allow_html=True)

                    if reset_btn:
                        try:
                            # Delete database files
                            db_files = [
                                DB_PATH,
                                Path(str(DB_PATH) + "-wal"),
                                Path(str(DB_PATH) + "-shm")
                            ]

                            for db_file in db_files:
                                if db_file.exists():
                                    db_file.unlink()

                            st.cache_data.clear()
                            st.success("✅ Database reset complete! Reloading...")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Reset error: {e}")

        st.markdown('<div class="nexus-divider"></div>', unsafe_allow_html=True)

        if date_stats['sst']['max']:
            freshness = get_data_freshness_status(
                date_stats['sst']['max'] + "T12:00:00")
            sc = 'online' if freshness[0] == 'Fresh' else 'stale' if freshness[0] == 'Stale' else 'offline'
            st.markdown(f"""
            <div style="font-size: 0.75rem; color: var(--text-secondary);">
                <span class="status-dot {sc}"></span>
                Latest: {date_stats['sst']['max']}<br>
                <span style="margin-left: 14px;">Next update: {calculate_time_until_next_update()}</span>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("ℹ️  About"):
            st.markdown("""
            **Data Sources**
            - **SST:** NOAA GLSEA ACSPO (2007–present)
            - **Chlorophyll:** Great Lakes Daily CHL (2012-03 to 2023-12)

            **Quick Controls:**
            - 🔄 Refresh: Latest 7 days
            - 📥 Demo: Aug 2023 bloom event (first-time only)
            - 🗄️ Archive: Historical data builder
            - ⚠️ Reset: Delete all data (troubleshooting)
            """)


    # ── Alert banner ──
    if data_loaded and risk_data['date']:
        msg = get_alert_message(risk_data)
        if risk_data['risk_level'] == 'High':
            st.error(msg)
        elif risk_data['risk_level'] == 'Moderate':
            st.warning(msg)
        else:
            st.success(msg)

    # ── Welcome (no data) ──
    if not data_loaded:
        st.markdown("""
        ### Welcome to Lake Erie Monitor
        Load satellite data to begin — use **"Load Demo Dataset"** in the sidebar,
        or **"Refresh from ERDDAP"** for the latest 7 days.
        """)
        return

    # ── Tabs ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "Trends & Analysis", "Spatial Explorer", "Field Guide", "System Status"])

    # ═══════════ TAB 1: Overview ═══════════
    with tab1:
        if len(sst_df) > 0:
            latest = sst_df.iloc[0]['date']
            st.markdown(f'<div class="section-label">Latest Observation — {format_date(latest)}</div>',
                        unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Lake-Wide SST",
                          format_metric(sst_df.iloc[0]['lake_mean'], 1, "°C"),
                          help="Average surface temperature across the entire lake. "
                               "Temperatures above 20°C create favorable conditions for "
                               "harmful algal blooms (HABs) in western Lake Erie.")
            with c2:
                st.metric("Peak SST",
                          format_metric(sst_df.iloc[0]['lake_max'], 1, "°C"),
                          help="Highest temperature observed anywhere on the lake surface. "
                               "Localized hotspots, especially in shallow western areas, "
                               "can trigger bloom growth even when the lake average is moderate.")
            with c3:
                st.metric("West Basin SST",
                          format_metric(sst_df.iloc[0]['west_basin_mean'], 1, "°C"),
                          help="Average temperature in the western basin (west of ~82°W). "
                               "This shallow, nutrient-rich area is where most HABs originate. "
                               "It warms faster than the deeper eastern basin.")
            with c4:
                val = chl_df.iloc[0]['west_basin_mean'] if len(chl_df) > 0 else None
                st.metric("West Basin Chl-a",
                          format_metric(val, 1, "mg/m³") if val is not None else "—",
                          help="Chlorophyll-a concentration in the western basin — a proxy for "
                               "algae density. Values above 10 mg/m³ indicate elevated algal "
                               "activity; above 40 mg/m³ suggests an active bloom.")

            # ── Dynamic insight callout ──
            sst_val = sst_df.iloc[0]['lake_mean']
            chl_val = val
            if sst_val is not None:
                insight_parts = []
                if sst_val < 10:
                    insight_parts.append(
                        "🧊 **Cold season** — Lake Erie is well below bloom-supporting "
                        "temperatures. Ice formation may be occurring in shallow areas.")
                elif sst_val < 15:
                    insight_parts.append(
                        "🌊 **Cool conditions** — Water temperatures are below the 20°C "
                        "threshold needed for significant bloom development. Bloom risk is minimal.")
                elif sst_val < 20:
                    insight_parts.append(
                        "🌤️ **Transitional period** — Temperatures are approaching but haven't "
                        "reached the 20°C bloom threshold. This is the watch zone — conditions "
                        "could shift within days during warming trends.")
                elif sst_val < 25:
                    insight_parts.append(
                        "🌡️ **Bloom-favorable temperatures** — Lake-wide SST exceeds 20°C, "
                        "which supports cyanobacteria growth. Monitor chlorophyll levels closely, "
                        "especially in the western basin.")
                else:
                    insight_parts.append(
                        "🔥 **Peak summer heat** — Surface temperatures are at or near annual "
                        "maximums. Combined with nutrient loading from the Maumee River, these "
                        "conditions strongly favor harmful algal bloom development.")

                if chl_val is not None:
                    if chl_val > 40:
                        insight_parts.append(
                            "🟥 **Active bloom detected** — Western basin chlorophyll exceeds "
                            "40 mg/m³, indicating a significant algal bloom is underway.")
                    elif chl_val > 20:
                        insight_parts.append(
                            "🟧 **Elevated chlorophyll** — Western basin levels suggest moderate "
                            "algal activity. Conditions should be monitored for escalation.")
                    elif chl_val > 10:
                        insight_parts.append(
                            "🟨 **Slightly elevated chlorophyll** — Above baseline but not yet "
                            "at concerning levels. Normal for productive summer months.")

                st.markdown('<div class="nexus-divider"></div>', unsafe_allow_html=True)
                for part in insight_parts:
                    st.info(part)

            # ── Historical anomaly context ──
            hist_avgs = load_historical_averages()
            if hist_avgs and sst_val is not None and len(sst_df) > 0:
                latest_date = sst_df.iloc[0]['date']
                try:
                    doy = pd.to_datetime(latest_date).dayofyear
                    if doy in hist_avgs:
                        hist_avg = hist_avgs[doy]
                        anomaly = sst_val - hist_avg
                        direction = "above" if anomaly > 0 else "below"
                        abs_anomaly = abs(anomaly)
                        if abs_anomaly >= 0.5:
                            st.markdown(
                                f'<div style="background: rgba(0,212,255,0.06); '
                                f'border-left: 3px solid #00d4ff; padding: 10px 14px; '
                                f'border-radius: 4px; margin: 4px 0; font-size: 0.88rem;">'
                                f'📊 <strong>Historical context:</strong> '
                                f'Current SST ({sst_val:.1f}°C) is '
                                f'<strong>{abs_anomaly:.1f}°C {direction}</strong> the '
                                f'long-term average of {hist_avg:.1f}°C for this day of year '
                                f'(based on {len(hist_avgs)} days of archive data).'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                except Exception:
                    pass  # Graceful degradation

        conditions = get_recent_conditions(7)
        if conditions['sst_avg'] is not None:
            st.markdown('<div class="nexus-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-label">7-Day Summary</div>',
                        unsafe_allow_html=True)
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.metric("Avg SST", format_metric(conditions['sst_avg'], 1, "°C"),
                          help="7-day running average of lake-wide surface temperature. "
                               "Sustained warmth is more significant for bloom development "
                               "than a single hot day.")
            with s2:
                st.metric("Max SST", format_metric(conditions['sst_max'], 1, "°C"),
                          help="Highest single-day peak temperature in the past week. "
                               "Extreme values can indicate localized heating events.")
            with s3:
                st.metric(f"Days >{CONFIG['thresholds']['sst_bloom']}°C",
                          f"{conditions['days_sst_above_threshold']} / {conditions['period_days']}",
                          help="How many of the last 7 days had lake-wide SST above the "
                               f"{CONFIG['thresholds']['sst_bloom']}°C bloom threshold. "
                               "Consecutive warm days compound bloom risk.")
            with s4:
                st.metric("Avg Chl (West)",
                          format_metric(conditions['chl_avg'], 1, "mg/m³")
                          if conditions['chl_avg'] else "—",
                          help="Average chlorophyll-a in the western basin over the past 7 days. "
                               "Rising trends here are an early warning of bloom intensification.")

            # ── 7-day trend direction ──
            if len(sst_df) >= 3:
                recent_vals = list(reversed(sst_df['lake_mean'].tolist()))
                # Compare last 3 days vs previous 3 days (or whatever's available)
                midpoint = min(3, len(recent_vals) // 2)
                if midpoint >= 1:
                    recent_avg = sum(recent_vals[-midpoint:]) / midpoint
                    earlier_avg = sum(recent_vals[:midpoint]) / midpoint
                    trend_diff = recent_avg - earlier_avg
                    if abs(trend_diff) >= 0.3:
                        if trend_diff > 0:
                            trend_icon = "📈"
                            trend_text = f"SST trending **upward** (+{trend_diff:.1f}°C over the past week)"
                            trend_color = "#ffb347"
                        else:
                            trend_icon = "📉"
                            trend_text = f"SST trending **downward** ({trend_diff:.1f}°C over the past week)"
                            trend_color = "#00d4ff"
                        st.markdown(
                            f'<div style="background: rgba({int(trend_color[1:3],16)},'
                            f'{int(trend_color[3:5],16)},{int(trend_color[5:7],16)},0.08); '
                            f'border-left: 3px solid {trend_color}; padding: 8px 14px; '
                            f'border-radius: 4px; margin-top: 8px; font-size: 0.85rem;">'
                            f'{trend_icon} {trend_text}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

        if len(sst_df) >= 2:
            st.markdown('<div class="nexus-divider"></div>', unsafe_allow_html=True)
            st.plotly_chart(build_trends_chart(sst_df, chl_df, "Recent 7-Day Trends"),
                            width='stretch', config={'displayModeBar': False})

    # ═══════════ TAB 2: Trends & Analysis ═══════════
    with tab2:
        st.markdown('<div class="section-label">Trend Analysis</div>',
                    unsafe_allow_html=True)

        range_opt = st.radio("Range", ["Last 30 days", "Last 90 days", "Full Archive"],
                             horizontal=True, label_visibility="collapsed")

        if range_opt == "Full Archive":
            sst_full = load_full_sst_series()
            if len(sst_full) > 0:
                st.plotly_chart(build_long_term_chart(sst_full), width='stretch')
                st.caption(f"{len(sst_full)} observations · {sst_full.iloc[0]['date']} — "
                           f"{sst_full.iloc[-1]['date']}")
            else:
                st.info("No archive data. Run `python build_archive.py`.")
        else:
            days = 30 if "30" in range_opt else 90
            ext_sst, ext_chl, ext_risk = load_extended_data(days)
            if len(ext_sst) >= 2:
                st.plotly_chart(build_trends_chart(ext_sst, ext_chl, f"SST & Chlorophyll — {range_opt}"),
                                width='stretch')
                if len(ext_risk) >= 2:
                    st.plotly_chart(build_risk_chart(ext_risk), width='stretch')
            else:
                st.info("Not enough data for this range.")

        # Basin comparison
        if len(sst_df) >= 2:
            st.markdown('<div class="nexus-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Basin Comparison</div>',
                        unsafe_allow_html=True)
            bfig = go.Figure()
            dr = list(reversed(sst_df['date'].tolist()))
            bfig.add_trace(go.Scatter(
                x=dr, y=list(reversed(sst_df['west_basin_mean'].tolist())),
                name='West Basin', line=dict(color='#ff6b6b', width=2),
                mode='lines+markers', marker=dict(size=5)))
            bfig.add_trace(go.Scatter(
                x=dr, y=list(reversed(sst_df['east_basin_mean'].tolist())),
                name='East Basin', line=dict(color='#00d4ff', width=2),
                mode='lines+markers', marker=dict(size=5)))
            basin_layout = {**PLOTLY_LAYOUT,
                'height': 280,
                'title': dict(text='West vs East Basin SST', font=dict(size=14, color='#e8f4f8')),
                'legend': dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0,
                               bgcolor='rgba(0,0,0,0)', borderwidth=0),
                'yaxis_title': 'SST (°C)',
            }
            bfig.update_layout(**basin_layout)
            bfig.update_xaxes(tickformat='%b %d')
            st.plotly_chart(bfig, width='stretch', config={'displayModeBar': False})

        # ── Seasonal Heatmap ──
        monthly_data = load_monthly_matrix()
        if 'sst' in monthly_data and len(monthly_data['sst']) > 1:
            st.markdown('<div class="nexus-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Seasonal Patterns</div>',
                        unsafe_allow_html=True)

            hm_type = st.radio(
                "Variable",
                ["SST (Lake-wide)", "Chlorophyll (West Basin)"],
                horizontal=True, key="heatmap_type", label_visibility="collapsed",
            )

            if hm_type == "SST (Lake-wide)" and len(monthly_data.get('sst', pd.DataFrame())) > 0:
                st.plotly_chart(
                    build_seasonal_heatmap(
                        monthly_data['sst'],
                        'Monthly Average SST — All Years',
                        [[0, '#0a1628'], [0.3, '#00d4ff'], [0.5, '#2dd4a8'],
                         [0.7, '#ffb347'], [1.0, '#ff6b6b']],
                        '°C',
                    ),
                    width='stretch', config={'displayModeBar': False},
                )
            elif hm_type == "Chlorophyll (West Basin)" and len(monthly_data.get('chl', pd.DataFrame())) > 0:
                st.plotly_chart(
                    build_seasonal_heatmap(
                        monthly_data['chl'],
                        'Monthly Average Chlorophyll (West Basin) — All Years',
                        [[0, '#0a1628'], [0.25, '#1a3a2a'], [0.5, '#2dd4a8'],
                         [0.75, '#ffb347'], [1.0, '#ff6b6b']],
                        'mg/m³',
                    ),
                    width='stretch', config={'displayModeBar': False},
                )
            else:
                st.info("No chlorophyll data available for heatmap.")

            st.caption("Each cell shows the monthly mean. Gaps appear where satellite data was unavailable (cloud cover, ice, or outside dataset coverage).")

        # ── Year-over-Year Comparison ──
        sst_full_yoy = load_full_sst_series()
        if len(sst_full_yoy) > 60:
            st.markdown('<div class="nexus-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Year-over-Year Comparison</div>',
                        unsafe_allow_html=True)

            available_years = sorted(pd.to_datetime(sst_full_yoy['date']).dt.year.unique())

            yoy_c1, yoy_c2 = st.columns([3, 1])
            with yoy_c1:
                # Default to most recent 3 years
                default_years = available_years[-3:] if len(available_years) >= 3 else available_years
                sel_years = st.multiselect(
                    "Select years to compare",
                    options=available_years,
                    default=default_years,
                    key="yoy_years",
                    max_selections=8,
                )
            with yoy_c2:
                season_opt = st.selectbox(
                    "Season",
                    ["Full Year (Jan-Dec)", "Bloom Season (May-Oct)",
                     "Summer Peak (Jun-Sep)", "Shoulder (Mar-May)"],
                    key="yoy_season",
                )

            month_ranges = {
                "Full Year (Jan-Dec)": (1, 12),
                "Bloom Season (May-Oct)": (5, 10),
                "Summer Peak (Jun-Sep)": (6, 9),
                "Shoulder (Mar-May)": (3, 5),
            }
            m_range = month_ranges.get(season_opt, (1, 12))

            if sel_years and len(sel_years) >= 1:
                st.plotly_chart(
                    build_yoy_chart(sst_full_yoy, sel_years, m_range),
                    width='stretch',
                    config={'displayModeBar': False},
                )
            else:
                st.info("Select at least one year to compare.")

        # ── Data Export ──
        st.markdown('<div class="nexus-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Data Export</div>',
                    unsafe_allow_html=True)

        exp_c1, exp_c2 = st.columns(2)
        with exp_c1:
            sst_export = load_full_sst_series()
            if len(sst_export) > 0:
                sst_csv = sst_export.to_csv(index=False)
                st.download_button(
                    label=f"Download SST Data ({len(sst_export):,} records)",
                    data=sst_csv,
                    file_name="bluenexus_sst_export.csv",
                    mime="text/csv",
                    width='stretch',
                )
            else:
                st.info("No SST data to export.")
        with exp_c2:
            chl_export = load_full_chl_series()
            if len(chl_export) > 0:
                chl_csv = chl_export.to_csv(index=False)
                st.download_button(
                    label=f"Download CHL Data ({len(chl_export):,} records)",
                    data=chl_csv,
                    file_name="bluenexus_chl_export.csv",
                    mime="text/csv",
                    width='stretch',
                )
            else:
                st.info("No chlorophyll data to export.")

        st.caption("Exported as CSV with columns: date, lake_mean, lake_max, west_basin_mean, east_basin_mean. Ready for import into Python, R, Excel, or any data science tool.")

        # ── Bloom Season Report Card ──
        bloom_stats = load_bloom_season_stats()
        if len(bloom_stats) >= 2:
            st.markdown('<div class="nexus-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Bloom Season Report Card</div>',
                        unsafe_allow_html=True)
            st.caption("Annual comparison of bloom-season severity (June–October). "
                       "Higher peak SST and more days above the bloom threshold "
                       "correlate with larger, more intense harmful algal blooms.")

            st.plotly_chart(
                build_bloom_report_chart(bloom_stats),
                width='stretch',
                config={'displayModeBar': False},
            )

            # Summary table
            display_df = bloom_stats.copy()
            display_df = display_df.rename(columns={
                'year': 'Year',
                'peak_sst': 'Peak SST (°C)',
                'avg_sst': 'Avg SST (°C)',
                'days_above_20': 'Days ≥20°C',
                'peak_chl': 'Peak Chl (mg/m³)',
                'avg_chl': 'Avg Chl (mg/m³)',
                'data_days': 'Data Days',
            })
            display_df['Year'] = display_df['Year'].astype(int)

            with st.expander("View detailed annual statistics"):
                st.dataframe(
                    display_df,
                    hide_index=True,
                    width='stretch',
                )

    # ═══════════ TAB 3: Spatial Explorer ═══════════
    with tab3:
        st.markdown('<div class="section-label">Spatial Data Explorer</div>',
                    unsafe_allow_html=True)

        if not available_dates:
            st.info("No spatial data available. Load demo data or run the archive builder.")
        else:
            grouped = group_dates_by_month(available_dates)
            years = sorted(grouped.keys(), reverse=True)
            MONTH_NAMES = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                           7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}

            nc1, nc2, nc3 = st.columns([1, 1, 2])

            with nc1:
                sel_year = st.selectbox("Year", years, key="map_year")

            with nc2:
                months_avail = sorted(grouped.get(sel_year, {}).keys(), reverse=True)
                mo_labels = [f"{MONTH_NAMES[m]} ({len(grouped[sel_year][m])}d)"
                             for m in months_avail]
                sel_mo_idx = st.selectbox("Month", range(len(months_avail)),
                                          format_func=lambda i: mo_labels[i],
                                          key="map_month")
                sel_month = months_avail[sel_mo_idx]

            with nc3:
                days_list = grouped[sel_year][sel_month]
                sel_day_idx = st.selectbox("Day", range(len(days_list)),
                                           format_func=lambda i: format_date(days_list[i]),
                                           key="map_day")
                selected_date = days_list[sel_day_idx]

            # Availability summary
            total_possible = monthrange(sel_year, sel_month)[1]
            st.caption(f"📊 {len(days_list)} of {total_possible} days available in "
                       f"{MONTH_NAMES[sel_month]} {sel_year}")

            # Quick jumps for bloom season
            bloom_dates = [d for d in available_dates
                           if datetime.strptime(d, "%Y-%m-%d").month in [7, 8, 9]]
            if bloom_dates and len(bloom_dates) > 5:
                with st.expander("⚡ Quick Jump — Bloom Season"):
                    jcols = st.columns(6)
                    for i, d in enumerate(bloom_dates[:18]):
                        with jcols[i % 6]:
                            if st.button(d, key=f"j_{d}", width='stretch'):
                                dt = datetime.strptime(d, "%Y-%m-%d")
                                st.session_state['map_year'] = dt.year
                                ma = sorted(grouped.get(dt.year, {}).keys(), reverse=True)
                                if dt.month in ma:
                                    st.session_state['map_month'] = ma.index(dt.month)
                                dl = grouped.get(dt.year, {}).get(dt.month, [])
                                if d in dl:
                                    st.session_state['map_day'] = dl.index(d)
                                st.rerun()

            st.markdown('<div class="nexus-divider"></div>', unsafe_allow_html=True)

            # ── Legend bar (compact, above maps) ──
            lg1, lg2 = st.columns(2)
            with lg1:
                st.markdown("""
                <div class="nexus-card" style="padding: 12px 18px;">
                    <div class="nexus-card-header" style="margin-bottom: 4px;">SST Scale</div>
                    <span style="font-size: 0.8rem;">
                        🔴 Warm (&gt;25°C) &nbsp;·&nbsp;
                        🟡 Moderate (15–25°C) &nbsp;·&nbsp;
                        🔵 Cool (&lt;15°C)
                    </span>
                </div>
                """, unsafe_allow_html=True)
            with lg2:
                st.markdown("""
                <div class="nexus-card" style="padding: 12px 18px;">
                    <div class="nexus-card-header" style="margin-bottom: 4px;">Bloom Thresholds</div>
                    <span style="font-size: 0.8rem;">
                        SST &gt; 20°C promotes growth &nbsp;·&nbsp;
                        Chl &gt; 10 elevated &nbsp;·&nbsp;
                        Chl &gt; 20 moderate &nbsp;·&nbsp;
                        Chl &gt; 40 active bloom
                    </span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="nexus-divider"></div>', unsafe_allow_html=True)

            # ── Map rendering (full-width, matching Plotly chart proportions) ──
            conn_sp = sqlite3.connect(f'file:{DB_PATH}?mode=ro', uri=True)
            cur = conn_sp.cursor()

            # -- SST map --
            cur.execute("SELECT lats, lons, data, shape_rows, shape_cols "
                        "FROM sst_spatial WHERE date = ?", (selected_date,))
            sst_row = cur.fetchone()
            if sst_row:
                try:
                    data, lats, lons = _load_spatial_grid(sst_row)
                    st.markdown(f"**🌡️ Sea Surface Temperature — {format_date(selected_date)}**")
                    m = create_interactive_map(data, lats, lons,
                                              label="SST", unit="°C", colors="temperature")
                    st_folium(m, use_container_width=True, height=520, returned_objects=[])
                except ValueError as e:
                    st.warning(f"SST spatial data issue for this date — skipping render. ({e})")
            else:
                st.info("No SST spatial data for this date.")

            # -- Chlorophyll map --
            cur.execute("SELECT lats, lons, data, shape_rows, shape_cols "
                        "FROM chl_spatial WHERE date = ?", (selected_date,))
            chl_row = cur.fetchone()
            if chl_row:
                try:
                    dc, lc, lonc = _load_spatial_grid(chl_row)
                    st.markdown(f"**🦠 Chlorophyll-a — {format_date(selected_date)}**")
                    m2 = create_interactive_map(dc, lc, lonc,
                                               label="Chlorophyll", unit="mg/m³",
                                               colors="viridis")
                    st_folium(m2, use_container_width=True, height=520, returned_objects=[])
                except ValueError as e:
                    st.warning(f"Chlorophyll spatial data issue for this date — skipping render. ({e})")

            conn_sp.close()

    # ═══════════ TAB 4: Field Guide ═══════════
    with tab4:
        st.markdown('<div class="section-label">Welcome Aboard BlueNexus</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        This guide helps you understand every measurement on this dashboard — whether
        you're a limnologist tracking bloom dynamics or someone seeing these numbers
        for the first time.
        """)

        st.markdown('<div class="nexus-divider"></div>', unsafe_allow_html=True)

        # ── Section 1: What We're Watching ──
        st.markdown("### 🔭 What We're Monitoring & Why")
        st.markdown("""
        Lake Erie — the shallowest and warmest of the Great Lakes — has experienced
        increasingly severe **harmful algal blooms (HABs)** since the mid-2000s. These
        blooms, dominated by the cyanobacterium *Microcystis*, can produce **microcystin**,
        a liver toxin that threatens drinking water supplies and recreation.

        In 2014, a bloom forced Toledo, Ohio to issue a **"Do Not Drink" advisory** for
        500,000 residents. Since then, monitoring lake conditions has become critical
        infrastructure.

        This dashboard tracks two satellite-derived measurements that together predict
        bloom risk: **sea surface temperature (SST)** and **chlorophyll-a concentration**.
        """)

        st.markdown('<div class="nexus-divider"></div>', unsafe_allow_html=True)

        # ── Section 2: Reading the Metrics ──
        st.markdown("### 📊 Understanding the Measurements")

        with st.expander("🌡️ Sea Surface Temperature (SST)", expanded=True):
            st.markdown(f"""
            **What it is:** The temperature of the lake's surface layer as measured by
            infrared satellite sensors (NOAA's ACSPO system).

            **Why it matters:** Cyanobacteria like *Microcystis* thrive in warm water.
            Below ~15°C they're mostly dormant. Between 15–20°C they grow slowly. Above
            **{CONFIG['thresholds']['sst_bloom']}°C** growth rates accelerate significantly,
            and above 25°C conditions are ideal for rapid bloom expansion.

            **What to look for:**
            - **Lake-Wide SST** — The average across all of Lake Erie. Gives the big picture.
            - **West Basin SST** — The western basin (west of ~82°W longitude) is shallower
              and warms faster. It's also where the Maumee River delivers nutrient-rich
              agricultural runoff, making it ground zero for blooms.
            - **Peak SST** — The hottest point on the lake. Localized hotspots can seed
              bloom development even when the average is moderate.
            - **Days above {CONFIG['thresholds']['sst_bloom']}°C** — Sustained warmth matters
              more than a single hot day. Multiple consecutive warm days compound bloom risk.

            **Limitations:** Satellites measure the skin layer (~top 1mm). Cloud cover and
            ice prevent measurements, which is why winter data has gaps.
            """)

        with st.expander("🦠 Chlorophyll-a (Chl-a)"):
            st.markdown(f"""
            **What it is:** A green pigment found in all photosynthetic organisms, including
            algae and cyanobacteria. Satellite sensors detect it by measuring the color of
            reflected light from the lake surface.

            **Why it matters:** Chlorophyll-a concentration is a **proxy for algal biomass** —
            more chlorophyll means more algae in the water. It doesn't distinguish between
            harmful cyanobacteria and harmless green algae, but elevated levels warrant attention.

            **Thresholds used in this dashboard:**
            | Level | Chl-a (mg/m³) | Interpretation |
            |-------|---------------|----------------|
            | Normal | < {CONFIG['thresholds']['chl_low']} | Background levels, healthy lake |
            | Elevated | {CONFIG['thresholds']['chl_low']}–{CONFIG['thresholds']['chl_moderate']} | Above baseline, worth monitoring |
            | Moderate Bloom | {CONFIG['thresholds']['chl_moderate']}–{CONFIG['thresholds']['chl_high']} | Significant algal activity |
            | Active Bloom | > {CONFIG['thresholds']['chl_high']} | Likely HAB in progress |

            **Limitations:** Satellite chlorophyll works best in open water. Near-shore readings
            can be skewed by sediment. Cloud cover creates data gaps. The current chlorophyll
            dataset covers 2012–2023.
            """)

        with st.expander("⚠️ Bloom Risk Score"):
            st.markdown(f"""
            **What it is:** A composite score (0–6) calculated daily from SST and chlorophyll
            conditions. It provides a quick at-a-glance risk assessment.

            **How it's calculated:**

            | Condition | Points |
            |-----------|--------|
            | Lake-wide SST > {CONFIG['thresholds']['sst_bloom']}°C | +1 |
            | Western basin chlorophyll > {CONFIG['thresholds']['chl_low']} mg/m³ | +1 |
            | Western basin chlorophyll > {CONFIG['thresholds']['chl_moderate']} mg/m³ | +2 (replaces +1) |
            | Western basin chlorophyll > {CONFIG['thresholds']['chl_high']} mg/m³ | +3 (replaces +2) |

            **Risk levels:**
            - 🟢 **Low (0–1):** Conditions not favorable for blooms
            - 🟡 **Moderate (2–3):** Bloom possible, monitor closely
            - 🔴 **High (4+):** Active bloom likely or confirmed

            **Important:** This is a simplified screening tool, not a toxin forecast.
            Actual bloom severity depends on additional factors like wind patterns,
            nutrient loading, and water column mixing that satellites can't measure.
            """)

        st.markdown('<div class="nexus-divider"></div>', unsafe_allow_html=True)

        # ── Section 3: Using the Dashboard ──
        st.markdown("### 🧭 Navigating the Dashboard")

        with st.expander("📋 Overview Tab"):
            st.markdown("""
            Your command center. Shows the most recent day's readings, a 7-day summary,
            and a trend chart. The **dynamic insight callouts** (blue info boxes) interpret
            the current conditions in plain language — they change automatically based on
            what the data shows.

            **Tip:** Hover over the small ⓘ icons next to each metric for a quick
            explanation of what that number means.
            """)

        with st.expander("📈 Trends & Analysis Tab"):
            st.markdown("""
            Explore temporal patterns at different scales. Use the radio buttons to switch
            between 30-day, 90-day, and full archive views.

            **Full Archive** mode shows your entire dataset with an interactive timeline
            slider — drag to zoom into any period. The 1M/3M/6M/1Y buttons above the chart
            are quick-zoom shortcuts.

            **Basin Comparison** reveals how the west and east basins diverge — the western
            basin typically warms earlier and cools later, creating a longer bloom window.

            **Tip:** Hover over any data point on the Plotly charts for exact values and dates.
            """)

        with st.expander("🗺️ Spatial Explorer Tab"):
            st.markdown("""
            View satellite imagery as interactive maps. The **Year → Month → Day** picker
            lets you navigate efficiently through hundreds of dates.

            **SST maps** use a thermal color scale: blue = cold, red = warm. Look for
            temperature gradients — the western basin warming first in spring is a classic
            pattern.

            **Chlorophyll maps** (available for 2012–2023) use a green color scale. Bright
            green hotspots in the western basin indicate areas of concentrated algal growth.

            **Quick Jump** (in the expander) takes you directly to summer bloom-season dates
            where you're most likely to see interesting patterns.
            """)

        st.markdown('<div class="nexus-divider"></div>', unsafe_allow_html=True)

        # ── Section 4: The Science ──
        st.markdown("### 🔬 The Science Behind the Data")

        with st.expander("How Satellites See the Lake"):
            st.markdown("""
            The SST data comes from **NOAA's ACSPO (Advanced Clear-Sky Processor for Oceans)**
            system, which processes infrared imagery from polar-orbiting satellites. These
            satellites pass over Lake Erie multiple times daily, and the ACSPO algorithm
            produces cloud-free composite images.

            Chlorophyll is measured using **ocean color sensors** that detect subtle shifts in
            the wavelengths of light reflected from the water surface. Healthy, clear water
            reflects mostly blue light. Water rich in chlorophyll absorbs blue and red light,
            reflecting more green — hence the characteristic green color of algal blooms.

            Both datasets are served through **NOAA GLERL's ERDDAP server**, a scientific
            data distribution system. This dashboard fetches data programmatically using
            NetCDF format requests.
            """)

        with st.expander("Why the Western Basin?"):
            st.markdown("""
            Lake Erie's western basin is uniquely vulnerable to harmful blooms due to a
            combination of factors:

            **Shallow depth** — Average depth is only ~7.4 meters (vs. 19m for the central
            basin and 24m for the east). Shallow water warms faster and allows light to
            penetrate to the bottom, promoting algal growth throughout the water column.

            **Nutrient loading** — The Maumee River, which drains 16,395 km² of heavily
            agricultural land in Ohio and Indiana, delivers large quantities of dissolved
            reactive phosphorus (DRP) directly into the western basin. This phosphorus is
            the primary fuel for *Microcystis* blooms.

            **Circulation patterns** — Prevailing winds tend to concentrate surface water
            (and floating cyanobacteria) along the southern and western shores, where
            population centers like Toledo are located.
            """)

        with st.expander("What Can We Infer from This Data?"):
            st.markdown(f"""
            **Seasonal patterns:** Lake Erie follows a predictable annual cycle — warming
            from April through August, peaking in late July/August, then cooling through
            December. Bloom season typically runs **June through October**, with peak risk
            in **August–September**.

            **Year-to-year variability:** Not every warm summer produces a severe bloom.
            The critical factor is nutrient availability, especially spring phosphorus loading
            from agricultural runoff. A wet spring with heavy rain → more nutrients → larger
            bloom potential, even if temperatures are average.

            **Trend detection:** With multiple years of data in the archive, you can compare
            bloom seasons across years. Are summers getting warmer? Are blooms starting earlier?
            The Trends tab lets you explore these questions directly.

            **Early warning signals:**
            - Western basin SST crossing {CONFIG['thresholds']['sst_bloom']}°C in early June = early bloom potential
            - Chlorophyll rising above 10 mg/m³ in July = bloom is developing
            - Multi-day sustained warmth + rising chlorophyll = escalation likely

            **What this dashboard cannot tell you:** Toxin concentrations (requires water
            sampling), bloom species composition, nutrient levels, wind-driven transport
            patterns, or drinking water safety status. For those, consult NOAA's official
            Lake Erie HAB Forecast bulletins.
            """)

        st.markdown('<div class="nexus-divider"></div>', unsafe_allow_html=True)

        # ── Section 5: Data sources ──
        st.markdown("### 📡 Data Sources & Credits")
        st.markdown("""
        | Dataset | Source | Coverage | Resolution |
        |---------|--------|----------|------------|
        | Sea Surface Temperature | NOAA GLSEA ACSPO (GLERL) | 2006–present | ~1.3 km, daily |
        | Chlorophyll-a | Great Lakes Daily CHL (GLERL) | 2012–2023 | ~1 km, daily |

        Data is accessed via [NOAA GLERL ERDDAP](https://apps.glerl.noaa.gov/erddap) and
        cached locally in a SQLite database for fast dashboard rendering.

        **BlueNexus** is an independent research project. It is not affiliated with or
        endorsed by NOAA, GLERL, or any government agency.
        """)

    # ═══════════ TAB 5: System Status ═══════════
    with tab5:
        st.markdown('<div class="section-label">Data Pipeline Status</div>',
                    unsafe_allow_html=True)

        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.metric("SST Records", f"{date_stats['sst']['count']:,}")
        with p2:
            st.metric("CHL Records", f"{date_stats['chl']['count']:,}")
        with p3:
            st.metric("Spatial Maps", f"{len(available_dates):,}")
        with p4:
            if date_stats['sst']['min'] and date_stats['sst']['max']:
                d1 = datetime.strptime(date_stats['sst']['min'], "%Y-%m-%d")
                d2 = datetime.strptime(date_stats['sst']['max'], "%Y-%m-%d")
                st.metric("Date Span", f"{(d2-d1).days} days")

        rc1, rc2 = st.columns(2)
        with rc1:
            if date_stats['sst']['min']:
                st.caption(f"SST: {date_stats['sst']['min']} → {date_stats['sst']['max']}")
        with rc2:
            if date_stats['chl']['min']:
                st.caption(f"CHL: {date_stats['chl']['min']} → {date_stats['chl']['max']}")

        st.markdown('<div class="nexus-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Recent Fetch Log</div>',
                    unsafe_allow_html=True)

        if len(history_df) > 0:
            st.dataframe(history_df, width='stretch', hide_index=True, height=300)
        else:
            st.info("No fetch history recorded yet.")

        # ── Data Coverage Heatmap ──
        coverage_df = load_data_coverage()
        if len(coverage_df) > 1:
            st.markdown('<div class="nexus-divider"></div>', unsafe_allow_html=True)
            st.plotly_chart(
                build_coverage_heatmap(coverage_df),
                width='stretch',
                config={'displayModeBar': False},
            )

            total_days = int(coverage_df.values.sum())
            total_possible = sum(
                (pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(0)).day
                for y in coverage_df.index
                for m in coverage_df.columns
                if coverage_df.loc[y, m] > 0
            )
            if total_possible > 0:
                pct = total_days / total_possible * 100
                st.caption(
                    f"Archive contains {total_days:,} observation days across "
                    f"{len(coverage_df)} years ({pct:.0f}% coverage in months with data). "
                    f"Gaps are typically caused by cloud cover, ice, or satellite downtime."
                )

        st.markdown('<div class="nexus-divider"></div>', unsafe_allow_html=True)

        with st.expander("⚙️  Active Configuration"):
            st.json({
                'erddap_base_url': CONFIG['erddap']['base_url'],
                'sst_dataset': CONFIG['erddap']['sst_dataset'],
                'chl_dataset': CONFIG['erddap']['chl_dataset'],
                'bloom_thresholds': CONFIG['thresholds'],
                'schedule': CONFIG['schedule'],
            })

        with st.expander("📋  Registered Datasets"):
            datasets = CONFIG.get("datasets", [])
            if datasets:
                for ds in datasets:
                    cov = ds.get("coverage", {})
                    cov_str = f"{cov.get('start', '?')} → {cov.get('end', '?')}"
                    st.markdown(
                        f'<div style="background: rgba(0,212,255,0.04); '
                        f'border-left: 3px solid {ds.get("color", "#666")}; '
                        f'padding: 8px 14px; border-radius: 4px; margin: 6px 0;">'
                        f'<strong>{ds.get("name", ds.get("id"))}</strong> '
                        f'<span style="color: var(--text-secondary); font-size: 0.8rem;">'
                        f'({ds.get("erddap_id", "?")}) · {ds.get("unit", "")} · '
                        f'{cov_str} · {ds.get("region", "")}'
                        f'</span></div>',
                        unsafe_allow_html=True,
                    )
                st.caption(
                    "Datasets are defined in config.yaml under the 'datasets' key. "
                    "Add new data sources by following the template in the config file."
                )
            else:
                st.info("No datasets registered in config.yaml yet.")

        st.markdown(f"""
        <div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 16px;">
            Database: <code>{DB_PATH}</code><br>
            Next scheduled update: {calculate_time_until_next_update()}
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
