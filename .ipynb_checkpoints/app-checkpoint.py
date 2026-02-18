"""
Lake Erie Live Monitor - Streamlit Dashboard
Real-time monitoring of sea surface temperature and harmful algal bloom risk
"""

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import time
from apscheduler.schedulers.background import BackgroundScheduler
import xarray as xr

# Import custom modules
from data_fetcher import fetch_latest_data, init_database, DB_PATH
from alert_engine import (
    get_current_risk,
    get_risk_trend,
    get_alert_message,
    get_risk_color,
    get_risk_emoji,
    get_recent_conditions
)
from utils import (
    format_date,
    format_metric,
    get_data_freshness_status,
    calculate_time_until_next_update,
    plot_time_series
)

# Page configuration
st.set_page_config(
    page_title="Lake Erie Live Monitor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)


# Initialize database on first run
@st.cache_resource
def initialize_system():
    """Initialize database and scheduler"""
    init_database()
    
    # Create and start scheduler
    scheduler = BackgroundScheduler(timezone=CONFIG['schedule']['timezone'])
    scheduler.add_job(
        fetch_latest_data,
        'cron',
        hour=CONFIG['schedule']['hour'],
        minute=CONFIG['schedule']['minute'],
        id='daily_fetch'
    )
    scheduler.start()
    
    return scheduler


# Load data with caching
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_latest_data():
    """Load the most recent 7 days of data from database"""
    conn = sqlite3.connect(DB_PATH)
    
    # SST data
    sst_df = pd.read_sql_query("""
        SELECT date, lake_mean, lake_max, west_basin_mean, 
               east_basin_mean, data_coverage, fetch_timestamp
        FROM sst_data
        ORDER BY date DESC
        LIMIT 7
    """, conn)
    
    # Chlorophyll data
    chl_df = pd.read_sql_query("""
        SELECT date, lake_mean, lake_max, west_basin_mean, 
               east_basin_mean, data_coverage, fetch_timestamp
        FROM chl_data
        ORDER BY date DESC
        LIMIT 7
    """, conn)
    
    # Bloom risk data
    risk_df = pd.read_sql_query("""
        SELECT date, risk_score, risk_level
        FROM bloom_risk
        ORDER BY date DESC
        LIMIT 7
    """, conn)
    
    # Fetch history (last 10 fetches)
    history_df = pd.read_sql_query("""
        SELECT fetch_timestamp, data_type, date_range_start, 
               date_range_end, status, error_message
        FROM fetch_history
        ORDER BY fetch_timestamp DESC
        LIMIT 10
    """, conn)
    
    conn.close()
    
    return sst_df, chl_df, risk_df, history_df


def main():
    """Main dashboard function"""
    
    # Initialize system
    scheduler = initialize_system()
    
    # Header
    st.title("🌊 Lake Erie Monitor — Historical Demo")
    st.markdown("**Monitoring system demonstration using December 2023 data** (most recent complete dataset)")
    st.caption("💡 For live monitoring, update `data_fetcher.py` to use current dates once real-time SST data source is configured")
    st.markdown("---")
    
    # Load data
    try:
        sst_df, chl_df, risk_df, history_df = load_latest_data()
        data_loaded = len(sst_df) > 0 or len(chl_df) > 0
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Click 'Initialize Data' below to fetch initial data.")
        data_loaded = False
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Data refresh button
        if st.button("🔄 Refresh Now", use_container_width=True):
            with st.spinner("Fetching latest data..."):
                try:
                    fetch_latest_data()
                    st.cache_data.clear()
                    time.sleep(2)  # Give database time to release locks
                    st.success("Data updated!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during refresh: {str(e)}")
                    st.info("Check cache/fetch_log.txt for details")
        
        # Initialize data button (for first run)
        if not data_loaded:
            if st.button("🚀 Initialize Data", type="primary", use_container_width=True):
                with st.spinner("Downloading initial dataset (7 days)..."):
                    try:
                        fetch_latest_data()
                        st.cache_data.clear()
                        time.sleep(2)  # Give database time to release locks
                        st.success("Initial data loaded!")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error during initialization: {str(e)}")
                        st.info("Check cache/fetch_log.txt for details")
        
        st.markdown("---")
        
        # Current risk status
        if data_loaded:
            risk_data = get_current_risk()
            
            if risk_data['date']:
                emoji = get_risk_emoji(risk_data['risk_level'])
                color = get_risk_color(risk_data['risk_level'])
                
                st.markdown(f"### {emoji} Bloom Risk")
                st.markdown(
                    f"<h2 style='color: {color}; margin: 0;'>{risk_data['risk_level'].upper()}</h2>",
                    unsafe_allow_html=True
                )
                st.markdown(f"**Score:** {risk_data['risk_score']}/6")
                st.caption(f"As of {format_date(risk_data['date'])}")
                
                st.markdown("---")
        
        # Data freshness indicator
        st.markdown("### 📊 Data Status")
        
        if data_loaded and len(sst_df) > 0:
            last_update = sst_df.iloc[0]['fetch_timestamp']
            status_text, status_color, status_emoji = get_data_freshness_status(last_update)
            
            st.markdown(f"{status_emoji} **Status:** {status_text}")
            
            try:
                update_time = datetime.fromisoformat(last_update)
                st.caption(f"Last updated: {update_time.strftime('%Y-%m-%d %H:%M')}")
            except:
                pass
            
            next_update = calculate_time_until_next_update()
            st.caption(f"Next update in: {next_update}")
        else:
            st.markdown("⚪ **Status:** No data")
            st.caption("Initialize data to begin monitoring")
        
        st.markdown("---")
        
        # Lake-wide metrics
        if data_loaded and len(sst_df) > 0:
            st.markdown("### 📈 Current Metrics")
            
            latest_sst = sst_df.iloc[0]
            st.metric(
                "Lake-Wide SST",
                format_metric(latest_sst['lake_mean'], 1, "°C"),
                delta=None
            )
            
            if len(chl_df) > 0:
                latest_chl = chl_df.iloc[0]
                st.metric(
                    "West Basin Chlorophyll",
                    format_metric(latest_chl['west_basin_mean'], 1, "mg/m³"),
                    delta=None
                )
            
            # Days above threshold
            threshold = CONFIG['thresholds']['sst_bloom']
            days_above = len(sst_df[sst_df['lake_mean'] > threshold])
            st.metric(
                f"Days >{threshold}°C (7-day)",
                f"{days_above}/7",
                delta=None
            )
        
        st.markdown("---")
        
        # Debug info toggle
        if CONFIG['dashboard'].get('show_debug_info', False):
            with st.expander("🔧 Debug Info"):
                st.caption(f"Scheduler running: {scheduler.running}")
                st.caption(f"DB path: {DB_PATH}")
                if data_loaded:
                    st.caption(f"SST records: {len(sst_df)}")
                    st.caption(f"Chl records: {len(chl_df)}")
    
    # Main content area
    if not data_loaded:
        st.info("👈 Click **Initialize Data** in the sidebar to download the initial dataset (7 days of satellite data).")
        st.markdown("""
        ### Welcome to Lake Erie Live Monitor
        
        This dashboard provides real-time monitoring of Lake Erie's environmental conditions:
        
        - **🌡️ Sea Surface Temperature** — Daily satellite measurements
        - **🦠 Chlorophyll-a Concentration** — Algal bloom indicator
        - **⚠️ Bloom Risk Alerts** — Automated scoring based on environmental thresholds
        
        Data is automatically updated daily at **8:00 AM EST** from NOAA's satellite systems.
        
        **To get started:** Click "Initialize Data" in the sidebar to download the last 7 days of data.
        """)
        return
    
    # Alert message
    if risk_data['date']:
        alert_msg = get_alert_message(risk_data)
        
        if risk_data['risk_level'] == 'High':
            st.error(alert_msg)
        elif risk_data['risk_level'] == 'Moderate':
            st.warning(alert_msg)
        else:
            st.success(alert_msg)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "📈 Trends",
        "🗺️ Historical Data",
        "⚙️ System Status"
    ])
    
    with tab1:
        st.header("Current Conditions Overview")
        
        # Latest data summary
        if len(sst_df) > 0:
            latest_date = sst_df.iloc[0]['date']
            st.subheader(f"Latest Available Data: {format_date(latest_date)}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Lake-Wide SST",
                    format_metric(sst_df.iloc[0]['lake_mean'], 1, "°C"),
                    delta=None,
                    help="Average temperature across entire lake"
                )
            
            with col2:
                st.metric(
                    "Peak SST",
                    format_metric(sst_df.iloc[0]['lake_max'], 1, "°C"),
                    delta=None,
                    help="Highest temperature recorded"
                )
            
            with col3:
                if len(chl_df) > 0:
                    st.metric(
                        "West Basin Chlorophyll",
                        format_metric(chl_df.iloc[0]['west_basin_mean'], 1, "mg/m³"),
                        delta=None,
                        help="Average chlorophyll in western basin"
                    )
            
            with col4:
                if len(chl_df) > 0:
                    st.metric(
                        "Peak Chlorophyll",
                        format_metric(chl_df.iloc[0]['lake_max'], 1, "mg/m³"),
                        delta=None,
                        help="Highest chlorophyll recorded"
                    )
            
            st.markdown("---")
            
            # Recent conditions summary
            st.subheader("7-Day Summary")
            conditions = get_recent_conditions(7)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Temperature**")
                st.write(f"• Average SST: {format_metric(conditions['sst_avg'], 1, '°C')}")
                st.write(f"• Peak SST: {format_metric(conditions['sst_max'], 1, '°C')}")
                st.write(f"• Days above {CONFIG['thresholds']['sst_bloom']}°C: {conditions['days_sst_above_threshold']}")
            
            with col2:
                if conditions['chl_avg']:
                    st.markdown("**Chlorophyll**")
                    st.write(f"• Average (West Basin): {format_metric(conditions['chl_avg'], 1, 'mg/m³')}")
                    st.write(f"• Peak: {format_metric(conditions['chl_max'], 1, 'mg/m³')}")
                    st.write(f"• Days elevated (>{CONFIG['thresholds']['chl_low']} mg/m³): {conditions['days_chl_elevated']}")
    
    with tab2:
        st.header("7-Day Trends")
        
        if len(sst_df) > 0:
            # Prepare data for time series plot
            dates = sst_df['date'].tolist()[::-1]  # Reverse for chronological order
            sst_values = sst_df['lake_mean'].tolist()[::-1]
            
            # Match chlorophyll dates to SST dates
            chl_values = []
            for date in dates:
                chl_row = chl_df[chl_df['date'] == date]
                if len(chl_row) > 0:
                    chl_values.append(chl_row.iloc[0]['west_basin_mean'])
                else:
                    chl_values.append(np.nan)
            
            # Create plot
            fig = plot_time_series(dates, sst_values, chl_values)
            st.pyplot(fig)
            
            st.caption("""
            **Note:** SST values are lake-wide averages. Chlorophyll values are from western basin only 
            (primary HAB region). Dashed lines show bloom risk thresholds.
            """)
            
            st.markdown("---")
            
            # Risk trend table
            st.subheader("Daily Bloom Risk History")
            
            if len(risk_df) > 0:
                # Format risk dataframe for display
                display_df = risk_df.copy()
                display_df['date'] = display_df['date'].apply(lambda x: format_date(x))
                display_df['emoji'] = display_df['risk_level'].apply(get_risk_emoji)
                display_df = display_df[['date', 'emoji', 'risk_level', 'risk_score']]
                display_df.columns = ['Date', '', 'Risk Level', 'Score']
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
    
    with tab3:
        st.header("Historical Data Table")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sea Surface Temperature")
            if len(sst_df) > 0:
                display_sst = sst_df[['date', 'lake_mean', 'west_basin_mean', 
                                      'east_basin_mean', 'data_coverage']].copy()
                display_sst.columns = ['Date', 'Lake Mean (°C)', 'West Basin (°C)', 
                                       'East Basin (°C)', 'Coverage (%)']
                display_sst['Date'] = display_sst['Date'].apply(lambda x: format_date(x))
                
                st.dataframe(
                    display_sst,
                    use_container_width=True,
                    hide_index=True
                )
        
        with col2:
            st.subheader("Chlorophyll-a Concentration")
            if len(chl_df) > 0:
                display_chl = chl_df[['date', 'lake_mean', 'west_basin_mean', 
                                      'east_basin_mean', 'data_coverage']].copy()
                display_chl.columns = ['Date', 'Lake Mean (mg/m³)', 'West Basin (mg/m³)', 
                                       'East Basin (mg/m³)', 'Coverage (%)']
                display_chl['Date'] = display_chl['Date'].apply(lambda x: format_date(x))
                
                st.dataframe(
                    display_chl,
                    use_container_width=True,
                    hide_index=True
                )
    
    with tab4:
        st.header("System Status")
        
        st.subheader("Fetch History (Last 10 Updates)")
        
        if len(history_df) > 0:
            display_history = history_df.copy()
            display_history['fetch_timestamp'] = pd.to_datetime(
                display_history['fetch_timestamp']
            ).dt.strftime('%Y-%m-%d %H:%M')
            
            # Add status emoji
            display_history['status_emoji'] = display_history['status'].apply(
                lambda x: '✅' if x == 'success' else '❌'
            )
            
            display_history = display_history[[
                'fetch_timestamp', 'data_type', 'status_emoji', 
                'status', 'date_range_start', 'date_range_end'
            ]]
            display_history.columns = [
                'Timestamp', 'Data Type', '', 'Status', 
                'Start Date', 'End Date'
            ]
            
            st.dataframe(
                display_history,
                use_container_width=True,
                hide_index=True
            )
            
            # Show errors if any
            errors = history_df[history_df['status'] == 'failed']
            if len(errors) > 0:
                with st.expander("⚠️ View Error Details"):
                    for _, row in errors.iterrows():
                        st.error(f"**{row['data_type']} at {row['fetch_timestamp']}:** {row['error_message']}")
        
        st.markdown("---")
        
        st.subheader("Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Data Sources**")
            st.write(f"• SST Dataset: {CONFIG['erddap']['sst_dataset']}")
            st.write(f"• Chlorophyll Dataset: {CONFIG['erddap']['chl_dataset']}")
            st.write(f"• Server: {CONFIG['erddap']['base_url']}")
        
        with col2:
            st.markdown("**Thresholds**")
            st.write(f"• SST Bloom Threshold: {CONFIG['thresholds']['sst_bloom']}°C")
            st.write(f"• Chlorophyll Low: {CONFIG['thresholds']['chl_low']} mg/m³")
            st.write(f"• Chlorophyll Moderate: {CONFIG['thresholds']['chl_moderate']} mg/m³")
            st.write(f"• Chlorophyll High: {CONFIG['thresholds']['chl_high']} mg/m³")
        
        st.markdown("---")
        
        st.subheader("About This Dashboard")
        st.markdown("""
        **Lake Erie Live Monitor** provides near-real-time tracking of environmental conditions 
        that contribute to harmful algal blooms (HABs).
        
        **Data Sources:**
        - Sea Surface Temperature: NOAA GLERL GLSEA dataset (~1.4 km resolution, daily)
        - Chlorophyll-a: NOAA GLERL VIIRS dataset (~0.6 km resolution, daily composite)
        
        **Limitations:**
        - Chlorophyll-a is a proxy for cyanobacteria, not a direct measurement of toxins
        - Satellite data has ~10% cloud cover gaps
        - Temperature alone does not predict blooms (see Project 02 findings: r² = 0.01)
        - Nutrient loading (dissolved phosphorus) is the primary driver but not measured here
        
        **For official bloom status, visit:** [NOAA HAB Bulletin](https://www.glerl.noaa.gov/res/HABs_and_Hypoxia/)
        
        ---
        
        Part of [Project Blue Nexus](https://github.com/rynjon1993/BlueNexus) by Ryan Jones
        """)


if __name__ == "__main__":
    main()