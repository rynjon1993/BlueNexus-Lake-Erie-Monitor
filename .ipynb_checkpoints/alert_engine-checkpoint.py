"""
Alert Engine for Lake Erie Live Monitor
Bloom risk scoring, alert generation, and notification logic
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import yaml

# Load configuration
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)

DB_PATH = Path(__file__).parent / "cache" / "realtime_data.db"


def get_current_risk():
    """
    Get the most recent bloom risk assessment
    
    Returns:
        dict with risk_score, risk_level, date, and contributing factors
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT date, risk_score, risk_level, sst_above_threshold, 
               chl_above_threshold, calculation_timestamp
        FROM bloom_risk
        ORDER BY date DESC
        LIMIT 1
    """)
    
    row = cursor.fetchone()
    conn.close()
    
    if row is None:
        return {
            'date': None,
            'risk_score': 0,
            'risk_level': 'Unknown',
            'sst_above_threshold': False,
            'chl_above_threshold': False,
            'calculation_timestamp': None
        }
    
    return {
        'date': row[0],
        'risk_score': row[1],
        'risk_level': row[2],
        'sst_above_threshold': bool(row[3]),
        'chl_above_threshold': bool(row[4]),
        'calculation_timestamp': row[5]
    }


def get_risk_trend(days=7):
    """
    Get bloom risk trend over specified number of days
    
    Args:
        days: Number of days to look back
    
    Returns:
        list of dicts with date, risk_score, risk_level
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT date, risk_score, risk_level
        FROM bloom_risk
        ORDER BY date DESC
        LIMIT ?
    """, (days,))
    
    rows = cursor.fetchall()
    conn.close()
    
    trend = []
    for row in rows:
        trend.append({
            'date': row[0],
            'risk_score': row[1],
            'risk_level': row[2]
        })
    
    return trend


def get_alert_message(risk_data):
    """
    Generate human-readable alert message based on risk data
    
    Args:
        risk_data: dict from get_current_risk()
    
    Returns:
        str with alert message
    """
    level = risk_data['risk_level']
    score = risk_data['risk_score']
    sst_high = risk_data['sst_above_threshold']
    chl_high = risk_data['chl_above_threshold']
    
    if level == 'Unknown':
        return "No recent data available. Check data fetch status."
    
    messages = {
        'High': "⚠️ HIGH BLOOM RISK — Active bloom likely or confirmed.",
        'Moderate': "⚡ MODERATE BLOOM RISK — Conditions favorable for bloom development.",
        'Low': "✅ LOW BLOOM RISK — Conditions not currently favorable for blooms."
    }
    
    base_message = messages.get(level, "Unknown risk level")
    
    # Add contributing factors
    factors = []
    if sst_high:
        factors.append(f"water temperature above {CONFIG['thresholds']['sst_bloom']}°C")
    if chl_high:
        factors.append("elevated chlorophyll detected")
    
    if factors:
        base_message += f"\n\n**Contributing factors:** {', '.join(factors)}"
    
    return base_message


def get_risk_color(risk_level):
    """
    Get color code for risk level
    
    Args:
        risk_level: 'High', 'Moderate', or 'Low'
    
    Returns:
        str with hex color code
    """
    colors = {
        'High': '#dc3545',      # Red
        'Moderate': '#ffc107',  # Yellow
        'Low': '#28a745',       # Green
        'Unknown': '#6c757d'    # Gray
    }
    return colors.get(risk_level, '#6c757d')


def get_risk_emoji(risk_level):
    """
    Get emoji for risk level
    
    Args:
        risk_level: 'High', 'Moderate', or 'Low'
    
    Returns:
        str with emoji
    """
    emojis = {
        'High': '🔴',
        'Moderate': '🟡',
        'Low': '🟢',
        'Unknown': '⚪'
    }
    return emojis.get(risk_level, '⚪')


def format_risk_badge(risk_data):
    """
    Format risk data as a styled badge for display
    
    Args:
        risk_data: dict from get_current_risk()
    
    Returns:
        tuple of (emoji, level_text, score_text)
    """
    emoji = get_risk_emoji(risk_data['risk_level'])
    level_text = risk_data['risk_level'].upper()
    score_text = f"Score: {risk_data['risk_score']}/6"
    
    return emoji, level_text, score_text


def should_alert(risk_data, previous_risk_level='Low'):
    """
    Determine if an alert notification should be sent
    
    Args:
        risk_data: dict from get_current_risk()
        previous_risk_level: Previous risk level for comparison
    
    Returns:
        bool - True if alert should be sent
    """
    current_level = risk_data['risk_level']
    
    # Alert conditions:
    # 1. Risk level increased from Low to Moderate/High
    # 2. Risk level increased from Moderate to High
    # 3. Risk level is High (always alert)
    
    if current_level == 'High':
        return True
    
    if previous_risk_level == 'Low' and current_level in ['Moderate', 'High']:
        return True
    
    if previous_risk_level == 'Moderate' and current_level == 'High':
        return True
    
    return False


def get_recent_conditions(days=7):
    """
    Get summary of recent environmental conditions
    
    Args:
        days: Number of days to summarize
    
    Returns:
        dict with SST and chlorophyll statistics
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Use most recent data date as reference (not datetime.now()) 
    # so historical demo mode works correctly
    cursor.execute("SELECT MAX(date) FROM sst_data")
    row = cursor.fetchone()
    
    if row[0] is None:
        conn.close()
        return {
            'sst_avg': None, 'sst_max': None, 'sst_west_avg': None,
            'chl_avg': None, 'chl_max': None, 'days_chl_elevated': 0,
            'days_sst_above_threshold': 0, 'period_days': days
        }
    
    max_date = datetime.strptime(row[0], "%Y-%m-%d")
    cutoff_date = (max_date - timedelta(days=days)).strftime("%Y-%m-%d")
    
    # SST statistics
    cursor.execute("""
        SELECT AVG(lake_mean), MAX(lake_max), AVG(west_basin_mean)
        FROM sst_data
        WHERE date >= ?
    """, (cutoff_date,))
    sst_row = cursor.fetchone()
    
    # Chlorophyll statistics
    cursor.execute("""
        SELECT AVG(west_basin_mean), MAX(west_basin_mean), 
               COUNT(CASE WHEN west_basin_mean > ? THEN 1 END)
        FROM chl_data
        WHERE date >= ?
    """, (CONFIG['thresholds']['chl_low'], cutoff_date))
    chl_row = cursor.fetchone()
    
    # Days above threshold
    cursor.execute("""
        SELECT COUNT(CASE WHEN lake_mean > ? THEN 1 END)
        FROM sst_data
        WHERE date >= ?
    """, (CONFIG['thresholds']['sst_bloom'], cutoff_date))
    days_above_threshold = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        'sst_avg': sst_row[0] if sst_row[0] else None,
        'sst_max': sst_row[1] if sst_row[1] else None,
        'sst_west_avg': sst_row[2] if sst_row[2] else None,
        'chl_avg': chl_row[0] if chl_row[0] else None,
        'chl_max': chl_row[1] if chl_row[1] else None,
        'days_chl_elevated': chl_row[2],
        'days_sst_above_threshold': days_above_threshold,
        'period_days': days
    }


if __name__ == "__main__":
    # Test the alert engine
    print("Current Bloom Risk Assessment")
    print("=" * 50)
    
    risk = get_current_risk()
    print(f"Date: {risk['date']}")
    print(f"Risk Level: {risk['risk_level']}")
    print(f"Risk Score: {risk['risk_score']}/6")
    print(f"\nAlert Message:\n{get_alert_message(risk)}")
    
    print("\n\nRecent Conditions (7-day summary)")
    print("=" * 50)
    conditions = get_recent_conditions(7)
    if conditions['sst_avg']:
        print(f"Average SST: {conditions['sst_avg']:.1f}°C")
        print(f"Max SST: {conditions['sst_max']:.1f}°C")
        print(f"Days above {CONFIG['thresholds']['sst_bloom']}°C: {conditions['days_sst_above_threshold']}")
    
    if conditions['chl_avg']:
        print(f"\nAverage Chlorophyll (West Basin): {conditions['chl_avg']:.1f} mg/m³")
        print(f"Max Chlorophyll: {conditions['chl_max']:.1f} mg/m³")
        print(f"Days elevated (>{CONFIG['thresholds']['chl_low']} mg/m³): {conditions['days_chl_elevated']}")