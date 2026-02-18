"""
Alert Engine for Lake Erie Live Monitor (BlueNexus)
Bloom risk scoring, alert generation, and notification logic.
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

DB_PATH = Path(__file__).parent / "cache" / "realtime_data.db"


# ---------------------------------------------------------------------------
# Risk queries
# ---------------------------------------------------------------------------

def get_current_risk():
    """Return the most recent bloom risk assessment as a dict."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
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
            "date": None,
            "risk_score": 0,
            "risk_level": "Unknown",
            "sst_above_threshold": False,
            "chl_above_threshold": False,
            "calculation_timestamp": None,
        }

    return {
        "date": row[0],
        "risk_score": row[1],
        "risk_level": row[2],
        "sst_above_threshold": bool(row[3]),
        "chl_above_threshold": bool(row[4]),
        "calculation_timestamp": row[5],
    }


def get_risk_trend(days=7):
    """Return a list of recent risk assessments (newest first)."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT date, risk_score, risk_level FROM bloom_risk "
        "ORDER BY date DESC LIMIT ?",
        (days,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [{"date": r[0], "risk_score": r[1], "risk_level": r[2]} for r in rows]


# ---------------------------------------------------------------------------
# Alert messages & formatting
# ---------------------------------------------------------------------------

def get_alert_message(risk_data):
    """Generate a human-readable alert message from a risk assessment."""
    level = risk_data["risk_level"]
    if level == "Unknown":
        return "No recent data available. Check data fetch status."

    messages = {
        "High": "HIGH BLOOM RISK — Active bloom likely or confirmed.",
        "Moderate": "MODERATE BLOOM RISK — Conditions favorable for bloom development.",
        "Low": "LOW BLOOM RISK — Conditions not currently favorable for blooms.",
    }
    base = messages.get(level, "Unknown risk level")

    factors = []
    if risk_data["sst_above_threshold"]:
        factors.append(
            f"water temperature above {CONFIG['thresholds']['sst_bloom']}°C"
        )
    if risk_data["chl_above_threshold"]:
        factors.append("elevated chlorophyll detected")

    if factors:
        base += f"\n\n**Contributing factors:** {', '.join(factors)}"

    return base


def get_risk_color(risk_level):
    """Map risk level to a hex color code."""
    return {
        "High": "#dc3545",
        "Moderate": "#ffc107",
        "Low": "#28a745",
        "Unknown": "#6c757d",
    }.get(risk_level, "#6c757d")


def get_risk_emoji(risk_level):
    """Map risk level to a status emoji."""
    return {
        "High": "\U0001f534",       # red circle
        "Moderate": "\U0001f7e1",   # yellow circle
        "Low": "\U0001f7e2",       # green circle
        "Unknown": "\u26aa",       # white circle
    }.get(risk_level, "\u26aa")


def format_risk_badge(risk_data):
    """Return (emoji, level_text, score_text) for badge display."""
    emoji = get_risk_emoji(risk_data["risk_level"])
    level_text = risk_data["risk_level"].upper()
    score_text = f"Score: {risk_data['risk_score']}/6"
    return emoji, level_text, score_text


def should_alert(risk_data, previous_risk_level="Low"):
    """Decide whether to fire a notification based on risk escalation."""
    current = risk_data["risk_level"]
    if current == "High":
        return True
    if previous_risk_level == "Low" and current in ("Moderate", "High"):
        return True
    if previous_risk_level == "Moderate" and current == "High":
        return True
    return False


# ---------------------------------------------------------------------------
# Environmental conditions summary
# ---------------------------------------------------------------------------

def get_recent_conditions(days=7):
    """
    Summarize SST and chlorophyll over the most recent *days* of data.

    Reference date is the latest date in the database, not wall-clock time,
    so historical/demo modes work correctly.
    """
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    cursor = conn.cursor()

    cursor.execute("SELECT MAX(date) FROM sst_data")
    row = cursor.fetchone()

    if row[0] is None:
        conn.close()
        return {
            "sst_avg": None, "sst_max": None, "sst_west_avg": None,
            "chl_avg": None, "chl_max": None, "days_chl_elevated": 0,
            "days_sst_above_threshold": 0, "period_days": days,
        }

    max_date = datetime.strptime(row[0], "%Y-%m-%d")
    cutoff = (max_date - timedelta(days=days)).strftime("%Y-%m-%d")

    cursor.execute(
        "SELECT AVG(lake_mean), MAX(lake_max), AVG(west_basin_mean) "
        "FROM sst_data WHERE date >= ?",
        (cutoff,),
    )
    sst_row = cursor.fetchone()

    cursor.execute(
        "SELECT AVG(west_basin_mean), MAX(west_basin_mean), "
        "COUNT(CASE WHEN west_basin_mean > ? THEN 1 END) "
        "FROM chl_data WHERE date >= ?",
        (CONFIG["thresholds"]["chl_low"], cutoff),
    )
    chl_row = cursor.fetchone()

    cursor.execute(
        "SELECT COUNT(CASE WHEN lake_mean > ? THEN 1 END) "
        "FROM sst_data WHERE date >= ?",
        (CONFIG["thresholds"]["sst_bloom"], cutoff),
    )
    days_above = cursor.fetchone()[0]

    conn.close()

    return {
        "sst_avg": sst_row[0],
        "sst_max": sst_row[1],
        "sst_west_avg": sst_row[2],
        "chl_avg": chl_row[0],
        "chl_max": chl_row[1],
        "days_chl_elevated": chl_row[2],
        "days_sst_above_threshold": days_above,
        "period_days": days,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Current Bloom Risk Assessment")
    print("=" * 50)
    risk = get_current_risk()
    print(f"Date:       {risk['date']}")
    print(f"Risk Level: {risk['risk_level']}")
    print(f"Risk Score: {risk['risk_score']}/6")
    print(f"\nAlert Message:\n{get_alert_message(risk)}")

    print("\n\nRecent Conditions (7-day summary)")
    print("=" * 50)
    cond = get_recent_conditions(7)
    if cond["sst_avg"]:
        thr = CONFIG["thresholds"]["sst_bloom"]
        print(f"Average SST:           {cond['sst_avg']:.1f} °C")
        print(f"Max SST:               {cond['sst_max']:.1f} °C")
        print(f"Days above {thr}°C:     {cond['days_sst_above_threshold']}")
    if cond["chl_avg"]:
        print(f"\nAvg Chl (West Basin):  {cond['chl_avg']:.1f} mg/m³")
        print(f"Max Chl:               {cond['chl_max']:.1f} mg/m³")
        print(f"Days elevated:         {cond['days_chl_elevated']}")
