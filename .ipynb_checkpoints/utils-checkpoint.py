"""
Utility functions for Lake Erie Live Monitor
Plotting, data formatting, and helper functions
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
import yaml
from pathlib import Path

# Load configuration
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)


def create_lake_erie_map(figsize=(10, 6)):
    """
    Create a base map for Lake Erie visualizations
    
    Args:
        figsize: Figure size tuple
    
    Returns:
        tuple of (fig, ax) with configured map
    """
    bbox = CONFIG['bbox']
    
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set extent (Lake Erie bounding box with small buffer)
    buffer = 0.2
    ax.set_extent([
        bbox['lon_min'] - buffer,
        bbox['lon_max'] + buffer,
        bbox['lat_min'] - buffer,
        bbox['lat_max'] + buffer
    ], crs=ccrs.PlateCarree())
    
    # Add geographic features
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
    ax.add_feature(cfeature.LAKES, facecolor='white', edgecolor='black', linewidth=1.5, zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle='--', edgecolor='gray', zorder=2)
    ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray', zorder=2)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    return fig, ax


def plot_sst_map(lons, lats, sst_data, date_str, vmin=None, vmax=None):
    """
    Create SST visualization map
    
    Args:
        lons: Longitude array
        lats: Latitude array
        sst_data: 2D SST data array
        date_str: Date string for title
        vmin: Minimum value for colorbar
        vmax: Maximum value for colorbar
    
    Returns:
        matplotlib figure
    """
    fig, ax = create_lake_erie_map(figsize=(12, 7))
    
    # Plot SST data
    im = ax.pcolormesh(
        lons, lats, sst_data,
        transform=ccrs.PlateCarree(),
        cmap='RdYlBu_r',
        vmin=vmin or 0,
        vmax=vmax or 30,
        zorder=3
    )
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7)
    cbar.set_label('Sea Surface Temperature (°C)', fontsize=11, weight='bold')
    
    # Title
    ax.set_title(
        f'Lake Erie Sea Surface Temperature\n{date_str}',
        fontsize=14,
        weight='bold',
        pad=10
    )
    
    # Add threshold line if SST bloom threshold visible
    threshold = CONFIG['thresholds']['sst_bloom']
    if vmin and vmin < threshold < vmax:
        ax.text(
            0.02, 0.98,
            f'Bloom threshold: {threshold}°C',
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    plt.tight_layout()
    return fig


def plot_chlorophyll_map(lons, lats, chl_data, date_str, vmin=None, vmax=None):
    """
    Create chlorophyll-a visualization map
    
    Args:
        lons: Longitude array
        lats: Latitude array
        chl_data: 2D chlorophyll data array
        date_str: Date string for title
        vmin: Minimum value for colorbar
        vmax: Maximum value for colorbar
    
    Returns:
        matplotlib figure
    """
    fig, ax = create_lake_erie_map(figsize=(12, 7))
    
    # Plot chlorophyll data
    im = ax.pcolormesh(
        lons, lats, chl_data,
        transform=ccrs.PlateCarree(),
        cmap='YlGn',
        vmin=vmin or 0,
        vmax=vmax or 50,
        zorder=3
    )
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7)
    cbar.set_label('Chlorophyll-a Concentration (mg/m³)', fontsize=11, weight='bold')
    
    # Title
    ax.set_title(
        f'Lake Erie Chlorophyll-a Concentration\n{date_str}',
        fontsize=14,
        weight='bold',
        pad=10
    )
    
    # Add bloom threshold annotations
    thresholds = CONFIG['thresholds']
    legend_text = (
        f"Bloom Risk Thresholds:\n"
        f"Low: >{thresholds['chl_low']} mg/m³\n"
        f"Moderate: >{thresholds['chl_moderate']} mg/m³\n"
        f"High: >{thresholds['chl_high']} mg/m³"
    )
    
    ax.text(
        0.02, 0.98,
        legend_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    return fig


def plot_time_series(dates, sst_values, chl_values, title="Lake Erie 7-Day Trends"):
    """
    Create dual-axis time series plot for SST and chlorophyll
    
    Args:
        dates: List of date strings
        sst_values: List of SST values
        chl_values: List of chlorophyll values
        title: Plot title
    
    Returns:
        matplotlib figure
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Convert date strings to datetime objects
    date_objs = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
    
    # Plot SST on primary y-axis
    color = '#d62728'  # Red
    ax1.set_xlabel('Date', fontsize=12, weight='bold')
    ax1.set_ylabel('SST (°C)', fontsize=12, weight='bold', color=color)
    line1 = ax1.plot(date_objs, sst_values, color=color, marker='o', 
                     linewidth=2, markersize=6, label='SST')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Add SST threshold line
    threshold_sst = CONFIG['thresholds']['sst_bloom']
    ax1.axhline(y=threshold_sst, color=color, linestyle='--', 
                alpha=0.5, linewidth=1.5, label=f'Bloom Threshold ({threshold_sst}°C)')
    
    # Plot chlorophyll on secondary y-axis
    ax2 = ax1.twinx()
    color = '#2ca02c'  # Green
    ax2.set_ylabel('Chlorophyll-a (mg/m³)', fontsize=12, weight='bold', color=color)
    line2 = ax2.plot(date_objs, chl_values, color=color, marker='s', 
                     linewidth=2, markersize=6, label='Chlorophyll-a (West Basin)')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add chlorophyll threshold lines
    threshold_chl = CONFIG['thresholds']['chl_moderate']
    ax2.axhline(y=threshold_chl, color=color, linestyle='--', 
                alpha=0.5, linewidth=1.5, label=f'Moderate Bloom ({threshold_chl} mg/m³)')
    
    # Title
    ax1.set_title(title, fontsize=14, weight='bold', pad=15)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    # Format x-axis
    fig.autofmt_xdate()
    plt.tight_layout()
    
    return fig


def format_date(date_str):
    """
    Format date string for display
    
    Args:
        date_str: Date string in YYYY-MM-DD format
    
    Returns:
        Formatted date string (e.g., "February 8, 2026")
    """
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        return date_obj.strftime("%B %d, %Y")
    except:
        return date_str


def format_metric(value, precision=1, unit=""):
    """
    Format numeric metric for display
    
    Args:
        value: Numeric value
        precision: Decimal places
        unit: Unit string to append
    
    Returns:
        Formatted string
    """
    if value is None or np.isnan(value):
        return "N/A"
    
    formatted = f"{value:.{precision}f}"
    if unit:
        formatted += f" {unit}"
    
    return formatted


def get_data_freshness_status(last_update_str):
    """
    Determine data freshness status
    
    Args:
        last_update_str: ISO format timestamp string
    
    Returns:
        tuple of (status_text, status_color, emoji)
    """
    try:
        last_update = datetime.fromisoformat(last_update_str)
        hours_old = (datetime.now() - last_update).total_seconds() / 3600
        
        if hours_old < 24:
            return "Fresh", "#28a745", "🟢"
        elif hours_old < 48:
            return "Stale", "#ffc107", "🟡"
        else:
            return "Outdated", "#dc3545", "🔴"
    except:
        return "Unknown", "#6c757d", "⚪"


def calculate_time_until_next_update():
    """
    Calculate time until next scheduled update
    
    Returns:
        str with human-readable time (e.g., "23h 45m")
    """
    now = datetime.now()
    schedule = CONFIG['schedule']
    
    # Calculate next update time
    next_update = now.replace(
        hour=schedule['hour'],
        minute=schedule['minute'],
        second=0,
        microsecond=0
    )
    
    # If we've passed today's update time, schedule for tomorrow
    if now >= next_update:
        next_update = next_update.replace(day=next_update.day + 1)
    
    # Calculate time difference
    time_diff = next_update - now
    hours = int(time_diff.total_seconds() // 3600)
    minutes = int((time_diff.total_seconds() % 3600) // 60)
    
    return f"{hours}h {minutes}m"


if __name__ == "__main__":
    # Test utility functions
    print("Lake Erie Live Monitor - Utility Functions Test")
    print("=" * 50)
    
    print(f"\nDate formatting test:")
    print(f"  Input: 2026-02-08")
    print(f"  Output: {format_date('2026-02-08')}")
    
    print(f"\nMetric formatting test:")
    print(f"  SST: {format_metric(22.4, 1, '°C')}")
    print(f"  Chlorophyll: {format_metric(38.2, 1, 'mg/m³')}")
    
    print(f"\nTime until next update:")
    print(f"  {calculate_time_until_next_update()}")
    
    print(f"\nBloom thresholds from config:")
    print(f"  SST: {CONFIG['thresholds']['sst_bloom']}°C")
    print(f"  Chlorophyll (Low): {CONFIG['thresholds']['chl_low']} mg/m³")
    print(f"  Chlorophyll (Moderate): {CONFIG['thresholds']['chl_moderate']} mg/m³")
    print(f"  Chlorophyll (High): {CONFIG['thresholds']['chl_high']} mg/m³")
