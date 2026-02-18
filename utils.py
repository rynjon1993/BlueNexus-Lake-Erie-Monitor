"""
Utility functions for Lake Erie Live Monitor (BlueNexus)
Plotting, data formatting, interactive mapping, and helper functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import folium
from streamlit_folium import st_folium
from branca.colormap import LinearColormap
from datetime import datetime, timedelta
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Static maps (matplotlib + Cartopy)
# ---------------------------------------------------------------------------

def create_lake_erie_map(figsize=(10, 6)):
    """Return (fig, ax) with a base map centred on Lake Erie."""
    bbox = CONFIG["bbox"]
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())

    buffer = 0.2
    ax.set_extent(
        [bbox["lon_min"] - buffer, bbox["lon_max"] + buffer,
         bbox["lat_min"] - buffer, bbox["lat_max"] + buffer],
        crs=ccrs.PlateCarree(),
    )

    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
    ax.add_feature(cfeature.LAKES, facecolor="white", edgecolor="black",
                   linewidth=1.5, zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle="--",
                   edgecolor="gray", zorder=2)
    ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor="gray", zorder=2)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    return fig, ax


def plot_sst_map(lons, lats, sst_data, date_str, vmin=None, vmax=None):
    """Create a static SST map with Cartopy."""
    fig, ax = create_lake_erie_map(figsize=(12, 7))

    im = ax.pcolormesh(
        lons, lats, sst_data,
        transform=ccrs.PlateCarree(),
        cmap="RdYlBu_r",
        vmin=vmin or 0, vmax=vmax or 30,
        zorder=3,
    )

    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
    cbar.set_label("Sea Surface Temperature (°C)", fontsize=11, weight="bold")
    ax.set_title(f"Lake Erie Sea Surface Temperature\n{date_str}",
                 fontsize=14, weight="bold", pad=10)

    plt.tight_layout()
    return fig


def plot_chlorophyll_map(lons, lats, chl_data, date_str, vmin=None, vmax=None):
    """Create a static chlorophyll-a map with Cartopy."""
    fig, ax = create_lake_erie_map(figsize=(12, 7))

    im = ax.pcolormesh(
        lons, lats, chl_data,
        transform=ccrs.PlateCarree(),
        cmap="YlGn",
        vmin=vmin or 0, vmax=vmax or 50,
        zorder=3,
    )

    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
    cbar.set_label("Chlorophyll-a Concentration (mg/m³)", fontsize=11, weight="bold")
    ax.set_title(f"Lake Erie Chlorophyll-a Concentration\n{date_str}",
                 fontsize=14, weight="bold", pad=10)

    thresholds = CONFIG["thresholds"]
    legend_text = (
        f"Bloom Risk Thresholds:\n"
        f"Low: >{thresholds['chl_low']} mg/m³\n"
        f"Moderate: >{thresholds['chl_moderate']} mg/m³\n"
        f"High: >{thresholds['chl_high']} mg/m³"
    )
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    return fig


def plot_time_series(dates, sst_values, chl_values,
                     title="Lake Erie 7-Day Trends"):
    """Dual-axis time series chart for SST and chlorophyll."""
    fig, ax1 = plt.subplots(figsize=(10, 4))
    date_objs = [datetime.strptime(d, "%Y-%m-%d") for d in dates]

    color_sst = "#d62728"
    ax1.set_xlabel("Date", fontsize=11, weight="bold")
    ax1.set_ylabel("SST (°C)", fontsize=11, weight="bold", color=color_sst)
    ax1.plot(date_objs, sst_values, color=color_sst, marker="o",
             linewidth=2, markersize=5, label="SST")
    ax1.tick_params(axis="y", labelcolor=color_sst)
    ax1.grid(True, alpha=0.3)

    threshold_sst = CONFIG["thresholds"]["sst_bloom"]
    ax1.axhline(y=threshold_sst, color=color_sst, linestyle="--",
                alpha=0.5, linewidth=1.5,
                label=f"Bloom Threshold ({threshold_sst}°C)")

    ax2 = ax1.twinx()
    color_chl = "#2ca02c"
    ax2.set_ylabel("Chlorophyll-a (mg/m³)", fontsize=11, weight="bold",
                   color=color_chl)
    if any(not np.isnan(v) for v in chl_values):
        ax2.plot(date_objs, chl_values, color=color_chl, marker="s",
                 linewidth=2, markersize=5, label="Chlorophyll-a (West Basin)")
    ax2.tick_params(axis="y", labelcolor=color_chl)

    threshold_chl = CONFIG["thresholds"]["chl_moderate"]
    ax2.axhline(y=threshold_chl, color=color_chl, linestyle="--",
                alpha=0.5, linewidth=1.5,
                label=f"Moderate Bloom ({threshold_chl} mg/m³)")

    ax1.set_title(title, fontsize=13, weight="bold", pad=10)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    fig.autofmt_xdate()
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Interactive maps (Folium)
# ---------------------------------------------------------------------------

def create_interactive_map(data, lats, lons, label="Value", unit="",
                           colors="viridis"):
    """
    Build a Folium map with a raster overlay and color legend.

    Returns a bare base map if any input is None (graceful degradation).
    """
    if data is None or lats is None or lons is None:
        return folium.Map(location=[42.2, -81.2], zoom_start=8)

    bounds = [[lats.min(), lons.min()], [lats.max(), lons.max()]]
    m = folium.Map(location=[42.2, -81.2], zoom_start=8, control_scale=True)

    v_min, v_max = float(np.nanmin(data)), float(np.nanmax(data))

    cmap_name = {
        "temperature": "RdYlBu_r",
        "viridis": "YlGn",
    }.get(colors, "viridis")
    cmap = plt.colormaps[cmap_name]

    norm = Normalize(vmin=v_min, vmax=v_max)
    rgba = cmap(norm(data))
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    rgb = np.flipud(rgb)

    folium.raster_layers.ImageOverlay(
        image=rgb, bounds=bounds, opacity=0.6, mercator_project=True
    ).add_to(m)

    legend_colors = (
        ["#00008b", "#0000ff", "#00ffff", "#ffff00", "#ff0000", "#8b0000"]
        if colors == "temperature"
        else ["#FFFFE0", "#90EE90", "#006400"]
    )
    colormap = LinearColormap(colors=legend_colors, vmin=v_min, vmax=v_max)
    colormap.caption = f"{label} ({unit})"
    colormap.add_to(m)

    return m


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_date(date_str):
    """Convert 'YYYY-MM-DD' to 'February 08, 2026'."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%B %d, %Y")
    except (ValueError, TypeError):
        return date_str


def format_metric(value, precision=1, unit=""):
    """Format a numeric value for dashboard display, or return 'N/A'."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    formatted = f"{value:.{precision}f}"
    if unit:
        formatted += f" {unit}"
    return formatted


def get_data_freshness_status(last_update_str):
    """Return (status_text, color_hex, emoji) indicating data age."""
    try:
        last_update = datetime.fromisoformat(last_update_str)
        hours_old = (datetime.now() - last_update).total_seconds() / 3600
        if hours_old < 24:
            return "Fresh", "#28a745", "\U0001f7e2"
        elif hours_old < 48:
            return "Stale", "#ffc107", "\U0001f7e1"
        else:
            return "Outdated", "#dc3545", "\U0001f534"
    except (ValueError, TypeError):
        return "Unknown", "#6c757d", "\u26aa"


def calculate_time_until_next_update():
    """Human-readable countdown to the next scheduled refresh."""
    now = datetime.now()
    schedule = CONFIG["schedule"]
    next_update = now.replace(
        hour=schedule["hour"], minute=schedule["minute"],
        second=0, microsecond=0,
    )
    if now >= next_update:
        next_update += timedelta(days=1)

    diff = next_update - now
    hours = int(diff.total_seconds() // 3600)
    minutes = int((diff.total_seconds() % 3600) // 60)
    return f"{hours}h {minutes}m"


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Lake Erie Live Monitor — Utility Functions Test")
    print("=" * 50)
    print(f"\nDate formatting:   {format_date('2026-02-08')}")
    print(f"Metric formatting: {format_metric(22.4, 1, '°C')}")
    print(f"Next update in:    {calculate_time_until_next_update()}")
    print(f"\nBloom thresholds:")
    print(f"  SST:       {CONFIG['thresholds']['sst_bloom']}°C")
    print(f"  Chl (Low): {CONFIG['thresholds']['chl_low']} mg/m³")
    print(f"  Chl (Mod): {CONFIG['thresholds']['chl_moderate']} mg/m³")
    print(f"  Chl (Hi):  {CONFIG['thresholds']['chl_high']} mg/m³")
