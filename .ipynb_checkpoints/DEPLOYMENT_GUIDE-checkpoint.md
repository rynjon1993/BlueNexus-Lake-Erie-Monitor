# 🚢 Phase 6 Deployment — Real-Time Monitoring Dashboard

**Captain's Orders:** Complete installation instructions for launching the live monitoring system.

---

## 📦 Files Downloaded

You should have received **10 files** from this session:

### Core Application Files (place in `E:\BlueNexus\03_realtime_monitor\`)
1. **app.py** — Main Streamlit dashboard (18 KB)
2. **data_fetcher.py** — Automated data pipeline (13 KB)
3. **alert_engine.py** — Bloom risk scoring (7.9 KB)
4. **utils.py** — Plotting utilities (9.8 KB)
5. **config.yaml** — Configuration file (1.4 KB)

### Documentation Files (place in `E:\BlueNexus\03_realtime_monitor\`)
6. **03_README.md** — Project README (rename to `README.md`)
7. **SETUP.md** — Detailed setup guide

### Repository Root Files (place in `E:\BlueNexus\`)
8. **README_updated.md** — Updated master README (rename to `README.md` after backing up current)
9. **requirements.txt** — Updated Python dependencies

### Git Configuration (place in `E:\BlueNexus\03_realtime_monitor\`)
10. **gitignore.txt** — Git ignore rules (rename to `.gitignore`)

---

## 🛠️ Installation Steps

### Step 1: Create Project Directory

```bash
cd E:\BlueNexus
mkdir 03_realtime_monitor
cd 03_realtime_monitor
```

### Step 2: Copy Files

Place the downloaded files in their correct locations:

```
E:\BlueNexus\
├── requirements.txt            ← Replace with new version
├── README.md                   ← Backup current, then replace with README_updated.md
│
└── 03_realtime_monitor\        ← New directory
    ├── README.md               ← From 03_README.md
    ├── SETUP.md
    ├── app.py
    ├── data_fetcher.py
    ├── alert_engine.py
    ├── utils.py
    ├── config.yaml
    └── .gitignore              ← From gitignore.txt
```

### Step 3: Install Dependencies

```bash
cd E:\BlueNexus
ocean_env\Scripts\activate
pip install streamlit apscheduler pyyaml
```

Verify installation:
```bash
pip list | findstr /I "streamlit apscheduler yaml"
```

You should see:
```
apscheduler      3.10.4
PyYAML           6.0.1
streamlit        1.31.1
```

### Step 4: Test Data Fetcher

Before launching the full dashboard, test the data pipeline:

```bash
cd E:\BlueNexus\03_realtime_monitor
python data_fetcher.py
```

**Expected output:**
```
[2026-02-08 XX:XX:XX] ====================================================
[2026-02-08 XX:XX:XX] Starting data fetch...
[2026-02-08 XX:XX:XX] Database initialized successfully
[2026-02-08 XX:XX:XX] Downloading GLSEA_GCS from 2026-02-01 to 2026-02-08...
[2026-02-08 XX:XX:XX] Successfully downloaded GLSEA_GCS
[2026-02-08 XX:XX:XX] Stored 7 records in sst_data
...
[2026-02-08 XX:XX:XX] Data fetch complete!
[2026-02-08 XX:XX:XX] ====================================================
```

This will:
- Create `cache/` directory
- Download 7 days of SST and chlorophyll data
- Initialize SQLite database (`cache/realtime_data.db`)
- Calculate bloom risk scores
- Create log file (`cache/fetch_log.txt`)

**Time estimate:** 3-5 minutes for initial download

---

### Step 5: Launch Dashboard

```bash
streamlit run app.py
```

The dashboard will:
1. Open automatically in your browser at `http://localhost:8501`
2. Start APScheduler for automated daily updates at 8:00 AM
3. Display current conditions, trends, and bloom risk

---

## 🎯 What You're Launching

### Dashboard Features

**Sidebar:**
- 🔄 Manual refresh button
- 🔴/🟡/🟢 Real-time bloom risk indicator
- 📊 Data freshness status
- 📈 Current lake-wide metrics

**Main Tabs:**
1. **Overview** — Current conditions, 7-day summary statistics
2. **Trends** — Time series plots (SST + chlorophyll), daily risk history
3. **Historical Data** — Tabular view of all cached data
4. **System Status** — Fetch history, configuration, troubleshooting info

### Automated Features

- **Daily Updates:** Fetches new data at 8:00 AM EST (configurable in `config.yaml`)
- **Bloom Risk Scoring:** Automatic calculation based on SST + chlorophyll thresholds
- **Data Retention:** Keeps 30 days of historical data (configurable)
- **Error Handling:** Automatic retries, logging, status reporting

---

## 📝 Configuration Options

Edit `config.yaml` to customize:

### Change Update Schedule
```yaml
schedule:
  hour: 8        # Change to your preferred hour (0-23)
  minute: 0
  timezone: "America/New_York"
```

### Adjust Bloom Thresholds
```yaml
thresholds:
  sst_bloom: 20.0              # °C - Temperature for HAB risk
  chl_low: 10.0                # mg/m³ - Low chlorophyll alert
  chl_moderate: 20.0           # mg/m³ - Moderate bloom
  chl_high: 40.0               # mg/m³ - Severe bloom
```

### Change Data Retention
```yaml
retention_days: 30             # Keep 30 days (increase for longer history)
```

---

## 🔧 Troubleshooting

### Issue: ModuleNotFoundError

**Symptom:** `ModuleNotFoundError: No module named 'streamlit'`

**Solution:**
```bash
ocean_env\Scripts\activate
pip install streamlit apscheduler pyyaml
```

---

### Issue: No Data in Dashboard

**Symptom:** Dashboard shows "No data" or "Initialize Data" button

**Solution:**
```bash
# Run data fetcher manually
python data_fetcher.py

# Check if database was created
dir cache\realtime_data.db
```

---

### Issue: ERDDAP Download Fails

**Symptom:** "502 Bad Gateway" or "Download failed" in logs

**Cause:** NOAA server throttling or temporary outage

**Solution:**
- Wait 5-10 minutes
- Click "Refresh Now" in dashboard
- Check server status: https://apps.glerl.noaa.gov/erddap
- Review `cache/fetch_log.txt` for details

---

### Issue: Scheduler Not Running

**Symptom:** Data not updating at 8:00 AM

**Cause:** Dashboard closed before scheduled time

**Solutions:**

**Option A:** Keep dashboard running in background
- Leave terminal and browser open
- Dashboard will fetch data at 8 AM automatically

**Option B:** Use Windows Task Scheduler (headless)
```bash
# Run as Administrator
schtasks /create /tn "Lake Erie Data Fetch" /tr "E:\BlueNexus\ocean_env\Scripts\python.exe E:\BlueNexus\03_realtime_monitor\data_fetcher.py" /sc daily /st 08:00
```

Now data fetcher runs daily even when dashboard is closed.

---

## 📊 Verification Checklist

Before considering Phase 6 complete, verify:

- [ ] `python data_fetcher.py` runs successfully
- [ ] `python alert_engine.py` shows risk assessment
- [ ] `python utils.py` runs without errors
- [ ] `streamlit run app.py` launches at localhost:8501
- [ ] Dashboard displays all 4 tabs without errors
- [ ] Sidebar shows bloom risk and metrics
- [ ] "Refresh Now" button triggers new data fetch
- [ ] Time series plot displays SST and chlorophyll
- [ ] Database created at `cache/realtime_data.db`
- [ ] Log file created at `cache/fetch_log.txt`
- [ ] Data covers last 7 days

---

## 🚀 Git Integration

Once verified, commit to GitHub:

```bash
cd E:\BlueNexus

# Add all Phase 6 files
git add 03_realtime_monitor/
git add requirements.txt
git add README.md

# Commit
git commit -m "feat: Phase 6 real-time monitoring dashboard

- Streamlit web app with live SST and chlorophyll visualization
- Automated daily data fetching via APScheduler
- Bloom risk scoring and alert system
- 7-day trend analysis with historical comparison
- SQLite caching for fast dashboard loads
- Configurable thresholds and update schedule"

# Push
git push origin main
```

---

## 🎓 Technical Architecture Summary

**Data Pipeline:**
```
NOAA ERDDAP API
    ↓
data_fetcher.py (downloads NetCDF files)
    ↓
xarray processing (calculate statistics)
    ↓
SQLite database (cache/realtime_data.db)
    ↓
alert_engine.py (risk scoring)
    ↓
Streamlit dashboard (visualization)
```

**Background Scheduler:**
```
APScheduler (starts with dashboard)
    ↓
Runs daily at 8:00 AM
    ↓
Calls data_fetcher.fetch_latest_data()
    ↓
Updates SQLite database
    ↓
Dashboard auto-refreshes on next page load
```

**Technology Stack:**
- **Streamlit 1.31+** — Web framework (pure Python)
- **APScheduler 3.10+** — Background job scheduling
- **SQLite 3** — Local database (built into Python)
- **xarray 2025.6+** — NetCDF data processing
- **matplotlib + cartopy** — Scientific visualization
- **pandas + numpy** — Data analysis

---

## 🎯 Phase 6 Success Criteria

By completing this phase, you've achieved:

✅ **Real-time capability** — Dashboard updates automatically without manual intervention  
✅ **Production-ready system** — Error handling, logging, retry logic, status monitoring  
✅ **Scalable architecture** — Easy to add new data sources, alerts, or visualizations  
✅ **Portfolio showcase** — Live web app demonstrates full-stack engineering skills  
✅ **Deployment-ready** — Can push to Streamlit Cloud for public access  

---

## 🔮 Future Enhancements (Phase 7+)

Now that automation is working, you can extend with:

1. **Email/SMS Alerts** — Notify when bloom risk goes High
2. **Historical Playback** — Slider to view conditions from any past date
3. **Multi-Lake Support** — Expand to all Great Lakes
4. **Predictive Models** — Use ML to forecast bloom risk 3-7 days ahead
5. **Export Reports** — Generate PDF summaries for stakeholders
6. **Public Deployment** — Push to Streamlit Cloud with custom domain

---

## 📞 Support

For questions or issues:

1. **Check logs:** `cache/fetch_log.txt`
2. **Review config:** `config.yaml`
3. **Read docs:** `SETUP.md` (comprehensive troubleshooting)
4. **Test components:** Run `python data_fetcher.py`, `python alert_engine.py`, `python utils.py` individually
5. **GitHub Issues:** Post to repository if problem persists

---

**⚓ Well done, Captain! Phase 6 complete — you've built a production-ready real-time monitoring system. The ship is now sailing on autopilot.**

🌊 **Project Blue Nexus** — From static analysis to live automation  
🚀 Built by Ryan Jones — February 2026
