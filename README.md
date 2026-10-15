# 🔥 FireWatch: NASA FIRMS Big Data Analytics & Wildfire Dispatch AI

An end-to-end Big Data & Machine Learning case study processing **9.51 Million satellite telemetry records (~731 MB)** from the **NASA FIRMS** global fire archive. Featuring distributed **PySpark** preprocessing and EDA, an operational **Wildfire Alert & Rapid Dispatch (WARD)** decision framework, predictive **Random Forest & Gradient Boosted Trees** models, **MongoDB NoSQL** integration, and a full-stack **Flask + Leaflet.js** interactive dashboard.

---

## 🌟 Key Features

- **Distributed Big Data Engine**: Ingests and cleans 9,511,402 records in ~120s using PySpark.
- **5 Frame Analytical Questions**: Resolves core spatiotemporal, diurnal vs. nocturnal, land cover source, thermal contrast, and extreme hotspot clustering dynamics.
- **WARD Decision Framework**: Real-time operational decision framework mapping physical fire parameters to actionable emergency dispatch protocols.
- **Dual ML Architecture**:
  - **Fire Severity Classifier**: Random Forest (74.7% 4-class accuracy, 0.874 ROC-AUC) & GBT.
  - **FRP Regressor**: Predicts continuous Fire Radiative Power (MW) for water/retardant drop volume estimation.
- **NoSQL Persistence**: MongoDB integration with embedded document store fallback, featuring geospatial indexing and bounding-box queries.
- **Interactive Full-Stack Web App**: Dark-mode glassmorphic dashboard with live Leaflet.js clustering, Chart.js analytics, and real-time inference simulator.

---

## 🏗️ Architecture & Pipeline

```text
[NASA FIRMS Raw VIIRS / MODIS Archive] (9.51M Records, ~731 MB)
                         │
                         ▼
             [PySpark Distributed Engine]
       ├── Data Cleansing & Schema Casting
       ├── Thermal Contrast ΔT = (Brightness - Bright_T31)
       └── Resolution of 5 Core Analytical Questions
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
  [Aggregated Data & JSON]       [ML Training Pipeline]
         │                       ├── Random Forest Classifier
         │                       ├── HistGradientBoosting
         │                       └── FRP Continuous Regressor
         ▼                               │
  [NoSQL MongoDB Store]                  │
  (Geospatial & Severity Index)          ▼
         │                        [Saved Models]
         └───────────────┬───────────────┘
                         ▼
        [Full-Stack Flask Web Application]
       ├── REST APIs (/api/stats, /api/predict, /api/hotspots)
       ├── Interactive Leaflet.js Clustered Map
       └── Real-Time WARD Operational Dispatch Simulator
```

---

## 📊 Analytical Findings (5 Frame Questions)

1. **Spatiotemporal Wildfire Dynamics**: Peak global wildfire activity is concentrated between **July and October** (1.25M to 1.45M fires/month).
2. **Diurnal vs. Nocturnal Thermal Energetics**: Daytime detections represent 70.3% of events, but nocturnal detections with $\text{FRP} \ge 30\text{ MW}$ signify aggressive crown fires requiring highest-priority dispatch.
3. **Land Cover Source Screening**: 90.1% of detections are genuine vegetation fires, while 8.9% are static industrial gas flares requiring spatial polygon masking.
4. **Thermal Radiative Correlation**: VIIRS Channel I-4 brightness and $\Delta T$ exhibit strong correlation ($r = +0.658$) with physical FRP.
5. **Extreme Hotspot Clustering**: Establishes the 4-tier WARD operational dispatch framework (Tier 0: $\ge 50\text{ MW}$ immediate aerial drop down to Tier 3: $<5\text{ MW}$ surveillance).

---

## 🚀 Quickstart & Usage

### 1. Requirements
- Python 3.10+
- Java 11 / 17 / 21+ (for PySpark)
- MongoDB (optional, automatic fallback to embedded SQLite document store)

### 2. Installation
```bash
pip install pyspark pandas numpy matplotlib seaborn scikit-learn flask pymongo plotly joblib
```

### 3. Run Pipeline
```bash
# Step 1: Run PySpark Preprocessing and EDA
python spark_analysis/pyspark_eda.py

# Step 2: Run Member 2 Distributed Temporal Analysis (PySpark / Spark SQL)
python spark_analysis/member2_temporal_analysis.py

# Step 3: Generate Publication-Quality Visualizations
python spark_analysis/generate_visualizations.py
python spark_analysis/generate_member2_visualizations.py

# Step 4: Run Unit Tests for Temporal Parsing & Thresholds
python -m unittest tests/test_temporal_analysis.py

# Step 5: Train Machine Learning Models & Evaluate Performance (Includes Temporal Holdout)
python ml_model/train_and_evaluate.py

# Step 6: Populate NoSQL Document Store (Hotspots & Temporal Aggregates)
python nosql/nosql_store.py

# Step 7: Launch Full-Stack Web Dashboard
python webapp/app.py
```
Open **[http://localhost:5000](http://localhost:5000)** in your browser.

---

## 📈 Model Performance Summary

| Architecture | Target Variable | Accuracy | Weighted F1 | ROC-AUC | Error / RMSE | Latency |
|---|---|---|---|---|---|---|
| **Random Forest Classifier** | Severity Tier (4-Class) | **74.72%** | **0.7309** | **0.8742** | — | **0.08 ms** |
| **HistGradientBoosting** | Severity Tier (4-Class) | **74.70%** | **0.7297** | **0.8747** | — | **0.04 ms** |
| **Random Forest Regressor** | Fire Radiative Power (MW) | — | — | $R^2 = 0.474$ | **13.45 MW** | **0.12 ms** |

---

## 📜 Documentation
For the complete technical case study and analysis, see [CASE_STUDY_REPORT.md](CASE_STUDY_REPORT.md).
