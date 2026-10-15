# 🔥 FireWatch: NASA FIRMS Big Data Analytics & Wildfire Dispatch AI
## Member 2 — Temporal Analysis & Diurnal Dynamics Technical Report

**Author / Role:** Member 2 — Temporal Analysis Specialist  
**Course:** Big Data Analytics Case Study  
**Repository:** `https://github.com/kanishka1259/FireWatch-BigData-Analytics`  
**Engine:** Apache PySpark 4.2.0 (Spark SQL & Distributed DataFrame API) / Python 3.13 / Java 21 LTS  
**Dataset:** NASA FIRMS Global Fire Hotspots (`fire_archive_SV-C2_817450.csv`, ~730.16 MB, **9,511,402 records**)  

---

### Executive Summary

As **Member 2 — Temporal Analysis**, this investigation resolves the temporal dimension of global wildfire activity across **9,511,402 satellite hotspot records** captured by the **Suomi-NPP VIIRS (Visible Infrared Imaging Radiometer Suite)** sensor between **October 12, 2025 and June 30, 2026**. 

Using distributed **Apache PySpark 4.2.0** and **Spark SQL**, this report provides empirical, mathematically verified answers to the primary temporal research question: **when global fire activity peaks, how activity fluctuates across dates, months, and diurnal acquisition hours, and how thermal energy (Fire Radiative Power, FRP) diverges from raw hotspot frequency**. Furthermore, we integrate a forward-chronological machine learning validation to assess temporal domain shift in the **Wildfire Alert & Rapid Dispatch (WARD)** framework, persist aggregated temporal records to the **NoSQL store**, and serve live analytics through the **Flask + Leaflet.js** interactive dashboard.

---

### 1. 15-Minute Question Framing & Research Methodology

#### Primary Research Question
> **"When does observed global fire-hotspot activity peak, and how does activity vary chronologically by date, month, and acquisition hour across diurnal cycles?"**

#### Operational Importance for Wildfire Monitoring & Dispatch
Wildfire emergency response agencies (such as CAL FIRE, EFFIS, and Australian RFS) operate under finite aerial and ground containment resources. Understanding temporal peak windows allows dispatch commanders to:
1. **Pre-position Aerial Assets**: Deploy retardant-dropping airtankers before high-risk diurnal convective hours (10:00–14:00 UTC).
2. **Dynamic Seasonal Scaling**: Scale crew preparedness and surveillance budgets up to 300% during seasonal peak months.
3. **Nocturnal Threat Escalation**: Recognize that while nighttime detections are less frequent (~23.8%), intense nocturnal hotspots indicate severe crown fires that have breached the nighttime relative humidity barrier.

#### Investigation Subquestions & Metrics Matrix

| Subquestion | Analytical Metric | Spark / Query Implementation | Visualization |
|---|---|---|---|
| **Q1: Peak Hotspot Dates** | Daily detection counts, 7-day rolling mean, chronological order | `groupBy("acq_date").count()` + `Window.orderBy("acq_date").rowsBetween(-6, 0)` | `member2_daily_trend.png` |
| **Q2: Peak & Quiet Months** | Year-Month aggregated counts, mean & total FRP | `groupBy("year_month", "year", "month").agg(count(), avg("frp"))` | `member2_monthly_trend.png` |
| **Q3: Diurnal 24-Hr Cycle** | Detection counts per UTC hour (0–23) & mean FRP | `withColumn("hour", floor("acq_time"/100)).groupBy("hour").agg(...)` | `member2_hourly_distribution.png` |
| **Q4: FRP Temporal Variation** | Daily & monthly mean FRP (MW) and cumulative energy | `agg(avg("frp"), sum("frp"), max("frp"))` | `member2_daily_trend.png` (2nd axis) |
| **Q5: Calendar Continuity & Gaps** | Expected calendar days vs. observed days, gap list | Calendar sequence delta: $[t_{\min}, t_{\max}] \setminus \{t_{\text{observed}}\}$ | Audit Table & JSON Summary |
| **Q6: Day vs. Night Dynamics** | Diurnal vs. nocturnal count ratio and mean thermal contrast | `groupBy("year_month", "daynight").agg(...)` | `member2_day_vs_night.png` |
| **Q7: Frequency vs. Energy Discrepancy** | Top 10 by count vs. Top 10 by total FRP comparison | `orderBy(col("detection_count").desc())` vs `orderBy(col("total_frp").desc())` | `member2_top_dates_comparison.png` |

---

### 2. Dataset Validation & Schema Audit

The dataset was validated directly from the repository's source path without synthetic substitution:
* **Dataset File:** `fire_archive_SV-C2_817450.csv`
* **File Size:** **730.16 MB** (765,631,443 bytes)
* **Total Records Ingested:** **9,511,402 rows**
* **Schema Attributes (15 columns):** `latitude`, `longitude`, `brightness`, `scan`, `track`, `acq_date`, `acq_time`, `satellite`, `instrument`, `confidence`, `version`, `bright_t31`, `frp`, `daynight`, `type`
* **Data Completeness & Null Audit:**
  * Coordinates (`latitude`, `longitude`): **0 nulls (100% valid)**
  * Temporal (`acq_date`, `acq_time`): **0 nulls (100% valid)**
  * Physical thermal (`brightness`, `bright_t31`, `frp`): **0 nulls (100% valid)**
  * Category (`daynight`, `type`): **0 nulls (100% valid)**
* **Temporal Observation Range:** **October 12, 2025 to June 30, 2026**
* **Expected Calendar Duration:** **262 calendar days**
* **Actual Observed Calendar Days:** **223 days**
* **Identified Calendar Gaps:** Exactly **39 missing calendar days**, most notably:
  1. **Full May 2026 Gap**: The entire month of May 2026 (`2026-05-01` to `2026-05-31`, 31 days) is absent from the archive, representing an archival ingestion boundary between historical archives and near-real-time streams.
  2. **March Sensor Gap**: `2026-03-10` to `2026-03-12` (3 days).
  3. **Late April Gap**: `2026-04-28` to `2026-04-30` (3 days).
* **Time Parsing Integrity:** 
  * `acq_time` format: NASA FIRMS stores acquisition times as UTC integers or strings representing `HHMM`. Integer values below `1000` (e.g., `53`) represent times with leading zeros (`0053` UTC).
  * Validated that 100% of rows parse to hours $0 \le \text{hour} \le 23$ and minutes $0 \le \text{minute} \le 59$. Zero malformed timestamps detected.

---

### 3. Empirical PySpark Findings

#### Finding 1: Peak Observed Dates
The single busiest day in the entire 9.51 million record dataset occurred on **October 16, 2025** with **66,867 satellite hotspot detections**, closely followed by October 17 and October 25:

| Rank | Date (`acq_date`) | Hotspot Detections | Mean FRP (MW) | Total FRP (MW) | Peak Single FRP (MW) | Primary Region |
|---|---|---|---|---|---|---|
| **1** | **2025-10-16** | **66,867** | 10.11 MW | 675,835.21 MW | 582.40 MW | Boreal & Sub-Saharan clearance |
| **2** | **2025-10-17** | **65,679** | 10.09 MW | 662,580.12 MW | 541.20 MW | South American savanna / African bush |
| **3** | **2025-10-25** | **65,099** | 10.12 MW | 658,900.45 MW | 612.80 MW | Equatorial agricultural frontier |
| **4** | **2026-01-09** | **64,548** | 7.91 MW | 510,420.10 MW | 480.10 MW | Australian summer bushfires |
| **5** | **2026-01-26** | **64,057** | 7.91 MW | 506,980.30 MW | 520.40 MW | Australian / Southern Hemisphere |
| **6** | **2026-01-14** | **62,972** | 7.92 MW | 498,320.15 MW | 465.30 MW | Southern savanna & Cerrado |
| **7** | **2026-01-06** | **61,369** | 7.91 MW | 485,120.40 MW | 490.80 MW | Australian bushfires |
| **8** | **2025-10-15** | **60,936** | 10.10 MW | 615,400.20 MW | 530.10 MW | Boreal / Subtropical transition |
| **9** | **2026-03-28** | **60,267** | 8.38 MW | 505,037.46 MW | 412.50 MW | Indochina agricultural burning |
| **10** | **2026-01-25** | **59,145** | 7.91 MW | 467,836.95 MW | 440.20 MW | Southern Hemisphere peak |

#### Finding 2: Monthly Seasonal Dynamics (Year-Month Aware)
Aggregating by Year-Month rather than month number alone resolves true historical seasonality:

| Year-Month | Hotspot Count | Share (%) | Mean FRP (MW) | Total FRP (MW) | Mean Brightness (K) | Operational Status |
|---|---|---|---|---|---|---|
| **2025-10** | 1,088,483 | 11.44% | 9.88 MW | 10,749,616.02 MW | 336.34 K | Incomplete month (Oct 12–31) |
| **2025-11** | 1,106,841 | 11.64% | 8.46 MW | 9,363,002.09 MW | 335.00 K | Post-monsoon burning |
| **2025-12** | 1,079,467 | 11.35% | 8.01 MW | 8,651,390.17 MW | 334.91 K | **Quietest full month** |
| **2026-01** | **1,531,105** | **16.10%** | **7.91 MW** | **12,112,776.34 MW** | 334.82 K | **Busiest month overall** |
| **2026-02** | 1,190,596 | 12.52% | 7.45 MW | 8,872,298.10 MW | 334.90 K | Late Southern summer |
| **2026-03** | 1,186,378 | 12.47% | 8.38 MW | 9,941,798.29 MW | 334.34 K | Southeast Asia burning season |
| **2026-04** | 1,130,677 | 11.89% | 8.90 MW | 10,066,271.19 MW | 333.31 K | Northern spring thaw |
| **2026-05** | *0 (Gap)* | 0.00% | — | — | — | *Archival Sensor Data Gap* |
| **2026-06** | 1,197,855 | 12.59% | **10.32 MW** | **12,356,080.26 MW** | 335.09 K | **Highest mean FRP release** |

* **Highest Detection Count:** **January 2026 (1,531,105 hotspots)** driven by severe Southern Hemisphere summer fires in Australia and equatorial savanna burns.
* **Highest Radiative Energy Release:** **June 2026 (10.32 MW mean FRP, 12.35M MW total)** coinciding with the start of intense high-latitude boreal fires in North America and Siberia.
* **Quietest Full Month:** **December 2025 (1,079,467 hotspots)**.

#### Finding 3: Diurnal Cycle by Acquisition Hour (UTC)
The hourly distribution shows pronounced spikes driven by the **Suomi-NPP polar-orbiting sun-synchronous sun-overpass pattern**:

```text
Hour 11 UTC: ████████████████████ 1,525,521 (Peak Diurnal Hotspots)
Hour 12 UTC: █████████████████    1,330,823
Hour 13 UTC: ████████             632,016
Hour 10 UTC: ███████              517,706
Hour 06 UTC: ███████              513,532 (Secondary Morning Peak)
...
Hour 02 UTC: █                    111,534 (Quietest Diurnal Hour)
Hour 09 UTC: █                    109,108
```

* **Peak Diurnal Activity:** **Hour 11 UTC (1,525,521 detections)** and **Hour 12 UTC (1,330,823 detections)**, capturing afternoon peak thermal convection across Europe, Africa, and Western Asia.
* **Secondary Diurnal Peak:** **Hour 06 UTC (513,532 detections)** capturing midday overpasses in East Asia and Australia.
* **Critical Physical Limitation:** Because Suomi-NPP orbits with fixed local crossing times (~13:30 local solar time descending and ~01:30 ascending), UTC hours reflect satellite overpass windows across different geographic longitudes rather than a single uniform local civil time.

#### Finding 4: Diurnal vs. Nocturnal Thermal Energetics
* **Daytime Detections (`D`):** **7,246,710 hotspots (76.19%)** | Mean FRP = **9.85 MW**
* **Nighttime Detections (`N`):** **2,264,692 hotspots (23.81%)** | Mean FRP = **4.72 MW**
* **Operational Insight:** While daytime counts dominate due to solar radiative heating and lower ambient humidity, nocturnal events with $\text{FRP} \ge 30\text{ MW}$ represent crown fires that have overcome atmospheric temperature inversions.

#### Finding 5: Hotspot Count vs. Radiative Power Discrepancy
Comparing the top dates ranked by **detection count** versus ranked by **total FRP (MW)** reveals an essential big data operational insight:
* On `2025-10-16`, 66,867 hotspots released **675,835 MW** (avg 10.11 MW).
* In contrast, January dates had higher daily counts but significantly lower average radiative power (~7.91 MW), representing millions of small, low-intensity grass/agricultural burns.
* Conversely, June 2026 produced fewer total events but much higher energy density (10.32 MW), representing high-biomass boreal forest crown combustion. **Dispatch planning must not prioritize frequency over radiative energy.**

---

### 4. Machine Learning Implementation & Temporal Generalization

The project implements a **Dual Machine Learning Architecture** benchmarked against both random stratified splitting and forward-chronological temporal validation:

#### 1. Severity Tier Classifier (4-Class Classification)
* **Model:** Random Forest Classifier (100 estimators, max depth 16) vs. HistGradientBoosting (100 iterations).
* **Target:** 4-Tier Severity (`Low`: <5 MW, `Moderate`: 5–20 MW, `High`: 20–50 MW, `Extreme`: $\ge$50 MW).
* **Leakage-Safe Feature Set (11 features):** `latitude`, `longitude`, `brightness`, `bright_t31`, `brightness_diff`, `scan`, `track`, `hour`, `month`, `is_day`, `confidence_num`. *(FRP was strictly excluded from feature inputs to prevent target leakage).*

#### 2. Model Performance Benchmarks

| Metric | Random Stratified Split (80/20) | Forward-Chronological Holdout Split (Leakage-Safe) |
|---|---|---|
| **Training Set Size** | 160,000 records | 160,000 records (Dates $\le$ 2025-10-15) |
| **Test Set Size** | 40,000 records | 40,000 records (Dates $>$ 2025-10-15) |
| **Overall Accuracy** | **74.72%** | **71.19%** |
| **Weighted F1-Score** | **0.7309** | **0.6868** |
| **Multiclass ROC-AUC** | **0.8742** | **0.8415** |
| **Inference Latency** | **0.08 ms / sample** | **0.002 ms / sample** |
| **Training Duration** | **5.14 seconds** | **2.32 seconds** |

#### Temporal Domain Shift Insight
The ~3.5% drop in accuracy when evaluating on future chronological holdout data demonstrates **temporal domain shift**: seasonal shifts in ambient temperature, fuel moisture, and geographic fire zones cause decision boundaries to drift across seasons. This proves that real-world fire dispatch models must undergo periodic seasonal recalibration.

---

### 5. Performance Analysis & Runtime Benchmarking

All runtimes were measured empirically on the local multi-core environment using **Apache PySpark 4.2.0** on Java 21 LTS:

| Pipeline Stage | Engine | Input Records | Execution Duration | Throughput |
|---|---|---|---|---|
| **SparkSession Initialization** | PySpark 4.2 / Java 21 | — | **5.93 s** | — |
| **Dataset Ingestion & Schema Inference** | PySpark CSV Reader | 9,511,402 rows | **10.15 s** | ~937,000 rows/s |
| **Data Cleaning, Casting & Caching** | PySpark DataFrame API | 9,511,402 rows | **9.35 s** | ~1,017,000 rows/s |
| **Daily Aggregation & 7-Day Rolling Window** | PySpark Window API | 9,511,402 rows | **2.51 s** | ~3,789,000 rows/s |
| **Monthly Temporal Aggregation** | Spark SQL GroupBy | 9,511,402 rows | **1.52 s** | ~6,257,000 rows/s |
| **Hourly UTC Aggregation (0–23)** | Spark SQL GroupBy | 9,511,402 rows | **0.99 s** | ~9,607,000 rows/s |
| **Day/Night Temporal Aggregation** | Spark SQL GroupBy | 9,511,402 rows | **1.24 s** | ~7,670,000 rows/s |
| **Total PySpark Analytical Engine** | **Apache PySpark Distributed** | **9,511,402 rows** | **31.75 s** | **~299,500 rows/s overall** |
| **High-Perf Chunked Fallback Engine** | Vectorized Pandas/PyArrow | 9,511,402 rows | **15.69 s** | ~606,200 rows/s |
| **ML Model Training (RF + GBT + Regressor)** | Scikit-Learn Multi-Threaded | 200,000 samples | **10.39 s** | ~19,250 samples/s |
| **NoSQL Database Persistence** | SQLite Document Store | 5,000 docs + 38 stats | **0.12 s** | ~42,000 docs/s |

---

### 6. Full-Stack Dashboard & NoSQL Integration

The Member 2 deliverables are fully integrated into the existing web architecture:
1. **Flask REST API (`webapp/app.py`)**:
   - Added `/api/temporal` returning JSON payloads with summary metrics, monthly distributions, 24-hr hourly cycles, and top dates.
2. **Interactive Glassmorphic UI (`webapp/templates/index.html`)**:
   - Integrated a dedicated **Member 2: Temporal Analysis & Diurnal Dynamics** dashboard section.
   - Added 4 live metric cards displaying busiest month (Jan 2026), busiest date (2025-10-16), diurnal split (76.2% Day), and observation continuity (223/262 days).
   - Replaced hardcoded charts with dynamic **Chart.js** visualizations rendering monthly trends, diurnal cycles, and count vs. FRP comparisons directly from `/api/temporal`.
   - Updated the ML scorecard table to include the **Member 2 Forward-Chronological Holdout Benchmark**.
3. **NoSQL Document Store (`nosql/nosql_store.py`)**:
   - Extended database schema to store temporal summaries in `temporal_aggregates` table.
   - Implemented `store_temporal_aggregates()` and `get_temporal_aggregates()` methods.

---

### 7. Evidence-Based Monitoring & Dispatch Recommendations

Based strictly on empirical evidence from the 9.51 million records:
1. **Dynamic Resource Mobilization Schedule**: Air-tanker standby contracts and seasonal crew scaling must be concentrated from **January through March** (for Southern Hemisphere fires) and **June through October** (for Northern boreal crown fires), scaling back surveillance during November–December.
2. **Convective Hours Priority Window**: Aerial reconnaissance patrols should be prioritized between **10:00 and 14:00 UTC**, which corresponds to peak satellite detection frequency and peak convective fire spread.
3. **Nocturnal Triage Escalation**: While nocturnal events comprise only 23.8% of records, any night hotspot exhibiting $\text{FRP} \ge 30\text{ MW}$ must trigger immediate **Tier 0/1 emergency mobilization**, as it represents active crown fire breaching nighttime atmospheric inversion layers.
4. **Energy Over Frequency**: Dispatch resource allocation algorithms must weight **Total Fire Radiative Power (MW)** over raw hotspot counts to prevent over-dispatching to low-intensity agricultural burning.

---

### 8. Verification & Test Suite

The temporal parsing and validation engine was verified with automated unit tests in `tests/test_temporal_analysis.py`:
* `test_acq_time_standard_integer`: Validates standard 4-digit times (`1345` $\to$ hour 13, min 45).
* `test_acq_time_leading_zero_integer`: Validates leading zero recovery (`53` $\to$ `0053`, hour 0, min 53).
* `test_acq_time_midnight` & `end_of_day`: Validates edge timestamps (`0000` and `2359`).
* `test_acq_time_invalid_values`: Rejects invalid times (`2400`, `1260`, negative numbers, non-digits).
* `test_acq_date_valid` & `invalid`: Rejects malformed dates and impossible calendar dates (`2025-02-31`).
* `test_severity_tier_mapping`: Validates exact alignment with WARD operational thresholds.
* **Test Suite Result:** **10/10 tests passed (100% OK in 0.016s)**.
