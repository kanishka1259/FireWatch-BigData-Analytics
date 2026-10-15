# NASA FIRMS Global Fire Hotspot Big Data Case Study
## Scalable PySpark Analytics, Machine Learning & WARD Full-Stack Decision Model

---

### Executive Summary
This case study provides an end-to-end Big Data and Machine Learning solution applied to the **NASA FIRMS (Fire Information for Resource Management System)** Global Fire Hotspot archive. Processing **9,511,402 satellite telemetry records (~731 MB)** using **PySpark**, the system extracts physical thermal patterns, resolves 5 core operational frame questions, builds an operational **Wildfire Alert & Rapid Dispatch (WARD)** decision framework, trains predictive Machine Learning models (Random Forest and Gradient Boosted Trees), integrates a **MongoDB NoSQL backend**, and serves results through an interactive **full-stack web application**.

---

### 1. Real-Time Dataset Specifications
* **Source:** NASA FIRMS (MODAPS EOSDIS / VIIRS S-NPP Sensor)
* **Dataset File:** `fire_archive_SV-C2_817450.csv`
* **Total Records Ingested:** 9,511,402 rows (100% clean data retention)
* **Dataset Size:** ~731 MB (15 attributes)
* **Telemetry Features:**
  * `latitude`, `longitude`: Geospatial location of detected hotspot.
  * `brightness`: VIIRS I-4 thermal channel (4 µm) brightness temperature (Kelvin).
  * `scan`, `track`: Geometric footprint dimensions of the satellite scan pixel.
  * `acq_date`, `acq_time`: Acquisition timestamp (UTC).
  * `satellite`, `instrument`: Suomi NPP VIIRS instrument telemetry.
  * `confidence`: Detection confidence tier (`l`: low, `n`: nominal, `h`: high).
  * `bright_t31`: Channel I-5 thermal background channel (11 µm) in Kelvin.
  * `frp`: Fire Radiative Power (MegaWatts, MW) — direct physical indicator of biomass consumption rate.
  * `daynight`: Diurnal (`D`) vs Nocturnal (`N`) overpass flag.
  * `type`: Source classification (0 = Presumed vegetation wildfire, 1 = Active volcano, 2 = Static industrial flare, 3 = Offshore oil flare).

---

### 2. Preprocessing & PySpark Pipeline
* **Engine:** Apache Spark / PySpark 4.2.0 local multi-core cluster (`local[*]`).
* **Data Cleansing:** Automated null filtering, type casting to double/integer schemas, and coordinate boundary validation.
* **Feature Engineering:**
  * $\Delta T_{\text{thermal}} = \text{brightness} - \text{bright\_t31}$ (Spectral thermal contrast between 4 µm and 11 µm channels; values $>30\text{ K}$ indicate flaming crown combustion).
  * Temporal decomposition: `hour` (UTC 0–23) and `month` (1–12).
  * Geospatial binning: $5^\circ \times 5^\circ$ lat/lon grid clusters for regional hotspot risk aggregation.
  * Operational Target: Multi-tier categorization based on Fire Radiative Power thresholds.

---

### 3. 15-Minute Analytical Frame Questions
The following five analytical questions were formulated and resolved across all 9.5 million rows:

#### **Q1: Spatiotemporal Dynamics**
* **Question:** What is the monthly and diurnal temporal distribution of global fire hotspots, and when do peak thermal energy spikes occur?
* **Findings:** Global fire incidence peaks strongly between July and October (averaging 1.2M to 1.45M fires per month), coinciding with boreal summers and tropical dry clearance seasons. Minimum activity occurs in February (~380K fires).
* **Operational Decision:** Aerial patrol flights and seasonal water-bomber readiness should scale by 300% during the July–October operational window.

#### **Q2: Diurnal vs. Nocturnal Thermal Energetics**
* **Question:** How does Fire Radiative Power (FRP) and flaming intensity vary between diurnal and nocturnal satellite overpasses?
* **Findings:** Daytime detections comprise 70.3% of total events due to lower solar humidity and solar heating. However, nocturnal detections with $\text{FRP} \ge 30\text{ MW}$ represent aggressive crown wildfires that have broken through the nighttime temperature inversion.
* **Operational Decision:** Any nighttime hotspot with $\text{FRP} \ge 30\text{ MW}$ bypasses ground validation and triggers immediate Level-1 emergency status.

#### **Q3: Land Cover Source & Confidence Screening**
* **Question:** What proportion of hotspots represent genuine wildfires versus industrial flaring or volcanic activity?
* **Findings:** 90.1% of detections represent genuine vegetation/wildfires (type 0), while 8.9% are static land sources (gas flares, industrial stacks, refineries).
* **Operational Decision:** Geospatial polygon masking of known industrial extraction fields prevents false-alarm resource mobilizations.

#### **Q4: Thermal Radiative Correlation Analysis**
* **Question:** How strongly do VIIRS spectral brightness temperatures correlate with physical Fire Radiative Power (FRP)?
* **Findings:** Channel I-4 brightness and the thermal delta ($\text{Brightness} - \text{Bright\_T31}$) correlate strongly ($r = +0.658$) with FRP, demonstrating that spectral thermal contrast is the premier physical predictor for fire energy release.
* **Operational Decision:** Spectral thermal contrast is selected as the top weighted feature in the automated ML dispatch pipeline.

#### **Q5: Extreme Wildfire Risk Clustering**
* **Question:** Where are persistent extreme fire clusters concentrated, and how does this define the operational dispatch framework?
* **Findings:** High-density extreme fire clusters ($\text{FRP} \ge 50\text{ MW}$) concentrate in sub-Saharan savanna regions, the Amazon-Cerrado agricultural frontier, and boreal Siberian forests.
* **Operational Decision:** Established the **Wildfire Alert & Rapid Dispatch (WARD)** framework with 4 actionable operational tiers:
  * **Tier 0: Extreme ($\ge 50\text{ MW}$):** Immediate aerial water/retardant bombing.
  * **Tier 1: High ($20 - 50\text{ MW}$):** Mobilize heavy ground firefighting crews.
  * **Tier 2: Moderate ($5 - 20\text{ MW}$):** Alert local forest patrol rangers.
  * **Tier 3: Low ($< 5\text{ MW}$):** Routine satellite surveillance.

---

### 4. Machine Learning Implementation & Performance Analysis
Two predictive models were implemented and benchmarked:

#### Model 1: Multi-Class Fire Severity Classifier
* **Architecture:** Random Forest Classifier (100 estimators, max depth 16) vs. Histogram Gradient Boosted Trees (benchmark).
* **Target:** 4-Tier Severity (`Low`, `Moderate`, `High`, `Extreme`).
* **Input Features:** `latitude`, `longitude`, `brightness`, `bright_t31`, `brightness_diff`, `scan`, `track`, `hour`, `month`, `is_day`, `confidence_num`.
* **Performance Metrics:**
  * **Accuracy:** 74.72% (4-class fine-grained boundary classification)
  * **Weighted F1-Score:** 0.7309
  * **Multiclass ROC-AUC:** 0.8742 (Excellent discrimination)
  * **Inference Latency:** 0.08 ms per sample

#### Model 2: Continuous FRP Regressor
* **Architecture:** Random Forest Regressor (80 estimators, max depth 14).
* **Target:** Fire Radiative Power in MegaWatts (continuous energy output).
* **Performance Metrics:**
  * **RMSE:** 13.45 MW
  * **MAE:** 5.23 MW
  * **$R^2$ Score:** 0.4739 (robust given high natural skewness of open wildfires)

---

### 5. NoSQL Backend Integration
* **Database:** **MongoDB** (connected to local server `localhost:27017`) with automatic fallback to embedded SQLite Document Store.
* **Capabilities:** Geospatial bounding-box queries, severity index retrieval, JSON document ingestion.
* **Ingestion:** 5,000 representative hotspots ingested in 0.21s with indexed query times $< 2\text{ ms}$.

---

### 6. Full-Stack Web Application Architecture
* **Backend:** Python Flask REST API (`webapp/app.py`) running on `http://127.0.0.1:5000`.
* **Frontend:** Glassmorphic dark-mode dashboard (`webapp/templates/index.html`) featuring:
  * **Live Leaflet.js Map:** Color-coded clustering and filtering by severity.
  * **Chart.js Visualizations:** Interactive monthly trends and day/night energetics.
  * **Real-Time WARD Dispatch Simulator:** Interactive tool allowing dispatchers to test coordinates and sensor readings to obtain instant ML predictions and operational protocols.
  * **Model Performance Scorecard:** Comprehensive metrics table.

---

### Summary of Artifacts Generated
| Artifact | Location | Purpose |
|---|---|---|
| **PySpark Engine** | `spark_analysis/pyspark_eda.py` | Distributed data processing across 9.5M rows |
| **Member 2 Temporal Engine** | `spark_analysis/member2_temporal_analysis.py` | Chronological series, diurnal 24-hr cycle & 7-day rolling window |
| **EDA Visualizations** | `spark_analysis/generate_visualizations.py` | 6 publication-quality charts in `outputs/charts/` |
| **Member 2 Visualizations** | `spark_analysis/generate_member2_visualizations.py` | 5 publication-quality temporal charts in `outputs/charts/` |
| **ML Models & Evaluation** | `ml_model/train_and_evaluate.py` | Random Forest & GBT models + temporal holdout evaluation |
| **NoSQL Backend** | `nosql/nosql_store.py` | MongoDB / Document store persistence with temporal aggregations |
| **Web Application** | `webapp/app.py` & `webapp/templates/index.html` | Full-stack interactive dashboard with live temporal metrics & Chart.js |
| **Member 2 Report** | `outputs/MEMBER2_TEMPORAL_ANALYSIS_REPORT.md` | Comprehensive empirical report resolving all temporal research questions |

