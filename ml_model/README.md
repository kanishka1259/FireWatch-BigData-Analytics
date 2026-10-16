# 🤖 Machine Learning Model Architecture & Benchmark Suite

This directory contains the machine learning pipelines for the **FireWatch Big Data Wildfire Analytics** system. The models classify fire severity into 4 operational tiers and predict continuous Fire Radiative Power (MW) across **9.51 million NASA FIRMS satellite observations**.

---

## 📊 Comprehensive Model Performance Scorecard

Evaluated on **200,000 stratified observations** with an 80/20 train/test split:

| Model Script | Architecture | Accuracy | Macro Precision | Macro Recall | Weighted F1 | ROC-AUC (OVR) | Training Time | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [`model_4_xgboost.py`](model_4_xgboost.py) | **XGBoost Classifier** *(Champion)* | **74.77%** | **70.69%** | **51.78%** | **0.7310** | **0.8752** | 5.37 s | **0.0056 ms** |
| [`model_1_random_forest.py`](model_1_random_forest.py) | **Random Forest Classifier** | 74.72% | 70.06% | 51.40% | 0.7309 | 0.8742 | 7.16 s | 0.0132 ms |
| [`model_2_hist_gradient_boosting.py`](model_2_hist_gradient_boosting.py) | **HistGradientBoosting** | 74.72% | 68.46% | 51.45% | 0.7297 | 0.8748 | 8.48 s | 0.0194 ms |
| [`model_3_extra_trees.py`](model_3_extra_trees.py) | **Extra Trees Classifier** *(Fastest)* | 74.02% | 69.89% | 49.28% | 0.7189 | 0.8678 | **2.72 s** | 0.0107 ms |

### Supporting Evaluators:
- **Continuous FRP Regressor ([`model_5_frp_regressor.py`](model_5_frp_regressor.py))**: Predicts continuous Fire Radiative Power in MegaWatts. Achieves $\text{RMSE} = 13.45\text{ MW}$, $\text{MAE} = 5.23\text{ MW}$, and $R^2 = 0.4736$.
- **Forward-Chronological Holdout ([`model_6_chronological_holdout.py`](model_6_chronological_holdout.py))**: Leakage-safe chronological validation (train on historical dates up to Oct 15, 2025; test on future dates). Demonstrates **71.19% Accuracy** and **0.8224 ROC-AUC**, quantifying **3.53% seasonal concept drift**.

---

## 📁 File Structure & Purpose

| File Name | Purpose | Simple Viva Explanation |
| :--- | :--- | :--- |
| **`train_and_evaluate.py`** | **Master Runner** | Trains all 4 classifiers + 1 regressor + holdout in one command; exports diagnostic charts (`outputs/charts/`) and metrics (`outputs/data/model_performance_metrics.json`). |
| **`model_1_random_forest.py`** | **Random Forest** | 100-tree bagging ensemble. Robust against noisy satellite pixels; outputs feature importance. |
| **`model_2_hist_gradient_boosting.py`** | **HistGBDT** | Bins continuous temperatures into 256 discrete bins for faster gradient boosting splits. |
| **`model_3_extra_trees.py`** | **Extra Trees** | Extremely randomized decision splits. **Fastest training ensemble (2.72s)**. |
| **`model_4_xgboost.py`** | **XGBoost** | Second-order Taylor series gradient boosting with L1/L2 regularization. **Highest accuracy (74.77%) and fastest latency (0.0056 ms)**. |
| **`model_5_frp_regressor.py`** | **FRP Regressor** | Predicts exact radiative energy (MW) rather than discrete categories. |
| **`model_6_chronological_holdout.py`** | **Holdout Test** | Trains on past dates and tests on future dates to eliminate temporal data leakage. |

---

## 🎯 Target Severity Classes (4-Tier WARD Framework)

The models predict four operational severity tiers governed by Fire Radiative Power (FRP):

1. **Tier 0 - Low (< 5 MW)**: Minor ground smoldering. Action: Routine satellite monitoring.
2. **Tier 1 - Moderate (5 – 20 MW)**: Surface brushfire. Action: Alert local forest ranger patrols.
3. **Tier 2 - High (20 – 50 MW)**: Active brush/forest fire. Action: Mobilize ground firefighting units.
4. **Tier 3 - Extreme (≥ 50 MW)**: Catastrophic crown fire. Action: **Immediate aerial water/retardant drop**.

---

## ⚙️ How to Run Any Model Individually

```bash
# Run Random Forest
python ml_model/model_1_random_forest.py

# Run HistGradientBoosting
python ml_model/model_2_hist_gradient_boosting.py

# Run Extra Trees (Fastest)
python ml_model/model_3_extra_trees.py

# Run XGBoost (Champion)
python ml_model/model_4_xgboost.py

# Run Continuous FRP Regressor
python ml_model/model_5_frp_regressor.py

# Run Chronological Holdout (Concept Drift)
python ml_model/model_6_chronological_holdout.py

# Run Master Suite (Trains all models, produces charts and metrics JSON)
python ml_model/train_and_evaluate.py
```

> **Note on `.pkl` files**: Large trained model weight binaries (`*.pkl`, up to 133 MB) exceed GitHub's 100 MB file size limit and are generated locally when running the scripts.
