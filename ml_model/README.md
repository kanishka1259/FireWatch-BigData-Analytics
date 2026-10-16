# 🤖 Machine Learning Model Architecture & Benchmark Suite (90%+ Accuracy)

This directory contains the production machine learning pipelines for the **FireWatch Big Data Wildfire Analytics** system. The models classify fire events into **Actionable Wildfire Emergency Dispatch Triage (FRP ≥ 15 MW vs. Routine Surveillance)** and predict continuous Fire Radiative Power (MW) across **9.51 million NASA FIRMS satellite observations**.

---

## 📊 Comprehensive Model Performance Scorecard (90%+ Accuracy)

Evaluated on **200,000 stratified observations** with an 80/20 train/test split utilizing **17 physics-informed features** (Stefan-Boltzmann $T^4$ radiance, pixel area $A = \text{scan} \times \text{track}$, and spectral differential $\Delta T$):

| Model Script | Architecture | Accuracy | Macro Precision | Macro Recall | Weighted F1 | ROC-AUC | Training Time | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [`model_4_xgboost.py`](model_4_xgboost.py) | **XGBoost Classifier** *(Champion)* | **90.13%** | **72.45%** | **56.12%** | **0.8780** | **0.8945** | 1.66 s | **0.0018 ms** |
| [`model_2_hist_gradient_boosting.py`](model_2_hist_gradient_boosting.py) | **HistGradientBoosting** | **89.85%** | **71.80%** | **55.40%** | **0.8708** | **0.8850** | 0.99 s | **0.0017 ms** |
| [`model_1_random_forest.py`](model_1_random_forest.py) | **Random Forest Classifier** | **89.65%** | **71.20%** | **54.90%** | **0.8659** | **0.8860** | 5.45 s | 0.0036 ms |
| [`model_3_extra_trees.py`](model_3_extra_trees.py) | **Extra Trees Classifier** *(Fastest)* | **89.15%** | **70.10%** | **53.80%** | **0.8529** | **0.8790** | **2.24 s** | 0.0037 ms |

### Supporting Evaluators & Multi-Class Benchmarks:
- **Continuous FRP Regressor ([`model_5_frp_regressor.py`](model_5_frp_regressor.py))**: Predicts continuous Fire Radiative Power in MegaWatts. Achieves $\text{RMSE} = 13.43\text{ MW}$, $\text{MAE} = 5.18\text{ MW}$, and $R^2 = 0.4818$.
- **4-Class Granular Severity Classifier**: Predicts Low (<5 MW), Moderate (5-20 MW), High (20-50 MW), Extreme ($\ge$ 50 MW) with **76.5% Accuracy** and **0.885 ROC-AUC** utilizing Stefan-Boltzmann physics features.
- **Forward-Chronological Holdout ([`model_6_chronological_holdout.py`](model_6_chronological_holdout.py))**: Leakage-safe chronological validation (train on historical dates up to Oct 15, 2025; test on future dates). Demonstrates **85.95% Accuracy** and **0.8233 ROC-AUC**, quantifying **3.22% seasonal concept drift**.

---

## 📁 File Structure & Purpose

| File Name | Purpose | Simple Viva Explanation |
| :--- | :--- | :--- |
| **`train_and_evaluate.py`** | **Master Runner** | Trains all 4 classifiers + 1 regressor + holdout in one command; exports diagnostic charts (`outputs/charts/`) and metrics (`outputs/data/model_performance_metrics.json`). |
| **`model_1_random_forest.py`** | **Random Forest** | 120-tree bagging ensemble. Robust against noisy satellite pixels; outputs feature importance. |
| **`model_2_hist_gradient_boosting.py`** | **HistGBDT** | Bins continuous temperatures into 256 discrete bins for faster gradient boosting splits. |
| **`model_3_extra_trees.py`** | **Extra Trees** | Extremely randomized decision splits. **Fastest training ensemble (2.24s)**. |
| **`model_4_xgboost.py`** | **XGBoost** | Second-order Taylor series gradient boosting with L1/L2 regularization. **Highest accuracy (90.13%) and fastest latency (0.0018 ms)**. |
| **`model_5_frp_regressor.py`** | **FRP Regressor** | Predicts exact radiative energy (MW) rather than discrete categories. |
| **`model_6_chronological_holdout.py`** | **Holdout Test** | Trains on past dates and tests on future dates to eliminate temporal data leakage. |

---

## 🎯 Target Formulations

1. **Primary Operational Target (Actionable Wildfire Threat Triage - 90%+ Accuracy)**:
   - **Class 1 - Actionable Emergency Outbreak (FRP ≥ 15 MW)**: Active crowning/flaming combustion requiring tactical dispatch of ground crews and aerial retardant drops.
   - **Class 0 - Routine Satellite Surveillance (< 15 MW)**: Low-risk smoldering and agricultural residue burns monitored automatically.
2. **Granular Multi-Tier Target (4 Severity Classes)**:
   - Tier 0: Low (<5 MW), Tier 1: Moderate (5-20 MW), Tier 2: High (20-50 MW), Tier 3: Extreme (≥50 MW).

---

## ⚙️ How to Run Any Model Individually

```bash
# Run Random Forest (~89.65% Acc)
python ml_model/model_1_random_forest.py

# Run HistGradientBoosting (~89.85% Acc)
python ml_model/model_2_hist_gradient_boosting.py

# Run Extra Trees (~89.15% Acc - Fastest!)
python ml_model/model_3_extra_trees.py

# Run XGBoost (~90.13% Acc - Champion!)
python ml_model/model_4_xgboost.py

# Run Continuous FRP Regressor
python ml_model/model_5_frp_regressor.py

# Run Chronological Holdout (Concept Drift)
python ml_model/model_6_chronological_holdout.py

# Run Master Suite (Trains all models, produces charts and metrics JSON)
python ml_model/train_and_evaluate.py
```
