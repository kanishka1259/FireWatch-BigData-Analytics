"""
Machine Learning Model Training & Comprehensive Performance Analysis
NASA FIRMS Wildfire Severity Classification & FRP Regression
Includes 4 Advanced Models:
1. Random Forest Classifier
2. Histogram Gradient Boosted Trees (HistGBDT)
3. Extra Trees Classifier (Extremely Randomized Trees)
4. XGBoost Classifier (Extreme Gradient Boosting)
Plus Continuous FRP Regressor and Forward-Chronological Holdout Validation.
"""

import os
import sys
import time
import json

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# Check for XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

DATA_FILE = "fire_archive_SV-C2_817450.csv"
if not os.path.exists(DATA_FILE):
    for alt in [
        "../fire_archive_SV-C2_817450.csv",
        os.path.join(os.path.dirname(__file__), "..", "fire_archive_SV-C2_817450.csv"),
        os.path.join(os.path.dirname(__file__), "..", "..", "fire_archive_SV-C2_817450.csv")
    ]:
        if os.path.exists(alt):
            DATA_FILE = os.path.abspath(alt)
            break

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "ml_model")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "data")
CHART_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "charts")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

def train_and_evaluate_all():
    print("=" * 75)
    print("🔥 Multi-Model ML Benchmark: 4 Classifier Architectures + Regressor")
    print("   Models: Random Forest | HistGBDT | Extra Trees | XGBoost")
    print("=" * 75)

    # 1. Load Data Sample for ML Training
    print("\n[Step 1/5] Loading representative sample (200,000 records)...")
    t0 = time.time()
    df_sample = pd.read_csv(DATA_FILE, nrows=200000)
    print(f"  Loaded {len(df_sample):,} rows in {time.time() - t0:.2f}s")

    # 2. Preprocessing & Feature Engineering
    print("\n[Step 2/5] Engineering ML Features and 4-Tier Target...")
    df_clean = df_sample.dropna(subset=['latitude', 'longitude', 'brightness', 'bright_t31', 'frp']).copy()

    df_clean['brightness_diff'] = df_clean['brightness'] - df_clean['bright_t31']
    df_clean['hour'] = (df_clean['acq_time'] // 100).astype(int)
    df_clean['month'] = pd.to_datetime(df_clean['acq_date']).dt.month
    df_clean['is_day'] = (df_clean['daynight'] == 'D').astype(int)

    conf_map = {'l': 0, 'n': 1, 'h': 2}
    df_clean['confidence_num'] = df_clean['confidence'].map(lambda x: conf_map.get(str(x).lower(), 1))

    # Target: Fire Severity Tier (4-Class Classification)
    # Tier 0: Low (<5 MW), Tier 1: Moderate (5-20 MW), Tier 2: High (20-50 MW), Tier 3: Extreme (>=50 MW)
    conditions = [
        (df_clean['frp'] < 5.0),
        (df_clean['frp'] >= 5.0) & (df_clean['frp'] < 20.0),
        (df_clean['frp'] >= 20.0) & (df_clean['frp'] < 50.0),
        (df_clean['frp'] >= 50.0)
    ]
    tier_labels = [0, 1, 2, 3]
    df_clean['severity_tier'] = np.select(conditions, tier_labels, default=0)

    feature_cols = [
        'latitude', 'longitude', 'brightness', 'bright_t31',
        'brightness_diff', 'scan', 'track', 'hour', 'month',
        'is_day', 'confidence_num'
    ]

    X = df_clean[feature_cols]
    y_class = df_clean['severity_tier']
    y_reg = df_clean['frp']
    tier_names = ['Low (<5 MW)', 'Moderate (5-20 MW)', 'High (20-50 MW)', 'Extreme (>=50 MW)']

    print(f"  Feature Matrix Shape: {X.shape}")
    print("  Class Distribution:\n", pd.Series(y_class).value_counts().sort_index())

    # Split into Train / Test (80/20 Stratified Random Split)
    X_train, X_test, y_train_c, y_test_c, y_train_r, y_test_r = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
    )
    y_test_bin = label_binarize(y_test_c, classes=[0, 1, 2, 3])

    # 3. Train & Benchmark 4 Classifier Models
    print("\n[Step 3/5] Training and Benchmarking 4 Classifier Architectures...")
    models_dict = {}

    # Model 1: Random Forest Classifier
    print("  -> Training Model 1: Random Forest Classifier (100 Trees)...")
    t_start = time.time()
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train, y_train_c)
    rf_train_time = time.time() - t_start

    t_eval = time.time()
    y_pred_rf = rf_clf.predict(X_test)
    y_proba_rf = rf_clf.predict_proba(X_test)
    rf_latency_ms = ((time.time() - t_eval) / len(X_test)) * 1000

    models_dict['Random Forest'] = {
        'model': rf_clf,
        'pred': y_pred_rf,
        'proba': y_proba_rf,
        'train_time': rf_train_time,
        'latency_ms': rf_latency_ms,
        'spec': '100 Trees | Max Depth: 16 | n_jobs: -1'
    }

    # Model 2: Histogram Gradient Boosting (HistGBDT)
    print("  -> Training Model 2: Histogram Gradient Boosting (HistGBDT)...")
    t_start = time.time()
    hgb_clf = HistGradientBoostingClassifier(max_iter=100, max_depth=10, random_state=42)
    hgb_clf.fit(X_train, y_train_c)
    hgb_train_time = time.time() - t_start

    t_eval = time.time()
    y_pred_hgb = hgb_clf.predict(X_test)
    y_proba_hgb = hgb_clf.predict_proba(X_test)
    hgb_latency_ms = ((time.time() - t_eval) / len(X_test)) * 1000

    models_dict['HistGradientBoosting'] = {
        'model': hgb_clf,
        'pred': y_pred_hgb,
        'proba': y_proba_hgb,
        'train_time': hgb_train_time,
        'latency_ms': hgb_latency_ms,
        'spec': '100 Iterations | Max Depth: 10'
    }

    # Model 3: Extra Trees Classifier (Extremely Randomized Trees)
    print("  -> Training Model 3: Extra Trees Classifier (100 Trees)...")
    t_start = time.time()
    et_clf = ExtraTreesClassifier(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1)
    et_clf.fit(X_train, y_train_c)
    et_train_time = time.time() - t_start

    t_eval = time.time()
    y_pred_et = et_clf.predict(X_test)
    y_proba_et = et_clf.predict_proba(X_test)
    et_latency_ms = ((time.time() - t_eval) / len(X_test)) * 1000

    models_dict['Extra Trees'] = {
        'model': et_clf,
        'pred': y_pred_et,
        'proba': y_proba_et,
        'train_time': et_train_time,
        'latency_ms': et_latency_ms,
        'spec': '100 Trees | Max Depth: 16 | Extremely Randomized Splits'
    }

    # Model 4: XGBoost Classifier (or GradientBoosting fallback)
    if HAS_XGB:
        print("  -> Training Model 4: XGBoost Classifier (100 Estimators)...")
        t_start = time.time()
        xgb_clf = XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='mlogloss')
        xgb_clf.fit(X_train, y_train_c)
        xgb_train_time = time.time() - t_start

        t_eval = time.time()
        y_pred_xgb = xgb_clf.predict(X_test)
        y_proba_xgb = xgb_clf.predict_proba(X_test)
        xgb_latency_ms = ((time.time() - t_eval) / len(X_test)) * 1000

        models_dict['XGBoost'] = {
            'model': xgb_clf,
            'pred': y_pred_xgb,
            'proba': y_proba_xgb,
            'train_time': xgb_train_time,
            'latency_ms': xgb_latency_ms,
            'spec': '100 Trees | Depth: 8 | LR: 0.1 | Second-Order Gradient Optimization'
        }
    else:
        print("  -> Training Model 4: Gradient Boosting Classifier (Fallback)...")
        t_start = time.time()
        gb_clf = GradientBoostingClassifier(n_estimators=80, max_depth=6, random_state=42)
        gb_clf.fit(X_train, y_train_c)
        gb_train_time = time.time() - t_start

        t_eval = time.time()
        y_pred_gb = gb_clf.predict(X_test)
        y_proba_gb = gb_clf.predict_proba(X_test)
        gb_latency_ms = ((time.time() - t_eval) / len(X_test)) * 1000

        models_dict['Gradient Boosting'] = {
            'model': gb_clf,
            'pred': y_pred_gb,
            'proba': y_proba_gb,
            'train_time': gb_train_time,
            'latency_ms': gb_latency_ms,
            'spec': '80 Trees | Max Depth: 6 | Gradient Boosting'
        }

    # Calculate metrics for all 4 models
    comparison_table = []
    print("\n" + "=" * 95)
    print(f"{'Model Architecture':<24} | {'Accuracy':<9} | {'Weighted F1':<11} | {'ROC-AUC':<9} | {'Train (s)':<9} | {'Latency':<12}")
    print("=" * 95)

    for name, mdata in models_dict.items():
        y_pred = mdata['pred']
        y_proba = mdata['proba']
        acc = accuracy_score(y_test_c, y_pred)
        prec_m = precision_score(y_test_c, y_pred, average='macro', zero_division=0)
        rec_m = recall_score(y_test_c, y_pred, average='macro', zero_division=0)
        f1_w = f1_score(y_test_c, y_pred, average='weighted', zero_division=0)
        roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class='ovr', average='weighted')
        
        mdata['acc'] = acc
        mdata['prec_macro'] = prec_m
        mdata['rec_macro'] = rec_m
        mdata['f1_weighted'] = f1_w
        mdata['roc_auc'] = roc_auc

        comparison_table.append({
            "model_name": name,
            "specification": mdata['spec'],
            "accuracy": round(acc * 100, 2),
            "precision_macro": round(prec_m * 100, 2),
            "recall_macro": round(rec_m * 100, 2),
            "f1_weighted": round(f1_w, 4),
            "roc_auc": round(roc_auc, 4),
            "train_time_sec": round(mdata['train_time'], 2),
            "inference_latency_ms": round(mdata['latency_ms'], 4)
        })
        print(f"{name:<24} | {acc*100:6.2f}%   | {f1_w:9.4f}   | {roc_auc:7.4f}   | {mdata['train_time']:6.2f}s   | {mdata['latency_ms']:8.4f} ms")
    print("=" * 95)

    # 4. Continuous FRP Regressor
    print("\n[Step 4/5] Training Fire Radiative Power (FRP) Continuous Regressor...")
    t_reg = time.time()
    rf_reg = RandomForestRegressor(n_estimators=80, max_depth=14, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train, y_train_r)
    reg_train_time = time.time() - t_reg

    y_pred_reg = rf_reg.predict(X_test)
    reg_rmse = float(np.sqrt(mean_squared_error(y_test_r, y_pred_reg)))
    reg_mae = float(mean_absolute_error(y_test_r, y_pred_reg))
    reg_r2 = float(r2_score(y_test_r, y_pred_reg))
    reg_ev = float(explained_variance_score(y_test_r, y_pred_reg))

    print(f"  Regression Results: RMSE = {reg_rmse:.2f} MW | MAE = {reg_mae:.2f} MW | R2 = {reg_r2:.4f}")

    # 5. Forward-Chronological Holdout Validation (Concept Drift)
    print("\n[Temporal ML Evaluation] Benchmarking Forward-Chronological Holdout Split...")
    df_sorted = df_clean.sort_values('acq_date').reset_index(drop=True)
    split_idx = int(len(df_sorted) * 0.8)
    cutoff_date = str(df_sorted.iloc[split_idx]['acq_date'])

    X_train_time = df_sorted.iloc[:split_idx][feature_cols]
    y_train_c_time = df_sorted.iloc[:split_idx]['severity_tier']
    X_test_time = df_sorted.iloc[split_idx:][feature_cols]
    y_test_c_time = df_sorted.iloc[split_idx:]['severity_tier']

    t_temp_start = time.time()
    rf_temp = RandomForestClassifier(n_estimators=80, max_depth=14, random_state=42, n_jobs=-1)
    rf_temp.fit(X_train_time, y_train_c_time)
    temp_train_time = time.time() - t_temp_start

    t_temp_eval = time.time()
    y_pred_time = rf_temp.predict(X_test_time)
    temp_latency_ms = ((time.time() - t_temp_eval) / len(X_test_time)) * 1000

    temp_acc = accuracy_score(y_test_c_time, y_pred_time)
    temp_f1_w = f1_score(y_test_c_time, y_pred_time, average='weighted', zero_division=0)
    y_test_time_bin = label_binarize(y_test_c_time, classes=[0, 1, 2, 3])
    y_proba_time = rf_temp.predict_proba(X_test_time)
    temp_roc_auc = roc_auc_score(y_test_time_bin, y_proba_time, multi_class='ovr', average='weighted')

    print(f"  Chronological Cutoff Date: {cutoff_date}")
    print(f"  -> Temporal Train Size: {len(X_train_time):,} | Forward Test Size: {len(X_test_time):,}")
    print(f"  -> Temporal Accuracy: {temp_acc*100:.2f}%, F1-Weighted: {temp_f1_w:.4f}, ROC-AUC: {temp_roc_auc:.4f}")

    # 6. Save Models & Metrics JSON
    print("\n[Step 5/5] Exporting Models, Metrics JSON, and Publication Visualizations...")
    joblib.dump(rf_clf, os.path.join(MODEL_DIR, "fire_severity_rf.pkl"))
    joblib.dump(rf_reg, os.path.join(MODEL_DIR, "frp_regressor_rf.pkl"))
    if HAS_XGB:
        joblib.dump(models_dict['XGBoost']['model'], os.path.join(MODEL_DIR, "fire_severity_xgb.pkl"))
    joblib.dump(models_dict['Extra Trees']['model'], os.path.join(MODEL_DIR, "fire_severity_extra_trees.pkl"))

    metrics_payload = {
        "benchmark_sample_size": len(df_sample),
        "random_split_ratio": "80/20 Stratified",
        "models_comparison": comparison_table,
        "regression_metrics": {
            "model": "Random Forest Regressor",
            "rmse_mw": round(reg_rmse, 2),
            "mae_mw": round(reg_mae, 2),
            "r2_score": round(reg_r2, 4),
            "explained_variance": round(reg_ev, 4),
            "train_time_sec": round(reg_train_time, 2)
        },
        "temporal_holdout_metrics": {
            "validation_scheme": "Forward-Chronological Holdout (Leakage-Safe)",
            "cutoff_date": cutoff_date,
            "train_samples": len(X_train_time),
            "test_samples": len(X_test_time),
            "accuracy": round(temp_acc * 100, 2),
            "f1_weighted": round(temp_f1_w, 4),
            "roc_auc": round(temp_roc_auc, 4),
            "train_time_sec": round(temp_train_time, 2),
            "inference_latency_ms": round(temp_latency_ms, 4),
            "concept_drift_drop_pct": round((models_dict['Random Forest']['acc'] - temp_acc) * 100, 2)
        }
    }

    metrics_json_path = os.path.join(OUTPUT_DIR, "model_performance_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"  Saved updated metrics to {metrics_json_path}")

    # Plot 1: Multi-Model ROC Curves Comparison
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    palette = ['#1E3A8A', '#0284C7', '#10B981', '#F59E0B']
    for idx, (mname, mdata) in enumerate(models_dict.items()):
        fpr, tpr, _ = roc_curve(y_test_bin[:, 3], mdata['proba'][:, 3]) # Extreme tier ROC
        roc_score = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=palette[idx % len(palette)], lw=2, label=f"{mname} (Tier 3 AUC = {roc_score:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.6)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold')
    ax.set_title('Comparative Multi-Model ROC Curves (Extreme Wildfire Tier >= 50 MW)', fontsize=12, fontweight='bold', pad=12)
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.grid(True, linestyle=':', alpha=0.5)
    fig.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "ml_roc_curve.png"))
    plt.close()
    print("  [1/4] Saved updated ml_roc_curve.png")

    # Plot 2: Confusion Matrix (Primary Model - Random Forest / XGBoost)
    cm = confusion_matrix(y_test_c, models_dict['Random Forest']['pred'])
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=tier_names, yticklabels=tier_names, ax=ax)
    plt.title('Random Forest - Fire Severity Confusion Matrix', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Predicted Severity Tier', fontsize=11, fontweight='bold')
    plt.ylabel('Actual Severity Tier', fontsize=11, fontweight='bold')
    plt.xticks(rotation=20, ha='right')
    fig.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "ml_confusion_matrix.png"))
    plt.close()
    print("  [2/4] Saved ml_confusion_matrix.png")

    # Plot 3: Feature Importance Comparison
    importances = rf_clf.feature_importances_
    feat_indices = np.argsort(importances)[::-1]
    sorted_features = [feature_cols[i] for i in feat_indices]
    sorted_importances = importances[feat_indices]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    colors = plt.cm.viridis(np.linspace(0.85, 0.2, len(feature_cols)))
    bars = ax.barh(sorted_features[::-1], sorted_importances[::-1], color=colors, edgecolor='#1e293b', height=0.6)
    ax.set_xlabel('Gini Feature Importance Score', fontsize=11, fontweight='bold')
    ax.set_title('Feature Importance Across Thermal & Temporal Predictors', fontsize=12, fontweight='bold', pad=12)
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f' {width:.3f}', xy=(width, bar.get_y() + bar.get_height() / 2),
                    ha='left', va='center', fontweight='bold', fontsize=9)
    fig.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "ml_feature_importance.png"))
    plt.close()
    print("  [3/4] Saved ml_feature_importance.png")

    # Plot 4: Regression Residuals
    residuals = y_test_r - y_pred_reg
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.scatter(y_pred_reg, residuals, alpha=0.35, color='#0284c7', edgecolors='none', s=15)
    ax.axhline(0, color='#dc2626', linestyle='--', linewidth=1.5)
    ax.set_xlim(0, 150)
    ax.set_ylim(-60, 60)
    ax.set_xlabel('Predicted Fire Radiative Power (MW)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Residuals (Actual - Predicted, MW)', fontsize=11, fontweight='bold')
    ax.set_title('FRP Regressor Residual Plot (Error Distribution)', fontsize=12, fontweight='bold', pad=12)
    fig.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "ml_regression_residuals.png"))
    plt.close()
    print("  [4/4] Saved ml_regression_residuals.png")

    print("\n✅ Multi-Model ML Benchmark complete!")

if __name__ == "__main__":
    train_and_evaluate_all()
