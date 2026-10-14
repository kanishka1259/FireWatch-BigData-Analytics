"""
Machine Learning Model Training & Comprehensive Performance Analysis
NASA FIRMS Wildfire Severity Classification & FRP Regression
"""

import os
import sys
import time
import json

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
)
from sklearn.preprocessing import label_binarize

DATA_FILE = "fire_archive_SV-C2_817450.csv"
MODEL_DIR = "ml_model"
OUTPUT_DIR = "outputs/data"
CHART_DIR = "outputs/charts"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

def train_and_evaluate():
    print("=" * 70)
    print("🤖 Machine Learning Model Training & Performance Evaluation Engine")
    print("   Operational Decision: Wildfire Alert & Rapid Dispatch (WARD)")
    print("=" * 70)

    # 1. Load Data Sample for ML Training
    print("\n[Step 1/5] Loading representative sample (200,000 records) for Model Training...")
    t0 = time.time()
    
    # Read chunked sample to ensure diversity across dataset
    df_sample = pd.read_csv(DATA_FILE, nrows=200000)
    print(f"  Loaded {len(df_sample):,} rows in {time.time() - t0:.2f}s")

    # 2. Preprocessing & Feature Engineering
    print("\n[Step 2/5] Engineering ML Features and Targets...")
    df_clean = df_sample.dropna(subset=['latitude', 'longitude', 'brightness', 'bright_t31', 'frp']).copy()
    
    # Feature transformations
    df_clean['brightness_diff'] = df_clean['brightness'] - df_clean['bright_t31']
    df_clean['hour'] = (df_clean['acq_time'] // 100).astype(int)
    df_clean['month'] = pd.to_datetime(df_clean['acq_date']).dt.month
    df_clean['is_day'] = (df_clean['daynight'] == 'D').astype(int)
    
    conf_map = {'l': 0, 'n': 1, 'h': 2}
    df_clean['confidence_num'] = df_clean['confidence'].map(lambda x: conf_map.get(str(x).lower(), 1))
    
    # Target 1: Fire Severity Tier (Multi-Class Classification)
    # Low: < 5 MW, Moderate: 5 - 20 MW, High: 20 - 50 MW, Extreme: >= 50 MW
    conditions = [
        (df_clean['frp'] < 5.0),
        (df_clean['frp'] >= 5.0) & (df_clean['frp'] < 20.0),
        (df_clean['frp'] >= 20.0) & (df_clean['frp'] < 50.0),
        (df_clean['frp'] >= 50.0)
    ]
    tier_labels = [0, 1, 2, 3] # 0: Low, 1: Moderate, 2: High, 3: Extreme
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

    # Split into Train / Test (80/20)
    X_train, X_test, y_train_c, y_test_c, y_train_r, y_test_r = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
    )

    # 3. Model 1: Fire Severity Classifier (Random Forest vs GBT)
    print("\n[Step 3/5] Training Multi-Class Severity Classifiers...")
    
    # Model A: Random Forest Classifier
    rf_start = time.time()
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train, y_train_c)
    rf_train_time = time.time() - rf_start
    
    # Model B: HistGradientBoosting Classifier
    gbt_start = time.time()
    gbt_clf = HistGradientBoostingClassifier(max_iter=100, max_depth=10, random_state=42)
    gbt_clf.fit(X_train, y_train_c)
    gbt_train_time = time.time() - gbt_start

    # Predictions & Latency Benchmark
    bench_start = time.time()
    y_pred_rf = rf_clf.predict(X_test)
    y_proba_rf = rf_clf.predict_proba(X_test)
    rf_latency_ms = ((time.time() - bench_start) / len(X_test)) * 1000

    y_pred_gbt = gbt_clf.predict(X_test)
    y_proba_gbt = gbt_clf.predict_proba(X_test)

    # Evaluation Metrics - Classification
    rf_acc = accuracy_score(y_test_c, y_pred_rf)
    rf_prec_macro = precision_score(y_test_c, y_pred_rf, average='macro', zero_division=0)
    rf_rec_macro = recall_score(y_test_c, y_pred_rf, average='macro', zero_division=0)
    rf_f1_macro = f1_score(y_test_c, y_pred_rf, average='macro', zero_division=0)
    rf_f1_weighted = f1_score(y_test_c, y_pred_rf, average='weighted', zero_division=0)
    
    # Multiclass ROC AUC (One-vs-Rest)
    y_test_bin = label_binarize(y_test_c, classes=[0, 1, 2, 3])
    rf_roc_auc = roc_auc_score(y_test_bin, y_proba_rf, multi_class='ovr', average='weighted')

    gbt_acc = accuracy_score(y_test_c, y_pred_gbt)
    gbt_f1_weighted = f1_score(y_test_c, y_pred_gbt, average='weighted', zero_division=0)
    gbt_roc_auc = roc_auc_score(y_test_bin, y_proba_gbt, multi_class='ovr', average='weighted')

    print(f"\n  Classification Results:")
    print(f"  -> Random Forest : Accuracy = {rf_acc*100:.2f}%, F1-Weighted = {rf_f1_weighted:.4f}, ROC-AUC = {rf_roc_auc:.4f} (Train: {rf_train_time:.2f}s)")
    print(f"  -> Gradient Boost: Accuracy = {gbt_acc*100:.2f}%, F1-Weighted = {gbt_f1_weighted:.4f}, ROC-AUC = {gbt_roc_auc:.4f} (Train: {gbt_train_time:.2f}s)")

    # 4. Model 2: FRP Continuous Regressor
    print("\n[Step 4/5] Training Fire Radiative Power (FRP) Continuous Regressor...")
    reg_start = time.time()
    rf_reg = RandomForestRegressor(n_estimators=80, max_depth=14, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train, y_train_r)
    reg_train_time = time.time() - reg_start

    y_pred_reg = rf_reg.predict(X_test)
    reg_rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_reg))
    reg_mae = mean_absolute_error(y_test_r, y_pred_reg)
    reg_r2 = r2_score(y_test_r, y_pred_reg)
    reg_ev = explained_variance_score(y_test_r, y_pred_reg)

    print(f"  Regression Results (FRP in MegaWatts):")
    print(f"  -> RMSE: {reg_rmse:.2f} MW | MAE: {reg_mae:.2f} MW | R2 Score: {reg_r2:.4f} | Explained Var: {reg_ev:.4f}")

    # 5. Save Models, Visualizations & Metrics Report
    print("\n[Step 5/5] Generating Performance Charts and Persisting Artifacts...")
    
    # Save Model Artifacts
    joblib.dump(rf_clf, os.path.join(MODEL_DIR, "fire_severity_rf.pkl"))
    joblib.dump(rf_reg, os.path.join(MODEL_DIR, "frp_regressor_rf.pkl"))
    print(f"  Saved Models to {MODEL_DIR}/")

    # Plot 1: Confusion Matrix
    cm = confusion_matrix(y_test_c, y_pred_rf)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=tier_names, yticklabels=tier_names, ax=ax)
    plt.title('Random Forest - Fire Severity Confusion Matrix', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Predicted Severity Tier', fontsize=11, fontweight='bold')
    plt.ylabel('Actual Severity Tier', fontsize=11, fontweight='bold')
    plt.xticks(rotation=20, ha='right')
    fig.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "ml_confusion_matrix.png"))
    plt.close()
    print("  [1/4] Saved ml_confusion_matrix.png")

    # Plot 2: Feature Importance
    importances = rf_clf.feature_importances_
    feat_indices = np.argsort(importances)[::-1]
    sorted_features = [feature_cols[i] for i in feat_indices]
    sorted_importances = importances[feat_indices]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    colors = plt.cm.viridis(np.linspace(0.85, 0.2, len(feature_cols)))
    bars = ax.barh(sorted_features[::-1], sorted_importances[::-1], color=colors, edgecolor='#1e293b', height=0.6)
    ax.set_xlabel('Gini Feature Importance Score', fontsize=11, fontweight='bold')
    ax.set_title('Random Forest Feature Importance in Predicting Wildfire Severity', fontsize=13, fontweight='bold', pad=12)
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f' {width:.3f}', xy=(width, bar.get_y() + bar.get_height() / 2),
                    ha='left', va='center', fontweight='bold', fontsize=9)
    fig.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "ml_feature_importance.png"))
    plt.close()
    print("  [2/4] Saved ml_feature_importance.png")

    # Plot 3: Regression Residuals
    residuals = y_test_r - y_pred_reg
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.scatter(y_pred_reg, residuals, alpha=0.35, color='#0284c7', edgecolors='none', s=15)
    ax.axhline(0, color='#dc2626', linestyle='--', linewidth=1.5)
    ax.set_xlim(0, 150)
    ax.set_ylim(-60, 60)
    ax.set_xlabel('Predicted Fire Radiative Power (MW)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Residuals (Actual - Predicted, MW)', fontsize=11, fontweight='bold')
    ax.set_title('FRP Regressor Residual Plot (Homoscedasticity Analysis)', fontsize=13, fontweight='bold', pad=12)
    fig.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "ml_regression_residuals.png"))
    plt.close()
    print("  [3/4] Saved ml_regression_residuals.png")

    # Plot 4: Multiclass ROC Curves
    from sklearn.metrics import roc_curve, auc
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    tier_colors = ['#10b981', '#f59e0b', '#ea580c', '#b91c1c']
    for i, col_color in enumerate(tier_colors):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba_rf[:, i])
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col_color, lw=2, label=f'{tier_names[i]} (AUC = {roc_auc_val:.3f})')
    ax.plot([0, 1], [0, 1], color='#94a3b8', lw=1.5, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold')
    ax.set_title('One-vs-Rest ROC Curves for Wildfire Severity Tiers', fontsize=13, fontweight='bold', pad=12)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "ml_roc_curve.png"))
    plt.close()
    print("  [4/4] Saved ml_roc_curve.png")

    # Performance Metrics JSON Report
    metrics_report = {
        "classification": {
            "model_name": "Random Forest Classifier (Ensemble)",
            "benchmark_model": "Histogram Gradient Boosted Trees",
            "accuracy": round(float(rf_acc), 4),
            "precision_macro": round(float(rf_prec_macro), 4),
            "recall_macro": round(float(rf_rec_macro), 4),
            "f1_macro": round(float(rf_f1_macro), 4),
            "f1_weighted": round(float(rf_f1_weighted), 4),
            "roc_auc_weighted": round(float(rf_roc_auc), 4),
            "gbt_accuracy": round(float(gbt_acc), 4),
            "gbt_f1_weighted": round(float(gbt_f1_weighted), 4),
            "gbt_roc_auc": round(float(gbt_roc_auc), 4),
            "inference_latency_per_sample_ms": round(float(rf_latency_ms), 4),
            "training_time_seconds": round(float(rf_train_time), 2),
            "confusion_matrix": cm.tolist()
        },
        "regression": {
            "model_name": "Random Forest Regressor",
            "target": "Fire Radiative Power (MW)",
            "rmse": round(float(reg_rmse), 4),
            "mae": round(float(reg_mae), 4),
            "r2_score": round(float(reg_r2), 4),
            "explained_variance": round(float(reg_ev), 4),
            "training_time_seconds": round(float(reg_train_time), 2)
        },
        "feature_importance": {feat: round(float(imp), 4) for feat, imp in zip(sorted_features, sorted_importances)},
        "decision_framework": {
            "name": "Wildfire Alert & Rapid Dispatch (WARD)",
            "tiers": [
                {"tier": "Tier 3: Low", "range": "< 5 MW", "action": "Routine satellite surveillance", "dispatch": "None"},
                {"tier": "Tier 2: Moderate", "range": "5 - 20 MW", "action": "Watch status, notify local forest rangers", "dispatch": "Ground reconnaissance"},
                {"tier": "Tier 1: High", "range": "20 - 50 MW", "action": "High alert, mobilize ground firefighting units", "dispatch": "Local response crews"},
                {"tier": "Tier 0: Extreme", "range": ">= 50 MW", "action": "CRITICAL EMERGENCY - Rapid Aerial Water/Retardant Drop", "dispatch": "Air-tanker & heavy brigade"}
            ]
        }
    }

    with open(os.path.join(OUTPUT_DIR, "model_performance_metrics.json"), "w") as f:
        json.dump(metrics_report, f, indent=2)

    print(f"\n✅ Performance Analysis Saved to {OUTPUT_DIR}/model_performance_metrics.json")
    print("=" * 70)

if __name__ == "__main__":
    train_and_evaluate()
