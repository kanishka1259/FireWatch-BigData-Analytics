"""
Model 2: Histogram Gradient Boosted Trees (HistGBDT) for Severity Triage
========================================================================
NASA FIRMS Telemetry Analytics - FireWatch AI

Simple Viva Explanation for Evaluators:
- Inspired by LightGBM. Bins continuous floating-point features into 256 integer bins.
- Makes evaluating split points lightning fast compared to standard Gradient Boosting.
- Builds successive decision trees that correct residual errors of earlier trees.
- Accuracy: ~74.72% | Weighted F1: 0.7297 | ROC-AUC: 0.8748 | Latency: 0.0194 ms
"""

import os
import sys
import time

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize
import joblib

DATA_FILE = "fire_archive_SV-C2_817450.csv"
for candidate in [
    DATA_FILE,
    os.path.join("..", DATA_FILE),
    os.path.join(os.path.dirname(__file__), "..", DATA_FILE),
    os.path.join(os.path.dirname(__file__), "..", "..", DATA_FILE)
]:
    if os.path.exists(candidate):
        DATA_FILE = os.path.abspath(candidate)
        break

def load_data(sample_size=100000):
    """Loads and preprocesses NASA satellite telemetry data."""
    print(f"Loading {sample_size:,} samples from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE, nrows=sample_size)
    df = df.dropna(subset=['latitude', 'longitude', 'brightness', 'bright_t31', 'frp']).copy()

    # Feature Engineering
    df['brightness_diff'] = df['brightness'] - df['bright_t31']
    df['hour'] = (df['acq_time'] // 100).astype(int)
    df['month'] = pd.to_datetime(df['acq_date']).dt.month
    df['is_day'] = (df['daynight'] == 'D').astype(int)
    conf_map = {'l': 0, 'n': 1, 'h': 2}
    df['confidence_num'] = df['confidence'].map(lambda x: conf_map.get(str(x).lower(), 1))

    # 4 Severity Tiers Target
    conditions = [
        (df['frp'] < 5.0),
        (df['frp'] >= 5.0) & (df['frp'] < 20.0),
        (df['frp'] >= 20.0) & (df['frp'] < 50.0),
        (df['frp'] >= 50.0)
    ]
    df['severity_tier'] = np.select(conditions, [0, 1, 2, 3], default=0)

    feature_cols = [
        'latitude', 'longitude', 'brightness', 'bright_t31',
        'brightness_diff', 'scan', 'track', 'hour', 'month',
        'is_day', 'confidence_num'
    ]
    X = df[feature_cols]
    y = df['severity_tier']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_hist_gradient_boosting():
    print("=" * 70)
    print("⚡ Model 2: Histogram Gradient Boosting Classifier (100 Iterations)")
    print("=" * 70)
    X_train, X_test, y_train, y_test = load_data()

    # Initialize HistGradientBoosting
    hgb = HistGradientBoostingClassifier(max_iter=100, max_depth=10, random_state=42)

    t0 = time.time()
    hgb.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Training Time: {train_time:.2f} seconds")

    t_eval = time.time()
    y_pred = hgb.predict(X_test)
    y_proba = hgb.predict_proba(X_test)
    eval_time = time.time() - t_eval
    latency_ms = (eval_time / len(X_test)) * 1000

    acc = accuracy_score(y_test, y_pred)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    auc = roc_auc_score(y_test_bin, y_proba, multi_class='ovr', average='weighted')

    print(f"  Accuracy Score   : {acc * 100:.2f}%")
    print(f"  ROC-AUC (OVR)    : {auc:.4f}")
    print(f"  Inference Latency: {latency_ms:.4f} ms per sample")
    print("\nClassification Report:")
    tier_names = ['Low (<5 MW)', 'Moderate (5-20 MW)', 'High (20-50 MW)', 'Extreme (>=50 MW)']
    print(classification_report(y_test, y_pred, target_names=tier_names, digits=4))

if __name__ == "__main__":
    train_hist_gradient_boosting()
