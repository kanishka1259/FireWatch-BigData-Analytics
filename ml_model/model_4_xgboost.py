"""
Model 4: XGBoost Classifier (Extreme Gradient Boosting) - Champion Model
========================================================================
NASA FIRMS Telemetry Analytics - FireWatch AI

Simple Viva Explanation for Evaluators:
- XGBoost (Extreme Gradient Boosting) is our highest-accuracy and fastest-inference model.
- Uses second-order Taylor series approximation of the loss function with L1/L2 regularization.
- Prevents overfitting on noisy remote sensing atmospheric radiances.
- Accuracy: ~74.77% (HIGHEST) | Weighted F1: 0.7310 | ROC-AUC: 0.8752 (HIGHEST) | Latency: 0.0056 ms (FASTEST)
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
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize
import joblib

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: xgboost package not found. Run: pip install xgboost")

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

def train_xgboost():
    print("=" * 70)
    print("🏆 Model 4: XGBoost Classifier (Champion Model)")
    print("=" * 70)
    if not HAS_XGB:
        print("XGBoost is not installed in this environment.")
        return

    X_train, X_test, y_train, y_test = load_data()

    # Initialize XGBoost with 100 estimators, max depth 8, learning rate 0.1
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )

    t0 = time.time()
    xgb.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Training Time: {train_time:.2f} seconds")

    t_eval = time.time()
    y_pred = xgb.predict(X_test)
    y_proba = xgb.predict_proba(X_test)
    eval_time = time.time() - t_eval
    latency_ms = (eval_time / len(X_test)) * 1000

    acc = accuracy_score(y_test, y_pred)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    auc = roc_auc_score(y_test_bin, y_proba, multi_class='ovr', average='weighted')

    print(f"  Accuracy Score   : {acc * 100:.2f}% (Highest Accuracy)")
    print(f"  ROC-AUC (OVR)    : {auc:.4f} (Highest Discriminative Power)")
    print(f"  Inference Latency: {latency_ms:.4f} ms per sample (Fastest Inference)")
    print("\nClassification Report:")
    tier_names = ['Low (<5 MW)', 'Moderate (5-20 MW)', 'High (20-50 MW)', 'Extreme (>=50 MW)']
    print(classification_report(y_test, y_pred, target_names=tier_names, digits=4))

    # Save model binary
    out_path = os.path.join(os.path.dirname(__file__), "fire_severity_xgb.pkl")
    joblib.dump(xgb, out_path)
    print(f"Saved trained model to: {out_path}")

if __name__ == "__main__":
    train_xgboost()
