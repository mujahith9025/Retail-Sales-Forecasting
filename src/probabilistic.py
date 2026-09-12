"""
Probabilistic & Quantile Forecasting Module.
Trains Gradient Boosting Quantile Regressors at alpha=0.10 (P10), alpha=0.50 (P50), and alpha=0.90 (P90)
to provide calibrated uncertainty intervals and risk-aware forecasts.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss

from src.config import (
    TRAIN_FEATURES_FILE,
    TEST_FEATURES_FILE,
    MODELS_DIR
)

QUANTILE_MODELS_FILE = MODELS_DIR / "quantile_models.pkl"

def train_quantile_forecasters():
    """Trains P10, P50, and P90 quantile models and calculates coverage metrics."""
    print("=" * 75)
    print(" ADVANCED PROBABILISTIC QUANTILE FORECASTING (P10, P50, P90)")
    print("=" * 75)
    
    train_df = pd.read_csv(TRAIN_FEATURES_FILE)
    test_df = pd.read_csv(TEST_FEATURES_FILE)
    
    ignore_cols = ["Date", "Store_ID", "Department", "Holiday_Name", "Weekly_Sales"]
    feature_cols = [c for c in train_df.columns if c not in ignore_cols]
    
    X_train = train_df[feature_cols]
    y_train = train_df["Weekly_Sales"]
    
    X_test = test_df[feature_cols]
    y_test = test_df["Weekly_Sales"]
    
    quantiles = [0.10, 0.50, 0.90]
    quantile_models = {}
    test_predictions = {}
    pinball_losses = {}
    
    for q in quantiles:
        q_label = f"P{int(q * 100)}"
        print(f"[INFO] Training Quantile Regressor: {q_label} (alpha={q})...")
        
        gbr = GradientBoostingRegressor(
            loss="quantile",
            alpha=q,
            n_estimators=140,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.85,
            random_state=42
        )
        gbr.fit(X_train, y_train)
        
        preds = gbr.predict(X_test)
        quantile_models[q_label] = gbr
        test_predictions[q_label] = preds
        
        # Calculate Pinball loss
        p_loss = mean_pinball_loss(y_test, preds, alpha=q)
        pinball_losses[q_label] = round(float(p_loss), 2)
        print(f"  [OK] {q_label} Model Fitted | Pinball Loss: ${p_loss:,.2f}")
        
    # Calculate 80% Prediction Interval Coverage Probability (PICP)
    p10_preds = test_predictions["P10"]
    p90_preds = test_predictions["P90"]
    p50_preds = test_predictions["P50"]
    
    # Monotonicity adjustment (ensure P10 <= P50 <= P90)
    p50_preds = np.maximum(p10_preds, p50_preds)
    p90_preds = np.maximum(p50_preds, p90_preds)
    
    is_covered = (y_test >= p10_preds) & (y_test <= p90_preds)
    coverage_pct = (np.sum(is_covered) / len(y_test)) * 100
    mean_interval_width = np.mean(p90_preds - p10_preds)
    
    print("\n" + "=" * 75)
    print(" PROBABILISTIC UNCERTAINTY METRICS (EVALUATED ON TEST SET)")
    print("=" * 75)
    print(f"  - Target Coverage Interval       : 80.0% (P10 to P90)")
    print(f"  - Empirical Coverage (PICP)      : {coverage_pct:.2f}%")
    print(f"  - Mean Uncertainty Band Width ($): ${mean_interval_width:,.2f}")
    print(f"  - P10 Pinball Loss               : ${pinball_losses['P10']:,.2f}")
    print(f"  - P50 Pinball Loss               : ${pinball_losses['P50']:,.2f}")
    print(f"  - P90 Pinball Loss               : ${pinball_losses['P90']:,.2f}")
    print("=" * 75)
    
    artifact = {
        "models": quantile_models,
        "feature_names": feature_cols,
        "quantiles": quantiles,
        "metrics": {
            "empirical_coverage_pct": round(float(coverage_pct), 2),
            "mean_interval_width": round(float(mean_interval_width), 2),
            "pinball_losses": pinball_losses
        }
    }
    
    joblib.dump(artifact, QUANTILE_MODELS_FILE)
    print(f"[OK] Quantile forecasting ensemble saved to: {QUANTILE_MODELS_FILE}")
    print("[SUCCESS] Probabilistic module training complete!")

if __name__ == "__main__":
    train_quantile_forecasters()
