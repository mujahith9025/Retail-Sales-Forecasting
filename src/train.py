"""
Phase 5: Multi-Model Training, Evaluation & Benchmarking.
Trains Baseline, Linear Regression, Random Forest, and XGBoost models.
Evaluates MAE, RMSE, MAPE, WMAPE, and R2 on unseen test data.
Persists the champion model and evaluation metrics.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

try:
    from src.config import (
        TRAIN_FEATURES_FILE,
        TEST_FEATURES_FILE,
        BEST_MODEL_FILE,
        MODEL_METRICS_FILE,
        MODELS_DIR,
        BASE_DIR
    )
except (ImportError, ModuleNotFoundError):
    from config import (
        TRAIN_FEATURES_FILE,
        TEST_FEATURES_FILE,
        BEST_MODEL_FILE,
        MODEL_METRICS_FILE,
        MODELS_DIR,
        BASE_DIR
    )

REPORTS_DIR = BASE_DIR / "reports" / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def calculate_metrics(y_true, y_pred, model_name: str) -> dict:
    """Calculates comprehensive forecasting error metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100
    wmape = (np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    return {
        "Model": model_name,
        "MAE ($)": round(float(mae), 2),
        "RMSE ($)": round(float(rmse), 2),
        "MAPE (%)": round(float(mape), 2),
        "WMAPE (%)": round(float(wmape), 2),
        "R2 Score": round(float(r2), 4)
    }

def train_and_evaluate_models():
    """Trains all models, evaluates on test set, and saves champion model."""
    print("=" * 75)
    print(" PHASE 5: MULTI-MODEL TRAINING & EVALUATION")
    print("=" * 75)
    
    if not TRAIN_FEATURES_FILE.exists() or not TEST_FEATURES_FILE.exists():
        raise FileNotFoundError("Train or test features not found! Run Phase 4 first.")
        
    train_df = pd.read_csv(TRAIN_FEATURES_FILE)
    test_df = pd.read_csv(TEST_FEATURES_FILE)
    
    # Exclude metadata and target columns from predictor feature matrix
    ignore_cols = ["Date", "Store_ID", "Department", "Holiday_Name", "Weekly_Sales"]
    feature_cols = [c for c in train_df.columns if c not in ignore_cols]
    
    X_train = train_df[feature_cols]
    y_train = train_df["Weekly_Sales"]
    
    X_test = test_df[feature_cols]
    y_test = test_df["Weekly_Sales"]
    
    print(f"[INFO] Total Training Samples: {len(X_train):,}")
    print(f"[INFO] Total Testing Samples:  {len(X_test):,}")
    print(f"[INFO] Number of Predictor Features: {len(feature_cols)}")
    
    models = {}
    predictions = {}
    metrics_list = []
    
    # -------------------------------------------------------------
    # 1. Baseline Model (Moving Average / Lag 1)
    # -------------------------------------------------------------
    print("\n[1/4] Evaluating Naive Baseline (Lag-1 Benchmark)...")
    baseline_pred = test_df["Sales_Lag_1"].values
    predictions["Baseline (Lag-1)"] = baseline_pred
    metrics_list.append(calculate_metrics(y_test, baseline_pred, "Baseline (Lag-1)"))
    
    # -------------------------------------------------------------
    # 2. Linear Model (Ridge Regression with Standard Scaling)
    # -------------------------------------------------------------
    print("[2/4] Training Ridge Linear Regression...")
    ridge_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=1.0))
    ])
    ridge_pipeline.fit(X_train, y_train)
    ridge_pred = ridge_pipeline.predict(X_test)
    models["Ridge Regression"] = ridge_pipeline
    predictions["Ridge Regression"] = ridge_pred
    metrics_list.append(calculate_metrics(y_test, ridge_pred, "Ridge Regression"))
    
    # -------------------------------------------------------------
    # 3. Random Forest Regressor
    # -------------------------------------------------------------
    print("[3/4] Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=14,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    models["Random Forest"] = rf_model
    predictions["Random Forest"] = rf_pred
    metrics_list.append(calculate_metrics(y_test, rf_pred, "Random Forest"))
    
    # -------------------------------------------------------------
    # 4. XGBoost Regressor
    # -------------------------------------------------------------
    print("[4/4] Training XGBoost Regressor...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=160,
        learning_rate=0.07,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    models["XGBoost"] = xgb_model
    predictions["XGBoost"] = xgb_pred
    metrics_list.append(calculate_metrics(y_test, xgb_pred, "XGBoost"))
    
    # -------------------------------------------------------------
    # Model Comparison Leaderboard
    # -------------------------------------------------------------
    metrics_df = pd.DataFrame(metrics_list).sort_values(by="RMSE ($)", ascending=True).reset_index(drop=True)
    print("\n" + "=" * 75)
    print(" MODEL PERFORMANCE LEADERBOARD (EVALUATED ON TEST SET)")
    print("=" * 75)
    print(metrics_df.to_string(index=False))
    print("=" * 75)
    
    # Determine best model (lowest RMSE)
    best_model_name = metrics_df.iloc[0]["Model"]
    if best_model_name == "Baseline (Lag-1)":
        best_model_name = metrics_df.iloc[1]["Model"]
        
    best_model_obj = models[best_model_name]
    print(f"\n[WINNER] Selected Champion Model: {best_model_name}")
    
    # Save Model Artifact
    model_artifact = {
        "model": best_model_obj,
        "model_name": best_model_name,
        "feature_names": feature_cols,
        "metrics": metrics_df.to_dict(orient="records"),
        "trained_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    joblib.dump(model_artifact, BEST_MODEL_FILE)
    print(f"[OK] Saved champion model artifact to: {BEST_MODEL_FILE}")
    
    # Save metrics JSON
    with open(MODEL_METRICS_FILE, "w") as f:
        json.dump(metrics_df.to_dict(orient="records"), f, indent=4)
    print(f"[OK] Saved metrics report to: {MODEL_METRICS_FILE}")
    
    # -------------------------------------------------------------
    # Feature Importance Analysis (from Tree Model)
    # -------------------------------------------------------------
    tree_model = models.get("XGBoost") or models.get("Random Forest")
    importances = tree_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    
    feat_imp_file = MODELS_DIR / "feature_importance.csv"
    feat_imp_df.to_csv(feat_imp_file, index=False)
    print(f"[OK] Feature importance table saved to: {feat_imp_file}")
    
    print("\n--- Top 10 Most Important Predictive Features ---")
    for i, row in feat_imp_df.head(10).iterrows():
        print(f"  {i+1:02d}. {row['Feature']:<26}: {row['Importance']*100:.2f}%")
        
    # -------------------------------------------------------------
    # Visualization: Actual vs Predicted & Metrics Comparison
    # -------------------------------------------------------------
    generate_model_evaluation_charts(test_df, predictions, metrics_df, feat_imp_df)
    
    print("\n[SUCCESS] Phase 5 Execution Complete: Models trained and evaluated!")

def generate_model_evaluation_charts(test_df, predictions, metrics_df, feat_imp_df):
    """Generates visual charts for model comparison and forecast trajectories."""
    print("\n[INFO] Generating model evaluation plots...")
    
    # --- Figure 5: Actual vs Predicted Trajectory (Aggregated Weekly Total) ---
    plt.figure(figsize=(14, 6))
    test_eval_df = test_df[["Date", "Weekly_Sales"]].copy()
    test_eval_df["Ridge"] = predictions["Ridge Regression"]
    test_eval_df["Random Forest"] = predictions["Random Forest"]
    test_eval_df["XGBoost"] = predictions["XGBoost"]
    
    agg_test = test_eval_df.groupby("Date").sum().reset_index()
    agg_test["Date"] = pd.to_datetime(agg_test["Date"])
    
    plt.plot(agg_test["Date"], agg_test["Weekly_Sales"] / 1000, label="Actual Total Sales", color="black", linewidth=2.5, marker="o", markersize=4)
    plt.plot(agg_test["Date"], agg_test["XGBoost"] / 1000, label="XGBoost Forecast", color="#2ca02c", linestyle="--", linewidth=2.0)
    plt.plot(agg_test["Date"], agg_test["Random Forest"] / 1000, label="Random Forest Forecast", color="#ff7f0e", linestyle=":", linewidth=2.0)
    plt.plot(agg_test["Date"], agg_test["Ridge"] / 1000, label="Ridge Linear Forecast", color="#1f77b4", linestyle="-.", linewidth=1.8)
    
    plt.title("Actual vs Predicted Weekly Sales (Test Set: July - Dec 2023)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Total Network Revenue ($ in Thousands)", fontsize=11)
    plt.legend(frameon=True, facecolor="white", loc="upper left")
    plt.tight_layout()
    fig5_path = REPORTS_DIR / "05_actual_vs_predicted.png"
    plt.savefig(fig5_path, dpi=200)
    plt.close()
    print(f"  [OK] Saved: {fig5_path}")
    
    # --- Figure 6: Top 12 Feature Importances ---
    plt.figure(figsize=(10, 6))
    top_feats = feat_imp_df.head(12).sort_values(by="Importance", ascending=True)
    plt.barh(top_feats["Feature"], top_feats["Importance"] * 100, color="#2b5c8f", height=0.6)
    plt.title("Top 12 Predictive Features (XGBoost Importance)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Relative Feature Importance (%)", fontsize=11)
    plt.tight_layout()
    fig6_path = REPORTS_DIR / "06_feature_importance.png"
    plt.savefig(fig6_path, dpi=200)
    plt.close()
    print(f"  [OK] Saved: {fig6_path}")

if __name__ == "__main__":
    train_and_evaluate_models()
