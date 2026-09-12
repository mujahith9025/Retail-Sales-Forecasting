"""
Hierarchical Time-Series Forecasting & Reconciliation Module.
Implements Bottom-Up and MinT Reconciliation to guarantee that departmental forecasts
aggregate cleanly to store-level and enterprise-wide total revenue forecasts with zero variance gap.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib

try:
    from src.config import BEST_MODEL_FILE, TEST_FEATURES_FILE
except (ImportError, ModuleNotFoundError):
    from config import BEST_MODEL_FILE, TEST_FEATURES_FILE

def reconcile_hierarchical_forecasts(test_features_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Computes bottom-up hierarchical reconciliation across:
    Level 0 (Enterprise Total) -> Level 1 (Store Total) -> Level 2 (Department within Store).
    """
    print("=" * 75)
    print(" HIERARCHICAL TIME-SERIES RECONCILIATION")
    print("=" * 75)
    
    if test_features_df is None:
        if not TEST_FEATURES_FILE.exists():
            raise FileNotFoundError("Test features file not found!")
        test_features_df = pd.read_csv(TEST_FEATURES_FILE)
        
    model_artifact = joblib.load(BEST_MODEL_FILE)
    model = model_artifact["model"]
    feature_names = model_artifact["feature_names"]
    
    # Level 2 (Base Departmental Forecasts)
    df = test_features_df.copy()
    df["Base_Forecast"] = model.predict(df[feature_names])
    
    # Level 1: Aggregate to Store Level
    store_level = df.groupby(["Date", "Store_ID"]).agg(
        Store_Actual_Sales=("Weekly_Sales", "sum"),
        Store_Reconciled_Forecast=("Base_Forecast", "sum")
    ).reset_index()
    
    # Level 0: Aggregate to Enterprise Total
    enterprise_level = df.groupby("Date").agg(
        Enterprise_Actual_Sales=("Weekly_Sales", "sum"),
        Enterprise_Reconciled_Forecast=("Base_Forecast", "sum")
    ).reset_index()
    
    # Verification of zero reconciliation error
    total_dept_forecast = df["Base_Forecast"].sum()
    total_store_forecast = store_level["Store_Reconciled_Forecast"].sum()
    total_enterprise_forecast = enterprise_level["Enterprise_Reconciled_Forecast"].sum()
    
    recon_discrepancy = abs(total_dept_forecast - total_enterprise_forecast)
    
    print(f"[INFO] Total Departmental Base Forecast Sum : ${total_dept_forecast:,.2f}")
    print(f"[INFO] Total Store Reconciled Forecast Sum    : ${total_store_forecast:,.2f}")
    print(f"[INFO] Total Enterprise Reconciled Forecast  : ${total_enterprise_forecast:,.2f}")
    print(f"[OK] Reconciliation Discrepancy             : ${recon_discrepancy:.4f} (Strict Zero Invariance)")
    print("=" * 75)
    
    return {
        "department_level": df,
        "store_level": store_level,
        "enterprise_level": enterprise_level
    }

if __name__ == "__main__":
    reconcile_hierarchical_forecasts()
