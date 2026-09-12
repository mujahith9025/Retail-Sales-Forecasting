"""
Phase 4: Time-Series Feature Engineering & Preprocessing.
Extracts calendar attributes, cyclical features, lag features, and rolling statistics.
Performs strict chronological train-test splitting to prevent temporal data leakage.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

try:
    from src.config import (
        RAW_DATA_FILE,
        TRAIN_FEATURES_FILE,
        TEST_FEATURES_FILE,
        PROCESSED_DATA_DIR
    )
except (ImportError, ModuleNotFoundError):
    from config import (
        RAW_DATA_FILE,
        TRAIN_FEATURES_FILE,
        TEST_FEATURES_FILE,
        PROCESSED_DATA_DIR
    )

def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts calendar and cyclical time features from Date."""
    df = df.copy()
    dt = df["Date"].dt
    
    df["Year"] = dt.year
    df["Month"] = dt.month
    df["Week_of_Year"] = dt.isocalendar().week.astype(int)
    df["Quarter"] = dt.quarter
    df["Is_Month_End"] = dt.is_month_end.astype(int)
    
    # Cyclical encoding for week of year (captures smooth 52-week annual cycles)
    df["Week_Sin"] = np.sin(2 * np.pi * df["Week_of_Year"] / 52.0)
    df["Week_Cos"] = np.cos(2 * np.pi * df["Week_of_Year"] / 52.0)
    
    # Cyclical encoding for month
    df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12.0)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12.0)
    
    return df

def create_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates lag features and rolling window statistics per Store & Department.
    Crucial: Rolling windows use shift(1) to avoid including the current week's target!
    """
    df = df.copy()
    
    # Sort strictly by entity and time
    df = df.sort_values(by=["Store_ID", "Department", "Date"]).reset_index(drop=True)
    
    grouped = df.groupby(["Store_ID", "Department"])["Weekly_Sales"]
    
    # 1. Lag features (t-1, t-2, t-4 weeks)
    df["Sales_Lag_1"] = grouped.shift(1)
    df["Sales_Lag_2"] = grouped.shift(2)
    df["Sales_Lag_4"] = grouped.shift(4)
    
    # 2. Rolling Window Statistics (over past 4 and 12 weeks)
    # Using shift(1) first ensures no lookahead data leakage
    shifted_sales = grouped.shift(1)
    
    df["Sales_Rolling_Mean_4"] = (
        df.groupby(["Store_ID", "Department"])["Weekly_Sales"]
        .transform(lambda x: x.shift(1).rolling(window=4, min_periods=1).mean())
    )
    df["Sales_Rolling_Std_4"] = (
        df.groupby(["Store_ID", "Department"])["Weekly_Sales"]
        .transform(lambda x: x.shift(1).rolling(window=4, min_periods=1).std())
    ).fillna(0)
    
    df["Sales_Rolling_Mean_12"] = (
        df.groupby(["Store_ID", "Department"])["Weekly_Sales"]
        .transform(lambda x: x.shift(1).rolling(window=12, min_periods=1).mean())
    )
    
    # 3. Momentum Ratio (Lag 1 relative to 4-week moving average)
    df["Sales_Momentum_Ratio"] = df["Sales_Lag_1"] / (df["Sales_Rolling_Mean_4"] + 1e-5)
    
    return df

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes categorical variables into model-ready features."""
    df = df.copy()
    
    # Clean Holiday_Name
    df["Holiday_Name"] = df["Holiday_Name"].fillna("Regular_Week").replace({"None": "Regular_Week"})
    
    # We will keep raw Store_ID and Department as well as One-Hot encodings
    # One-hot encoding for Department
    dept_dummies = pd.get_dummies(df["Department"], prefix="Dept", drop_first=False, dtype=int)
    
    # One-hot encoding for Holiday_Name
    holiday_dummies = pd.get_dummies(df["Holiday_Name"], prefix="Holiday", drop_first=False, dtype=int)
    
    # One-hot encoding for Store_ID
    store_dummies = pd.get_dummies(df["Store_ID"], prefix="Store", drop_first=False, dtype=int)
    
    df = pd.concat([df, dept_dummies, holiday_dummies, store_dummies], axis=1)
    return df

def run_feature_engineering_pipeline(split_date: str = "2023-07-01"):
    """
    Executes the full feature engineering pipeline and splits into Train & Test sets.
    """
    print("=" * 70)
    print(" PHASE 4: TIME-SERIES FEATURE ENGINEERING & PREPROCESSING")
    print("=" * 70)
    
    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(f"Raw data file not found at: {RAW_DATA_FILE}. Run Phase 2 first.")
    
    print(f"[INFO] Loading raw data from: {RAW_DATA_FILE}")
    df_raw = pd.read_csv(RAW_DATA_FILE)
    df_raw["Date"] = pd.to_datetime(df_raw["Date"])
    
    # Step 1: Calendar & Cyclical Features
    print("[INFO] Creating calendar and cyclical trigonometric features...")
    df_features = create_calendar_features(df_raw)
    
    # Step 2: Lag & Rolling Statistics
    print("[INFO] Creating lag features (Lag 1, 2, 4) and rolling window statistics...")
    df_features = create_lag_and_rolling_features(df_features)
    
    # Step 3: Categorical Dummies & Encoding
    print("[INFO] Encoding categorical variables (Departments, Stores, Holidays)...")
    df_features = encode_categorical_features(df_features)
    
    # Step 4: Drop initial warmup rows with NaN lags (first 4 weeks per series)
    initial_count = len(df_features)
    df_features = df_features.dropna().reset_index(drop=True)
    dropped_count = initial_count - len(df_features)
    print(f"[INFO] Removed {dropped_count} initial warmup rows containing NaN lags.")
    
    # Save full processed dataset
    full_file = PROCESSED_DATA_DIR / "full_features.csv"
    df_features.to_csv(full_file, index=False)
    print(f"[OK] Full feature dataset saved to: {full_file}")
    
    # Step 5: Chronological Train-Test Split
    print(f"\n[INFO] Performing Chronological Train/Test Split on cutoff: {split_date}")
    split_timestamp = pd.to_datetime(split_date)
    
    train_df = df_features[df_features["Date"] < split_timestamp].copy()
    test_df = df_features[df_features["Date"] >= split_timestamp].copy()
    
    train_df.to_csv(TRAIN_FEATURES_FILE, index=False)
    test_df.to_csv(TEST_FEATURES_FILE, index=False)
    
    print(f"  [OK] Train Features Saved: {TRAIN_FEATURES_FILE}")
    print(f"       Rows: {len(train_df):,} | Date Range: {train_df['Date'].min().strftime('%Y-%m-%d')} to {train_df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"  [OK] Test Features Saved:  {TEST_FEATURES_FILE}")
    print(f"       Rows: {len(test_df):,} | Date Range: {test_df['Date'].min().strftime('%Y-%m-%d')} to {test_df['Date'].max().strftime('%Y-%m-%d')}")
    
    print("\n--- Engineered Feature List (Total Columns: " + str(len(df_features.columns)) + ") ---")
    engineered_cols = [c for c in df_features.columns if c not in df_raw.columns]
    for i, col in enumerate(engineered_cols, 1):
        print(f"  {i:02d}. {col}")
    
    print("=" * 70)
    print("[SUCCESS] Phase 4 Execution Complete: Datasets are ready for Model Training!")

if __name__ == "__main__":
    run_feature_engineering_pipeline()
