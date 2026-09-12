"""
Phase 2: Realistic Retail Sales Dataset Generator and Ingestion Script.
Generates a multi-store, multi-department time-series retail dataset spanning 3 years.
Incorporates seasonal trends, holiday surges, promotional markdowns, and macroeconomic indicators.
"""

import sys
from pathlib import Path

# Add project root directory to sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
try:
    from src.config import RAW_DATA_FILE, STORES, DEPARTMENTS, START_DATE, END_DATE
except (ImportError, ModuleNotFoundError):
    from config import RAW_DATA_FILE, STORES, DEPARTMENTS, START_DATE, END_DATE

# Set random seed for reproducibility
np.random.seed(42)

# Department baseline revenue multipliers and seasonality profiles
DEPT_CONFIG = {
    "Grocery": {"base": 28000, "promo_sensitivity": 0.8, "holiday_boost": 1.35},
    "Electronics": {"base": 22000, "promo_sensitivity": 1.6, "holiday_boost": 2.20},
    "Apparel": {"base": 18000, "promo_sensitivity": 1.4, "holiday_boost": 1.80},
    "Home_Garden": {"base": 15000, "promo_sensitivity": 1.1, "holiday_boost": 1.25},
    "Pharmacy": {"base": 12000, "promo_sensitivity": 0.5, "holiday_boost": 1.10}
}

# Store characteristics (size in sq ft, location factor)
STORE_CONFIG = {
    store: {
        "size_sqft": np.random.randint(70000, 210000),
        "scale_factor": np.random.uniform(0.85, 1.35)
    }
    for store in STORES
}

def identify_holiday(dt: pd.Timestamp):
    """
    Returns (is_holiday: bool, holiday_name: str) for major retail sales events.
    """
    month = dt.month
    day = dt.day
    
    # Super Bowl week (early Feb)
    if month == 2 and 5 <= day <= 14:
        return True, "Super_Bowl"
    # Easter period (late Mar - mid Apr)
    elif (month == 3 and day >= 25) or (month == 4 and day <= 15):
        return True, "Easter"
    # Labor Day (early Sept)
    elif month == 9 and 1 <= day <= 10:
        return True, "Labor_Day"
    # Thanksgiving & Black Friday (late Nov)
    elif month == 11 and 20 <= day <= 30:
        return True, "Thanksgiving_BlackFriday"
    # Christmas & New Year (late Dec)
    elif month == 12 and 18 <= day <= 31:
        return True, "Christmas_Holiday"
    
    return False, "None"

def generate_retail_sales_data() -> pd.DataFrame:
    """
    Generates realistic weekly sales data for all stores and departments.
    """
    print("[INFO] Starting retail sales dataset generation...")
    print(f"[INFO] Time Range: {START_DATE} to {END_DATE}")
    print(f"[INFO] Stores: {len(STORES)} | Departments: {len(DEPARTMENTS)}")
    
    # Weekly date range (Fridays)
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq="W-FRI")
    num_weeks = len(dates)
    
    records = []
    
    # Base macroeconomic trends over time
    cpi_base = 215.0 + np.linspace(0, 30, num_weeks) + np.random.normal(0, 0.5, num_weeks)
    unemp_base = 8.5 - np.linspace(0, 2.5, num_weeks) + np.random.normal(0, 0.1, num_weeks)
    fuel_base = 3.20 + 0.6 * np.sin(np.linspace(0, 3 * np.pi, num_weeks)) + np.random.normal(0, 0.05, num_weeks)
    
    for store_id in STORES:
        store_info = STORE_CONFIG[store_id]
        store_scale = store_info["scale_factor"]
        store_size = store_info["size_sqft"]
        
        for dept in DEPARTMENTS:
            dept_info = DEPT_CONFIG[dept]
            dept_base = dept_info["base"]
            promo_sens = dept_info["promo_sensitivity"]
            holiday_boost_max = dept_info["holiday_boost"]
            
            for i, date in enumerate(dates):
                week_of_year = date.isocalendar().week
                is_holiday, holiday_name = identify_holiday(date)
                
                # 1. Base annual growth (3% per year)
                year_offset = (date.year - 2021)
                growth_multiplier = 1.0 + (0.035 * year_offset)
                
                # 2. Seasonality curve (Spring uptick, Summer plateau, Q4 Holiday surge)
                seasonal_factor = 1.0 + 0.15 * np.sin(2 * np.pi * (week_of_year - 10) / 52.0)
                if dept == "Home_Garden" and 15 <= week_of_year <= 28:
                    seasonal_factor += 0.35  # Gardening boost in spring/early summer
                elif dept == "Apparel" and 32 <= week_of_year <= 36:
                    seasonal_factor += 0.25  # Back to school
                
                # 3. Holiday Spike
                holiday_multiplier = 1.0
                if is_holiday:
                    if holiday_name in ["Thanksgiving_BlackFriday", "Christmas_Holiday"]:
                        holiday_multiplier = holiday_boost_max + np.random.uniform(-0.05, 0.1)
                    else:
                        holiday_multiplier = 1.0 + (holiday_boost_max - 1.0) * 0.45
                
                # 4. Promotional discounts (promotions happen more frequently near holidays)
                has_promo = np.random.rand() < (0.45 if is_holiday else 0.20)
                promo_discount = round(np.random.uniform(0.05, 0.30), 2) if has_promo else 0.0
                promo_multiplier = 1.0 + (promo_discount * promo_sens * 1.5)
                
                # 5. Temperature (Seasonal weather in deg F)
                temperature = 60.0 - 25.0 * np.cos(2 * np.pi * (week_of_year - 3) / 52.0) + np.random.normal(0, 3.0)
                
                # 6. Economic values for this week
                fuel_price = round(float(fuel_base[i]), 2)
                cpi = round(float(cpi_base[i]), 2)
                unemployment = round(float(unemp_base[i]), 2)
                
                # 7. Total Weekly Sales calculation
                base_sales = dept_base * store_scale * growth_multiplier * seasonal_factor
                adjusted_sales = base_sales * holiday_multiplier * promo_multiplier
                
                # Add realistic noise (+/- 4%)
                noise = np.random.normal(1.0, 0.04)
                weekly_sales = max(500.0, round(adjusted_sales * noise, 2))
                
                records.append({
                    "Date": date.strftime("%Y-%m-%d"),
                    "Store_ID": store_id,
                    "Store_Size_SqFt": store_size,
                    "Department": dept,
                    "Is_Holiday": int(is_holiday),
                    "Holiday_Name": holiday_name,
                    "Promotion_Discount": promo_discount,
                    "Temperature": round(temperature, 1),
                    "Fuel_Price": fuel_price,
                    "CPI": cpi,
                    "Unemployment_Rate": unemployment,
                    "Weekly_Sales": weekly_sales
                })
                
    df = pd.DataFrame(records)
    
    # Sort chronologically by Store, Department, Date
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by=["Store_ID", "Department", "Date"]).reset_index(drop=True)
    
    return df

def save_and_summarize(df: pd.DataFrame):
    """
    Saves the generated dataset to raw data folder and prints quality inspection summary.
    """
    df.to_csv(RAW_DATA_FILE, index=False)
    print(f"\n[OK] Dataset successfully saved to: {RAW_DATA_FILE}")
    print("=" * 70)
    print(f"Total Records: {len(df):,}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Date Range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Unique Stores: {df['Store_ID'].nunique()} | Unique Departments: {df['Department'].nunique()}")
    print("=" * 70)
    
    print("\n--- Dataset Sample Preview (First 5 Rows) ---")
    print(df.head())
    
    print("\n--- Target Variable ('Weekly_Sales') Statistics ---")
    print(df["Weekly_Sales"].describe().to_string())
    
    print("\n--- Missing Value Check ---")
    null_counts = df.isnull().sum()
    print(null_counts.to_string())
    print("=" * 70)
    print("[SUCCESS] Phase 2 Execution Complete: Dataset is ready for EDA and Preprocessing!")

if __name__ == "__main__":
    df = generate_retail_sales_data()
    save_and_summarize(df)
