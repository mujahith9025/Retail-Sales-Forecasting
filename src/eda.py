"""
Phase 3: Exploratory Data Analysis (EDA) Script.
Performs comprehensive data exploration, statistical analysis, and generates visualization charts.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for generating figures
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from src.config import RAW_DATA_FILE, BASE_DIR
except (ImportError, ModuleNotFoundError):
    from config import RAW_DATA_FILE, BASE_DIR

# Reports output directory
REPORTS_DIR = BASE_DIR / "reports" / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Set clean visualization theme
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_palette("tab10")

def load_data() -> pd.DataFrame:
    """Loads raw retail sales data and parses dates."""
    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(f"Raw data file not found at: {RAW_DATA_FILE}. Run Phase 2 first.")
    df = pd.read_csv(RAW_DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Holiday_Name"] = df["Holiday_Name"].fillna("Regular_Week").replace({"None": "Regular_Week"})
    return df

def perform_statistical_analysis(df: pd.DataFrame):
    """Computes and prints essential statistical insights."""
    print("\n" + "=" * 70)
    print(" PHASE 3: EXPLORATORY DATA ANALYSIS (EDA) SUMMARY")
    print("=" * 70)
    
    # 1. High-level Summary
    total_sales = df["Weekly_Sales"].sum()
    avg_sales = df["Weekly_Sales"].mean()
    print(f"\n[1] OVERALL SALES METRICS:")
    print(f"  - Total Revenue Recorded: ${total_sales:,.2f}")
    print(f"  - Average Weekly Sales per Dept: ${avg_sales:,.2f}")
    print(f"  - Min / Max Weekly Sales: ${df['Weekly_Sales'].min():,.2f} / ${df['Weekly_Sales'].max():,.2f}")

    # 2. Performance by Department
    dept_summary = df.groupby("Department")["Weekly_Sales"].agg(["count", "mean", "std", "sum"]).reset_index()
    dept_summary["Revenue_Share_%"] = (dept_summary["sum"] / total_sales) * 100
    dept_summary = dept_summary.sort_values(by="sum", ascending=False)
    print(f"\n[2] SALES BY DEPARTMENT:")
    for _, row in dept_summary.iterrows():
        print(f"  - {row['Department']:<12}: Avg = ${row['mean']:,.2f} | Share = {row['Revenue_Share_%']:.1f}%")

    # 3. Holiday vs Non-Holiday Impact
    holiday_summary = df.groupby("Holiday_Name")["Weekly_Sales"].agg(["count", "mean"]).reset_index()
    reg_week_row = holiday_summary.loc[holiday_summary["Holiday_Name"] == "Regular_Week", "mean"]
    non_holiday_mean = reg_week_row.values[0] if len(reg_week_row) > 0 else avg_sales
    holiday_summary["Lift_%"] = ((holiday_summary["mean"] - non_holiday_mean) / non_holiday_mean) * 100
    holiday_summary = holiday_summary.sort_values(by="mean", ascending=False)
    print(f"\n[3] HOLIDAY IMPACT ON SALES:")
    for _, row in holiday_summary.iterrows():
        lift_str = f"+{row['Lift_%']:.1f}%" if row['Lift_%'] >= 0 else f"{row['Lift_%']:.1f}%"
        print(f"  - {row['Holiday_Name']:<26}: Avg = ${row['mean']:,.2f} (Lift: {lift_str})")

    # 4. Promotion Impact
    promo_stats = df.groupby(df["Promotion_Discount"] > 0)["Weekly_Sales"].agg(["count", "mean"])
    no_promo_mean = promo_stats.loc[False, "mean"]
    promo_mean = promo_stats.loc[True, "mean"]
    promo_lift = ((promo_mean - no_promo_mean) / no_promo_mean) * 100
    print(f"\n[4] PROMOTIONAL MARKDOWN IMPACT:")
    print(f"  - Without Promotion Avg: ${no_promo_mean:,.2f}")
    print(f"  - With Promotion Avg:    ${promo_mean:,.2f} (Lift: +{promo_lift:.1f}%)")
    print("=" * 70)

def generate_visualizations(df: pd.DataFrame):
    """Creates and exports key EDA charts to reports/figures/."""
    print("\n[INFO] Generating and saving visualization charts...")

    # --- Figure 1: Weekly Total Sales Trend Over Time ---
    plt.figure(figsize=(14, 5))
    weekly_total = df.groupby("Date")["Weekly_Sales"].sum().reset_index()
    plt.plot(weekly_total["Date"], weekly_total["Weekly_Sales"] / 1000, color="#1f77b4", linewidth=2.2, label="Total Weekly Sales ($k)")
    
    # Highlight Thanksgiving / Black Friday & Christmas peaks
    holiday_weeks = df[df["Is_Holiday"] == 1][["Date", "Holiday_Name"]].drop_duplicates()
    for _, row in holiday_weeks.iterrows():
        if row["Holiday_Name"] in ["Thanksgiving_BlackFriday", "Christmas_Holiday"]:
            sales_val = weekly_total.loc[weekly_total["Date"] == row["Date"], "Weekly_Sales"].values
            if len(sales_val) > 0:
                plt.scatter(row["Date"], sales_val[0] / 1000, color="crimson", s=70, zorder=5)
                plt.annotate(row["Holiday_Name"].replace("_", " "), (row["Date"], (sales_val[0] / 1000) + 15),
                             fontsize=8, ha='center', color="darkred", fontweight="bold", rotation=30)
                
    plt.title("Total Retail Sales Trend (2021 - 2023) with Holiday Peaks", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Total Weekly Revenue ($ in Thousands)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig1_path = REPORTS_DIR / "01_overall_sales_trend.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"  [OK] Saved: {fig1_path}")

    # --- Figure 2: Sales Distribution by Department ---
    plt.figure(figsize=(10, 5))
    dept_order = df.groupby("Department")["Weekly_Sales"].mean().sort_values(ascending=False).index
    sns.boxplot(data=df, x="Department", y="Weekly_Sales", order=dept_order, hue="Department", palette="viridis", legend=False)
    plt.title("Weekly Sales Distribution Across Departments", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Department", fontsize=11)
    plt.ylabel("Weekly Sales ($)", fontsize=11)
    plt.tight_layout()
    fig2_path = REPORTS_DIR / "02_sales_by_department.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"  [OK] Saved: {fig2_path}")

    # --- Figure 3: Average Sales by Holiday Event ---
    plt.figure(figsize=(10, 5))
    holiday_avg = df.groupby("Holiday_Name")["Weekly_Sales"].mean().reset_index()
    holiday_avg = holiday_avg.sort_values(by="Weekly_Sales", ascending=True)
    colors = ["#4575b4" if name == "Regular_Week" else "#d73027" for name in holiday_avg["Holiday_Name"]]
    
    bars = plt.barh(holiday_avg["Holiday_Name"], holiday_avg["Weekly_Sales"], color=colors, height=0.6)
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 800, bar.get_y() + bar.get_height()/2, f"${width:,.0f}", 
                 va='center', ha='left', fontsize=9, fontweight='bold')
                 
    plt.title("Average Weekly Revenue by Holiday Event", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Average Weekly Sales ($)", fontsize=11)
    plt.ylabel("Event Name", fontsize=11)
    plt.xlim(0, max(holiday_avg["Weekly_Sales"]) * 1.18)
    plt.tight_layout()
    fig3_path = REPORTS_DIR / "03_holiday_impact_comparison.png"
    plt.savefig(fig3_path, dpi=200)
    plt.close()
    print(f"  [OK] Saved: {fig3_path}")

    # --- Figure 4: Feature Correlation Heatmap ---
    plt.figure(figsize=(9, 7))
    numeric_cols = ["Weekly_Sales", "Store_Size_SqFt", "Promotion_Discount", "Is_Holiday", "Temperature", "Fuel_Price", "CPI", "Unemployment_Rate"]
    corr = df[numeric_cols].corr()
    
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, linewidths=0.5,
                annot_kws={"size": 10, "weight": "bold"})
    plt.title("Feature Correlation Heatmap", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    fig4_path = REPORTS_DIR / "04_correlation_heatmap.png"
    plt.savefig(fig4_path, dpi=200)
    plt.close()
    print(f"  [OK] Saved: {fig4_path}")

    print("[SUCCESS] All Phase 3 EDA figures generated successfully!")

if __name__ == "__main__":
    data = load_data()
    perform_statistical_analysis(data)
    generate_visualizations(data)
