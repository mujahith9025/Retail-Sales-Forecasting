"""
Retail Sales Forecasting - Beginner's Interactive Tutorial Script.
This tutorial walks you through each step of the Machine Learning pipeline with explanations.
"""

import pandas as pd
import numpy as np
import joblib

def step_1_explain_problem():
    print("""
================================================================================
STEP 1: UNDERSTANDING THE PROBLEM
================================================================================
Retail Sales Forecasting is a Time-Series regression problem where the goal is
to predict the total weekly revenue ($) for retail store departments.

Why is this important for businesses?
- Inventory Management: Prevents stockouts (running out of items) and overstocking.
- Staffing & Labor: Helps schedule store staff based on anticipated customer surges.
- Promotion Planning: Measures expected revenue lift from marketing discounts.
""")

def step_2_explore_dataset():
    print("""
================================================================================
STEP 2: EXPLORING THE DATASET
================================================================================
""")
    df = pd.read_csv("data/raw/retail_sales_data.csv")
    print(f"Total Rows: {len(df):,} | Columns: {list(df.columns)}")
    print("\nSample records:")
    print(df[["Date", "Store_ID", "Department", "Is_Holiday", "Promotion_Discount", "Weekly_Sales"]].head(5))

def step_3_feature_engineering_concept():
    print("""
================================================================================
STEP 3: WHY FEATURE ENGINEERING MATTERS FOR TIME-SERIES
================================================================================
Standard machine learning models (like Random Forest and XGBoost) don't naturally
understand the sequence of time unless we engineer temporal features:

1. Lags (t-1, t-2, t-4): What were the sales 1 week ago, 2 weeks ago, etc.?
2. Rolling Windows: What was the average sales over the last 4 weeks?
3. Cyclical Trigonometry: Representing Week of Year using sin() and cos() so
   Week 52 connects smoothly back to Week 1.
""")

def step_4_make_a_prediction():
    print("""
================================================================================
STEP 4: LOADING TRAINED CHAMPION MODEL & INFERENCE
================================================================================
""")
    model_artifact = joblib.load("models/best_forecasting_model.pkl")
    model = model_artifact["model"]
    feature_names = model_artifact["feature_names"]
    
    test_df = pd.read_csv("data/processed/test_features.csv")
    sample_row = test_df[feature_names].iloc[[0]]
    actual_sales = test_df["Weekly_Sales"].iloc[0]
    
    predicted_sales = model.predict(sample_row)[0]
    
    print(f"Sample Store & Dept: {test_df['Store_ID'].iloc[0]} - {test_df['Department'].iloc[0]}")
    print(f"Date: {test_df['Date'].iloc[0]}")
    print(f"Actual Sales:    ${actual_sales:,.2f}")
    print(f"Predicted Sales: ${predicted_sales:,.2f}")
    print(f"Absolute Error:  ${abs(predicted_sales - actual_sales):,.2f} ({abs(predicted_sales - actual_sales)/actual_sales * 100:.2f}%)")

if __name__ == "__main__":
    step_1_explain_problem()
    step_2_explore_dataset()
    step_3_feature_engineering_concept()
    step_4_make_a_prediction()
    print("""
================================================================================
TUTORIAL COMPLETE!
Run 'streamlit run app/app.py' to explore the full interactive visual dashboard!
================================================================================
""")
