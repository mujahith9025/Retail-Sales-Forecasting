"""
Store & Category Health Scorecard Engine.
Evaluates 10 retail branches and 5 commercial departments across 5 performance pillars:
1. Revenue Velocity (0-25 pts)
2. Space Efficiency ($/SqFt) (0-20 pts)
3. Trend Momentum & Growth (0-20 pts)
4. Forecast Stability & Low Volatility (0-20 pts)
5. Promotional & Holiday Agility (0-15 pts)

Computes composite 0-100 scores, A+ to F letter grades, traffic light statuses,
and AI prescriptive recommendations.
"""

import pandas as pd
import numpy as np

GRADE_THRESHOLDS = [
    (95, "A+", "#10B981", "Elite Leader", "High revenue velocity & top space yield."),
    (88, "A", "#059669", "High Performer", "Strong consistency and dependable demand."),
    (78, "B", "#3B82F6", "Stable / Solid", "Steady revenue; upside via targeted promotions."),
    (68, "C", "#F59E0B", "Needs Focus", "Under-indexing yield; optimize space & stock."),
    (55, "D", "#EA580C", "Underperforming", "Elevated volatility; review inventory & staffing."),
    (0, "F", "#DC2626", "Critical Risk", "Severe revenue lag; executive review required.")
]

def calculate_letter_grade(score: float) -> dict:
    """Assigns letter grade, color, badge, and descriptor based on 0-100 score."""
    for min_val, grade, color, badge, desc in GRADE_THRESHOLDS:
        if score >= min_val:
            return {
                "grade": grade,
                "color": color,
                "badge": badge,
                "description": desc,
                "score": round(score, 1)
            }
    return {
        "grade": "F",
        "color": "#DC2626",
        "badge": "Critical Risk",
        "description": "Severe revenue lag; review needed.",
        "score": round(score, 1)
    }

def compute_store_health_scorecard(raw_df: pd.DataFrame, store_locations: dict) -> pd.DataFrame:
    """
    Computes comprehensive 5-pillar health scorecards for all stores.
    """
    store_cols = ["Rank", "Store_ID", "City", "State", "Health_Score", "Grade", "Status", "Color", 
                  "Total_Revenue ($)", "Avg_Weekly_Sales ($)", "Sales_per_SqFt ($)", "Growth_Pace (%)", 
                  "Top_Category", "Pillar_Revenue", "Pillar_Efficiency", "Pillar_Growth", 
                  "Pillar_Stability", "Pillar_Agility", "Prescription"]
                  
    if raw_df is None or len(raw_df) == 0 or "Weekly_Sales" not in raw_df.columns:
        return pd.DataFrame(columns=store_cols)

    total_network_rev = raw_df["Weekly_Sales"].sum()
    store_rows = []
    
    unique_stores = raw_df["Store_ID"].unique() if "Store_ID" in raw_df.columns else []
    for s_id in unique_stores:
        s_data = raw_df[raw_df["Store_ID"] == s_id].sort_values(by="Date") if "Date" in raw_df.columns else raw_df[raw_df["Store_ID"] == s_id]
        if len(s_data) == 0:
            continue
            
        s_info = store_locations.get(s_id, {}) if store_locations else {}
        city_name = s_info.get("city", str(s_data["City"].iloc[0]) if ("City" in s_data.columns and pd.notna(s_data["City"].iloc[0])) else str(s_id))
        state_name = s_info.get("state", str(s_data["State"].iloc[0]) if ("State" in s_data.columns and pd.notna(s_data["State"].iloc[0])) else "")

        tot_rev = float(s_data["Weekly_Sales"].sum())
        avg_rev = float(s_data.groupby("Date")["Weekly_Sales"].sum().mean()) if "Date" in s_data.columns else (tot_rev / max(1, len(s_data)))
        sqft = float(s_data["Store_Size_SqFt"].iloc[0]) if ("Store_Size_SqFt" in s_data.columns and pd.notna(s_data["Store_Size_SqFt"].iloc[0])) else float(s_info.get("sqft", 100000.0))
        if sqft <= 0: sqft = 100000.0
        sales_per_sqft = tot_rev / sqft
        
        # Weekly aggregates
        if "Date" in s_data.columns:
            weekly_totals = s_data.groupby("Date")["Weekly_Sales"].sum()
            r4 = weekly_totals.tail(4).mean()
            r12 = weekly_totals.tail(12).mean()
            growth_ratio = (r4 - r12) / (r12 + 1e-5)
            cov = weekly_totals.std() / (weekly_totals.mean() + 1e-5)
        else:
            growth_ratio = 0.03
            cov = 0.12
        
        # Promo & Holiday Lift
        if "Promotion_Discount" in s_data.columns:
            promo_sub = s_data[s_data["Promotion_Discount"] > 0]
            reg_sub = s_data[s_data["Promotion_Discount"] == 0]
            promo_mean = promo_sub["Weekly_Sales"].mean() if len(promo_sub) > 0 else avg_rev
            reg_mean = reg_sub["Weekly_Sales"].mean() if len(reg_sub) > 0 else avg_rev
            promo_lift = (promo_mean - reg_mean) / (reg_mean + 1e-5) if (pd.notna(promo_mean) and pd.notna(reg_mean) and reg_mean > 0) else 0.0
        else:
            reg_mean = avg_rev
            promo_lift = 0.0
            
        if "Is_Holiday" in s_data.columns:
            hol_sub = s_data[s_data["Is_Holiday"] == 1]
            hol_mean = hol_sub["Weekly_Sales"].mean() if len(hol_sub) > 0 else avg_rev
            hol_lift = (hol_mean - reg_mean) / (reg_mean + 1e-5) if (pd.notna(hol_mean) and pd.notna(reg_mean) and reg_mean > 0) else 0.0
        else:
            hol_lift = 0.0
        
        # Pillar Scoring (0 to Max)
        p1_rev = min(25.0, max(5.0, (tot_rev / max(1.0, total_network_rev * 0.15)) * 25.0))
        p2_eff = min(20.0, max(4.0, (sales_per_sqft / 85.0) * 20.0))
        p3_growth = min(20.0, max(5.0, 12.0 + (growth_ratio * 40.0)))
        p4_stability = min(20.0, max(5.0, 20.0 - (cov * 15.0)))
        p5_agility = min(15.0, max(3.0, (promo_lift * 15.0 + hol_lift * 10.0)))
        
        composite_score = p1_rev + p2_eff + p3_growth + p4_stability + p5_agility
        composite_score = min(99.4, max(42.0, composite_score))
        
        grade_info = calculate_letter_grade(composite_score)
        top_category = s_data.groupby("Department")["Weekly_Sales"].sum().idxmax() if ("Department" in s_data.columns and len(s_data) > 0) else "General"
        
        store_rows.append({
            "Store_ID": s_id,
            "City": city_name,
            "State": state_name,
            "Health_Score": round(composite_score, 1),
            "Grade": grade_info["grade"],
            "Status": grade_info["badge"],
            "Color": grade_info["color"],
            "Total_Revenue ($)": tot_rev,
            "Avg_Weekly_Sales ($)": avg_rev,
            "Sales_per_SqFt ($)": round(sales_per_sqft, 2),
            "Growth_Pace (%)": round(growth_ratio * 100, 1),
            "Top_Category": top_category,
            "Pillar_Revenue": round(p1_rev, 1),
            "Pillar_Efficiency": round(p2_eff, 1),
            "Pillar_Growth": round(p3_growth, 1),
            "Pillar_Stability": round(p4_stability, 1),
            "Pillar_Agility": round(p5_agility, 1),
            "Prescription": grade_info["description"]
        })
        
    if not store_rows:
        return pd.DataFrame(columns=store_cols)
        
    df_scores = pd.DataFrame(store_rows).sort_values(by="Health_Score", ascending=False).reset_index(drop=True)
    df_scores["Rank"] = range(1, len(df_scores) + 1)
    return df_scores


def compute_category_health_scorecard(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes comprehensive health scorecards for all product departments.
    """
    cat_cols = ["Rank", "Department", "Health_Score", "Grade", "Status", "Color", 
                "Total_Revenue ($)", "Avg_Weekly ($)", "Revenue_Share (%)", 
                "Growth_Pace (%)", "Promo_Lift (%)", "Holiday_Lift (%)", 
                "Pillar_Share", "Pillar_Growth", "Pillar_Stability", "Pillar_Agility", "Prescription"]
                
    if raw_df is None or len(raw_df) == 0 or "Weekly_Sales" not in raw_df.columns:
        return pd.DataFrame(columns=cat_cols)
        
    total_rev = raw_df["Weekly_Sales"].sum()
    dept_rows = []
    
    unique_depts = raw_df["Department"].unique() if "Department" in raw_df.columns else []
    for dept in unique_depts:
        d_data = raw_df[raw_df["Department"] == dept].sort_values(by="Date") if "Date" in raw_df.columns else raw_df[raw_df["Department"] == dept]
        if len(d_data) == 0:
            continue
            
        tot_d = float(d_data["Weekly_Sales"].sum())
        avg_d = float(d_data["Weekly_Sales"].mean())
        rev_share = (tot_d / max(1.0, total_rev)) * 100
        
        # Time trend
        if "Date" in d_data.columns:
            weekly_d = d_data.groupby("Date")["Weekly_Sales"].sum()
            r4 = weekly_d.tail(4).mean()
            r12 = weekly_d.tail(12).mean()
            growth = (r4 - r12) / (r12 + 1e-5)
            cov = weekly_d.std() / (weekly_d.mean() + 1e-5)
        else:
            growth = 0.03
            cov = 0.12
        
        if "Promotion_Discount" in d_data.columns:
            promo_sub = d_data[d_data["Promotion_Discount"] > 0]
            reg_sub = d_data[d_data["Promotion_Discount"] == 0]
            promo_sales = promo_sub["Weekly_Sales"].mean() if len(promo_sub) > 0 else avg_d
            reg_sales = reg_sub["Weekly_Sales"].mean() if len(reg_sub) > 0 else avg_d
            promo_lift = (promo_sales - reg_sales) / (reg_sales + 1e-5) if (pd.notna(promo_sales) and pd.notna(reg_sales) and reg_sales > 0) else 0.0
        else:
            reg_sales = avg_d
            promo_lift = 0.0
            
        if "Is_Holiday" in d_data.columns:
            hol_sub = d_data[d_data["Is_Holiday"] == 1]
            hol_sales = hol_sub["Weekly_Sales"].mean() if len(hol_sub) > 0 else avg_d
            hol_lift = (hol_sales - reg_sales) / (reg_sales + 1e-5) if (pd.notna(hol_sales) and pd.notna(reg_sales) and reg_sales > 0) else 0.0
        else:
            hol_lift = 0.0
        
        # Pillars
        p1 = min(30.0, max(8.0, (rev_share / 25.0) * 30.0))
        p2 = min(25.0, max(5.0, 15.0 + (growth * 30.0)))
        p3 = min(25.0, max(5.0, 25.0 - (cov * 18.0)))
        p4 = min(20.0, max(4.0, (promo_lift * 12.0 + hol_lift * 10.0)))
        
        score = min(98.5, max(45.0, p1 + p2 + p3 + p4))
        grade_info = calculate_letter_grade(score)
        
        dept_rows.append({
            "Department": dept,
            "Health_Score": round(score, 1),
            "Grade": grade_info["grade"],
            "Status": grade_info["badge"],
            "Color": grade_info["color"],
            "Total_Revenue ($)": tot_d,
            "Avg_Weekly ($)": avg_d,
            "Revenue_Share (%)": round(rev_share, 1),
            "Growth_Pace (%)": round(growth * 100, 1),
            "Promo_Lift (%)": round(promo_lift * 100, 1),
            "Holiday_Lift (%)": round(hol_lift * 100, 1),
            "Pillar_Share": round(p1, 1),
            "Pillar_Growth": round(p2, 1),
            "Pillar_Stability": round(p3, 1),
            "Pillar_Agility": round(p4, 1),
            "Prescription": grade_info["description"]
        })
        
    if not dept_rows:
        return pd.DataFrame(columns=cat_cols)
        
    df_cat = pd.DataFrame(dept_rows).sort_values(by="Health_Score", ascending=False).reset_index(drop=True)
    df_cat["Rank"] = range(1, len(df_cat) + 1)
    return df_cat
