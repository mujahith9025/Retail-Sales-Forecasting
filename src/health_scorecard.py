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
    (95, "A+", "#10B981", "Elite Leader", "Outstanding revenue velocity, high footprint efficiency, and robust growth trajectory."),
    (88, "A", "#059669", "High Performer", "Strong commercial performance with consistent sales and dependable demand response."),
    (78, "B", "#3B82F6", "Stable / Solid", "Steady operations and average space yield. Minor upside available via targeted promos."),
    (68, "C", "#F59E0B", "Needs Optimization", "Under-indexing on space efficiency or experiencing sales momentum slowdown."),
    (55, "D", "#EA580C", "Underperforming", "High volatility or weak category traction. Immediate inventory & staffing review needed."),
    (0, "F", "#DC2626", "Critical Risk", "Severe revenue lag or high stockout variance. Requires urgent executive intervention.")
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
        "description": "Severe revenue lag.",
        "score": round(score, 1)
    }

def compute_store_health_scorecard(raw_df: pd.DataFrame, store_locations: dict) -> pd.DataFrame:
    """
    Computes comprehensive 5-pillar health scorecards for all stores.
    """
    total_network_rev = raw_df["Weekly_Sales"].sum()
    store_rows = []
    
    for s_id, s_info in store_locations.items():
        s_data = raw_df[raw_df["Store_ID"] == s_id].sort_values(by="Date")
        if len(s_data) == 0:
            continue
            
        tot_rev = s_data["Weekly_Sales"].sum()
        avg_rev = s_data.groupby("Date")["Weekly_Sales"].sum().mean()
        sqft = s_data["Store_Size_SqFt"].iloc[0]
        sales_per_sqft = tot_rev / sqft
        
        # Weekly aggregates
        weekly_totals = s_data.groupby("Date")["Weekly_Sales"].sum()
        r4 = weekly_totals.tail(4).mean()
        r12 = weekly_totals.tail(12).mean()
        growth_ratio = (r4 - r12) / (r12 + 1e-5)
        
        # Volatility
        cov = weekly_totals.std() / (weekly_totals.mean() + 1e-5)
        
        # Promo & Holiday Lift
        promo_mean = s_data[s_data["Promotion_Discount"] > 0]["Weekly_Sales"].mean()
        reg_mean = s_data[s_data["Promotion_Discount"] == 0]["Weekly_Sales"].mean()
        promo_lift = (promo_mean - reg_mean) / (reg_mean + 1e-5)
        
        hol_mean = s_data[s_data["Is_Holiday"] == 1]["Weekly_Sales"].mean()
        hol_lift = (hol_mean - reg_mean) / (reg_mean + 1e-5)
        
        # Pillar Scoring (0 to Max)
        # 1. Revenue Velocity (0-25)
        p1_rev = min(25.0, max(5.0, (tot_rev / (total_network_rev * 0.15)) * 25.0))
        
        # 2. Footprint Efficiency (0-20): Target ~$70-$100/sqft
        p2_eff = min(20.0, max(4.0, (sales_per_sqft / 85.0) * 20.0))
        
        # 3. Growth Momentum (0-20): Positive growth awards high score
        p3_growth = min(20.0, max(5.0, 12.0 + (growth_ratio * 40.0)))
        
        # 4. Stability (0-20): Lower COV = higher stability
        p4_stability = min(20.0, max(5.0, 20.0 - (cov * 15.0)))
        
        # 5. Promotional & Holiday Agility (0-15)
        p5_agility = min(15.0, max(3.0, (promo_lift * 15.0 + hol_lift * 10.0)))
        
        composite_score = p1_rev + p2_eff + p3_growth + p4_stability + p5_agility
        composite_score = min(99.4, max(42.0, composite_score))
        
        grade_info = calculate_letter_grade(composite_score)
        top_category = s_data.groupby("Department")["Weekly_Sales"].sum().idxmax()
        
        store_rows.append({
            "Store_ID": s_id,
            "City": s_info["city"],
            "State": s_info["state"],
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
        
    df_scores = pd.DataFrame(store_rows).sort_values(by="Health_Score", ascending=False).reset_index(drop=True)
    df_scores["Rank"] = range(1, len(df_scores) + 1)
    return df_scores

def compute_category_health_scorecard(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes comprehensive health scorecards for all 5 product departments.
    """
    total_rev = raw_df["Weekly_Sales"].sum()
    dept_rows = []
    
    for dept in raw_df["Department"].unique():
        d_data = raw_df[raw_df["Department"] == dept].sort_values(by="Date")
        tot_d = d_data["Weekly_Sales"].sum()
        avg_d = d_data["Weekly_Sales"].mean()
        rev_share = (tot_d / total_rev) * 100
        
        # Time trend
        weekly_d = d_data.groupby("Date")["Weekly_Sales"].sum()
        r4 = weekly_d.tail(4).mean()
        r12 = weekly_d.tail(12).mean()
        growth = (r4 - r12) / (r12 + 1e-5)
        
        promo_sales = d_data[d_data["Promotion_Discount"] > 0]["Weekly_Sales"].mean()
        reg_sales = d_data[d_data["Promotion_Discount"] == 0]["Weekly_Sales"].mean()
        promo_lift = (promo_sales - reg_sales) / (reg_sales + 1e-5)
        
        hol_sales = d_data[d_data["Is_Holiday"] == 1]["Weekly_Sales"].mean()
        hol_lift = (hol_sales - reg_sales) / (reg_sales + 1e-5)
        
        cov = weekly_d.std() / (weekly_d.mean() + 1e-5)
        
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
        
    df_cat = pd.DataFrame(dept_rows).sort_values(by="Health_Score", ascending=False).reset_index(drop=True)
    df_cat["Rank"] = range(1, len(df_cat) + 1)
    return df_cat
