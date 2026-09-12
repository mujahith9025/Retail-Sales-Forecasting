"""
Goal-Seek & Reverse Target Revenue Solver Engine.
Given a user's target sales revenue for any store and department (or entire store network),
calculates the optimal promotional discount, inventory safety buffer, staffing allocation,
financial feasibility, and 3 strategic tier execution plans.
"""

import numpy as np
import pandas as pd
import joblib
from datetime import datetime
try:
    from src.config import BEST_MODEL_FILE, DEPARTMENTS, STORES
except (ImportError, ModuleNotFoundError):
    from config import BEST_MODEL_FILE, DEPARTMENTS, STORES


def solve_target_revenue_plan(store_id: str, dept: str, target_sales: float, raw_df: pd.DataFrame) -> dict:
    """
    Reverse-engineers the required promotional discount, staffing, inventory buffer,
    and financial margin to hit a target revenue goal. Supports both single department
    and entire store aggregation.
    """
    champion_artifact = joblib.load(BEST_MODEL_FILE)
    model = champion_artifact["model"]
    feature_names = champion_artifact["feature_names"]
    
    dt_target = pd.to_datetime("2024-01-05")
    week_num = dt_target.isocalendar().week
    month_num = dt_target.month
    
    # Check if store-wide (all departments)
    is_store_wide = (dept in ["All Departments (Entire Store)", "All Departments", "All", "Store Total"])
    
    if is_store_wide:
        # Evaluate baseline across all departments
        store_subset = raw_df[raw_df["Store_ID"] == store_id]
        store_size = store_subset["Store_Size_SqFt"].iloc[0] if len(store_subset) > 0 else 120000
        
        dept_baselines = {}
        dept_recent_sales = {}
        for d in DEPARTMENTS:
            sub = store_subset[store_subset["Department"] == d].sort_values(by="Date")
            rec = sub["Weekly_Sales"].tail(4).values
            base = float(np.mean(rec)) if len(rec) > 0 else 25000.0
            dept_baselines[d] = base
            dept_recent_sales[d] = rec
            
        total_baseline = sum(dept_baselines.values())
        gap = target_sales - total_baseline
        pct_gap = (gap / (total_baseline + 1e-5)) * 100
        
        def evaluate_store_combo(promo_pct: float, is_hol: int, hol_name: str) -> dict:
            dept_preds = {}
            for d in DEPARTMENTS:
                rec = dept_recent_sales[d]
                base_d = dept_baselines[d]
                row = {
                    "Store_Size_SqFt": store_size,
                    "Is_Holiday": is_hol,
                    "Promotion_Discount": promo_pct,
                    "Temperature": 65.0,
                    "Fuel_Price": 3.45,
                    "CPI": 245.0,
                    "Unemployment_Rate": 5.5,
                    "Year": dt_target.year,
                    "Month": month_num,
                    "Week_of_Year": int(week_num),
                    "Quarter": dt_target.quarter,
                    "Is_Month_End": int(dt_target.is_month_end),
                    "Week_Sin": np.sin(2 * np.pi * week_num / 52.0),
                    "Week_Cos": np.cos(2 * np.pi * week_num / 52.0),
                    "Month_Sin": np.sin(2 * np.pi * month_num / 12.0),
                    "Month_Cos": np.cos(2 * np.pi * month_num / 12.0),
                    "Sales_Lag_1": rec[-1] if len(rec) >= 1 else 25000.0,
                    "Sales_Lag_2": rec[-2] if len(rec) >= 2 else 24500.0,
                    "Sales_Lag_4": rec[0] if len(rec) >= 4 else 24000.0,
                    "Sales_Rolling_Mean_4": base_d,
                    "Sales_Rolling_Std_4": 1200.0,
                    "Sales_Rolling_Mean_12": base_d,
                    "Sales_Momentum_Ratio": 1.0
                }
                for dept_k in DEPARTMENTS: row[f"Dept_{dept_k}"] = 1 if d == dept_k else 0
                for h in ["Christmas_Holiday", "Easter", "Labor_Day", "Regular_Week", "Super_Bowl", "Thanksgiving_BlackFriday"]:
                    row[f"Holiday_{h}"] = 1 if hol_name == h else 0
                for s in STORES: row[f"Store_{s}"] = 1 if store_id == s else 0
                
                in_df = pd.DataFrame([row])
                for col in feature_names:
                    if col not in in_df.columns:
                        in_df[col] = 0
                dept_preds[d] = float(model.predict(in_df[feature_names])[0])
            return {"total": sum(dept_preds.values()), "by_dept": dept_preds}
            
        promo_steps = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
        best_promo = 0.0
        best_eval = evaluate_store_combo(0.0, 0, "Regular_Week")
        best_hol = "Regular_Week"
        best_is_hol = 0
        
        for p in promo_steps:
            res_eval = evaluate_store_combo(p, 0, "Regular_Week")
            if res_eval["total"] >= target_sales:
                best_promo = p
                best_eval = res_eval
                break
            elif res_eval["total"] > best_eval["total"]:
                best_promo = p
                best_eval = res_eval
                
        if best_eval["total"] < target_sales:
            for p in promo_steps:
                res_eval = evaluate_store_combo(p, 1, "Thanksgiving_BlackFriday")
                if res_eval["total"] >= target_sales:
                    best_promo = p
                    best_eval = res_eval
                    best_hol = "Thanksgiving_BlackFriday"
                    best_is_hol = 1
                    break
                elif res_eval["total"] > best_eval["total"]:
                    best_promo = p
                    best_eval = res_eval
                    best_hol = "Thanksgiving_BlackFriday"
                    best_is_hol = 1
                    
        best_pred = best_eval["total"]
        lift_achieved = best_pred - total_baseline
        pct_lift_achieved = (lift_achieved / (total_baseline + 1e-5)) * 100
        
        # Staffing recommendation across whole store
        if pct_lift_achieved > 30:
            staff_req = "+6 Store Staff (3 Extra Cashiers + 3 Floor Restockers)"
            labor_cost = 2100.0
        elif pct_lift_achieved > 15:
            staff_req = "+4 Store Staff (2 Extra Cashiers + 2 Floor Associates)"
            labor_cost = 1400.0
        elif pct_lift_achieved > 5:
            staff_req = "+2 Store Staff (Peak Support Staffing)"
            labor_cost = 700.0
        else:
            staff_req = "Standard Base Staffing (No extra store labor required)"
            labor_cost = 0.0
            
        buffer_pct = max(10, int(round(pct_lift_achieved * 1.15)))
        buffer_req = f"+{buffer_pct}% Safety Stock Buffer Across All Depts"
        
        if best_pred >= target_sales and pct_gap <= 35:
            feasibility = "Highly Achievable"
            feasibility_color = "#10B981"
            feasibility_desc = "Goal can be met comfortably across departments using targeted promotional discounts and standard staffing."
        elif best_pred >= target_sales * 0.90:
            feasibility = "Moderately Challenging"
            feasibility_color = "#F59E0B"
            feasibility_desc = "Requires store-wide promotional push (20-30% off) and holiday traffic surge to approach target."
        else:
            feasibility = "Highly Ambitious / Stretch Goal"
            feasibility_color = "#DC2626"
            feasibility_desc = "Target is significantly above total historical capacity (>40% lift). Recommend multi-quarter expansion."
            
        cogs_est = best_pred * 0.58
        discount_cost = best_pred * (best_promo / (1.0 - best_promo + 1e-5)) if best_promo > 0 else 0.0
        net_operating_profit = best_pred - cogs_est - discount_cost - labor_cost
        net_margin_pct = (net_operating_profit / (best_pred + 1e-5)) * 100
        
        # Department breakdown table
        dept_rows = []
        for d in DEPARTMENTS:
            b_d = dept_baselines[d]
            p_d = best_eval["by_dept"][d]
            l_d = p_d - b_d
            pl_d = (l_d / (b_d + 1e-5)) * 100
            dept_rows.append({
                "Department": d,
                "Baseline ($)": round(b_d, 2),
                "Target / Forecast ($)": round(p_d, 2),
                "Projected Lift ($)": round(l_d, 2),
                "Lift (%)": round(pl_d, 1),
                "Revenue Share (%)": round((p_d / best_pred) * 100, 1)
            })
        dept_breakdown_df = pd.DataFrame(dept_rows)
        
        # 3 Plan tiers
        plan_options = [
            {
                "Tier": "1. Conservative Plan",
                "Promo": "5% Store Discount",
                "Forecast ($)": round(evaluate_store_combo(0.05, 0, "Regular_Week")["total"], 2),
                "Staff": "+2 Store Associates",
                "Buffer": "+10% Stock Buffer"
            },
            {
                "Tier": "2. Recommended Optimal Plan",
                "Promo": f"{int(best_promo * 100)}% Store Discount",
                "Forecast ($)": round(best_pred, 2),
                "Staff": staff_req,
                "Buffer": buffer_req
            },
            {
                "Tier": "3. Aggressive Surge Plan",
                "Promo": "25% Discount + Holiday Event",
                "Forecast ($)": round(evaluate_store_combo(0.25, 1, "Thanksgiving_BlackFriday")["total"], 2),
                "Staff": "+8 Store Staff",
                "Buffer": "+40% Stock Buffer"
            }
        ]
        
        return {
            "is_store_wide": True,
            "store_id": store_id,
            "dept": "All Departments (Entire Store)",
            "baseline_sales": round(total_baseline, 2),
            "target_sales": round(target_sales, 2),
            "projected_sales": round(best_pred, 2),
            "gap": round(gap, 2),
            "pct_gap": round(pct_gap, 1),
            "pct_lift_achieved": round(pct_lift_achieved, 1),
            "recommended_promo_pct": int(best_promo * 100),
            "recommended_event": "Holiday Event Campaign" if best_is_hol else "Standard Operating Week",
            "staff_recommendation": staff_req,
            "buffer_recommendation": buffer_req,
            "feasibility": feasibility,
            "feasibility_color": feasibility_color,
            "feasibility_desc": feasibility_desc,
            "gross_sales": round(best_pred, 2),
            "cogs_est": round(cogs_est, 2),
            "discount_cost": round(discount_cost, 2),
            "labor_cost": round(labor_cost, 2),
            "net_profit": round(net_operating_profit, 2),
            "net_margin_pct": round(net_margin_pct, 1),
            "plans_df": pd.DataFrame(plan_options),
            "dept_breakdown_df": dept_breakdown_df
        }
        
    else:
        # Single department solver
        subset = raw_df[(raw_df["Store_ID"] == store_id) & (raw_df["Department"] == dept)].sort_values(by="Date")
        recent_sales = subset["Weekly_Sales"].tail(4).values
        baseline_sales = float(np.mean(recent_sales)) if len(recent_sales) > 0 else 25000.0
        store_size = subset["Store_Size_SqFt"].iloc[0] if len(subset) > 0 else 120000
        
        gap = target_sales - baseline_sales
        pct_gap = (gap / (baseline_sales + 1e-5)) * 100
        
        def evaluate_single_combo(promo_pct: float, is_hol: int, hol_name: str) -> float:
            row = {
                "Store_Size_SqFt": store_size,
                "Is_Holiday": is_hol,
                "Promotion_Discount": promo_pct,
                "Temperature": 65.0,
                "Fuel_Price": 3.45,
                "CPI": 245.0,
                "Unemployment_Rate": 5.5,
                "Year": dt_target.year,
                "Month": month_num,
                "Week_of_Year": int(week_num),
                "Quarter": dt_target.quarter,
                "Is_Month_End": int(dt_target.is_month_end),
                "Week_Sin": np.sin(2 * np.pi * week_num / 52.0),
                "Week_Cos": np.cos(2 * np.pi * week_num / 52.0),
                "Month_Sin": np.sin(2 * np.pi * month_num / 12.0),
                "Month_Cos": np.cos(2 * np.pi * month_num / 12.0),
                "Sales_Lag_1": recent_sales[-1] if len(recent_sales) >= 1 else 25000.0,
                "Sales_Lag_2": recent_sales[-2] if len(recent_sales) >= 2 else 24500.0,
                "Sales_Lag_4": recent_sales[0] if len(recent_sales) >= 4 else 24000.0,
                "Sales_Rolling_Mean_4": baseline_sales,
                "Sales_Rolling_Std_4": 1200.0,
                "Sales_Rolling_Mean_12": baseline_sales,
                "Sales_Momentum_Ratio": 1.0
            }
            for d in DEPARTMENTS: row[f"Dept_{d}"] = 1 if dept == d else 0
            for h in ["Christmas_Holiday", "Easter", "Labor_Day", "Regular_Week", "Super_Bowl", "Thanksgiving_BlackFriday"]:
                row[f"Holiday_{h}"] = 1 if hol_name == h else 0
            for s in STORES: row[f"Store_{s}"] = 1 if store_id == s else 0
            
            in_df = pd.DataFrame([row])
            for col in feature_names:
                if col not in in_df.columns:
                    in_df[col] = 0
            return float(model.predict(in_df[feature_names])[0])
        
        promo_steps = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
        best_promo = 0.0
        best_pred = evaluate_single_combo(0.0, 0, "Regular_Week")
        best_hol = "Regular_Week"
        best_is_hol = 0
        
        for p in promo_steps:
            pred_val = evaluate_single_combo(p, 0, "Regular_Week")
            if pred_val >= target_sales:
                best_promo = p
                best_pred = pred_val
                break
            elif pred_val > best_pred:
                best_promo = p
                best_pred = pred_val
                
        if best_pred < target_sales:
            for p in promo_steps:
                pred_val = evaluate_single_combo(p, 1, "Thanksgiving_BlackFriday")
                if pred_val >= target_sales:
                    best_promo = p
                    best_pred = pred_val
                    best_hol = "Thanksgiving_BlackFriday"
                    best_is_hol = 1
                    break
                elif pred_val > best_pred:
                    best_promo = p
                    best_pred = pred_val
                    best_hol = "Thanksgiving_BlackFriday"
                    best_is_hol = 1

        lift_achieved = best_pred - baseline_sales
        pct_lift_achieved = (lift_achieved / (baseline_sales + 1e-5)) * 100
        
        if pct_lift_achieved > 30:
            staff_req = "+4 Staff (2 Extra Cashiers + 2 Restockers)"
            labor_cost = 1400.0
        elif pct_lift_achieved > 15:
            staff_req = "+2 Staff (1 Extra Cashier + 1 Floor Associate)"
            labor_cost = 700.0
        elif pct_lift_achieved > 5:
            staff_req = "+1 Staff (Peak Hour Floor Support)"
            labor_cost = 350.0
        else:
            staff_req = "Standard Base Staffing (No extra labor required)"
            labor_cost = 0.0
            
        buffer_pct = max(10, int(round(pct_lift_achieved * 1.15)))
        buffer_req = f"+{buffer_pct}% Safety Stock Buffer"
        
        if best_pred >= target_sales and pct_gap <= 35:
            feasibility = "Highly Achievable"
            feasibility_color = "#10B981"
            feasibility_desc = "Goal can be met comfortably using targeted promotional discounts and standard staffing."
        elif best_pred >= target_sales * 0.90:
            feasibility = "Moderately Challenging"
            feasibility_color = "#F59E0B"
            feasibility_desc = "Requires aggressive discounting and a major holiday traffic boost to approach goal."
        else:
            feasibility = "Highly Ambitious / Stretch Goal"
            feasibility_color = "#DC2626"
            feasibility_desc = "Target is significantly above historical capacity (>50% lift). Consider pacing over multiple quarters."
            
        cogs_est = best_pred * 0.58
        discount_cost = best_pred * (best_promo / (1.0 - best_promo + 1e-5)) if best_promo > 0 else 0.0
        net_operating_profit = best_pred - cogs_est - discount_cost - labor_cost
        net_margin_pct = (net_operating_profit / (best_pred + 1e-5)) * 100
        
        plan_options = [
            {
                "Tier": "1. Conservative Plan",
                "Promo": "5% Discount",
                "Forecast ($)": round(evaluate_single_combo(0.05, 0, "Regular_Week"), 2),
                "Staff": "+1 Floor Staff",
                "Buffer": "+10% Stock"
            },
            {
                "Tier": "2. Recommended Optimal Plan",
                "Promo": f"{int(best_promo * 100)}% Discount",
                "Forecast ($)": round(best_pred, 2),
                "Staff": staff_req,
                "Buffer": buffer_req
            },
            {
                "Tier": "3. Aggressive Surge Plan",
                "Promo": "25% Discount + Holiday",
                "Forecast ($)": round(evaluate_single_combo(0.25, 1, "Thanksgiving_BlackFriday"), 2),
                "Staff": "+5 Staff",
                "Buffer": "+40% Stock"
            }
        ]
        
        return {
            "is_store_wide": False,
            "store_id": store_id,
            "dept": dept,
            "baseline_sales": round(baseline_sales, 2),
            "target_sales": round(target_sales, 2),
            "projected_sales": round(best_pred, 2),
            "gap": round(gap, 2),
            "pct_gap": round(pct_gap, 1),
            "pct_lift_achieved": round(pct_lift_achieved, 1),
            "recommended_promo_pct": int(best_promo * 100),
            "recommended_event": "Holiday Event Window" if best_is_hol else "Standard Promotional Week",
            "staff_recommendation": staff_req,
            "buffer_recommendation": buffer_req,
            "feasibility": feasibility,
            "feasibility_color": feasibility_color,
            "feasibility_desc": feasibility_desc,
            "gross_sales": round(best_pred, 2),
            "cogs_est": round(cogs_est, 2),
            "discount_cost": round(discount_cost, 2),
            "labor_cost": round(labor_cost, 2),
            "net_profit": round(net_operating_profit, 2),
            "net_margin_pct": round(net_margin_pct, 1),
            "plans_df": pd.DataFrame(plan_options),
            "dept_breakdown_df": None
        }


def generate_goal_seek_playbook_text(res: dict) -> str:
    """Generates a plain-text downloadable Target Execution Action Playbook for store leaders."""
    scope_str = f"Store {res['store_id']} - {res['dept']}"
    
    text = f"""================================================================================
RETAIL PULSE AI - TARGET REVENUE EXECUTION PLAYBOOK
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Target Scope: {scope_str}
Feasibility Rating: {res['feasibility']}

1. EXECUTIVE SUMMARY & TARGET GOAL
--------------------------------------------------------------------------------
- Historical 4-Week Baseline:  ${res['baseline_sales']:,.2f}
- User Target Revenue:         ${res['target_sales']:,.2f} ({res['pct_gap']:+.1f}% vs Baseline)
- AI Optimized Projection:     ${res['projected_sales']:,.2f} ({res['pct_lift_achieved']:+.1f}% Net Lift)
- Revenue Gap to Fill:         ${res['gap']:,.2f}
- Feasibility Assessment:      {res['feasibility_desc']}

2. OPERATIONAL REQUIREMENTS (TO HIT TARGET)
--------------------------------------------------------------------------------
- Recommended Promo Discount:  {res['recommended_promo_pct']}% Markdown
- Campaign Operating Window:   {res['recommended_event']}
- Labor Staffing Allocation:   {res['staff_recommendation']}
- Warehouse Inventory Buffer:  {res['buffer_recommendation']}

3. PROJECTED FINANCIAL CONTRIBUTION & MARGINS
--------------------------------------------------------------------------------
- Gross Revenue Achieved:      ${res['gross_sales']:,.2f} (100.0%)
- Estimated COGS (~58%):       -${res['cogs_est']:,.2f}
- Promotional Discount Cost:   -${res['discount_cost']:,.2f}
- Incremental Labor Cost:      -${res['labor_cost']:,.2f}
- Projected Net Profit:        ${res['net_profit']:,.2f}
- Net Operating Margin:        {res['net_margin_pct']:.1f}%

4. THREE-TIER STRATEGIC ACTION PLANS
--------------------------------------------------------------------------------
"""
    for _, r in res["plans_df"].iterrows():
        text += f"\n[{r['Tier'].upper()}]\n"
        text += f"  • Promo:      {r['Promo']}\n"
        text += f"  • Forecast:   ${r['Forecast ($)']:,.2f}\n"
        text += f"  • Staffing:   {r['Staff']}\n"
        text += f"  • Inventory:  {r['Buffer']}\n"
        
    text += f"""
================================================================================
END OF EXECUTION PLAYBOOK — RETAIL PULSE AI
================================================================================
"""
    return text
