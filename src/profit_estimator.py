"""
Profit & Operating Margin Estimator Engine.
Provides comprehensive financial P&L modeling, cost of goods sold (COGS) decomposition,
discount elasticity sensitivity curves, break-even analysis, and net operating margin optimization.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List

from src.config import DEPARTMENTS, STORES


DEPARTMENT_COST_PROFILES = {
    "Grocery": {
        "cogs_pct": 68.0,
        "base_margin_pct": 32.0,
        "labor_intensity": 1.15,
        "shrinkage_pct": 1.5,
        "description": "High-velocity essential staples with tight wholesale margins (32%) and moderate markdown resilience."
    },
    "Electronics": {
        "cogs_pct": 60.0,
        "base_margin_pct": 40.0,
        "labor_intensity": 1.0,
        "shrinkage_pct": 1.2,
        "description": "High ticket items with healthy 40% margin and strong volume surge during holiday sales."
    },
    "Apparel": {
        "cogs_pct": 42.0,
        "base_margin_pct": 58.0,
        "labor_intensity": 1.1,
        "shrinkage_pct": 2.0,
        "description": "High gross margin category (58%) capable of absorbing 25-30% clearance markdowns profitably."
    },
    "Home_Garden": {
        "cogs_pct": 48.0,
        "base_margin_pct": 52.0,
        "labor_intensity": 0.95,
        "shrinkage_pct": 1.4,
        "description": "Balanced 52% gross margin with strong spring/summer seasonal volume surges."
    },
    "Pharmacy": {
        "cogs_pct": 55.0,
        "base_margin_pct": 45.0,
        "labor_intensity": 1.25,
        "shrinkage_pct": 0.8,
        "description": "Resilient inelastic healthcare staples with steady 45% margin and low promotional sensitivity."
    }
}


def compute_profit_and_loss(
    gross_sales: float,
    dept: str = "Grocery",
    promo_discount: float = 0.0,
    custom_cogs_pct: float = None,
    hourly_labor_rate: float = 18.50,
    extra_staff_count: int = 0,
    opex_pct: float = 8.0
) -> Dict[str, Any]:
    """
    Computes a detailed financial P&L statement, net operating profit, and break-even metrics.
    """
    profile = DEPARTMENT_COST_PROFILES.get(dept, DEPARTMENT_COST_PROFILES["Grocery"])
    cogs_base_pct = custom_cogs_pct if custom_cogs_pct is not None else profile["cogs_pct"]
    
    # 1. Revenue & Discount Breakdown
    # If promo discount was given, gross list value was higher:
    gross_list_val = gross_sales / (1.0 - promo_discount + 1e-5) if promo_discount > 0 else gross_sales
    discount_cost = gross_list_val - gross_sales
    net_sales = gross_sales
    
    # 2. Direct Cost of Goods Sold (COGS) based on base unit wholesale cost
    cogs_amount = gross_list_val * (cogs_base_pct / 100.0) * 0.90
    shrinkage_amount = net_sales * (profile["shrinkage_pct"] / 100.0)
    total_cogs = cogs_amount + shrinkage_amount
    gross_profit = net_sales - total_cogs
    gross_margin_pct = (gross_profit / (net_sales + 1e-5)) * 100
    
    # 3. Store Labor Costs (Base 110 hrs/week per dept + extra staff * 35 hrs/wk)
    base_labor_hours = 110.0 * profile["labor_intensity"]
    extra_labor_hours = extra_staff_count * 35.0
    total_labor_hours = base_labor_hours + extra_labor_hours
    labor_cost = total_labor_hours * hourly_labor_rate
    
    # 4. Store Fixed Overhead / OPEX Allocation (Utilities, rent, POS ~8%)
    fixed_opex = net_sales * (opex_pct / 100.0)
    
    # 5. Net Operating Profit & Margin (EBITDA)
    total_operating_costs = total_cogs + labor_cost + fixed_opex
    net_operating_profit = net_sales - total_operating_costs
    net_margin_pct = (net_operating_profit / (net_sales + 1e-5)) * 100
    
    # 6. Break-Even Sales ($ needed to cover labor + fixed opex at current gross margin)
    effective_gm_ratio = max(0.05, gross_profit / (net_sales + 1e-5))
    fixed_costs = labor_cost + fixed_opex
    break_even_sales = fixed_costs / effective_gm_ratio
    
    # 7. Margin Health Rating
    if net_margin_pct >= 20.0:
        margin_grade = "A+ (Elite Profitability)"
        margin_color = "#10B981"
        margin_badge = "🟢 High Profit Margin"
    elif net_margin_pct >= 14.0:
        margin_grade = "A (Healthy & Robust)"
        margin_color = "#059669"
        margin_badge = "🟢 Solid Margin"
    elif net_margin_pct >= 8.0:
        margin_grade = "B (Moderate / Stable)"
        margin_color = "#3B82F6"
        margin_badge = "🔵 Moderate Margin"
    elif net_margin_pct >= 0.0:
        margin_grade = "C (Narrow / Risk of Erosion)"
        margin_color = "#F59E0B"
        margin_badge = "🟡 Thin Margin"
    else:
        margin_grade = "F (Loss-Making / Negative Margin)"
        margin_color = "#DC2626"
        margin_badge = "🔴 Unprofitable"

    return {
        "dept": dept,
        "gross_list_val": round(gross_list_val, 2),
        "gross_sales": round(gross_sales, 2),
        "discount_cost": round(discount_cost, 2),
        "promo_discount_pct": round(promo_discount * 100, 1),
        "net_sales": round(net_sales, 2),
        "cogs_amount": round(cogs_amount, 2),
        "cogs_pct": round(cogs_base_pct, 1),
        "shrinkage_amount": round(shrinkage_amount, 2),
        "total_cogs": round(total_cogs, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_margin_pct": round(gross_margin_pct, 1),
        "labor_cost": round(labor_cost, 2),
        "total_labor_hours": round(total_labor_hours, 1),
        "fixed_opex": round(fixed_opex, 2),
        "opex_pct": round(opex_pct, 1),
        "total_operating_costs": round(total_operating_costs, 2),
        "net_operating_profit": round(net_operating_profit, 2),
        "net_margin_pct": round(net_margin_pct, 1),
        "break_even_sales": round(break_even_sales, 2),
        "margin_grade": margin_grade,
        "margin_color": margin_color,
        "margin_badge": margin_badge,
        "cogs_desc": profile["description"]
    }


def generate_financial_waterfall_chart(pl: Dict[str, Any]) -> go.Figure:
    """Creates a visual financial P&L waterfall chart decomposing revenue into net profit."""
    wf_labels = [
        "Gross List Value",
        "Promotional Markdowns",
        "Cost of Goods (COGS)",
        "Store Labor Costs",
        "Overhead (OPEX)",
        "Net Operating Profit"
    ]
    wf_values = [
        pl["gross_list_val"],
        -pl["discount_cost"],
        -pl["total_cogs"],
        -pl["labor_cost"],
        -pl["fixed_opex"],
        pl["net_operating_profit"]
    ]
    wf_measures = ["absolute", "relative", "relative", "relative", "relative", "total"]
    
    text_labels = [
        f"${pl['gross_list_val']:,.0f}",
        f"-${pl['discount_cost']:,.0f}" if pl['discount_cost']>0 else "$0",
        f"-${pl['total_cogs']:,.0f}",
        f"-${pl['labor_cost']:,.0f}",
        f"-${pl['fixed_opex']:,.0f}",
        f"${pl['net_operating_profit']:,.0f}"
    ]
    
    fig = go.Figure(go.Waterfall(
        name="Financial P&L Waterfall",
        orientation="v",
        measure=wf_measures,
        x=wf_labels,
        y=wf_values,
        textposition="outside",
        text=text_labels,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#DC2626"}},
        increasing={"marker": {"color": "#10B981"}},
        totals={"marker": {"color": "#2563EB" if pl["net_operating_profit"] > 0 else "#DC2626"}}
    ))
    fig.update_layout(
        title="Weekly Financial P&L Waterfall Decomposition ($)",
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        height=340
    )
    return fig


def simulate_discount_elasticity_curve(
    baseline_sales: float,
    dept: str = "Grocery",
    custom_cogs_pct: float = None,
    opex_pct: float = 8.0
) -> pd.DataFrame:
    """
    Simulates revenue and net profit across a 0% to 40% discount ladder
    to find the profit-maximizing optimal discount rate.
    """
    try:
        baseline_sales = float(baseline_sales)
    except (ValueError, TypeError):
        baseline_sales = 25000.0
        
    discounts = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    
    elasticity_multipliers = {
        "Apparel": 2.2,
        "Electronics": 1.8,
        "Home_Garden": 1.6,
        "Grocery": 1.3,
        "Pharmacy": 0.9
    }
    el_mult = elasticity_multipliers.get(dept, 1.5)
    
    records = []
    for d in discounts:
        lift = 1.0 + (d * el_mult)
        sim_gross = baseline_sales * lift
        staff_add = 2 if d >= 0.25 else (1 if d >= 0.15 else 0)
        
        pl = compute_profit_and_loss(
            gross_sales=sim_gross,
            dept=dept,
            promo_discount=d,
            custom_cogs_pct=custom_cogs_pct,
            extra_staff_count=staff_add,
            opex_pct=opex_pct
        )
        records.append({
            "Discount (%)": int(d * 100),
            "Discount_Rate": d,
            "Gross Revenue ($)": pl["gross_sales"],
            "Revenue Lift (%)": round((lift - 1.0) * 100, 1),
            "COGS ($)": pl["total_cogs"],
            "Labor ($)": pl["labor_cost"],
            "Net Profit ($)": pl["net_operating_profit"],
            "Net Margin (%)": pl["net_margin_pct"],
            "Is_Profitable": pl["net_operating_profit"] > 0
        })
        
    df = pd.DataFrame(records)
    best_row = df.loc[df["Net Profit ($)"].idxmax()]
    df["Is_Optimal"] = (df["Discount (%)"] == best_row["Discount (%)"])
    return df


def plot_discount_elasticity_curve(df_curve: pd.DataFrame) -> go.Figure:
    """Plots the dual-axis Gross Revenue vs Net Operating Profit elasticity curve."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_curve["Discount (%)"].map("{}%".format),
        y=df_curve["Gross Revenue ($)"],
        name="Gross Revenue ($)",
        marker_color="rgba(37, 99, 235, 0.75)"
    ))
    
    fig.add_trace(go.Scatter(
        x=df_curve["Discount (%)"].map("{}%".format),
        y=df_curve["Net Profit ($)"],
        name="Net Operating Profit ($)",
        mode="lines+markers",
        line=dict(color="#10B981", width=3.5),
        marker=dict(size=8, color="#059669"),
        yaxis="y2"
    ))
    
    fig.add_trace(go.Scatter(
        x=df_curve["Discount (%)"].map("{}%".format),
        y=df_curve["Net Margin (%)"],
        name="Net Margin (%)",
        mode="lines",
        line=dict(color="#F59E0B", width=2, dash="dash"),
        yaxis="y2"
    ))
    
    fig.update_layout(
        title="Promotional Discount Elasticity: Revenue vs Net Profit Sweet Spot",
        template="plotly_white",
        yaxis=dict(title="Gross Sales Revenue ($)", showgrid=True),
        yaxis2=dict(title="Net Operating Profit ($) / Margin (%)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        height=340,
        hovermode="x unified"
    )
    return fig


def export_financial_statement_text(pl: Dict[str, Any], df_curve: pd.DataFrame) -> str:
    """Generates a downloadable plain-text Executive Financial P&L Statement."""
    best_row = df_curve.loc[df_curve["Net Profit ($)"].idxmax()]
    
    text = f"""================================================================================
RETAIL PULSE AI - EXECUTIVE FINANCIAL P&L AND OPERATING MARGIN STATEMENT
================================================================================
Department: {pl['dept']}
Active Discount: {pl['promo_discount_pct']}% Markdown
Financial Health Grade: {pl['margin_grade']}

1. EXECUTIVE REVENUE & PROFIT SUMMARY
--------------------------------------------------------------------------------
- Gross List Value:               ${pl['gross_list_val']:,.2f}
- Promotional Markdowns:         -${pl['discount_cost']:,.2f} ({pl['promo_discount_pct']}%)
- Net Register Sales:             ${pl['net_sales']:,.2f}
- Cost of Goods Sold (COGS):     -${pl['total_cogs']:,.2f} ({pl['cogs_pct']}% base wholesale)
- Gross Profit:                   ${pl['gross_profit']:,.2f} (Gross Margin: {pl['gross_margin_pct']:.1f}%)
- Store Floor Labor:             -${pl['labor_cost']:,.2f} ({pl['total_labor_hours']:.1f} Hours)
- Store Overhead (OPEX ~{pl['opex_pct']}%):    -${pl['fixed_opex']:,.2f}
--------------------------------------------------------------------------------
= NET OPERATING PROFIT (EBITDA):  ${pl['net_operating_profit']:,.2f}
= NET OPERATING MARGIN:           {pl['net_margin_pct']:.1f}%

2. BREAK-EVEN & RISK THRESHOLDS
--------------------------------------------------------------------------------
- Break-Even Weekly Sales:        ${pl['break_even_sales']:,.2f}
- Margin Safety Buffer:           ${(pl['net_sales'] - pl['break_even_sales']):,.2f} ({((pl['net_sales'] - pl['break_even_sales'])/pl['net_sales'])*100:.1f}%)

3. PROMOTIONAL DISCOUNT ELASTICITY & SWEET SPOT
--------------------------------------------------------------------------------
- Profit-Maximizing Discount:     {best_row['Discount (%)']}% Markdown
- Maximum Achievable Net Profit:  ${best_row['Net Profit ($)']:,.2f} ({best_row['Net Margin (%)']:.1f}% Margin)
- Recommendation:                 {pl['cogs_desc']}

================================================================================
END OF FINANCIAL STATEMENT - RETAIL PULSE AI
================================================================================
"""
    return text
