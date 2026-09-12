"""
AI-Powered Executive Demand Briefing Engine.
Generates natural-language operational insights, stockout risk alerts,
labor scheduling recommendations, and commercial margin strategies.
"""

import pandas as pd
import numpy as np

def generate_executive_briefing(store_id: str, dept: str, predicted_sales: float, 
                                baseline_sales: float, p10_val: float, p90_val: float, 
                                promo_discount: float, is_holiday: bool, holiday_name: str,
                                persona: str = "Chief Executive (CFO / CEO)") -> dict:
    """
    Generates persona-tailored natural language executive briefings based on forecast telemetry.
    """
    lift_pct = ((predicted_sales - baseline_sales) / (baseline_sales + 1e-5)) * 100
    uncertainty_spread = p90_val - p10_val
    spread_pct = (uncertainty_spread / (predicted_sales + 1e-5)) * 100
    
    # 1. Headline Assessment
    if lift_pct >= 20:
        headline = f"🚀 HIGH DEMAND SURGE DETECTED (+{lift_pct:.1f}% vs Baseline)"
        risk_level = "High Stockout Risk"
        risk_color = "#DC2626"
    elif lift_pct >= 5:
        headline = f"📈 MODERATE GROWTH TRAJECTORY (+{lift_pct:.1f}% vs Baseline)"
        risk_level = "Balanced Demand"
        risk_color = "#059669"
    elif lift_pct >= -5:
        headline = f"⚖️ STABLE REVENUE CONTINUITY ({lift_pct:+.1f}% vs Baseline)"
        risk_level = "Low Volatility"
        risk_color = "#2563EB"
    else:
        headline = f"⚠️ DEMAND CONTRACTION WARNING ({lift_pct:.1f}% vs Baseline)"
        risk_level = "Overstock Risk"
        risk_color = "#D97706"

    # 2. Persona-Specific Narrative & Strategic Action Items
    if persona == "Chief Executive (CFO / CEO)":
        summary = (
            f"For **{store_id} ({dept})**, expected weekly revenue is projected at **${predicted_sales:,.2f}** "
            f"({lift_pct:+.1f}% variance against the recent 4-week benchmark of ${baseline_sales:,.2f}). "
            f"Demand uncertainty envelope spans from **${p10_val:,.2f}** (P10 minimum) to **${p90_val:,.2f}** (P90 surge ceiling)."
        )
        recommendations = [
            f"💰 **Revenue Capitalization**: Total expected contribution of **${predicted_sales:,.0f}** accounts for significant branch volume.",
            f"📊 **Promotional ROI**: Active discount of **{promo_discount*100:.0f}%** generates an incremental ~**${max(0, predicted_sales - baseline_sales):,.0f}** in gross top-line volume.",
            f"🛡️ **Financial Risk Hedge**: Maintain cash reserves to absorb variance spread of **${uncertainty_spread:,.0f}**."
        ]
        
    elif persona == "VP of Supply Chain & Logistics":
        summary = (
            f"Supply chain capacity evaluation for **{store_id} ({dept})**: Forecasted weekly volume requires throughput for "
            f"**${predicted_sales:,.2f}**. Surge risk threshold is calibrated at **${p90_val:,.2f}**."
        )
        buffer_units = round(max(0, (p90_val - predicted_sales) / 50))
        recommendations = [
            f"📦 **Safety Stock Target**: Maintain **+{max(15, round(lift_pct * 1.2))}%** safety stock buffer to cover P90 peak surge (${p90_val:,.0f}).",
            f"🚚 **Fulfillment Lead Time**: Trigger replenishment purchase orders **4 days earlier** if active holiday is '{holiday_name}'.",
            f"🏬 **Warehouse Allocation**: Staging area should accommodate roughly **~{buffer_units:,} extra unit equivalents**."
        ]
        
    else:  # Regional Store Manager
        summary = (
            f"Store operational briefing for **{store_id} - {dept}**: Scheduled week requires floor readiness for "
            f"**${predicted_sales:,.2f}** in sales."
        )
        extra_staff = "+3 to +4 staff" if lift_pct >= 25 else ("+2 staff" if lift_pct >= 10 else "Standard staffing")
        recommendations = [
            f"👥 **Floor Labor Scheduling**: Deploy **{extra_staff}** during peak afternoon shopping hours (12 PM - 6 PM).",
            f"🛒 **Merchandising Display**: Front-load promotional end-caps for **{dept}** items with active {promo_discount*100:.0f}% markdown tag.",
            f"⏱️ **Checkout Queue Management**: Open **+2 express cashier registers** if weekend traffic surges as predicted."
        ]

    return {
        "headline": headline,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "summary": summary,
        "recommendations": recommendations,
        "lift_pct": round(lift_pct, 1),
        "uncertainty_spread": round(uncertainty_spread, 2),
        "p10": round(p10_val, 2),
        "p50": round(predicted_sales, 2),
        "p90": round(p90_val, 2)
    }
