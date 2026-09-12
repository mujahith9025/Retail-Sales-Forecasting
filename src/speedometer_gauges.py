"""
Visual Inventory & Labor Speedometer Gauges Engine.
Generates publication-grade Plotly indicator speedometer gauges for:
1. Inventory Capacity & Stockout Risk Gauge
2. Labor Workload & Staffing Pressure Speedometer
3. Store Sales Demand Velocity Gauge
4. On-Shelf Availability & Fill Rate SLA Gauge
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any

try:
    from src.config import DEPARTMENTS, STORES
except (ImportError, ModuleNotFoundError):
    from config import DEPARTMENTS, STORES


def compute_operational_gauges(
    store_id: str,
    dept: str,
    predicted_sales: float,
    baseline_sales: float,
    promo_discount: float,
    is_holiday: bool,
    holiday_name: str
) -> Dict[str, Any]:
    """
    Calculates operational stress indices and generates 4 interactive Plotly speedometer gauges.
    """
    pct_lift = ((predicted_sales - baseline_sales) / (baseline_sales + 1e-5)) * 100
    
    # 1. Inventory Capacity & Stress Index (0 to 150%)
    inv_stress_index = min(150.0, max(20.0, 100.0 + (pct_lift * 0.95) + (promo_discount * 40.0)))
    
    if inv_stress_index >= 130:
        inv_status = "CRITICAL STOCKOUT RISK"
        inv_color = "#DC2626"
        inv_rec = f"+{int(pct_lift*1.2)}% Safety Stock Buffer Required Immediately"
        inv_alert_desc = "Warehouse depletion rate exceeds replenishment cycle. High probability of empty shelves on top SKUs."
    elif inv_stress_index >= 110:
        inv_status = "ELEVATED DEMAND"
        inv_color = "#F59E0B"
        inv_rec = f"+{int(pct_lift*1.1)}% Safety Stock Buffer Staged"
        inv_alert_desc = "Fast shelf turnover. Schedule mid-day backroom restock runs to avoid stockouts."
    elif inv_stress_index >= 60:
        inv_status = "OPTIMAL CAPACITY"
        inv_color = "#10B981"
        inv_rec = "Standard Safety Buffer (+10%)"
        inv_alert_desc = "Inventory velocity aligns with standard supplier delivery schedules."
    else:
        inv_status = "STAGNANT / OVERSTOCK"
        inv_color = "#64748B"
        inv_rec = "Reduce Reorder Quantities (-15%)"
        inv_alert_desc = "Slow inventory turnover. Consider promotional markdown to free working capital."

    fig_inv = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=inv_stress_index,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "<b>📦 Inventory Capacity & Stockout Risk</b><br><span style='font-size:0.75rem;color:#64748B;'>Normal = 100% | Critical Risk > 130%</span>", 'font': {'size': 14, 'color': '#0F172A'}},
        delta={'reference': 100.0, 'suffix': "% vs Normal", 'increasing': {'color': "#DC2626"}, 'decreasing': {'color': "#10B981"}},
        number={'suffix': "%", 'font': {'size': 26, 'color': '#0F172A', 'family': 'Plus Jakarta Sans'}},
        gauge={
            'axis': {'range': [0, 150], 'tickwidth': 1.5, 'tickcolor': "#475569", 'ticksuffix': "%"},
            'bar': {'color': inv_color, 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 1.5,
            'bordercolor': "#E2E8F0",
            'steps': [
                {'range': [0, 60], 'color': "rgba(100, 116, 139, 0.15)"},
                {'range': [60, 110], 'color': "rgba(16, 185, 129, 0.20)"},
                {'range': [110, 130], 'color': "rgba(245, 158, 11, 0.25)"},
                {'range': [130, 150], 'color': "rgba(220, 38, 38, 0.30)"}
            ],
            'threshold': {
                'line': {'color': "#DC2626", 'width': 3.5},
                'thickness': 0.85,
                'value': 130
            }
        }
    ))
    fig_inv.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=240, paper_bgcolor="rgba(0,0,0,0)")

    # 2. Labor Workload & Staffing Pressure Index (0 to 100)
    labor_index = min(100.0, max(15.0, 50.0 + (pct_lift * 0.90)))
    
    if labor_index >= 85:
        labor_status = "SEVERE OVERTIME / BURNOUT RISK"
        labor_color = "#DC2626"
        labor_rec = "+4 to +5 Extra Floor Staff (2 Cashiers + 2 Restockers)"
        labor_desc = "Cashier queues exceed 8 minutes. Severe risk of checkout bottlenecks and customer walkouts."
    elif labor_index >= 70:
        labor_status = "ELEVATED WORKLOAD"
        labor_color = "#F59E0B"
        labor_rec = "+2 Extra Floor Associates"
        labor_desc = "Peak traffic hours require dedicated restocking and front-end queue management."
    elif labor_index >= 35:
        labor_status = "BALANCED STAFFING"
        labor_color = "#10B981"
        labor_rec = "Standard Staffing Roster"
        labor_desc = "Associate labor hours match customer foot-traffic flow comfortably."
    else:
        labor_status = "LOW TRAFFIC / OVERSTAFFED"
        labor_color = "#3B82F6"
        labor_rec = "Optimize Shifts (-1 Floor Staff)"
        labor_desc = "Customer volume is light. Reassign associates to inventory audits or merchandising."

    fig_labor = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=labor_index,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "<b>👥 Labor Workload & Staffing Pressure</b><br><span style='font-size:0.75rem;color:#64748B;'>Optimal = 50 | Overtime Surge > 75</span>", 'font': {'size': 14, 'color': '#0F172A'}},
        delta={'reference': 50.0, 'suffix': " pts", 'increasing': {'color': "#DC2626"}, 'decreasing': {'color': "#10B981"}},
        number={'suffix': "/100", 'font': {'size': 26, 'color': '#0F172A', 'family': 'Plus Jakarta Sans'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1.5, 'tickcolor': "#475569"},
            'bar': {'color': labor_color, 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 1.5,
            'bordercolor': "#E2E8F0",
            'steps': [
                {'range': [0, 35], 'color': "rgba(59, 130, 246, 0.15)"},
                {'range': [35, 70], 'color': "rgba(16, 185, 129, 0.20)"},
                {'range': [70, 85], 'color': "rgba(245, 158, 11, 0.25)"},
                {'range': [85, 100], 'color': "rgba(220, 38, 38, 0.30)"}
            ],
            'threshold': {
                'line': {'color': "#DC2626", 'width': 3.5},
                'thickness': 0.85,
                'value': 85
            }
        }
    ))
    fig_labor.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=240, paper_bgcolor="rgba(0,0,0,0)")

    # 3. Store Sales Demand Velocity ($/week)
    max_gauge_sales = max(50000.0, float(round(baseline_sales * 2.0, -3)))
    
    fig_sales = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=predicted_sales,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "<b>🚀 Weekly Sales Demand Velocity</b><br><span style='font-size:0.75rem;color:#64748B;'>Forecast Run-Rate vs Baseline</span>", 'font': {'size': 14, 'color': '#0F172A'}},
        delta={'reference': baseline_sales, 'valueformat': "$,.0f", 'increasing': {'color': "#10B981"}, 'decreasing': {'color': "#DC2626"}},
        number={'prefix': "$", 'valueformat': ",.0f", 'font': {'size': 26, 'color': '#0F172A', 'family': 'Plus Jakarta Sans'}},
        gauge={
            'axis': {'range': [0, max_gauge_sales], 'tickwidth': 1.5, 'tickcolor': "#475569", 'tickprefix': "$"},
            'bar': {'color': "#2563EB", 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 1.5,
            'bordercolor': "#E2E8F0",
            'steps': [
                {'range': [0, baseline_sales * 0.8], 'color': "rgba(100, 116, 139, 0.15)"},
                {'range': [baseline_sales * 0.8, baseline_sales * 1.15], 'color': "rgba(37, 99, 235, 0.20)"},
                {'range': [baseline_sales * 1.15, max_gauge_sales], 'color': "rgba(16, 185, 129, 0.25)"}
            ],
            'threshold': {
                'line': {'color': "#10B981", 'width': 3.5},
                'thickness': 0.85,
                'value': baseline_sales
            }
        }
    ))
    fig_sales.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=240, paper_bgcolor="rgba(0,0,0,0)")

    # 4. On-Shelf Availability & Fill Rate Confidence (%)
    # In high demand surges without buffer, fill rate drops from 99.5% down to 88%
    if inv_stress_index > 130:
        fill_rate = max(86.0, 99.5 - ((inv_stress_index - 130) * 0.65))
    elif inv_stress_index > 110:
        fill_rate = 96.5
    else:
        fill_rate = 99.2
        
    fill_color = "#10B981" if fill_rate >= 98.0 else ("#F59E0B" if fill_rate >= 94.0 else "#DC2626")

    fig_fill = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=fill_rate,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "<b>🛡️ On-Shelf Availability & Fill Rate</b><br><span style='font-size:0.75rem;color:#64748B;'>Target SLA = 98.0% Minimum</span>", 'font': {'size': 14, 'color': '#0F172A'}},
        delta={'reference': 98.0, 'suffix': "% SLA", 'increasing': {'color': "#10B981"}, 'decreasing': {'color': "#DC2626"}},
        number={'suffix': "%", 'valueformat': ".1f", 'font': {'size': 26, 'color': '#0F172A', 'family': 'Plus Jakarta Sans'}},
        gauge={
            'axis': {'range': [80.0, 100.0], 'tickwidth': 1.5, 'tickcolor': "#475569", 'ticksuffix': "%"},
            'bar': {'color': fill_color, 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 1.5,
            'bordercolor': "#E2E8F0",
            'steps': [
                {'range': [80.0, 92.0], 'color': "rgba(220, 38, 38, 0.25)"},
                {'range': [92.0, 97.0], 'color': "rgba(245, 158, 11, 0.25)"},
                {'range': [97.0, 100.0], 'color': "rgba(16, 185, 129, 0.25)"}
            ],
            'threshold': {
                'line': {'color': "#10B981", 'width': 3.5},
                'thickness': 0.85,
                'value': 98.0
            }
        }
    ))
    fig_fill.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=240, paper_bgcolor="rgba(0,0,0,0)")

    return {
        "inv_stress_index": round(inv_stress_index, 1),
        "inv_status": inv_status,
        "inv_color": inv_color,
        "inv_rec": inv_rec,
        "inv_desc": inv_alert_desc,
        "fig_inv": fig_inv,
        "labor_index": round(labor_index, 1),
        "labor_status": labor_status,
        "labor_color": labor_color,
        "labor_rec": labor_rec,
        "labor_desc": labor_desc,
        "fig_labor": fig_labor,
        "predicted_sales": round(predicted_sales, 2),
        "baseline_sales": round(baseline_sales, 2),
        "pct_lift": round(pct_lift, 1),
        "fig_sales": fig_sales,
        "fill_rate": round(fill_rate, 1),
        "fill_color": fill_color,
        "fig_fill": fig_fill
    }
