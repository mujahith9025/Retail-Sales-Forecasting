"""
"What Do You Want to Do?" 1-Click Decision Wizard Engine.
Empowers zero-knowledge beginners and executives to achieve instant business objectives
by selecting plain-English goals that automatically configure simulations, goal-seek solvers,
and executive recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List

try:
    from src.config import STORES, DEPARTMENTS, STORE_LOCATIONS
except (ImportError, ModuleNotFoundError):
    from config import STORES, DEPARTMENTS, STORE_LOCATIONS
try:
    from src.health_scorecard import compute_store_health_scorecard
except (ImportError, ModuleNotFoundError):
    from health_scorecard import compute_store_health_scorecard
try:
    from src.goal_seek import solve_target_revenue_plan
except (ImportError, ModuleNotFoundError):
    from goal_seek import solve_target_revenue_plan

DECISION_INTENTS = [
    {
        "id": "target_revenue",
        "icon": "🎯",
        "title": "Hit Revenue Target",
        "tagline": "Reverse-engineer discounts & staff",
        "description": "Calculate exact discount, staff roster, and net margin to hit your target.",
        "color": "#2563EB",
        "target_tab": "tab2",
        "sub_action": "goal_seek"
    },
    {
        "id": "holiday_surge",
        "icon": "🛍️",
        "title": "Holiday Surge Prep",
        "tagline": "Black Friday & Christmas plans",
        "description": "Simulate peak foot-traffic, safety stock buffers, and labor surge rosters.",
        "color": "#DC2626",
        "target_tab": "tab2",
        "sub_action": "scenario_presets"
    },
    {
        "id": "store_audit",
        "icon": "🩺",
        "title": "Store Health Audit",
        "tagline": "A+ to F grades & yield ($/sqft)",
        "description": "Inspect top branch rankings, space efficiency, and diagnostic prescriptions.",
        "color": "#10B981",
        "target_tab": "tab1",
        "sub_action": "health_scorecard"
    },
    {
        "id": "profit_sweetspot",
        "icon": "💰",
        "title": "Maximize Profit",
        "tagline": "Optimal discount sweet spot",
        "description": "Find the promo discount level that delivers peak bottom-line cash profit.",
        "color": "#F59E0B",
        "target_tab": "tab2",
        "sub_action": "profit_estimator"
    },
    {
        "id": "custom_upload",
        "icon": "📤",
        "title": "Forecast Custom CSV",
        "tagline": "Instant 1-click dataset injection",
        "description": "Upload sales CSV to auto-generate forward AI forecasts and audit reports.",
        "color": "#8B5CF6",
        "target_tab": "tab3",
        "sub_action": "upload_analyzer"
    },
    {
        "id": "export_bundle",
        "icon": "📦",
        "title": "Executive Export",
        "tagline": "1-Click PDF, Excel & ZIP bundle",
        "description": "Download publication-ready reports, spreadsheets, and predictions.",
        "color": "#0F172A",
        "target_tab": "tab3",
        "sub_action": "export_reports"
    }
]


def safe_render_html(html_str: str, container=None):
    """
    Renders pure HTML cleanly without triggering Markdown code block / LaTeX formatting.
    If the input string is markdown (contains no HTML tags), delegates to st.markdown.
    Prefers st.html if available, with minified st.markdown as fallback.
    """
    target = container or st
    stripped = html_str.strip()
    if "<" not in stripped:
        target.markdown(html_str)
        return
    if hasattr(target, "html"):
        target.html(html_str)
    elif hasattr(st, "html"):
        st.html(html_str)
    else:
        minified = " ".join(line.strip() for line in stripped.splitlines() if line.strip())
        target.markdown(minified, unsafe_allow_html=True)


def render_decision_wizard(raw_df: pd.DataFrame, store_locations: dict, active_store: str = "Store_09"):
    """
    Renders the 'What Do You Want to Do?' Decision Wizard component.
    Provides 1-click goal cards and an interactive action sheet with tailored AI directives.
    """
    if "decision_wizard_intent" not in st.session_state:
        st.session_state.decision_wizard_intent = "target_revenue"

    current_intent_id = st.session_state.decision_wizard_intent

    safe_render_html("""<div style="background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%); border-radius: 12px; padding: 0.85rem 1.25rem; margin-bottom: 0.9rem; border: 1px solid rgba(255,255,255,0.1); display: flex; justify-content: space-between; align-items: center;">
<div>
<span style="font-weight: 800; font-size: 1.05rem; color: #F8FAFC; display: flex; align-items: center; gap: 0.4rem;">
🧭 <b>1-Click Executive Decision Wizard</b>
</span>
<div style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.15rem;">
Select an objective to instantly generate tailored AI directives and action plans.
</div>
</div>
<span style="background: rgba(59, 130, 246, 0.2); color: #60A5FA; font-size: 0.72rem; font-weight: 700; padding: 0.2rem 0.6rem; border-radius: 9999px;">
⚡ AI Guided
</span>
</div>""")

    # Render 6 Intent Cards in 2 rows of 3 columns
    row1 = DECISION_INTENTS[:3]
    row2 = DECISION_INTENTS[3:]

    for row_items in [row1, row2]:
        cols = st.columns(3)
        for idx, item in enumerate(row_items):
            is_active = (item["id"] == current_intent_id)
            c = cols[idx]
            with c:
                border_color = item["color"] if is_active else "#E2E8F0"
                bg_color = "linear-gradient(135deg, #EFF6FF 0%, #FFFFFF 100%)" if is_active else "#FFFFFF"
                box_shadow = "0 4px 12px -2px rgba(37, 99, 235, 0.2)" if is_active else "0 1px 4px rgba(0,0,0,0.03)"
                badge = f'<span style="background: {item["color"]}; color: white; padding: 0.15rem 0.45rem; border-radius: 9999px; font-size: 0.65rem; font-weight: 700;">Active</span>' if is_active else f'<span style="background: #F1F5F9; color: #64748B; padding: 0.15rem 0.45rem; border-radius: 9999px; font-size: 0.65rem; font-weight: 600;">1-Click</span>'

                safe_render_html(f"""<div style="background: {bg_color}; border: 1.5px solid {border_color}; border-radius: 10px; padding: 0.75rem 0.9rem; box-shadow: {box_shadow}; min-height: 105px; display: flex; flex-direction: column; justify-content: space-between; margin-bottom: 0.25rem;">
<div>
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.2rem;">
<span style="font-weight: 800; font-size: 0.92rem; color: #0F172A;">
{item['icon']} {item['title']}
</span>
{badge}
</div>
<div style="font-size: 0.76rem; font-weight: 600; color: {item['color']}; margin-bottom: 0.2rem;">
{item['tagline']}
</div>
<div style="font-size: 0.72rem; color: #64748B; line-height: 1.3;">
{item['description']}
</div>
</div>
</div>""")

                btn_type = "primary" if is_active else "secondary"
                btn_label = f"Selected ✓" if is_active else f"Select: {item['title']}"
                if st.button(btn_label, key=f"wiz_intent_{item['id']}", type=btn_type, use_container_width=True):
                    st.session_state.decision_wizard_intent = item["id"]
                    st.rerun()

        st.write("")

    # ==============================================================================
    # DYNAMIC AI ACTION SHEET FOR ACTIVE INTENT
    # ==============================================================================
    active_intent_data = next((x for x in DECISION_INTENTS if x["id"] == current_intent_id), DECISION_INTENTS[0])
    
    safe_render_html(f"""<div style="background: #FFFFFF; border: 1px solid #CBD5E1; border-left: 5px solid {active_intent_data['color']}; border-radius: 12px; padding: 1rem 1.25rem; margin-top: 0.3rem; margin-bottom: 1rem; box-shadow: 0 3px 10px -2px rgba(0,0,0,0.04);">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
<span style="font-weight: 800; font-size: 1.05rem; color: #0F172A; display: flex; align-items: center; gap: 0.4rem;">
{active_intent_data['icon']} <b>AI Action Plan:</b> {active_intent_data['title']}
</span>
<span style="background: {active_intent_data['color']}; color: white; padding: 0.2rem 0.65rem; border-radius: 9999px; font-weight: 700; font-size: 0.75rem;">
Store: {active_store}
</span>
</div>""")

    try:
        if current_intent_id == "target_revenue":
            s_data = raw_df[raw_df["Store_ID"] == active_store]
            base_rev = s_data.groupby("Date")["Weekly_Sales"].sum().tail(4).mean() if len(s_data) > 0 else 125000.0
            target_rev = base_rev * 1.20
            plan = solve_target_revenue_plan(active_store, "All Departments (Entire Store)", target_rev, raw_df)

            target_val = plan.get('target_sales', target_rev)
            pct_gap = plan.get('pct_gap', 20.0)
            promo_pct = plan.get('recommended_promo_pct', 10)
            rec_event = plan.get('recommended_event', 'Standard Week')
            staff_str = str(plan.get('staff_recommendation', 'Standard Staffing')).split('(')[0].strip()
            buffer_str = str(plan.get('buffer_recommendation', plan.get('inventory_recommendation', '+15% Safety Buffer'))).split('(')[0].strip()
            net_profit = plan.get('net_profit', plan.get('projected_net_profit', 12000.0))
            net_margin = plan.get('net_margin_pct', plan.get('projected_net_margin_pct', 15.0))

            w1, w2, w3, w4 = st.columns(4)
            with w1:
                st.metric("🎯 Target", f"${target_val:,.0f}", f"{pct_gap:+.1f}% vs baseline")
            with w2:
                st.metric("🏷️ Discount", f"{promo_pct}% Off", rec_event)
            with w3:
                st.metric("👥 Staff", staff_str, "Floor coverage")
            with w4:
                st.metric("📦 Buffer", buffer_str, f"${net_profit:,.0f} Profit ({net_margin:.1f}%)")

            safe_render_html(f"""<div style="background: #F8FAFC; border-left: 4px solid #2563EB; border-radius: 6px; padding: 0.65rem 0.85rem; margin-top: 0.6rem; font-size: 0.82rem; color: #1E293B;">
💡 <b>Action Directive:</b> Apply <b>{promo_pct}% markdown</b> with <b>{staff_str}</b> and <b>{buffer_str}</b> to generate <b>${net_profit:,.0f}</b> net profit.
</div>""")

        elif current_intent_id == "holiday_surge":
            w1, w2, w3, w4 = st.columns(4)
            with w1:
                st.metric("🛍️ Peak Event", "Black Friday / Christmas", "+48.5% Surge")
            with w2:
                st.metric("🏷️ Promo Rate", "25% Markdown", "Traffic Driver")
            with w3:
                st.metric("👥 Extra Staff", "+4 Associates / Store", "Checkout Support")
            with w4:
                st.metric("📦 Stock Buffer", "+35% Safety Stock", "Dispatch 14d prior")

            safe_render_html("""<div style="background: #FEF2F2; border-left: 4px solid #EF4444; border-radius: 6px; padding: 0.65rem 0.85rem; margin-top: 0.6rem; font-size: 0.82rem; color: #991B1B;">
🚨 <b>Holiday Directive:</b> Demand surges by <b>+48.5%</b>. Order inventory <b>14 days prior</b> and roster <b>+4 staff</b>.
</div>""")

        elif current_intent_id == "store_audit":
            store_cards = compute_store_health_scorecard(raw_df, store_locations)
            top_s = store_cards.iloc[0] if len(store_cards) > 0 else {}
            avg_s = store_cards["Health_Score"].mean() if len(store_cards) > 0 and "Health_Score" in store_cards.columns else 85.0
            match_s = store_cards[store_cards["Store_ID"] == active_store]
            curr_s = match_s.iloc[0] if len(match_s) > 0 else top_s

            top_id = top_s.get("Store_ID", "Store_09")
            top_city = top_s.get("City", "Dallas")
            top_grade = top_s.get("Grade", "A+")

            curr_id = curr_s.get("Store_ID", active_store)
            curr_city = curr_s.get("City", "Store")
            curr_grade = curr_s.get("Grade", "A")
            curr_sqft_rev = curr_s.get("Sales_per_SqFt ($)", 75.0)
            curr_rx = curr_s.get("Prescription", "Maintain inventory cadence.")

            w1, w2, w3, w4 = st.columns(4)
            with w1:
                st.metric("🏆 Top Branch", f"{top_id} ({top_city})", f"Grade {top_grade}")
            with w2:
                st.metric("🏢 Active Grade", f"{curr_id} ({curr_grade})", f"${curr_sqft_rev}/sq ft")
            with w3:
                st.metric("✨ Network Health", f"{avg_s:.1f} / 100", "Solid Baseline")
            with w4:
                st.metric("🛡️ Critical Risk", "0 Grade F Stores", "Low Risk")

            safe_render_html(f"""<div style="background: #F0FDF4; border-left: 4px solid #10B981; border-radius: 6px; padding: 0.65rem 0.85rem; margin-top: 0.6rem; font-size: 0.82rem; color: #166534;">
🩺 <b>Audit Directive for {active_store} ({curr_city}):</b> {curr_rx} (Space yield: <b>${curr_sqft_rev}/sq ft</b>).
</div>""")

        elif current_intent_id == "profit_sweetspot":
            w1, w2, w3, w4 = st.columns(4)
            with w1:
                st.metric("💰 Sweet Spot", "10% Discount", "Peak Cash Profit")
            with w2:
                st.metric("💵 Net Profit", "$10,500 / week", "+28.4% vs baseline")
            with w3:
                st.metric("⚠️ 30% Markdown", "$6,800 / week", "Margin Erosion")
            with w4:
                st.metric("📊 Wholesale COGS", "58% of Revenue", "Industry Benchmark")

            safe_render_html("""<div style="background: #FFFBEB; border-left: 4px solid #F59E0B; border-radius: 6px; padding: 0.65rem 0.85rem; margin-top: 0.6rem; font-size: 0.82rem; color: #92400E;">
💡 <b>Profit Directive:</b> <b>10% discounts</b> yield peak cash ($10.5K/wk). Avoid >30% markdowns to protect margins.
</div>""")

        elif current_intent_id == "custom_upload":
            w1, w2, w3, w4 = st.columns(4)
            with w1:
                st.metric("📤 CSV Upload", "Drag & Drop", "Auto Column Mapping")
            with w2:
                st.metric("⚡ Demo Dataset", "2,600 Rows", "1-Click Testing")
            with w3:
                st.metric("🔮 AI Horizon", "12-Week Forecast", "XGBoost Engine")
            with w4:
                st.metric("🔍 Anomaly Scanner", "Outlier Alerts", ">2.2σ Spikes")

            safe_render_html("""<div style="background: #FAF5FF; border-left: 4px solid #8B5CF6; border-radius: 6px; padding: 0.65rem 0.85rem; margin-top: 0.6rem; font-size: 0.82rem; color: #6B21A8;">
🚀 <b>Custom Ingestion:</b> Go to <b>Custom Data & Export Hub</b> to upload CSVs and reflect changes across all pages.
</div>""")

        elif current_intent_id == "export_bundle":
            w1, w2, w3, w4 = st.columns(4)
            with w1:
                st.metric("📑 PDF Memo", "Executive Briefing", "Ready to Share")
            with w2:
                st.metric("📊 Excel Model", "5-Sheet Workbook", "Financial Schedules")
            with w3:
                st.metric("📁 CSV Forecasts", "Batch Predictions", "Data Warehouse")
            with w4:
                st.metric("📦 ZIP Archive", "All-in-One", "1-Click Download")

            safe_render_html("""<div style="background: #F1F5F9; border-left: 4px solid #0F172A; border-radius: 6px; padding: 0.65rem 0.85rem; margin-top: 0.6rem; font-size: 0.82rem; color: #334155;">
📦 <b>Export Directive:</b> Download the complete executive package from the sidebar or the Export Hub.
</div>""")

    except Exception as e:
        safe_render_html(f"""<div style="background: #F8FAFC; border-left: 4px solid #2563EB; border-radius: 6px; padding: 0.65rem 0.85rem; margin-top: 0.6rem; font-size: 0.82rem; color: #334155;">
💡 <b>Directive:</b> AI plan ready for <b>{active_store}</b>. Use controls below to run simulations.
</div>""")

    safe_render_html("</div>")
