"""
Interactive Visual Store Card Deck Engine (Goodbye Boring Dropdowns).
Renders responsive, clickable city store cards with live letter grades,
footprint efficiency metrics ($/sq ft), operational health scores, and active selection rings.
"""

import pandas as pd
import streamlit as st
from typing import Dict, Any, List

from src.health_scorecard import compute_store_health_scorecard

STORE_PROFILES = {
    "Store_01": {"city": "New York", "state": "NY", "icon": "🗽", "tag": "Flagship Hub", "type": "Supercenter"},
    "Store_02": {"city": "Los Angeles", "state": "CA", "icon": "🌴", "tag": "Pacific Flagship", "type": "Supercenter"},
    "Store_03": {"city": "Chicago", "state": "IL", "icon": "🏙️", "tag": "Midwest Center", "type": "Standard"},
    "Store_04": {"city": "Houston", "state": "TX", "icon": "🚀", "tag": "Energy Corridor", "type": "Standard"},
    "Store_05": {"city": "Phoenix", "state": "AZ", "icon": "🌵", "tag": "Desert West", "type": "Standard"},
    "Store_06": {"city": "Philadelphia", "state": "PA", "icon": "🔔", "tag": "Historic Core", "type": "Compact"},
    "Store_07": {"city": "San Antonio", "state": "TX", "icon": "🤠", "tag": "Southwest Branch", "type": "Compact"},
    "Store_08": {"city": "San Diego", "state": "CA", "icon": "🌊", "tag": "Coastal Hub", "type": "Standard"},
    "Store_09": {"city": "Dallas", "state": "TX", "icon": "🏆", "tag": "#1 Network Leader", "type": "High Yield"},
    "Store_10": {"city": "Atlanta", "state": "GA", "icon": "🍑", "tag": "Southeast Hub", "type": "Standard"},
}

GRADE_COLORS = {
    "A+": "#10B981",
    "A": "#059669",
    "B": "#2563EB",
    "C": "#F59E0B",
    "D": "#EA580C",
    "F": "#DC2626"
}


def get_enriched_store_cards(raw_df: pd.DataFrame, store_locations: dict) -> List[Dict[str, Any]]:
    """
    Computes real-time metrics for all 10 stores to render live card stats,
    synchronizing with the official 5-pillar health scorecard engine.
    """
    card_df = compute_store_health_scorecard(raw_df, store_locations)
    card_map = {row["Store_ID"]: row for _, row in card_df.iterrows()}
    
    cards = []
    for s_id, s_prof in STORE_PROFILES.items():
        s_data = raw_df[raw_df["Store_ID"] == s_id]
        if len(s_data) == 0:
            continue
            
        tot_rev = s_data["Weekly_Sales"].sum()
        avg_weekly = s_data.groupby("Date")["Weekly_Sales"].sum().mean()
        sqft = s_data["Store_Size_SqFt"].iloc[0] if len(s_data) > 0 else 100000
        yield_sqft = tot_rev / sqft
        
        scorecard_info = card_map.get(s_id, {})
        grade = scorecard_info.get("Grade", "B")
        color = scorecard_info.get("Color", GRADE_COLORS.get(grade, "#2563EB"))
        health_score = scorecard_info.get("Health_Score", 80.0)
        growth_pace = scorecard_info.get("Growth_Pace (%)", 3.5)
        top_cat = scorecard_info.get("Top_Category", s_data.groupby("Department")["Weekly_Sales"].sum().idxmax())
        
        cards.append({
            "store_id": s_id,
            "city": s_prof["city"],
            "state": s_prof["state"],
            "icon": s_prof["icon"],
            "tag": s_prof["tag"],
            "grade": grade,
            "color": color,
            "health_score": health_score,
            "tot_rev": tot_rev,
            "avg_weekly": avg_weekly,
            "sqft": sqft,
            "yield_sqft": yield_sqft,
            "growth": growth_pace,
            "top_category": top_cat
        })
        
    return cards


def render_interactive_store_deck(
    raw_df: pd.DataFrame,
    store_locations: dict,
    active_store: str = None,
    key_prefix: str = "deck"
) -> str:
    """
    Renders an interactive visual card deck (2 rows x 5 columns) in Streamlit.
    Clicking any card updates st.session_state.active_store globally and triggers a rerun.
    Returns the currently active store ID.
    """
    if "active_store" not in st.session_state:
        st.session_state.active_store = "Store_09"  # Dallas, TX (#1 Network Leader)
        
    current_active = active_store or st.session_state.active_store
    if current_active not in STORE_PROFILES:
        current_active = "Store_09"
        st.session_state.active_store = current_active

    store_cards = get_enriched_store_cards(raw_df, store_locations)
    active_profile = STORE_PROFILES.get(current_active, STORE_PROFILES["Store_09"])

    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; background: #F8FAFC; border: 1px solid #E2E8F0; padding: 0.6rem 1rem; border-radius: 10px;">
        <span style="font-weight: 700; font-size: 0.95rem; color: #0F172A; display: flex; align-items: center; gap: 0.5rem;">
            🏢 <b>Visual Store Card Deck:</b> Click Any Branch to Select
        </span>
        <span style="font-size: 0.82rem; color: #334155; font-weight: 600;">
            Currently Active: <span style="background: #2563EB; color: white; padding: 0.2rem 0.6rem; border-radius: 6px; font-weight: 700;">{active_profile['icon']} {active_profile['city']}, {active_profile['state']} ({current_active})</span>
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Render in 2 rows of 5 columns
    row1 = store_cards[:5]
    row2 = store_cards[5:]

    clicked_store = None

    for row_cards in [row1, row2]:
        cols = st.columns(5)
        for idx, card in enumerate(row_cards):
            is_active = (card["store_id"] == current_active)
            c = cols[idx]
            with c:
                # Active card border glow styling
                border_color = "#2563EB" if is_active else "#E2E8F0"
                bg_color = "linear-gradient(135deg, #EFF6FF 0%, #FFFFFF 100%)" if is_active else "rgba(255, 255, 255, 0.95)"
                box_shadow = "0 8px 24px -4px rgba(37, 99, 235, 0.3)" if is_active else "0 2px 6px rgba(0,0,0,0.03)"
                active_badge = f'<span style="background: #2563EB; color: white; padding: 0.15rem 0.5rem; border-radius: 9999px; font-size: 0.68rem; font-weight: 700;">🟢 Active</span>' if is_active else f'<span style="background: {card["color"]}; color: white; padding: 0.15rem 0.5rem; border-radius: 9999px; font-size: 0.68rem; font-weight: 800;">{card["grade"]}</span>'

                st.markdown(f"""
                <div style="background: {bg_color}; border: 2px solid {border_color}; border-radius: 12px; padding: 0.8rem 0.9rem; box-shadow: {box_shadow}; min-height: 142px; display: flex; flex-direction: column; justify-content: space-between; transition: all 0.2s ease;">
                    <div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.2rem;">
                            <span style="font-weight: 800; font-size: 0.95rem; color: #0F172A;">
                                {card['icon']} {card['city']}
                            </span>
                            {active_badge}
                        </div>
                        <div style="font-size: 0.72rem; color: #64748B; font-weight: 600; margin-bottom: 0.4rem;">
                            {card['store_id']} • {card['state']} • {card['tag']}
                        </div>
                    </div>
                    <div style="border-top: 1px solid #E2E8F0; padding-top: 0.4rem; font-size: 0.75rem; color: #334155; line-height: 1.35;">
                        <div>⚡ <b>Yield:</b> ${card['yield_sqft']:.1f}/sq ft</div>
                        <div>📈 <b>Sales:</b> ${card['tot_rev']/1e6:.1f}M Total</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                btn_type = "primary" if is_active else "secondary"
                btn_label = f"Selected ✓" if is_active else f"Select {card['city']}"
                if st.button(btn_label, key=f"{key_prefix}_btn_{card['store_id']}", type=btn_type, use_container_width=True):
                    clicked_store = card["store_id"]

        st.write("")

    if clicked_store and clicked_store != current_active:
        st.session_state.active_store = clicked_store
        st.rerun()

    return st.session_state.get("active_store", current_active)
