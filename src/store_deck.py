"""
Interactive Visual Store Card Deck Engine (Goodbye Boring Dropdowns).
Renders responsive, clickable city store cards with live letter grades,
footprint efficiency metrics ($/sq ft), operational health scores, and active selection rings.
"""

import pandas as pd
import streamlit as st
from typing import Dict, Any, List
try:
    from src.health_scorecard import compute_store_health_scorecard
except (ImportError, ModuleNotFoundError):
    from health_scorecard import compute_store_health_scorecard

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
    Computes real-time metrics for all stores present in raw_df to render live card stats,
    synchronizing with the official 5-pillar health scorecard engine.
    Supports both default US stores and custom uploaded datasets (e.g. India Region or user CSVs).
    """
    card_df = compute_store_health_scorecard(raw_df, store_locations)
    card_map = {row["Store_ID"]: row for _, row in card_df.iterrows()} if len(card_df) > 0 else {}
    
    unique_stores = list(raw_df["Store_ID"].unique()) if "Store_ID" in raw_df.columns else list(STORE_PROFILES.keys())
    
    cards = []
    for idx, s_id in enumerate(unique_stores):
        s_data = raw_df[raw_df["Store_ID"] == s_id]
        if len(s_data) == 0:
            continue
            
        tot_rev = float(s_data["Weekly_Sales"].sum())
        avg_weekly = float(s_data.groupby("Date")["Weekly_Sales"].sum().mean()) if "Date" in s_data.columns else (tot_rev / max(1, len(s_data)))
        
        # Determine store size
        if "Store_Size_SqFt" in s_data.columns and pd.notna(s_data["Store_Size_SqFt"].iloc[0]):
            sqft = float(s_data["Store_Size_SqFt"].iloc[0])
        elif s_id in store_locations and "sqft" in store_locations[s_id]:
            sqft = float(store_locations[s_id]["sqft"])
        else:
            sqft = 100000.0
            
        yield_sqft = tot_rev / max(1.0, sqft)
        
        # Profile metadata lookup
        loc_prof = store_locations.get(s_id, {})
        def_prof = STORE_PROFILES.get(s_id, {})
        
        city = loc_prof.get("city", s_data["City"].iloc[0] if ("City" in s_data.columns and pd.notna(s_data["City"].iloc[0])) else def_prof.get("city", str(s_id)))
        state = loc_prof.get("state", s_data["State"].iloc[0] if ("State" in s_data.columns and pd.notna(s_data["State"].iloc[0])) else def_prof.get("state", ""))
        
        icons = ["🏬", "🏙️", "🏛️", "🏪", "🏬", "🏢", "🛍️", "🎯", "⚓", "🌟"]
        icon = loc_prof.get("icon", def_prof.get("icon", icons[idx % len(icons)]))
        tag = loc_prof.get("tag", def_prof.get("tag", f"{city} Branch"))
        
        scorecard_info = card_map.get(s_id, {})
        grade = scorecard_info.get("Grade", "B")
        color = scorecard_info.get("Color", GRADE_COLORS.get(grade, "#2563EB"))
        health_score = float(scorecard_info.get("Health_Score", 80.0))
        growth_pace = float(scorecard_info.get("Growth_Pace (%)", 3.5))
        
        top_cat = scorecard_info.get("Top_Category", s_data.groupby("Department")["Weekly_Sales"].sum().idxmax() if "Department" in s_data.columns and len(s_data) > 0 else "General")
        
        cards.append({
            "store_id": s_id,
            "city": city,
            "state": state,
            "icon": icon,
            "tag": tag,
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


def render_interactive_store_deck(
    raw_df: pd.DataFrame,
    store_locations: dict,
    active_store: str = None,
    key_prefix: str = "deck"
) -> str:
    """
    Renders an interactive visual card deck in Streamlit for all branches in the active dataset.
    Clicking any card updates st.session_state.active_store globally and triggers a rerun.
    Returns the currently active store ID.
    """
    store_cards = get_enriched_store_cards(raw_df, store_locations)
    if not store_cards:
        st.info("ℹ️ No store branch records available to display.")
        return active_store or "Store_01"

    valid_store_ids = [c["store_id"] for c in store_cards]
    
    if "active_store" not in st.session_state or st.session_state.active_store not in valid_store_ids:
        st.session_state.active_store = valid_store_ids[0]
        
    current_active = active_store if (active_store in valid_store_ids) else st.session_state.active_store
    active_card = next((c for c in store_cards if c["store_id"] == current_active), store_cards[0])

    header_deck_html = f"""<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; background: #F8FAFC; border: 1px solid #E2E8F0; padding: 0.6rem 1rem; border-radius: 10px;">
<span style="font-weight: 700; font-size: 0.95rem; color: #0F172A; display: flex; align-items: center; gap: 0.5rem;">
🏢 <b>Visual Store Card Deck:</b> Click Any Branch to Select ({len(store_cards)} Stores Active)
</span>
<span style="font-size: 0.82rem; color: #334155; font-weight: 600;">
Currently Active: <span style="background: #2563EB; color: white; padding: 0.2rem 0.6rem; border-radius: 6px; font-weight: 700;">{active_card['icon']} {active_card['city']}, {active_card['state']} ({current_active})</span>
</span>
</div>"""
    safe_render_html(header_deck_html)

    # Dynamically chunk cards into rows of up to 5 columns
    chunk_size = 5
    card_rows = [store_cards[i:i + chunk_size] for i in range(0, len(store_cards), chunk_size)]

    clicked_store = None

    for row_cards in card_rows:
        num_cols = len(row_cards)
        cols = st.columns(num_cols if num_cols <= 5 else 5)
        for idx, card in enumerate(row_cards):
            is_active = (card["store_id"] == current_active)
            c = cols[idx]
            with c:
                # Active card border glow styling
                border_color = "#2563EB" if is_active else "#E2E8F0"
                bg_color = "linear-gradient(135deg, #EFF6FF 0%, #FFFFFF 100%)" if is_active else "rgba(255, 255, 255, 0.95)"
                box_shadow = "0 8px 24px -4px rgba(37, 99, 235, 0.3)" if is_active else "0 2px 6px rgba(0,0,0,0.03)"
                active_badge = f'<span style="background: #2563EB; color: white; padding: 0.15rem 0.5rem; border-radius: 9999px; font-size: 0.68rem; font-weight: 700;">🟢 Active</span>' if is_active else f'<span style="background: {card["color"]}; color: white; padding: 0.15rem 0.5rem; border-radius: 9999px; font-size: 0.68rem; font-weight: 800;">{card["grade"]}</span>'

                sales_display = f"${card['tot_rev']/1e6:.2f}M" if card['tot_rev'] >= 1e6 else f"${card['tot_rev']/1e3:.1f}K"

                card_html = f"""<div style="background: {bg_color}; border: 2px solid {border_color}; border-radius: 12px; padding: 0.8rem 0.9rem; box-shadow: {box_shadow}; min-height: 142px; display: flex; flex-direction: column; justify-content: space-between; transition: all 0.2s ease;">
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
<div>📈 <b>Sales:</b> {sales_display} Total</div>
</div>
</div>"""
                safe_render_html(card_html)
                
                btn_type = "primary" if is_active else "secondary"
                btn_label = f"Selected ✓" if is_active else f"Select {card['city']}"
                if st.button(btn_label, key=f"{key_prefix}_btn_{card['store_id']}", type=btn_type, use_container_width=True):
                    clicked_store = card["store_id"]

        st.write("")

    if clicked_store and clicked_store != current_active:
        st.session_state.active_store = clicked_store
        st.rerun()

    return st.session_state.get("active_store", current_active)
