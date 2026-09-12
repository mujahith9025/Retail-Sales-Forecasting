"""
Interactive US Mini-Map Pinboard Engine.
Renders a compact, interactive US Map with color-coded pulsating dots (🟢 A+, 🔵 B, 🟡 C)
and 1-click pin chips to select and synchronize stores globally.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, Any, List

from src.config import STORE_LOCATIONS, STORES
from src.store_deck import STORE_PROFILES, GRADE_COLORS, get_enriched_store_cards


def generate_us_minimap_figure(cards: List[Dict[str, Any]], active_store_id: str) -> go.Figure:
    """
    Generates a high-precision Plotly USA geo scatter map with color-coded grade pins
    and glowing highlight rings for the active store.
    """
    lats = []
    lons = []
    texts = []
    colors = []
    sizes = []
    hover_texts = []
    
    active_lat, active_lon, active_city = None, None, None
    
    for card in cards:
        s_id = card["store_id"]
        loc = STORE_LOCATIONS.get(s_id, {})
        lat = loc.get("lat", 38.0)
        lon = loc.get("lon", -97.0)
        grade = card["grade"]
        color = GRADE_COLORS.get(grade, "#2563EB")
        
        is_active = (s_id == active_store_id)
        if is_active:
            active_lat = lat
            active_lon = lon
            active_city = card["city"]
            
        lats.append(lat)
        lons.append(lon)
        colors.append(color)
        sizes.append(22 if is_active else 15)
        
        hover_info = (
            f"<b>{card['icon']} {s_id}: {card['city']}, {card['state']}</b><br>"
            f"Grade: <b>{grade}</b> ({card['health_score']:.1f} pts)<br>"
            f"Total Sales: <b>${card['tot_rev']/1e6:,.2f}M</b><br>"
            f"Space Yield: <b>${card['yield_sqft']:.2f}/sq ft</b><br>"
            f"Status: <i>{card['tag']}</i>"
        )
        hover_texts.append(hover_info)

    fig = go.Figure()

    # Base stores scatter trace
    fig.add_trace(go.Scattergeo(
        lon=lons,
        lat=lats,
        text=[c["city"] for c in cards],
        customdata=[c["store_id"] for c in cards],
        hovertext=hover_texts,
        hoverinfo="text",
        mode="markers+text",
        textposition="top center",
        textfont=dict(size=10, family="Inter, sans-serif", color="#1E293B"),
        marker=dict(
            size=sizes,
            color=colors,
            opacity=0.92,
            line=dict(color="#FFFFFF", width=2.5)
        ),
        name="Store Branches"
    ))

    # Active store pulsating highlight ring
    if active_lat is not None and active_lon is not None:
        fig.add_trace(go.Scattergeo(
            lon=[active_lon],
            lat=[active_lat],
            hoverinfo="skip",
            mode="markers",
            marker=dict(
                size=34,
                color="rgba(37, 99, 235, 0.25)",
                line=dict(color="#2563EB", width=2.5)
            ),
            showlegend=False,
            name="Selected Branch"
        ))

    fig.update_layout(
        geo_scope="usa",
        geo=dict(
            bgcolor="rgba(0,0,0,0)",
            lakecolor="#EFF6FF",
            landcolor="#F8FAFC",
            subunitcolor="#CBD5E1",
            countrycolor="#94A3B8",
            showlakes=True,
            showland=True,
            showsubunits=True,
            showcountries=False,
            projection_type="albers usa"
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        dragmode=False
    )
    
    return fig


def render_us_minimap_pinboard(
    raw_df: pd.DataFrame,
    store_locations: dict,
    active_store: str = None,
    key_prefix: str = "pinboard"
) -> str:
    """
    Renders the Interactive US Mini-Map Pinboard with colored pulsating dots (🟢 A+, 🔵 B, 🟡 C)
    and quick-select pin buttons in Streamlit.
    """
    if "active_store" not in st.session_state:
        st.session_state.active_store = "Store_09"  # Dallas, TX
        
    current_active = active_store or st.session_state.active_store
    if current_active not in STORE_PROFILES:
        current_active = "Store_09"
        st.session_state.active_store = current_active

    cards = get_enriched_store_cards(raw_df, store_locations)
    card_dict = {c["store_id"]: c for c in cards}
    active_c = card_dict.get(current_active, cards[0])

    # Header Card
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%); border: 1.5px solid #BFDBFE; border-left: 6px solid {active_c['color']}; border-radius: 12px; padding: 0.75rem 1.1rem; margin-bottom: 0.75rem; display: flex; justify-content: space-between; align-items: center;">
        <div style="display: flex; align-items: center; gap: 0.6rem;">
            <span style="font-size: 1.3rem;">📍</span>
            <div>
                <span style="font-weight: 800; font-size: 0.95rem; color: #0F172A;">Interactive US Pinboard: <b>{active_c['icon']} {current_active} ({active_c['city']}, {active_c['state']})</b></span>
                <span style="font-size: 0.8rem; color: #64748B; margin-left: 0.5rem;">Grade: <b style="color: {active_c['color']};">{active_c['grade']}</b> ({active_c['health_score']:.1f} pts)</span>
            </div>
        </div>
        <span style="background: {active_c['color']}; color: white; padding: 0.2rem 0.65rem; border-radius: 9999px; font-weight: 800; font-size: 0.72rem;">
            ${active_c['yield_sqft']:.2f}/sq ft Yield
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Mini-Map Plotly Chart
    fig = generate_us_minimap_figure(cards, current_active)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Interactive 1-Click Store Pin Chips (2 rows x 5 cols)
    st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #475569; margin-bottom: 0.35rem;'>📌 Click any store pin below to select:</div>", unsafe_allow_html=True)
    
    cols_row1 = st.columns(5)
    cols_row2 = st.columns(5)
    
    for i, c in enumerate(cards):
        col = cols_row1[i] if i < 5 else cols_row2[i - 5]
        s_id = c["store_id"]
        is_sel = (s_id == current_active)
        
        # Color dot emoji by grade
        dot = "🟢" if "A" in c["grade"] else ("🔵" if "B" in c["grade"] else ("🟡" if "C" in c["grade"] else "🔴"))
        btn_label = f"{dot} {c['city']} ({c['grade']})"
        btn_type = "primary" if is_sel else "secondary"
        
        if col.button(btn_label, key=f"{key_prefix}_pin_{s_id}", type=btn_type, use_container_width=True):
            st.session_state.active_store = s_id
            st.rerun()

    return current_active
