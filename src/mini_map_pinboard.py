"""
Interactive US Mini-Map Pinboard Engine.
Renders a compact, interactive US Map with color-coded pulsating dots (🟢 A+, 🔵 B, 🟡 C)
and 1-click pin chips to select and synchronize stores globally.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, Any, List
try:
    from src.config import STORE_LOCATIONS, STORES
    from src.store_deck import STORE_PROFILES, GRADE_COLORS, get_enriched_store_cards
except (ImportError, ModuleNotFoundError):
    from config import STORE_LOCATIONS, STORES
    from store_deck import STORE_PROFILES, GRADE_COLORS, get_enriched_store_cards


def generate_us_minimap_figure(cards: List[Dict[str, Any]], active_store_id: str, store_locations: dict = None) -> go.Figure:
    """
    Generates a high-precision Plotly geo scatter map with color-coded grade pins
    and glowing highlight rings for the active store.
    Automatically adapts projection and bounding box for US, India, or global datasets.
    """
    lats = []
    lons = []
    texts = []
    colors = []
    sizes = []
    hover_texts = []
    
    active_lat, active_lon, active_city = None, None, None
    loc_dict = store_locations or STORE_LOCATIONS
    
    for card in cards:
        s_id = card["store_id"]
        loc = loc_dict.get(s_id, {})
        lat = float(loc.get("lat", 38.0))
        lon = float(loc.get("lon", -97.0))
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
        
        tot_rev_fmt = f"${card['tot_rev']/1e6:,.2f}M" if card['tot_rev'] >= 1e6 else f"${card['tot_rev']/1e3:,.1f}K"
        hover_info = (
            f"<b>{card['icon']} {s_id}: {card['city']}, {card['state']}</b><br>"
            f"Grade: <b>{grade}</b> ({card['health_score']:.1f} pts)<br>"
            f"Total Sales: <b>{tot_rev_fmt}</b><br>"
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

    # Geographic region detection
    is_india = len(lons) > 0 and all(60.0 <= x <= 100.0 for x in lons) and all(5.0 <= y <= 40.0 for y in lats)
    is_usa = len(lons) > 0 and all(-135.0 <= x <= -60.0 for x in lons) and all(20.0 <= y <= 55.0 for y in lats)

    if is_india:
        geo_layout = dict(
            scope="asia",
            bgcolor="rgba(0,0,0,0)",
            lakecolor="#EFF6FF",
            landcolor="#F8FAFC",
            subunitcolor="#CBD5E1",
            countrycolor="#94A3B8",
            showlakes=True,
            showland=True,
            showsubunits=True,
            showcountries=True,
            center=dict(lat=21.5, lon=79.5),
            projection_scale=2.7
        )
    elif is_usa:
        geo_layout = dict(
            scope="usa",
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
        )
    else:
        geo_layout = dict(
            scope="world",
            bgcolor="rgba(0,0,0,0)",
            lakecolor="#EFF6FF",
            landcolor="#F8FAFC",
            subunitcolor="#CBD5E1",
            countrycolor="#94A3B8",
            showlakes=True,
            showland=True,
            showsubunits=True,
            showcountries=True
        )

    fig.update_layout(
        geo=geo_layout,
        margin=dict(l=0, r=0, t=10, b=0),
        height=270,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        dragmode=False
    )
    
    return fig


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


def render_us_minimap_pinboard(
    raw_df: pd.DataFrame,
    store_locations: dict,
    active_store: str = None,
    key_prefix: str = "pinboard"
) -> str:
    """
    Renders the Interactive Store Mini-Map Pinboard with colored pulsating dots (🟢 A+, 🔵 B, 🟡 C)
    and quick-select pin buttons in Streamlit.
    """
    cards = get_enriched_store_cards(raw_df, store_locations)
    if not cards:
        st.info("ℹ️ No store location coordinates available to render map pinboard.")
        return active_store or "Store_01"

    valid_store_ids = [c["store_id"] for c in cards]
    if "active_store" not in st.session_state or st.session_state.active_store not in valid_store_ids:
        st.session_state.active_store = valid_store_ids[0]
        
    current_active = active_store if (active_store in valid_store_ids) else st.session_state.active_store
    card_dict = {c["store_id"]: c for c in cards}
    active_c = card_dict.get(current_active, cards[0])

    # Header Card
    header_pin_html = f"""<div style="background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%); border: 1.5px solid #BFDBFE; border-left: 6px solid {active_c['color']}; border-radius: 12px; padding: 0.75rem 1.1rem; margin-bottom: 0.75rem; display: flex; justify-content: space-between; align-items: center;">
<div style="display: flex; align-items: center; gap: 0.6rem;">
<span style="font-size: 1.3rem;">📍</span>
<div>
<span style="font-weight: 800; font-size: 0.95rem; color: #0F172A;">Interactive Store Pinboard: <b>{active_c['icon']} {current_active} ({active_c['city']}, {active_c['state']})</b></span>
<span style="font-size: 0.8rem; color: #64748B; margin-left: 0.5rem;">Grade: <b style="color: {active_c['color']};">{active_c['grade']}</b> ({active_c['health_score']:.1f} pts)</span>
</div>
</div>
<span style="background: {active_c['color']}; color: white; padding: 0.2rem 0.65rem; border-radius: 9999px; font-weight: 800; font-size: 0.72rem;">
${active_c['yield_sqft']:.2f}/sq ft Yield
</span>
</div>"""
    safe_render_html(header_pin_html)

    # Mini-Map Plotly Chart
    fig = generate_us_minimap_figure(cards, current_active, store_locations)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Interactive 1-Click Store Pin Chips (dynamically chunked)
    safe_render_html("<div style='font-size: 0.82rem; font-weight: 700; color: #475569; margin-bottom: 0.35rem;'>📌 Click any store pin below to select:</div>")
    
    chunk_size = 5
    card_chunks = [cards[i:i + chunk_size] for i in range(0, len(cards), chunk_size)]
    
    for row_chunk in card_chunks:
        cols = st.columns(len(row_chunk) if len(row_chunk) <= 5 else 5)
        for idx, c in enumerate(row_chunk):
            col = cols[idx]
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
