"""
Store Battle Arena Engine.
Renders head-to-head side-by-side visual comparison bars, metrics, and Plotly charts
comparing two store branches in 1 glance.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, Any, List
try:
    from src.config import STORE_LOCATIONS, STORES, DEPARTMENTS
    from src.store_deck import STORE_PROFILES, GRADE_COLORS, get_enriched_store_cards
except (ImportError, ModuleNotFoundError):
    from config import STORE_LOCATIONS, STORES, DEPARTMENTS
    from store_deck import STORE_PROFILES, GRADE_COLORS, get_enriched_store_cards


def compute_store_battle_metrics(raw_df: pd.DataFrame, store_locations: dict, store_a_id: str, store_b_id: str) -> Dict[str, Any]:
    """
    Computes comparative metrics, victory scores, and head-to-head ratios for two stores.
    """
    cards = get_enriched_store_cards(raw_df, store_locations)
    card_dict = {c["store_id"]: c for c in cards}
    
    cA = card_dict.get(store_a_id, cards[0])
    cB = card_dict.get(store_b_id, cards[1] if len(cards) > 1 else cards[0])
    
    # 5 Head-to-Head Battle Dimensions
    metrics = [
        {
            "name": "💵 Total Gross Revenue",
            "val_a": cA["tot_rev"],
            "val_b": cB["tot_rev"],
            "fmt_a": f"${cA['tot_rev']/1e6:,.2f}M",
            "fmt_b": f"${cB['tot_rev']/1e6:,.2f}M",
            "winner": "A" if cA["tot_rev"] > cB["tot_rev"] else ("B" if cB["tot_rev"] > cA["tot_rev"] else "Tie"),
            "diff_pct": abs((cA["tot_rev"] - cB["tot_rev"]) / (cB["tot_rev"] + 1e-5)) * 100
        },
        {
            "name": "📐 Space Productivity ($/sq ft)",
            "val_a": cA["yield_sqft"],
            "val_b": cB["yield_sqft"],
            "fmt_a": f"${cA['yield_sqft']:.2f} / sqft",
            "fmt_b": f"${cB['yield_sqft']:.2f} / sqft",
            "winner": "A" if cA["yield_sqft"] > cB["yield_sqft"] else ("B" if cB["yield_sqft"] > cA["yield_sqft"] else "Tie"),
            "diff_pct": abs((cA["yield_sqft"] - cB["yield_sqft"]) / (cB["yield_sqft"] + 1e-5)) * 100
        },
        {
            "name": "🩺 5-Pillar Health Score",
            "val_a": cA["health_score"],
            "val_b": cB["health_score"],
            "fmt_a": f"{cA['health_score']:.1f} pts ({cA['grade']})",
            "fmt_b": f"{cB['health_score']:.1f} pts ({cB['grade']})",
            "winner": "A" if cA["health_score"] > cB["health_score"] else ("B" if cB["health_score"] > cA["health_score"] else "Tie"),
            "diff_pct": abs(cA["health_score"] - cB["health_score"])
        },
        {
            "name": "📈 Annual Growth Momentum",
            "val_a": cA["growth"],
            "val_b": cB["growth"],
            "fmt_a": f"{cA['growth']:+.1f}%",
            "fmt_b": f"{cB['growth']:+.1f}%",
            "winner": "A" if cA["growth"] > cB["growth"] else ("B" if cB["growth"] > cA["growth"] else "Tie"),
            "diff_pct": abs(cA["growth"] - cB["growth"])
        },
        {
            "name": "📦 Avg Weekly Throughput",
            "val_a": cA["avg_weekly"],
            "val_b": cB["avg_weekly"],
            "fmt_a": f"${cA['avg_weekly']:,.0f} / wk",
            "fmt_b": f"${cB['avg_weekly']:,.0f} / wk",
            "winner": "A" if cA["avg_weekly"] > cB["avg_weekly"] else ("B" if cB["avg_weekly"] > cA["avg_weekly"] else "Tie"),
            "diff_pct": abs((cA["avg_weekly"] - cB["avg_weekly"]) / (cB["avg_weekly"] + 1e-5)) * 100
        }
    ]
    
    wins_a = sum(1 for m in metrics if m["winner"] == "A")
    wins_b = sum(1 for m in metrics if m["winner"] == "B")
    
    if wins_a > wins_b:
        overall_winner = f"{cA['icon']} {cA['city']} ({cA['store_id']})"
        verdict_summary = f"🏆 <b>{cA['city']}</b> wins <b>{wins_a} of {len(metrics)}</b> battle categories (+{cA['yield_sqft'] - cB['yield_sqft']:+.2f}/sq ft space yield advantage)."
    elif wins_b > wins_a:
        overall_winner = f"{cB['icon']} {cB['city']} ({cB['store_id']})"
        verdict_summary = f"🏆 <b>{cB['city']}</b> wins <b>{wins_b} of {len(metrics)}</b> battle categories (+{cB['yield_sqft'] - cA['yield_sqft']:+.2f}/sq ft space yield advantage)."
    else:
        overall_winner = "Dead Heat Draw"
        verdict_summary = "⚖️ <b>Dead Heat Draw:</b> Both branch locations are evenly matched across operational velocity and efficiency."

    return {
        "store_a": cA,
        "store_b": cB,
        "metrics": metrics,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "overall_winner": overall_winner,
        "verdict_summary": verdict_summary
    }


def generate_battle_chart(battle_data: Dict[str, Any]) -> go.Figure:
    """
    Generates a grouped horizontal bar chart comparing the two stores across normalized metrics.
    """
    cA = battle_data["store_a"]
    cB = battle_data["store_b"]
    
    categories = [
        "Revenue ($M)",
        "Yield ($/sqft)",
        "Health Score",
        "Growth (%)",
        "Weekly Velocity ($k)"
    ]
    
    # Normalized 0-100 scores for visual symmetry
    max_rev = max(cA["tot_rev"], cB["tot_rev"], 1.0)
    max_yld = max(cA["yield_sqft"], cB["yield_sqft"], 1.0)
    max_wk = max(cA["avg_weekly"], cB["avg_weekly"], 1.0)
    
    scores_a = [
        (cA["tot_rev"] / max_rev) * 100,
        (cA["yield_sqft"] / max_yld) * 100,
        cA["health_score"],
        max(0, min(100, 50 + cA["growth"] * 10)),
        (cA["avg_weekly"] / max_wk) * 100
    ]
    
    scores_b = [
        (cB["tot_rev"] / max_rev) * 100,
        (cB["yield_sqft"] / max_yld) * 100,
        cB["health_score"],
        max(0, min(100, 50 + cB["growth"] * 10)),
        (cB["avg_weekly"] / max_wk) * 100
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=categories,
        x=scores_a,
        name=f"{cA['icon']} {cA['city']} ({cA['store_id']})",
        orientation="h",
        marker=dict(color="#2563EB", line=dict(color="#1D4ED8", width=1.5)),
        text=[f"{s:.0f}%" for s in scores_a],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="white", family="Inter, sans-serif", size=11)
    ))
    
    fig.add_trace(go.Bar(
        y=categories,
        x=scores_b,
        name=f"{cB['icon']} {cB['city']} ({cB['store_id']})",
        orientation="h",
        marker=dict(color="#8B5CF6", line=dict(color="#7C3AED", width=1.5)),
        text=[f"{s:.0f}%" for s in scores_b],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="white", family="Inter, sans-serif", size=11)
    ))
    
    fig.update_layout(
        barmode="group",
        template="plotly_white",
        margin=dict(l=10, r=10, t=25, b=15),
        height=250,
        xaxis=dict(title="Relative Strength Index (0-100)", range=[0, 110], gridcolor="#F1F5F9"),
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        font=dict(family="Inter, sans-serif")
    )
    
    return fig


def render_store_battle_arena(
    raw_df: pd.DataFrame,
    store_locations: dict,
    key_prefix: str = "arena"
):
    """
    Renders the complete Store Battle Arena Card with side-by-side visual comparison bars.
    """
    st.markdown("#### ⚔️ Store Battle Arena (1-Glance Head-to-Head Comparison)")
    st.caption("Compare any two retail branches side-by-side across revenue yield, operational health, growth velocity, and floor throughput.")

    # Selection Pickers
    col_a, col_swap, col_b = st.columns([2, 0.6, 2])
    
    if f"{key_prefix}_store_a" not in st.session_state:
        st.session_state[f"{key_prefix}_store_a"] = "Store_09"  # Dallas
    if f"{key_prefix}_store_b" not in st.session_state:
        st.session_state[f"{key_prefix}_store_b"] = "Store_01"  # New York

    val_a = st.session_state.get(f"{key_prefix}_store_a", "Store_09")
    val_b = st.session_state.get(f"{key_prefix}_store_b", "Store_01")
    idx_a = STORES.index(val_a) if val_a in STORES else 8
    idx_b = STORES.index(val_b) if val_b in STORES else 0

    with col_a:
        sel_a = st.selectbox(
            "🔵 Select Branch Fighter A:",
            options=STORES,
            index=idx_a,
            format_func=lambda s: f"{STORE_PROFILES.get(s, {}).get('icon', '🏢')} {s} — {STORE_PROFILES.get(s, {}).get('city', s)}",
            key=f"{key_prefix}_sel_box_a"
        )
        if sel_a != st.session_state[f"{key_prefix}_store_a"]:
            st.session_state[f"{key_prefix}_store_a"] = sel_a

    with col_swap:
        st.write("")
        st.write("")
        if st.button("⇄", key=f"{key_prefix}_btn_swap", help="Swap Branch Fighters"):
            temp = st.session_state[f"{key_prefix}_store_a"]
            st.session_state[f"{key_prefix}_store_a"] = st.session_state[f"{key_prefix}_store_b"]
            st.session_state[f"{key_prefix}_store_b"] = temp
            st.session_state[f"{key_prefix}_sel_box_a"] = st.session_state[f"{key_prefix}_store_a"]
            st.session_state[f"{key_prefix}_sel_box_b"] = st.session_state[f"{key_prefix}_store_b"]
            st.rerun()

    with col_b:
        sel_b = st.selectbox(
            "🟣 Select Branch Fighter B:",
            options=STORES,
            index=idx_b,
            format_func=lambda s: f"{STORE_PROFILES.get(s, {}).get('icon', '🏢')} {s} — {STORE_PROFILES.get(s, {}).get('city', s)}",
            key=f"{key_prefix}_sel_box_b"
        )
        if sel_b != st.session_state[f"{key_prefix}_store_b"]:
            st.session_state[f"{key_prefix}_store_b"] = sel_b

    # Compute battle data
    battle = compute_store_battle_metrics(raw_df, store_locations, sel_a, sel_b)
    cA = battle["store_a"]
    cB = battle["store_b"]

    # Matchup Hero Banner
    banner_html = f"""<div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); border-radius: 14px; padding: 1rem 1.4rem; margin-top: 0.5rem; margin-bottom: 1rem; box-shadow: 0 4px 15px rgba(15,23,42,0.15); color: white;">
<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.8rem;">
<div style="display: flex; align-items: center; gap: 0.75rem;">
<span style="font-size: 2rem;">{cA['icon']}</span>
<div>
<div style="font-size: 0.75rem; color: #93C5FD; font-weight: 800; text-transform: uppercase;">BLUE CORNER</div>
<div style="font-size: 1.15rem; font-weight: 900; color: white;">{cA['city']} ({cA['store_id']})</div>
<div style="font-size: 0.8rem; color: #E2E8F0;">Grade: <b style="color: #60A5FA;">{cA['grade']}</b> • {cA['tag']}</div>
</div>
</div>
<div style="background: rgba(255,255,255,0.1); border: 1.5px solid rgba(255,255,255,0.25); border-radius: 9999px; padding: 0.35rem 1rem; font-weight: 900; font-size: 0.95rem; letter-spacing: 0.08em; color: #F8FAFC;">
⚔️ VS
</div>
<div style="display: flex; align-items: center; gap: 0.75rem; text-align: right;">
<div>
<div style="font-size: 0.75rem; color: #C4B5FD; font-weight: 800; text-transform: uppercase;">PURPLE CORNER</div>
<div style="font-size: 1.15rem; font-weight: 900; color: white;">{cB['city']} ({cB['store_id']})</div>
<div style="font-size: 0.8rem; color: #E2E8F0;">Grade: <b style="color: #A78BFA;">{cB['grade']}</b> • {cB['tag']}</div>
</div>
<span style="font-size: 2rem;">{cB['icon']}</span>
</div>
</div>
<div style="border-top: 1px solid rgba(255,255,255,0.15); margin-top: 0.8rem; padding-top: 0.6rem; font-size: 0.88rem; color: #F1F5F9;">
{battle['verdict_summary']}
</div>
</div>"""
    st.markdown(banner_html, unsafe_allow_html=True)

    # 2-Column Layout: Side-by-Side Comparison Bars (Left) + Plotly Benchmark Chart (Right)
    arena_c1, arena_c2 = st.columns([1.4, 1.2])

    with arena_c1:
        st.markdown("##### 📊 Head-to-Head Comparison Bars:")
        for m in battle["metrics"]:
            win_a = (m["winner"] == "A")
            win_b = (m["winner"] == "B")
            
            border_a = "border-left: 4px solid #2563EB;" if win_a else "border-left: 1px solid #E2E8F0;"
            badge_a = "🏆 WINNER" if win_a else ""
            badge_b = "🏆 WINNER" if win_b else ""
            
            bar_html = f"""<div style="background: white; border: 1px solid #E2E8F0; border-radius: 10px; padding: 0.55rem 0.85rem; margin-bottom: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
<div style="font-size: 0.78rem; font-weight: 800; color: #64748B; text-transform: uppercase; margin-bottom: 0.25rem;">
{m['name']}
</div>
<div style="display: flex; justify-content: space-between; align-items: center; gap: 0.5rem;">
<div style="flex: 1; background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 6px; padding: 0.35rem 0.6rem; display: flex; justify-content: space-between; align-items: center;">
<span style="font-weight: 800; font-size: 0.88rem; color: #1E40AF;">{m['fmt_a']}</span>
<span style="font-size: 0.68rem; font-weight: 800; color: #2563EB;">{badge_a}</span>
</div>
<span style="font-size: 0.75rem; font-weight: 700; color: #94A3B8;">vs</span>
<div style="flex: 1; background: #F5F3FF; border: 1px solid #DDD6FE; border-radius: 6px; padding: 0.35rem 0.6rem; display: flex; justify-content: space-between; align-items: center;">
<span style="font-size: 0.68rem; font-weight: 800; color: #7C3AED;">{badge_b}</span>
<span style="font-weight: 800; font-size: 0.88rem; color: #6D28D9;">{m['fmt_b']}</span>
</div>
</div>
</div>"""
            st.markdown(bar_html, unsafe_allow_html=True)

    with arena_c2:
        st.markdown("##### 📈 Relative Strength Index:")
        fig_battle = generate_battle_chart(battle)
        st.plotly_chart(fig_battle, use_container_width=True, config={'displayModeBar': False})
        
        # 1-Glance Executive Takeaway Pill
        takeaway_html = f"""<div style="background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%); border: 1.5px solid #DBEAFE; border-left: 5px solid #2563EB; border-radius: 10px; padding: 0.75rem 0.95rem; margin-top: 0.4rem;">
<div style="font-size: 0.75rem; font-weight: 800; color: #2563EB; text-transform: uppercase;">
💡 1-Glance Strategic Takeaway
</div>
<div style="font-size: 0.85rem; color: #1E293B; line-height: 1.4; margin-top: 0.2rem;">
Benchmark top SKU assortments and promotional floor plans from <b>{battle['overall_winner']}</b> to lift space productivity in peer branches.
</div>
</div>"""
        st.markdown(takeaway_html, unsafe_allow_html=True)
