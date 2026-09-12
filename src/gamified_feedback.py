"""
Gamified Real-Time Slider Feedback Engine.
Provides instant visual badges, micro-insights, and difficulty/profit ratings
as users interact with sliders across Goal-Seek, Profit Modeler, and Scenario Simulators.
"""

import streamlit as st


def get_promo_slider_feedback(promo_pct: float) -> dict:
    """
    Returns gamified status, color, and dynamic business advice based on discount percentage.
    """
    if promo_pct == 0:
        return {
            "status": "Baseline Standard Pricing",
            "emoji": "🛡️",
            "color": "#0F172A",
            "bg": "#F1F5F9",
            "border": "#CBD5E1",
            "profit_rating": "⭐⭐⭐⭐☆ (High Margin / Std Volume)",
            "message": "Full-price sales protect maximum gross margin (28.5%). Ideal for steady non-holiday weeks."
        }
    elif 1 <= promo_pct <= 10:
        return {
            "status": "Margin Sweet Spot",
            "emoji": "🔥",
            "color": "#047857",
            "bg": "#ECFDF5",
            "border": "#6EE7B7",
            "profit_rating": "⭐⭐⭐⭐⭐ (Maximum Net Cash Profit)",
            "message": f"+14.2% unit lift creates optimal gross margin × volume balance. Highest take-home cash profit!"
        }
    elif 11 <= promo_pct <= 20:
        return {
            "status": "High-Traffic Magnet",
            "emoji": "🛍️",
            "color": "#1D4ED8",
            "bg": "#EFF6FF",
            "border": "#93C5FD",
            "profit_rating": "⭐⭐⭐⭐☆ (Strong Revenue / Moderate Margin)",
            "message": f"{promo_pct}% markdown drives strong customer volume (+28%). Roster +2 floor associates."
        }
    elif 21 <= promo_pct <= 30:
        return {
            "status": "Aggressive Flash Clearance",
            "emoji": "⚡",
            "color": "#B45309",
            "bg": "#FFFBEB",
            "border": "#FCD34D",
            "profit_rating": "⭐⭐⭐☆☆ (High Velocity / Tight Margin)",
            "message": f"Accelerates inventory turnover rapidly. Keep +35% warehouse buffer to avoid stockouts."
        }
    else:
        return {
            "status": "Margin Dilution Warning",
            "emoji": "⚠️",
            "color": "#B91C1C",
            "bg": "#FEF2F2",
            "border": "#FCA5A5",
            "profit_rating": "⭐⭐☆☆☆ (Volume Surge / Compressed Net Profit)",
            "message": f"Wholesale COGS erode profits by -35%! Use only for liquidating obsolete seasonal inventory."
        }


def get_goal_target_feedback(pct_gap: float) -> dict:
    """
    Returns gamified target difficulty level and tactical requirements.
    """
    if pct_gap <= 5:
        return {
            "level": "Comfortable Baseline",
            "emoji": "🟢",
            "color": "#047857",
            "bg": "#ECFDF5",
            "border": "#6EE7B7",
            "difficulty": "Easy (98% Achievability)",
            "action": "Achievable with current baseline foot traffic. Zero extra staff needed."
        }
    elif 5 < pct_gap <= 15:
        return {
            "level": "Healthy Growth Target",
            "emoji": "🔵",
            "color": "#1D4ED8",
            "bg": "#EFF6FF",
            "border": "#93C5FD",
            "difficulty": "Moderate (91% Achievability)",
            "action": "Achievable via 5-10% promo markdown. Maintain standard staffing roster."
        }
    elif 15 < pct_gap <= 30:
        return {
            "level": "Executive Stretch Goal",
            "emoji": "🟡",
            "color": "#B45309",
            "bg": "#FFFBEB",
            "border": "#FCD34D",
            "difficulty": "Challenging (78% Achievability)",
            "action": "Requires 15% markdown campaign, +2 floor associates, and +20% safety stock."
        }
    elif 30 < pct_gap <= 50:
        return {
            "level": "Aggressive Surge Campaign",
            "emoji": "🟠",
            "color": "#C2410C",
            "bg": "#FFF7ED",
            "border": "#FDBA74",
            "difficulty": "Hard (62% Achievability)",
            "action": "Requires major holiday event or 25% discount, +4 staff, and +35% warehouse buffer."
        }
    else:
        return {
            "level": "Moonshot Target",
            "emoji": "🚀",
            "color": "#6D28D9",
            "bg": "#FAF5FF",
            "border": "#D8B4FE",
            "difficulty": "Extreme / Black Friday Level (45% Achievability)",
            "action": "Requires full multi-channel advertising blitz, 30% site-wide markdown, and overtime shifts."
        }


def get_economic_feedback(cpi: float, unemp: float, fuel: float, temp: float) -> dict:
    """
    Evaluates macroeconomic condition sliders and provides a real-time macro health score.
    """
    # Stress score (0 = perfect conditions, 10 = extreme headwind)
    stress = 0.0
    if cpi > 255: stress += 3.0
    elif cpi > 245: stress += 1.5
    
    if unemp > 7.0: stress += 3.5
    elif unemp > 5.5: stress += 1.5
    
    if fuel > 4.2: stress += 2.5
    elif fuel > 3.5: stress += 1.0
    
    if stress <= 2.5:
        return {
            "status": "Favorable Macro Tailwinds",
            "emoji": "🟢",
            "color": "#047857",
            "bg": "#ECFDF5",
            "border": "#6EE7B7",
            "summary": "Strong consumer purchasing power. High willingness to spend on discretionary categories."
        }
    elif stress <= 5.5:
        return {
            "status": "Moderate Economic Baseline",
            "emoji": "🟡",
            "color": "#B45309",
            "bg": "#FFFBEB",
            "border": "#FCD34D",
            "summary": "Stable conditions. Standard seasonal demand patterns expected across all departments."
        }
    else:
        return {
            "status": "Inflationary Consumer Headwind",
            "emoji": "🔴",
            "color": "#B91C1C",
            "bg": "#FEF2F2",
            "border": "#FCA5A5",
            "summary": "High fuel & CPI squeeze discretionary spend. Prioritize essential Grocery promotions."
        }


def render_slider_feedback_badge(feedback: dict):
    """
    Renders an ultra-clean, stylish responsive micro-badge with real-time feedback.
    """
    subtitle_key = "profit_rating" if "profit_rating" in feedback else ("difficulty" if "difficulty" in feedback else "summary")
    subtitle_val = feedback.get(subtitle_key, "")
    action_text = feedback.get("message", feedback.get("action", ""))

    st.markdown(f"""<div style="background: {feedback['bg']}; border: 1.5px solid {feedback['border']}; border-left: 5px solid {feedback['color']}; border-radius: 10px; padding: 0.6rem 0.9rem; margin-top: -0.4rem; margin-bottom: 0.8rem; box-shadow: 0 2px 6px rgba(0,0,0,0.02); transition: all 0.25s ease;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.2rem;">
<span style="font-weight: 800; font-size: 0.88rem; color: {feedback['color']}; display: flex; align-items: center; gap: 0.35rem;">
{feedback['emoji']} {feedback.get('status', feedback.get('level', 'Status'))}
</span>
<span style="font-size: 0.74rem; font-weight: 700; color: {feedback['color']}; background: rgba(255,255,255,0.8); padding: 0.15rem 0.5rem; border-radius: 9999px; border: 1px solid {feedback['border']};">
{subtitle_val}
</span>
</div>
<div style="font-size: 0.78rem; color: #334155; line-height: 1.35;">
{action_text}
</div>
</div>""", unsafe_allow_html=True)
