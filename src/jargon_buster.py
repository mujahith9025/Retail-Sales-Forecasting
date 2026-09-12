"""
Retail & AI Jargon Buster (Plain-English Glossary Engine).
Translates retail, financial, supply chain, and data science terminology
into crystal-clear 1-sentence explanations with real-world examples.
"""

from typing import List, Dict, Any

JARGON_TERMS = [
    # --- 1. Financial & Profitability Terms ---
    {
        "term": "COGS (Cost of Goods Sold)",
        "category": "💰 Finance & Profit",
        "icon": "🏷️",
        "definition": "The wholesale purchase price paid to suppliers and manufacturers for merchandise before adding store markup.",
        "example": "If a jacket retails for $100 and wholesale cost was $45, the COGS is $45 (45%).",
        "why_it_matters": "Low COGS creates higher gross profit margins, giving more flexibility for discounts."
    },
    {
        "term": "Gross Margin ($ and %)",
        "category": "💰 Finance & Profit",
        "icon": "💵",
        "definition": "The money remaining from register sales after paying wholesale product costs (Gross Sales minus COGS).",
        "example": "Selling $50,000 with $30,000 in COGS leaves a Gross Margin of $20,000 (40.0%).",
        "why_it_matters": "Must be large enough to pay store associate wages, rent, and electricity."
    },
    {
        "term": "Net Operating Profit (EBITDA)",
        "category": "💰 Finance & Profit",
        "icon": "🏆",
        "definition": "The actual take-home cash profit after subtracting wholesale goods, floor staff wages, rent, and overhead.",
        "example": "$50,000 Sales - $30,000 COGS - $6,000 Labor - $4,000 Rent = $10,000 Net Profit (20% Net Margin).",
        "why_it_matters": "The ultimate measure of commercial business success and cash generation."
    },
    {
        "term": "Break-Even Sales Threshold",
        "category": "💰 Finance & Profit",
        "icon": "🛡️",
        "definition": "The minimum dollar volume a store branch must sell in a week to cover all bills and wages with zero net loss.",
        "example": "If fixed store rent and baseline payroll equal $12,500/week, weekly sales must reach $25,000 at 50% margin.",
        "why_it_matters": "Helps store managers know their safety threshold during slow seasonal weeks."
    },
    {
        "term": "Discount Elasticity",
        "category": "💰 Finance & Profit",
        "icon": "📈",
        "definition": "The measure of how many extra customer units are bought when prices are discounted by a specific percentage.",
        "example": "A 10% discount that boosts sales volume by +25% has strong positive price elasticity.",
        "why_it_matters": "Finds the peak profit 'Sweet Spot' so discounts boost cash without eroding net margins."
    },

    # --- 2. Supply Chain & Inventory Terms ---
    {
        "term": "Safety Stock / Inventory Buffer",
        "category": "📦 Supply Chain & Inventory",
        "icon": "📦",
        "definition": "Extra reserve boxes staged in the stockroom to prevent empty shelves during unexpected customer demand spikes.",
        "example": "Holding a +35% buffer before Black Friday ensures you don't run out of TVs before Saturday.",
        "why_it_matters": "Prevents lost revenue and maintains high shopper satisfaction."
    },
    {
        "term": "Stockout / Out-of-Stock (OOS)",
        "category": "📦 Supply Chain & Inventory",
        "icon": "⚠️",
        "definition": "When an item is completely sold out on the shelf, leaving shoppers unable to make their intended purchase.",
        "example": "A customer walks in for milk but the shelf is empty, leading to a lost $5.00 transaction.",
        "why_it_matters": "Stockouts cost US retailers billions annually and drive shoppers to competitors."
    },
    {
        "term": "On-Shelf Availability (Fill Rate SLA)",
        "category": "📦 Supply Chain & Inventory",
        "icon": "🎯",
        "definition": "The percentage of customer product requests that can be fulfilled immediately from shelf inventory (Target: >98%).",
        "example": "If 100 shoppers seek an item and 99 find it in stock, the fill-rate SLA is 99.0%.",
        "why_it_matters": "The gold standard operational benchmark for supply chain and warehouse excellence."
    },
    {
        "term": "Inventory Turnover Pace",
        "category": "📦 Supply Chain & Inventory",
        "icon": "🔄",
        "definition": "The speed at which a store sells through its entire stockroom inventory and replaces it with fresh goods.",
        "example": "Grocery turns over inventory every 10 days, while winter apparel may take 60 days.",
        "why_it_matters": "High turnover reduces warehouse storage fees and minimizes expired or damaged stock."
    },

    # --- 3. Store Operations & Footprint Terms ---
    {
        "term": "Space Efficiency Yield ($/Sq Ft)",
        "category": "🏢 Store Operations",
        "icon": "🏬",
        "definition": "Total annual store revenue divided by building square footage, evaluating how hard every square foot works.",
        "example": "Store 09 in Dallas generates $318.48 per square foot, making it the #1 most efficient branch in the network.",
        "why_it_matters": "Allows fair comparison between giant flagship branches (150k sq ft) and compact urban stores (80k sq ft)."
    },
    {
        "term": "Labor Pressure Index",
        "category": "🏢 Store Operations",
        "icon": "👥",
        "definition": "A 0-100 metric measuring whether store associates are comfortably staffed or overwhelmed with checkout lines.",
        "example": "A score of 85+ indicates cashiers and shelf restockers need +3 overtime shift helpers.",
        "why_it_matters": "Prevents staff burnout and long checkout lines that frustrate customers."
    },
    {
        "term": "Cross-Category Halo Effect",
        "category": "🏢 Store Operations",
        "icon": "🔗",
        "definition": "When promotional foot-traffic in one department (e.g. Snack Foods) drives spillover purchases in other categories.",
        "example": "Discounting Super Bowl chips by 15% causes shoppers to also buy big-screen TVs and electronics.",
        "why_it_matters": "Justifies 'loss-leader' discounts because overall basket sizes increase across the store."
    },

    # --- 4. AI & Machine Learning Terms ---
    {
        "term": "Forecast Accuracy (R² = 94.6%)",
        "category": "🤖 AI & Data Science",
        "icon": "🎯",
        "definition": "The percentage of real-world retail sales fluctuations successfully captured and explained by the machine learning model.",
        "example": "An R² of 0.968 means the AI understands 96.8% of what causes weekly sales to go up or down.",
        "why_it_matters": "Gives executive leadership high confidence when planning multi-million dollar purchasing orders."
    },
    {
        "term": "MAPE (Average Error Margin ±5.4%)",
        "category": "🤖 AI & Data Science",
        "icon": "📐",
        "definition": "Mean Absolute Percentage Error: On average, how close the AI's predictions are to actual register sales.",
        "example": "If actual sales were $10,000, a 5.4% MAPE means the AI forecasted between $9,460 and $10,540.",
        "why_it_matters": "Industry benchmark; error margins below 8% are considered world-class enterprise tier."
    },
    {
        "term": "P10 (Safety Floor / Conservative Bound)",
        "category": "🤖 AI & Data Science",
        "icon": "🛡️",
        "definition": "The conservative low-end demand estimate. There is a 90% statistical probability sales will be at least this high.",
        "example": "If P10 is $22,000, finance uses this floor to ensure debt and payroll obligations are covered.",
        "why_it_matters": "Protects against financial downside and over-ordering risk."
    },
    {
        "term": "P50 (Expected Median Forecast)",
        "category": "🤖 AI & Data Science",
        "icon": "⚖️",
        "definition": "The most probable, balanced sales forecast representing normal expected trading conditions.",
        "example": "The champion model predicts $28,500 for regular weekly baseline demand.",
        "why_it_matters": "Used for standard weekly purchasing orders, baseline rosters, and revenue targets."
    },
    {
        "term": "P90 (Surge Ceiling / Peak Demand)",
        "category": "🤖 AI & Data Science",
        "icon": "🚀",
        "definition": "The optimistic high-demand ceiling. Only a 10% chance demand will exceed this surge number during peak rushes.",
        "example": "If P90 is $38,000 for Christmas week, warehouse logistics stages inventory up to this ceiling.",
        "why_it_matters": "Prevents stockouts during massive holiday shopping waves like Black Friday."
    },
    {
        "term": "Outlier / Anomaly (>2.2σ)",
        "category": "🤖 AI & Data Science",
        "icon": "⚠️",
        "definition": "An abnormal spike or crash in sales that deviates dramatically from normal seasonal patterns (>2.2 standard deviations).",
        "example": "A sudden blizzard causing sales to drop -70% in 1 day is flagged as an outlier for management review.",
        "why_it_matters": "Alerts store managers to investigate supplier disruptions or localized demand surges."
    },
    {
        "term": "Lag Features (Lag-1, Lag-4)",
        "category": "🤖 AI & Data Science",
        "icon": "⏳",
        "definition": "Past historical sales (from 1 week ago and 4 weeks ago) fed into the AI to teach it current growth momentum.",
        "example": "If sales were $20k, $22k, $25k in consecutive weeks, Lag features signal an upward momentum trend.",
        "why_it_matters": "Enables the AI to adapt dynamically to real-time sales acceleration or slowdown."
    }
]


def search_jargon_terms(query: str = "", category_filter: str = "All Categories") -> List[Dict[str, Any]]:
    """
    Filters jargon glossary terms by category and keyword search query.
    """
    results = JARGON_TERMS
    if category_filter and category_filter != "All Categories":
        results = [t for t in results if t["category"] == category_filter]
        
    if query:
        q_clean = query.strip().lower()
        results = [
            t for t in results
            if q_clean in t["term"].lower()
            or q_clean in t["definition"].lower()
            or q_clean in t["example"].lower()
            or q_clean in t["why_it_matters"].lower()
        ]
        
    return results
