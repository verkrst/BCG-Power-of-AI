"""
Streamlit KPI Dashboard – French Barley Supply‑Chain Risk Monitor
Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
from pathlib import Path

# ── page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Barley Supply Risk Dashboard",
    page_icon="🌾",
    layout="wide",
)

# ── load data (cached) ──────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent / "dataset"


@st.cache_data
def load_predictions():
    return pd.read_csv(DATA_DIR / "predictions.csv")


@st.cache_data
def load_barley():
    return pd.read_csv(DATA_DIR / "barley_yield_from_1982.csv", sep=";", index_col=0)


@st.cache_data
def load_geojson():
    geo_path = Path(__file__).resolve().parent / "departements.geojson"
    with open(geo_path, encoding="utf-8") as f:
        geo = json.load(f)
    # keep only mainland France (code < 97)
    geo["features"] = [
        ft for ft in geo["features"]
        if ft["properties"]["code"] < "97"
    ]
    return geo


predictions = load_predictions()
barley = load_barley()
geojson = load_geojson()

# ── constants ────────────────────────────────────────────────────────────
EF_HA = 2500  # kg CO2e per hectare (fixed)

SCENARIO_LABELS = {
    "SSP1‑2.6 (Optimistic)":  "ssp1_2_6",
    "SSP2‑4.5 (Middle Road)": "ssp2_4_5",
    "SSP5‑8.5 (Pessimistic)": "ssp5_8_5",
}

# ── sidebar ──────────────────────────────────────────────────────────────
st.sidebar.markdown("## ⚙️ Parameters")

scenario_label = st.sidebar.selectbox(
    "Climate Scenario",
    list(SCENARIO_LABELS.keys()),
    index=1,  # default SSP2-4.5
)
selected_scenario = SCENARIO_LABELS[scenario_label]

st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 Financial Assumptions")

assumed_price_per_ton = st.sidebar.slider(
    "Assumed Barley Market Price (€/Ton)",
    min_value=130,
    max_value=400,
    value=180,
    step=10,
    help="Adjust this to simulate price spikes during climate-driven supply shortages.",
)

# ── main area ────────────────────────────────────────────────────────────
st.title("🌾 ClientCo Barley Yield and ESG Risk Dashboard")
st.caption(
    "Predicted national barley KPIs under CMIP6 climate scenarios · XGBoost model · "
    "Baseline = 2004–2014 average"
)

# year slider
available_years = sorted(predictions["year"].dropna().unique())
min_yr, max_yr = int(min(available_years)), int(max(available_years))

selected_year = st.slider(
    "Select Forecast Year",
    min_value=min_yr,
    max_value=max_yr,
    value=min(2025, max_yr),
    step=1,
)

# ── filter data ──────────────────────────────────────────────────────────
df_year = predictions[
    (predictions["year"] == selected_year)
    & (predictions["scenario"] == selected_scenario)
].copy()

# ── KPI calculations ────────────────────────────────────────────────────

# --- Production ---
total_production = df_year["pred_production"].sum()  # tonnes
baseline_production = (df_year["baseline_yield"] * df_year["area_2014"]).sum()
prod_evol_pct = (
    (total_production - baseline_production) / baseline_production * 100
    if baseline_production
    else 0
)

# --- Sourcing Spend ---
total_procurement_cost = total_production * assumed_price_per_ton
baseline_cost = baseline_production * assumed_price_per_ton
cost_evol_pct = (
    (total_procurement_cost - baseline_cost) / baseline_cost * 100
    if baseline_cost
    else 0
)

# --- CO₂e per ton ---
total_area = df_year["area_2014"].sum()
total_co2e = total_area * EF_HA  # kg CO2e

national_co2e_per_ton = total_co2e / total_production if total_production else 0
baseline_co2e_per_ton = total_co2e / baseline_production if baseline_production else 0
co2_evol_pct = (
    (national_co2e_per_ton - baseline_co2e_per_ton) / baseline_co2e_per_ton * 100
    if baseline_co2e_per_ton
    else 0
)

# ── display KPIs ─────────────────────────────────────────────────────────
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="🌾 Predicted Total Production",
        value=f"{total_production / 1_000_000:.2f} M tons",
        delta=f"{prod_evol_pct:+.1f}% vs baseline",
        delta_color="normal",
    )

with col2:
    st.metric(
        label="💰 Est. Sourcing Spend",
        value=f"€{total_procurement_cost / 1_000_000:.2f} M",
        delta=f"{cost_evol_pct:+.1f}% vs baseline",
        delta_color="inverse",  # cost increase → red
    )

with col3:
    st.metric(
        label="🏭 Est. CO₂e per Ton",
        value=f"{national_co2e_per_ton:.0f} kg",
        delta=f"{co2_evol_pct:+.1f}% vs baseline",
        delta_color="inverse",  # emission increase → red
    )

# ── interactive map ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader(f"🗺️ Yield Change by Department — {selected_year} ({scenario_label})")

# prepare map data — compute yield_change_pct on the fly for robustness
map_df = df_year.copy()
map_df["yield_change_pct"] = (
    (map_df["pred_yield"] - map_df["baseline_yield"]) / map_df["baseline_yield"] * 100
)

# discrete color bins matching the notebook style
BINS = [-np.inf, -20, -15, -5, 0, 5, 10, np.inf]
BIN_LABELS = [
    "< −20 %", "−20 % to −15 %", "−15 % to −5 %",
    "−5 % to 0 %", "0 % to 5 %", "5 % to 10 %", "> 10 %",
]
BIN_COLORS = {
    "< −20 %":        "#d7191c",  # red
    "−20 % to −15 %": "#e66101",  # dark orange
    "−15 % to −5 %":  "#fdae61",  # orange
    "−5 % to 0 %":    "#ffffbf",  # yellow
    "0 % to 5 %":     "#a6d96a",  # light green
    "5 % to 10 %":    "#33a02c",  # medium green
    "> 10 %":         "#006837",  # dark green
}
NO_DATA_COLOR = "#cccccc"

map_df["bin"] = pd.cut(
    map_df["yield_change_pct"], bins=BINS, labels=BIN_LABELS
).astype(str)
map_df.loc[map_df["yield_change_pct"].isna(), "bin"] = "No data"

# hover-friendly columns
map_df["pred_yield_round"] = map_df["pred_yield"].round(2)
map_df["baseline_yield_round"] = map_df["baseline_yield"].round(2)
map_df["yield_change_round"] = map_df["yield_change_pct"].round(1)
map_df["pred_production_k"] = (map_df["pred_production"] / 1_000).round(1)

# handle gdd / max_consec_dry if present
has_gdd = "gdd" in map_df.columns
has_dry = "max_consec_dry" in map_df.columns
if has_gdd:
    map_df["gdd_round"] = map_df["gdd"].round(0)
if has_dry:
    map_df["dry_days"] = map_df["max_consec_dry"]

# build hover text manually for full control
hover_parts = [
    "<b>%{customdata[0]}</b><br>",
    "Yield Change: %{customdata[1]:.1f}%<br>",
    "Pred. Yield: %{customdata[2]:.2f} t/ha<br>",
    "Baseline Yield: %{customdata[3]:.2f} t/ha<br>",
    "Pred. Production: %{customdata[4]:.1f} k tons<br>",
]
custom_cols = ["nom_dep", "yield_change_pct", "pred_yield", "baseline_yield", "pred_production_k"]
if has_gdd:
    hover_parts.append("GDD: %{customdata[5]:.0f}<br>")
    custom_cols.append("gdd_round")
if has_dry:
    idx = len(custom_cols)
    hover_parts.append(f"Max Consec. Dry Days: %{{customdata[{idx}]}}<br>")
    custom_cols.append("dry_days")
hover_template = "".join(hover_parts) + "<extra></extra>"

import plotly.graph_objects as go

fig = go.Figure()

# draw each color bin as a separate choropleth trace (discrete legend)
all_bins = BIN_LABELS + ["No data"]
all_colors = {**BIN_COLORS, "No data": NO_DATA_COLOR}

for bin_label in all_bins:
    sub = map_df[map_df["bin"] == bin_label]
    if sub.empty:
        continue
    fig.add_trace(
        go.Choroplethmapbox(
            geojson=geojson,
            locations=sub["code_dep"],
            featureidkey="properties.code",
            z=[1] * len(sub),  # dummy z for uniform fill
            customdata=sub[custom_cols].values,
            hovertemplate=hover_template,
            marker=dict(
                opacity=0.75,
                line=dict(width=0.5, color="white"),
            ),
            colorscale=[[0, all_colors[bin_label]], [1, all_colors[bin_label]]],
            showscale=False,
            name=bin_label,
            showlegend=True,
        )
    )

fig.update_layout(
    mapbox=dict(
        style="carto-positron",
        center={"lat": 46.6, "lon": 2.3},
        zoom=4.5,
    ),
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    height=650,
    legend=dict(
        title="Yield change (Δ%)",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#ccc",
        borderwidth=1,
        font=dict(size=12),
        x=0.01, y=0.99,
        xanchor="left", yanchor="top",
    ),
)

st.plotly_chart(fig, use_container_width=True)

# ── department detail panel ──────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔍 Department Detail")

# build sorted list of departments available for the selected year+scenario
dept_list = sorted(map_df["nom_dep"].dropna().unique().tolist())

selected_dept = st.selectbox(
    "Select a Department",
    options=dept_list,
    index=0,
)

dept_row = map_df[map_df["nom_dep"] == selected_dept]

if not dept_row.empty:
    row = dept_row.iloc[0]
    d_yield_change = (
        (row["pred_yield"] - row["baseline_yield"]) / row["baseline_yield"] * 100
        if row["baseline_yield"] else 0
    )
    d_pred_prod = row["pred_production"] / 1_000  # k tons
    d_co2_per_ton = (
        (row["area_2014"] * EF_HA) / row["pred_production"]
        if row["pred_production"] else 0
    )
    d_baseline_co2_per_ton = (
        (row["area_2014"] * EF_HA) / (row["baseline_yield"] * row["area_2014"])
        if row["baseline_yield"] and row["area_2014"] else 0
    )
    d_co2_evol = (
        (d_co2_per_ton - d_baseline_co2_per_ton) / d_baseline_co2_per_ton * 100
        if d_baseline_co2_per_ton else 0
    )

    dc1, dc2, dc3, dc4, dc5 = st.columns(5)

    with dc1:
        st.metric(
            label="📊 Yield Change",
            value=f"{d_yield_change:+.1f}%",
            delta=f"vs 2004–2014 baseline",
            delta_color="off",
        )
    with dc2:
        st.metric(
            label="🌾 Pred. Yield",
            value=f"{row['pred_yield']:.2f} t/ha",
        )
    with dc3:
        st.metric(
            label="📐 Baseline Yield",
            value=f"{row['baseline_yield']:.2f} t/ha",
        )
    with dc4:
        st.metric(
            label="📦 Pred. Production",
            value=f"{d_pred_prod:,.1f} k tons",
        )
    with dc5:
        st.metric(
            label="🏭 Est. CO₂e per Ton",
            value=f"{d_co2_per_ton:.0f} kg",
            delta=f"{d_co2_evol:+.1f}% vs baseline",
            delta_color="inverse",
        )
else:
    st.info(f"No data available for **{selected_dept}** in {selected_year} / {scenario_label}.")

# ── risk-to-action decision engine (quadrant chart) ─────────────────────────
st.markdown("---")
st.subheader("💡 Risk-to-Action Decision Engine")
st.markdown(
    "Map climate risk against procurement exposure to generate **targeted action packs** "
    "for ESG, procurement & operations teams."
)

# --- compute quadrant data (dynamic — tied to selected year) ---
quad_src = df_year.copy()  # already filtered by selected_year + selected_scenario

# Risk (X-axis): yield change % for the selected year
quad_src["yield_change_pct"] = (
    (quad_src["pred_yield"] - quad_src["baseline_yield"])
    / quad_src["baseline_yield"] * 100
)

# Exposure (Y-axis): predicted production share for the selected year
quad_src["pred_prod_q"] = quad_src["pred_yield"] * quad_src["area_2014"]
total_pred_prod_year = quad_src["pred_prod_q"].sum()
quad_src["procurement_share"] = (
    quad_src["pred_prod_q"] / total_pred_prod_year * 100
    if total_pred_prod_year else 0
)

quad_df = quad_src[["code_dep", "nom_dep", "yield_change_pct", "procurement_share", "pred_prod_q"]].copy()
quad_df = quad_df.dropna(subset=["yield_change_pct", "procurement_share"])

# Thresholds for quadrant lines
RISK_THRESHOLD = quad_df["yield_change_pct"].median()
EXPOSURE_THRESHOLD = quad_df["procurement_share"].median()

# Assign quadrants
def assign_quadrant(row):
    high_risk = row["yield_change_pct"] <= RISK_THRESHOLD
    high_exposure = row["procurement_share"] >= EXPOSURE_THRESHOLD
    if high_exposure and high_risk:
        return "Q1: High Exposure / High Risk"
    elif high_exposure and not high_risk:
        return "Q2: High Exposure / Low Risk"
    elif not high_exposure and high_risk:
        return "Q3: Low Exposure / High Risk"
    else:
        return "Q4: Low Exposure / Low Risk"

quad_df["quadrant"] = quad_df.apply(assign_quadrant, axis=1)

QUAD_COLORS = {
    "Q1: High Exposure / High Risk": "#d7191c",
    "Q2: High Exposure / Low Risk":  "#1a9850",
    "Q3: Low Exposure / High Risk":  "#fdae61",
    "Q4: Low Exposure / Low Risk":   "#91bfdb",
}

# --- scatter plot ---
fig_quad = px.scatter(
    quad_df,
    x="yield_change_pct",
    y="procurement_share",
    text="nom_dep",
    color="quadrant",
    color_discrete_map=QUAD_COLORS,
    size="procurement_share",
    size_max=30,
    hover_name="nom_dep",
    hover_data={
        "yield_change_pct": ":.1f",
        "procurement_share": ":.2f",
        "quadrant": True,
        "nom_dep": False,
    },
    labels={
        "yield_change_pct": f"Risk: Yield Change in {selected_year} (%)",
        "procurement_share": f"Exposure: Production Share in {selected_year} (%)",
        "quadrant": "Quadrant",
    },
)

fig_quad.add_vline(x=RISK_THRESHOLD, line_dash="dash", line_color="grey", line_width=1)
fig_quad.add_hline(y=EXPOSURE_THRESHOLD, line_dash="dash", line_color="grey", line_width=1)

# Quadrant labels as annotations
fig_quad.add_annotation(
    x=quad_df["yield_change_pct"].min() * 0.95,
    y=quad_df["procurement_share"].max() * 0.95,
    text="<b>Q1</b> Core Crisis Zone",
    showarrow=False, font=dict(size=11, color="#d7191c"),
    xanchor="left",
)
fig_quad.add_annotation(
    x=quad_df["yield_change_pct"].max() * 0.95,
    y=quad_df["procurement_share"].max() * 0.95,
    text="<b>Q2</b> Strategic Base",
    showarrow=False, font=dict(size=11, color="#1a9850"),
    xanchor="right",
)
fig_quad.add_annotation(
    x=quad_df["yield_change_pct"].min() * 0.95,
    y=quad_df["procurement_share"].min() + 0.1,
    text="<b>Q3</b> Marginal High-Risk",
    showarrow=False, font=dict(size=11, color="#e66101"),
    xanchor="left",
)
fig_quad.add_annotation(
    x=quad_df["yield_change_pct"].max() * 0.95,
    y=quad_df["procurement_share"].min() + 0.1,
    text="<b>Q4</b> Long-Tail Safe",
    showarrow=False, font=dict(size=11, color="#4575b4"),
    xanchor="right",
)

fig_quad.update_traces(textposition="top center", textfont_size=9)
fig_quad.update_layout(
    height=600,
    margin=dict(l=40, r=40, t=40, b=40),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
        font=dict(size=11),
    ),
)

st.plotly_chart(fig_quad, use_container_width=True)

# --- action cards ---
st.markdown("---")
st.subheader("🎯 Action Pack")

selected_action_dept = st.selectbox(
    "Select a department to view its Action Pack:",
    options=sorted(quad_df["nom_dep"].unique().tolist()),
    index=0,
    key="action_dept",
)

action_row = quad_df[quad_df["nom_dep"] == selected_action_dept]

if not action_row.empty:
    ar = action_row.iloc[0]

    ac1, ac2 = st.columns([1, 3])

    with ac1:
        st.metric(label=f"Risk (Yield Change {selected_year})", value=f"{ar['yield_change_pct']:+.1f}%")
        st.metric(label="Exposure (Procurement Share)", value=f"{ar['procurement_share']:.2f}%")
        quadrant_name = ar["quadrant"]
        q_color = QUAD_COLORS.get(quadrant_name, "#999")
        st.markdown(
            f'<div style="background-color:{q_color}20; border-left:4px solid {q_color}; '
            f'padding:10px; border-radius:4px;">'
            f'<b>{quadrant_name}</b></div>',
            unsafe_allow_html=True,
        )

    with ac2:
        if "Q1" in quadrant_name:
            st.error("**Core Strategy: Adaptation (Deep Intervention)**")
            st.markdown(
                """
**🌍 ESG:**
- Launch **Regenerative Agriculture (RegenAg) pilots** and subsidise cover crops to help retain soil moisture.
- Partner with financial institutions to provide **low-interest loans** for micro-irrigation infrastructure.

**🛒 Procurement & Supply Chain:**
- Secure **3–5 year long-term contracts** to stabilise farmer commitment.
- Actively seek alternative suppliers in Q2 or Q4 to offset anticipated yield drops.

**👨‍🌾 Farmer Empowerment:**
- Provide **drought-resistant seed varieties** at subsidised prices.
- Offer **premium buy-back** for farmers adopting water-saving RegenAg practices (verified via LCA data).
"""
            )

        elif "Q2" in quadrant_name:
            st.success("**Core Strategy: Operations (Optimise & Expand)**")
            st.markdown(
                """
**🏭 Operations & Infrastructure:**
- Increase the share of **long-term contracts** in this region to secure a more stable supply base.
- Evaluate relocating **primary processing or core warehousing** closer to this region.

**🌍 ESG & Governance:**
- Prioritise collecting **Primary Data** (instead of using Agribalyse industry averages) for more accurate carbon footprint accounting.
- Use this high-performing quadrant as the "gold standard" benchmark for your sustainability reports.

**👨‍🌾 Farmer Empowerment:**
- Fund **drones, sensors, or AI tools** to maximize the yield of these already-stable farms.
"""
            )

        elif "Q3" in quadrant_name:
            st.warning("**Core Strategy: Procurement (Agile & Substitute)**")
            st.markdown(
                """
**🛒 Supply Chain & Procurement:**
- Work with R&D to test if the specific crops sourced here can be replaced by alternatives from lower-risk regions.
- Move to seasonal or annual contracts to maintain flexibility and avoid being locked into failing harvests.

**🌍 ESG & Governance:**
- Provide **early climate warning** services and support farmers on switching to more drought-tolerant crops next season.

**👨‍🌾 Farmer Empowerment:**
- Fund **crop diversification trials** to help farmers transition to viable alternatives.
"""
            )

        else:  # Q4
            st.info("**Core Strategy: Governance (Monitor & Comply)**")
            st.markdown(
                """
**📋 Supply Chain & Procurement:**
- Gradually shift small test volumes from High-Risk (Q1) zones to these emerging regions.
- Assess and build local warehousing and transport infrastructure ahead of future scale-ups.

**🌍 ESG & Governance:**
- Mandate regenerative agriculture practices from the start to build resilience early on.

**👨‍🌾 Farmer Empowerment:**
- Deploy **agronomists** to help farmers optimize crops that are newly thriving due to climate shifts.
- Provide micro-grants or financing for **equipment** so farmers can handle growing yields.
"""
            )

# ── footer info ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"""
**Assumptions & Notes**
- **Area** held constant at 2014 levels ({total_area / 1_000:,.0f} k ha nationally)
- **CO₂e** fixed at {EF_HA:,} kg CO₂e / ha, which is sourced from authoritative French **LCA databases** (Agribalyse)
- **Baseline** = average over **2004–2014**
- **Market price assumption**: €{assumed_price_per_ton} / ton
- **Scenario**: {scenario_label}
"""
)
