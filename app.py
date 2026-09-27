from __future__ import annotations

import io
from dataclasses import dataclass

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from folium.plugins import Fullscreen, HeatMap
from streamlit_folium import st_folium


st.set_page_config(
    page_title="HeatLens | Urban Heat Intelligence",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Theme ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#10211b; --muted:#65736d; --mint:#dff5e9; --green:#176b4d; --orange:#ee7959; --yellow:#f5b942; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
    .stApp { background: #f6faf8; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.03em; }
    h1 { font-size: 2.8rem !important; margin-bottom: .2rem !important; }
    h2 { font-size: 1.55rem !important; }
    [data-testid="stSidebar"] { background: #10211b; }
    [data-testid="stSidebar"] * { color: #edf8f1 !important; }
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div { color: white !important; }
    .hero { background: linear-gradient(120deg,#143c2d 0%,#1d7554 62%,#53ae79 100%); color:white; padding:2.1rem 2.4rem; border-radius:24px; margin-bottom:1.2rem; box-shadow:0 14px 35px rgba(25,94,65,.14); }
    .hero p { max-width:740px; color:#d8f1e2; font-size:1.05rem; margin-top:.5rem; }
    .eyebrow { text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700; color:#a9e7c1; }
    .metric { background:white; border:1px solid #e3eee8; border-radius:18px; padding:1rem 1.1rem; min-height:115px; box-shadow:0 5px 16px rgba(28,70,51,.04); }
    .metric-label { color:var(--muted); font-size:.78rem; font-weight:600; text-transform:uppercase; letter-spacing:.06em; }
    .metric-value { font-family:'Space Grotesk'; font-size:1.75rem; font-weight:700; margin-top:.4rem; }
    .metric-note { color:#2d8a5e; font-size:.78rem; font-weight:600; margin-top:.25rem; }
    .section-note { color:var(--muted); margin-top:-.65rem; margin-bottom:1rem; }
    .pill { display:inline-block; padding:.32rem .65rem; border-radius:30px; background:#e5f6eb; color:#196542; font-weight:700; font-size:.75rem; }
    .recommendation { background:white; border:1px solid #e3eee8; border-left:5px solid #2f9d68; border-radius:14px; padding:1rem 1rem .9rem; margin:.5rem 0; }
    .recommendation h4 { margin:0 0 .25rem; font-family:'Space Grotesk'; }
    .recommendation p { color:var(--muted); margin:0; font-size:.9rem; }
    footer { text-align:center; color:#81918a; padding:2rem 0 1rem; font-size:.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@dataclass
class Intervention:
    name: str
    icon: str
    description: str
    cooling: float
    cost: str
    carbon: float
    best_for: str


INTERVENTIONS = [
    Intervention("Street tree canopy", "🌳", "Plant large-canopy trees along the hottest pedestrian routes.", 2.8, "$$", 18.4, "Parking lots & sidewalks"),
    Intervention("Cool roofs", "🏠", "Apply high-albedo coating to broad, low-slope rooftops.", 2.1, "$", 12.2, "Large roof footprints"),
    Intervention("Permeable surfaces", "🧱", "Replace dark asphalt with permeable, light-colored paving.", 1.7, "$$", 8.6, "Plazas & car parks"),
    Intervention("Shade structures", "⛱️", "Add solar shade sails or lightweight structures over gathering areas.", 3.4, "$$$", 6.1, "Transit & play areas"),
    Intervention("Rain garden", "💧", "Capture stormwater with planted bioswales and rain gardens.", 1.4, "$$", 10.8, "Low points & edges"),
]


@st.cache_data
def create_heat_surface(area: str, hour: int, tree_cover: int, reflective: int, water: int, seed: int):
    rng = np.random.default_rng(seed)
    configs = {
        "Campus core": (42.3601, -71.0589, 30, 0.0019),
        "Riverside neighbourhood": (34.0522, -118.2437, 36, 0.0023),
        "Market district": (40.7128, -74.0060, 40, 0.0018),
    }
    lat0, lon0, size, spread = configs[area]
    grid = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(grid, grid)
    # Hot hardscape clusters + a cooler green/water corridor.
    heat = 30.4 + 2.0 * (xx + 1) / 2 + 1.25 * np.sin(yy * 4)
    heat += 5.1 * np.exp(-((xx + .48) ** 2 + (yy - .12) ** 2) / .065)
    heat += 3.8 * np.exp(-((xx - .38) ** 2 + (yy + .44) ** 2) / .09)
    heat += 2.7 * np.exp(-((xx - .1) ** 2 + (yy - .05) ** 2) / .035)
    heat -= (tree_cover / 100) * 3.2 * np.exp(-((xx + .2) ** 2 + (yy + .55) ** 2) / .19)
    heat -= (water / 100) * 2.3 * np.exp(-((xx - .68) ** 2 + (yy + .05) ** 2) / .08)
    heat -= (reflective / 100) * 1.8
    heat += ((hour - 8) / 8) * 1.5
    heat += rng.normal(0, .18, heat.shape)
    lats = lat0 + yy * spread
    lons = lon0 + xx * spread * 1.22
    df = pd.DataFrame({"lat": lats.ravel(), "lon": lons.ravel(), "temp": heat.ravel()})
    df["risk"] = pd.cut(df.temp, bins=[0, 32, 35, 38, 100], labels=["Low", "Watch", "High", "Severe"])
    return df, (lat0, lon0)


def metric(label: str, value: str, note: str):
    st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)


def build_map(df: pd.DataFrame, center: tuple[float, float], show_cooling: bool):
    m = folium.Map(location=center, zoom_start=15, tiles="CartoDB positron", control_scale=True)
    heat_points = [[row.lat, row.lon, max(0.1, (row.temp - 29) / 10)] for row in df.itertuples()]
    HeatMap(heat_points, radius=22, blur=18, min_opacity=.32, gradient={"0.2":"#2b83ba","0.45":"#abdda4","0.65":"#fdae61","0.85":"#f46d43","1":"#d73027"}, name="Thermal intensity").add_to(m)
    if show_cooling:
        cool = df.nsmallest(90, "temp")
        HeatMap([[r.lat, r.lon, .35] for r in cool.itertuples()], radius=28, blur=22, gradient={"0":"#7fcdbb",".5":"#41b6c4","1":"#225ea8"}, name="Cooling potential").add_to(m)
    for label, subset, color in [("Severe hotspot", df.nlargest(2, "temp"), "#d73027"), ("Cooling opportunity", df.nsmallest(2, "temp"), "#168c62")]:
        for row in subset.itertuples():
            folium.CircleMarker([row.lat, row.lon], radius=8, color=color, fill=True, fill_opacity=.9, popup=f"<b>{label}</b><br>{row.temp:.1f}°C").add_to(m)
    folium.LayerControl().add_to(m)
    Fullscreen(position="topright").add_to(m)
    return m


# ---------- Sidebar ----------
st.sidebar.markdown("# 🌡️ HeatLens")
st.sidebar.caption("Urban heat intelligence & cooling planner")
st.sidebar.divider()
st.sidebar.markdown("### Study area")
area = st.sidebar.selectbox("Choose a demo area", ["Campus core", "Riverside neighbourhood", "Market district"])
hour = st.sidebar.slider("Time of day", 6, 20, 14, 1, format="%d:00")
st.sidebar.markdown("### What-if levers")
tree_cover = st.sidebar.slider("Tree canopy coverage", 0, 60, 18, 1, format="%d%%")
reflective = st.sidebar.slider("Reflective surface coverage", 0, 60, 12, 1, format="%d%%")
water = st.sidebar.slider("Water & rain gardens", 0, 35, 8, 1, format="%d%%")
show_cooling = st.sidebar.toggle("Show cooling opportunities", True)
st.sidebar.divider()
st.sidebar.info("💡 Tip: Move the sliders to test a cooling scenario. The map and impact estimates update instantly.")

# ---------- Data ----------
df, center = create_heat_surface(area, hour, tree_cover, reflective, water, seed=[2, 7, 11][["Campus core", "Riverside neighbourhood", "Market district"].index(area)])
hottest = df.temp.max()
avg = df.temp.mean()
severe = (df.temp >= 38).sum()
base_df, _ = create_heat_surface(area, hour, 0, 0, 0, seed=[2, 7, 11][["Campus core", "Riverside neighbourhood", "Market district"].index(area)])
cooled = base_df.temp.mean() - avg

# ---------- Main ----------
st.markdown('<div class="hero"><div class="eyebrow">Live urban climate intelligence</div><h1>See heat. Plan cool.</h1><p>HeatLens turns a neighbourhood-scale thermal model into an actionable cooling plan — so communities can focus resources where people feel heat most.</p></div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1: metric("Average surface temperature", f"{avg:.1f}°C", f"{cooled:+.1f}°C vs. baseline")
with c2: metric("Peak hotspot", f"{hottest:.1f}°C", "Priority for intervention")
with c3: metric("Severe heat cells", f"{severe:,}", "Above 38°C threshold")
with c4: metric("Cooling potential", f"{max(0, cooled + 1.8):.1f}°C", "With planned interventions")

st.write("")
left, right = st.columns([1.65, 1])
with left:
    st.markdown("## Thermal surface map")
    st.markdown(f'<span class="pill">● LIVE MODEL</span> &nbsp; {area} · {hour}:00 local time', unsafe_allow_html=True)
    st.caption("Red zones are heat hotspots; blue/green zones show cooler surfaces. Select a marker to inspect it.")
    st_folium(build_map(df, center, show_cooling), height=535, use_container_width=True, returned_objects=[])
with right:
    st.markdown("## Heat profile")
    st.markdown('<div class="section-note">Distribution of modeled surface temperatures</div>', unsafe_allow_html=True)
    fig = px.histogram(df, x="temp", nbins=22, color_discrete_sequence=["#ee7959"], labels={"temp": "Surface temperature (°C)", "count": "Cells"})
    fig.update_layout(height=245, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", bargap=.08, showlegend=False)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#edf2ef")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("### Fast read")
    high_share = (df.temp >= 35).mean() * 100
    st.write(f"**{high_share:.0f}%** of modeled cells are at or above the heat-watch threshold. The highest-impact first move is to target hardscape around the two red markers.")
    st.progress(min(1.0, high_share / 100), text="Area requiring heat-aware design")

st.divider()
st.markdown("## Cooling plan builder")
st.markdown('<div class="section-note">Prioritized interventions for this study area. Combine actions to turn heat intelligence into a project brief.</div>', unsafe_allow_html=True)
selected = st.multiselect("Add interventions to your plan", [f"{i.icon} {i.name}" for i in INTERVENTIONS], default=[f"{INTERVENTIONS[0].icon} {INTERVENTIONS[0].name}", f"{INTERVENTIONS[1].icon} {INTERVENTIONS[1].name}"], label_visibility="collapsed")
selected_names = {x.split(" ", 1)[1] for x in selected}
plan = [i for i in INTERVENTIONS if i.name in selected_names]
if plan:
    p1, p2, p3 = st.columns([1.2, 1, 1])
    with p1: metric("Combined cooling impact", f"{sum(i.cooling for i in plan):.1f}°C", "Estimated peak reduction")
    with p2: metric("Annual carbon benefit", f"{sum(i.carbon for i in plan):.0f} t", "Equivalent CO₂ avoided")
    with p3: metric("Implementation scale", "Medium", f"{len(plan)} actions selected")
    cols = st.columns(min(3, len(plan)))
    for col, intervention in zip(cols * ((len(plan) + 2) // 3), plan):
        with col:
            st.markdown(f'<div class="recommendation"><h4>{intervention.icon} {intervention.name}</h4><p>{intervention.description}</p><br><small><b>Impact:</b> {intervention.cooling:.1f}°C &nbsp; <b>Cost:</b> {intervention.cost}<br><b>Best for:</b> {intervention.best_for}</small></div>', unsafe_allow_html=True)
    impact_df = pd.DataFrame({"Intervention": [i.name for i in plan], "Cooling impact (°C)": [i.cooling for i in plan]})
    impact_fig = px.bar(impact_df, x="Cooling impact (°C)", y="Intervention", orientation="h", color="Cooling impact (°C)", color_continuous_scale=["#b8e6ca", "#168c62"])
    impact_fig.update_layout(height=230, margin=dict(l=0,r=0,t=10,b=0), coloraxis_showscale=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(impact_fig, use_container_width=True, config={"displayModeBar": False})
else:
    st.warning("Choose at least one intervention to build a cooling plan.")

st.divider()
with st.expander("About this prototype & data"):
    st.write("HeatLens uses a deterministic, synthetic surface-temperature model designed for a hackathon demonstration. It combines hardscape heat clusters, vegetation cooling, water features, reflectivity, time of day, and a small amount of spatial noise. Replace `create_heat_surface` with satellite, sensor, or GIS data when moving toward a field pilot.")
    download = df.round({"lat": 6, "lon": 6, "temp": 2}).to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download modeled heat cells (CSV)", data=download, file_name="heatlens_surface.csv", mime="text/csv")

st.markdown("<footer>HeatLens · Built for climate resilience · Prototype estimates are directional, not official measurements.</footer>", unsafe_allow_html=True)
