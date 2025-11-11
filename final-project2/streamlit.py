import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statistics
from datetime import datetime, timedelta

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="🏠 Housing Market Analysis Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# STYLE
# ==============================
st.markdown("""
<style>
:root { --bg: #1b1126; --card: #2a1630; --accent: #9b59b6; --muted: #d9cfe8; --panel: #241028; }
.stApp, .main, body { background-color: var(--bg) !important; color: var(--muted) !important; }
h1, h2, h3 { color: var(--accent) !important; }
.chart-container { background-color: var(--card); padding: 1rem; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); margin-bottom: 1rem; }
.metric-card { background-color: var(--panel); padding: 0.6rem 0.8rem; border-radius: 8px; text-align:center; }
.subheader-custom { color: var(--accent); font-weight:600; font-size:18px; margin-bottom:6px; }
.plotly-graph-div { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("housing.csv")
        return df
    except FileNotFoundError:
        uploaded = st.sidebar.file_uploader("📂 Upload housing.csv file", type=["csv"])
        if uploaded is not None:
            return pd.read_csv(uploaded)
        else:
            st.warning("⚠️ Please upload the housing.csv file to continue.")
            st.stop()

df = load_data()

# Check columns exist
required_cols = {"RM", "LSTAT", "PTRATIO", "MEDV"}
if not required_cols.issubset(df.columns):
    st.error(f"❌ Missing columns! Your CSV must include: {', '.join(required_cols)}")
    st.stop()

# ==============================
# DATA PROCESSING
# ==============================
def preprocess_data(df):
    processed = []
    for i, row in df.iterrows():
        base_date = datetime(2020, 1, 1)
        current_date = base_date + timedelta(days=(i % 48) * 30)
        price = row["MEDV"]
        rm = row["RM"]
        lstat = row["LSTAT"]
        ptratio = row["PTRATIO"]

        # Adjust categories to realistic Boston dataset scale
        if price < 20:
            price_category = "Budget"
        elif price < 30:
            price_category = "Mid-Range"
        elif price < 40:
            price_category = "Premium"
        else:
            price_category = "Luxury"

        room_category = "Small" if rm < 5 else ("Medium" if rm < 7 else "Large")
        socio_category = "High Income" if lstat < 10 else ("Middle Income" if lstat < 20 else "Low Income")
        school_category = "Excellent" if ptratio < 15 else ("Good" if ptratio < 20 else "Average")

        processed.append({
            "RM": rm,
            "LSTAT": lstat,
            "PTRATIO": ptratio,
            "MEDV": price,
            "Date": current_date,
            "Price_Category": price_category,
            "Room_Category": room_category,
            "Socio_Category": socio_category,
            "School_Category": school_category,
        })
    return processed

data_df = pd.DataFrame(preprocess_data(df))

# ==============================
# HEADER
# ==============================
st.markdown("<h1 style='text-align:center; color:#9b59b6;'>🏠 Housing Market Analysis Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# ==============================
# SIDEBAR FILTERS
# ==============================
st.sidebar.header("🔍 Filters")

def multi_filter(df, col_name, label):
    options = ["All"] + sorted(df[col_name].unique().tolist())
    selection = st.sidebar.multiselect(label, options, default=["All"])
    if "All" not in selection:
        df = df[df[col_name].isin(selection)]
    return df

for col, label in [
    ("Price_Category", "Price Category"),
    ("Room_Category", "Room Category"),
    ("Socio_Category", "Socioeconomic Category"),
    ("School_Category", "School Quality"),
]:
    data_df = multi_filter(data_df, col, label)

if data_df.empty:
    st.warning("⚠️ No data matches your filters.")
    st.stop()

# ==============================
# METRICS
# ==============================
st.markdown('<div class="chart-container"><div class="subheader-custom">💜 Key Indicators</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Average Price", f"${data_df['MEDV'].mean():,.1f}")
with c2:
    st.metric("Average Rooms", f"{data_df['RM'].mean():.1f}")
with c3:
    st.metric("Avg LSTAT (%)", f"{data_df['LSTAT'].mean():.1f}%")
with c4:
    st.metric("Total Properties", f"{len(data_df):,}")
st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# CHARTS
# ==============================
fig1 = px.histogram(data_df, x="MEDV", nbins=20, title="🏡 Price Distribution", color_discrete_sequence=["#9b59b6"])
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.scatter(data_df, x="RM", y="MEDV", color="Price_Category", title="Rooms vs Price", color_discrete_sequence=px.colors.qualitative.Pastel)
st.plotly_chart(fig2, use_container_width=True)

corr = data_df[["RM", "LSTAT", "PTRATIO", "MEDV"]].corr()
fig3 = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Correlation Heatmap")
st.plotly_chart(fig3, use_container_width=True)

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.markdown("<p style='text-align:center; color:#999;'>© 2025 Housing Dashboard | Designed by Salma 💜</p>", unsafe_allow_html=True)
