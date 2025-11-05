
import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ------------------ Page Setup ------------------
st.set_page_config(
    page_title="🌿 Clustering Ecosystem Dashboard",
    page_icon="🌍",
    layout="wide"
)

# ------------------ Styling ------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #f7fff9, #e9f7ef);
    color: #0a3d2e;
    font-family: 'Poppins', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #f0fff4;
    border-right: 3px solid #2a9d8f;
}

/* Headings */
h1, h2, h3 {
    color: #1b4332;
    font-weight: 600;
}

/* Buttons */
.stButton>button {
    background-color: #2a9d8f;
    color: white;
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #1f776b;
    transform: scale(1.04);
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    background-color: #ffffff;
    box-shadow: 0 0 10px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ------------------ Header ------------------
col1, col2 = st.columns([1, 6])
logo_path = os.path.join("assets", "logo.png")
with col1:
    if os.path.exists(logo_path):
        st.image(logo_path, width=100)
with col2:
    st.markdown("<h1>🌿 Clustering Ecosystem Patches</h1>", unsafe_allow_html=True)
    st.markdown("<h4>Analyzing Energy Flow Patterns Using Machine Learning</h4>", unsafe_allow_html=True)

st.markdown("---")

# ------------------ Refresh Button ------------------
if st.button("🔄 Refresh Dashboard"):
    st.rerun()



# ------------------ Load Dataset ------------------
csv_path = os.path.join("data", "large_sample_data.csv")
if not os.path.exists(csv_path):
    st.error("❌ No dataset found. Please place `sample_data.csv` inside `/data` folder.")
    st.stop()

df = pd.read_csv(csv_path)
st.success("✅ Dataset loaded successfully.")

import requests

# ---------------- Live Environmental Data (Open-Meteo API Demo) ----------------
st.markdown("### ☀️ Live Environmental Snapshot")
city = st.text_input("Enter a city name for current temperature:", "Delhi")

if st.button("Fetch Live Data"):
    try:
        resp = requests.get(
            f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        )
        geo = resp.json()["results"][0]
        lat, lon = geo["latitude"], geo["longitude"]
        weather = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        ).json()
        temp = weather["current_weather"]["temperature"]
        wind = weather["current_weather"]["windspeed"]
        st.success(f"🌡️ Temperature: {temp}°C | 💨 Wind Speed: {wind} km/h at {city}")
    except Exception as e:
        st.warning("Could not fetch data. Check your connection or city name.")


# ------------------ Sidebar Controls ------------------
st.sidebar.header("⚙️ Preprocessing & Clustering Settings")
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

features = st.sidebar.multiselect("Select features for clustering", numeric_cols, default=numeric_cols[:4])
scaling_option = st.sidebar.selectbox("Scaling Method", ["None", "StandardScaler", "MinMaxScaler"])
algo = st.sidebar.selectbox("Select Clustering Algorithm", ["KMeans", "DBSCAN", "Agglomerative Clustering"])

# Sidebar cluster controls
if algo == "KMeans":
    n_clusters = st.sidebar.slider("Number of Clusters (k)", 2, 10, 3)
elif algo == "DBSCAN":
    eps = st.sidebar.slider("Neighborhood Size (EPS)", 0.1, 5.0, 0.5, step=0.1)
    min_samples = st.sidebar.slider("Minimum Samples", 1, 20, 5)
else:
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)
    linkage = st.sidebar.selectbox("Linkage Method", ["ward", "complete", "average", "single"])

# ------------------ Preprocessing ------------------
if scaling_option == "StandardScaler":
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[features])
elif scaling_option == "MinMaxScaler":
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[features])
else:
    df_scaled = df[features].values

df_scaled = pd.DataFrame(df_scaled, columns=features)
if df_scaled.isnull().values.any():
    df_scaled = df_scaled.fillna(df_scaled.mean())

# ------------------ Clustering ------------------
if algo == "KMeans":
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(df_scaled)
elif algo == "DBSCAN":
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(df_scaled)
else:
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(df_scaled)

df["Cluster"] = labels

# ------------------ Data + Cluster Stats ------------------
st.subheader("📋 Data Overview")
st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
st.dataframe(df.head(10), use_container_width=True)

try:
    sil = silhouette_score(df_scaled, labels)
    st.metric("🌱 Silhouette Score", f"{sil:.3f}")
except:
    st.warning("Silhouette score unavailable for single cluster or noise labels.")

st.markdown("### 🌿 Cluster Statistics")
cluster_summary = df.groupby("Cluster")[features].mean().round(2)
st.dataframe(cluster_summary)

# ------------------ 2D PCA Visualization ------------------
st.markdown("### 🌍 2D Cluster Visualization")
pca = PCA(n_components=2)
df_pca = pd.DataFrame(pca.fit_transform(df_scaled), columns=["PCA1", "PCA2"])
df_pca["Cluster"] = labels
fig_2d = px.scatter(
    df_pca,
    x="PCA1", y="PCA2",
    color=df_pca["Cluster"].astype(str),
    color_discrete_sequence=px.colors.qualitative.Set2,
    title="2D Cluster View (PCA Projection)",
    template="plotly_white"
)
fig_2d.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
st.plotly_chart(fig_2d, use_container_width=True)

# ------------------ 3D Cluster Visualization ------------------
st.markdown("### 🪐 3D Cluster Visualization")
pca3 = PCA(n_components=3)
df_3d = pd.DataFrame(pca3.fit_transform(df_scaled), columns=["X", "Y", "Z"])
df_3d["Cluster"] = labels
fig_3d = px.scatter_3d(
    df_3d, x="X", y="Y", z="Z",
    color=df_3d["Cluster"].astype(str),
    color_discrete_sequence=px.colors.qualitative.Vivid,
    title="Interactive 3D Cluster Map",
    opacity=0.8
)
fig_3d.update_traces(marker=dict(size=8, symbol="circle", line=dict(width=1, color="black")))
fig_3d.update_layout(scene=dict(
    xaxis=dict(title='Component 1'),
    yaxis=dict(title='Component 2'),
    zaxis=dict(title='Component 3'),
    bgcolor='rgba(245,255,245,0.8)'
))
st.plotly_chart(fig_3d, use_container_width=True)

# ------------------ Map Visualization ------------------
if {"lat", "lon"}.issubset(df.columns):
    st.markdown("### 🗺️ Geo Map View")
    fig_map = px.scatter_mapbox(
        df, lat="lat", lon="lon",
        color=df["Cluster"].astype(str),
        hover_data=features,
        zoom=3,
        mapbox_style="carto-positron",
        title="Cluster Distribution on Map",
        height=450
    )
    st.plotly_chart(fig_map, use_container_width=True)


# ------------------ Download Results ------------------
st.markdown("---")
csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Clustered Data (CSV)", csv_data, "clustered_results.csv", "text/csv")

st.success("🌳 Analysis Complete! Scroll down to explore all visualizations.")
# ---------------- Community Feedback ----------------
st.markdown("### 💬 Community Feedback")
user_name = st.text_input("Your Name:")
user_feedback = st.text_area("Share your feedback or sustainability idea:")
if st.button("Submit Feedback"):
    if user_name and user_feedback:
        with open("data/user_feedback.csv", "a", encoding="utf-8") as f:
            f.write(f"{user_name},{user_feedback}\n")
        st.success("🌿 Thank you for sharing your thoughts!")
    else:
        st.warning("Please fill in both fields before submitting.")

 