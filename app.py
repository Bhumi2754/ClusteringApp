# # # # app.py
# # # import streamlit as st
# # # import pandas as pd
# # # from src.data_loader import load_csv
# # # from src.preprocess import preprocess_dataframe, generate_feature_summary
# # # from src.clustering import run_kmeans, run_dbscan, run_agglomerative, compute_cluster_metrics
# # # from src.visualize import scatter_plot_2d, plot_silhouette_scores, plot_dendrogram_plotly
# # # from src.utils import df_to_csv_download, ensure_data_folder
# # # import os

# # # st.set_page_config(page_title="Clustering Ecosystem Patches", layout="wide", page_icon="🌱")

# # # ensure_data_folder()

# # # st.title("Clustering Ecosystem Patches by Energy-Flow Patterns")
# # # st.markdown(
# # #     """
# # #     Upload a CSV of ecosystem patches (feature columns like NPP, biomass, trophic_efficiency, temperature, etc.)
# # #     or use the synthetic sample dataset. Then choose preprocessing and clustering options.
# # #     """
# # # )

# # # # --- Sidebar: data upload and options
# # # st.sidebar.header("Data / Settings")
# # # use_sample = st.sidebar.checkbox("Use sample dataset (generated)", value=True)

# # # uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# # # if use_sample and uploaded_file is None:
# # #     csv_path = os.path.join("data", "sample_data.csv")
# # #     if not os.path.exists(csv_path):
# # #         st.sidebar.info("Sample dataset not found — generating sample data.")
# # #         os.system("python data/generate_sample.py")
# # #     df = pd.read_csv(csv_path)
# # # else:
# # #     df = load_csv(uploaded_file)
# # #     if df is None:
# # #         st.stop()

# # # st.sidebar.markdown("---")
# # # st.sidebar.subheader("Preprocessing")
# # # impute_method = st.sidebar.selectbox("Imputation for missing values", ["mean", "median", "drop"], index=0)
# # # scaling = st.sidebar.selectbox("Feature scaling", ["None", "StandardScaler (z-score)", "MinMaxScaler"], index=1)
# # # select_features = st.sidebar.multiselect("Select feature columns for clustering", options=list(df.columns), default=list(df.columns)[:5])

# # # if len(select_features) < 2:
# # #     st.sidebar.error("Pick at least 2 features to cluster.")
# # #     st.stop()

# # # # Preprocess
# # # X, processed_df = preprocess_dataframe(df[select_features], impute_method=impute_method, scaling=scaling)
# # # st.subheader("Feature summary")
# # # st.dataframe(generate_feature_summary(processed_df))

# # # # --- Clustering options
# # # st.sidebar.markdown("---")
# # # st.sidebar.subheader("Clustering")
# # # algorithm = st.sidebar.selectbox("Algorithm", ["KMeans", "DBSCAN", "Agglomerative"])
# # # if algorithm == "KMeans":
# # #     n_clusters = st.sidebar.slider("n_clusters", 2, 10, 3)
# # # elif algorithm == "DBSCAN":
# # #     eps = st.sidebar.slider("eps (radius)", 0.1, 5.0, 0.5, step=0.1)
# # #     min_samples = st.sidebar.slider("min_samples", 1, 20, 5)
# # # else:
# # #     n_clusters = st.sidebar.slider("n_clusters (for agglo)", 2, 10, 3)
# # #     linkage = st.sidebar.selectbox("Linkage", ["ward", "complete", "average", "single"])

# # # run_button = st.sidebar.button("Run Clustering")

# # # # Show raw/sample data optional
# # # with st.expander("Raw / original data (first 10 rows)"):
# # #     st.dataframe(df.head(10))

# # # if run_button:
# # #     if algorithm == "KMeans":
# # #         labels, model = run_kmeans(X, n_clusters=n_clusters)
# # #     elif algorithm == "DBSCAN":
# # #         labels, model = run_dbscan(X, eps=eps, min_samples=min_samples)
# # #     else:
# # #         labels, model = run_agglomerative(X, n_clusters=n_clusters, linkage=linkage)

# # #     processed_df["cluster"] = labels
# # #     st.success("Clustering complete!")
# # #     # show counts
# # #     st.subheader("Cluster counts")
# # #     st.write(processed_df["cluster"].value_counts().sort_index())

# # #     # Show table of cluster means
# # #     st.subheader("Cluster summary (means)")
# # #     st.dataframe(processed_df.groupby("cluster").mean())

# # #     # Scatter matrix / 2D scatter: allow user pick x,y
# # #     st.subheader("Cluster visualization")
# # #     col1, col2 = st.columns([2, 1])
# # #     with col1:
# # #         x_feature = st.selectbox("X axis", select_features, index=0)
# # #         y_feature = st.selectbox("Y axis", select_features, index=1)
# # #         fig = scatter_plot_2d(processed_df, x_feature, y_feature, label_column="cluster")
# # #         st.plotly_chart(fig, use_container_width=True)

# # #     with col2:
# # #         # Metrics
# # #         st.subheader("Metrics")
# # #         metrics = compute_cluster_metrics(X, labels)
# # #         for k, v in metrics.items():
# # #             st.write(f"**{k}**: {v:.4f}" if isinstance(v, float) else f"**{k}**: {v}")

# # #     # Silhouette plot (only for 2..10 clusters and not for noisy DBSCAN labels with -1 only)
# # #     if algorithm in ["KMeans", "Agglomerative"] and len(set(labels)) > 1:
# # #         st.subheader("Silhouette / cluster quality")
# # #         fig_sil = plot_silhouette_scores(X, labels)
# # #         st.plotly_chart(fig_sil, use_container_width=True)

# # #     # Dendrogram for Agglomerative
# # #     if algorithm == "Agglomerative":
# # #         st.subheader("Dendrogram (Agglomerative)")
# # #         try:
# # #             fig_den = plot_dendrogram_plotly(X)
# # #             st.plotly_chart(fig_den, use_container_width=True)
# # #         except Exception as e:
# # #             st.warning(f"Dendrogram could not be computed: {e}")

# # #     # Download resulting CSV
# # #     st.subheader("Export clusters")
# # #     csv_bytes = df_to_csv_download(processed_df)
# # #     st.download_button("Download clustered data (CSV)", data=csv_bytes, file_name="clustered_results.csv", mime="text/csv")

# # #     # show model details
# # #     st.subheader("Model info")
# # #     st.write(model)

# # # st.sidebar.markdown("---")
# # # st.sidebar.caption("Built using the project synopsis provided. See README for details.")
# # # import streamlit as st

# # # st.title("Welcome to ClusteringApp!")
# # # st.write("This is a sample Streamlit application.")
# # # st.


# # # import streamlit as st
# # # import pandas as pd
# # # import os

# # # # Set title
# # # st.title("Welcome to ClusteringApp!")
# # # st.write("This is a sample Streamlit application.")

# # # # Define the path to your sample data
# # # csv_path = os.path.join("data", "sample_data.csv")

# # # # Check if the file exists
# # # if os.path.exists(csv_path):
# # #     # Load CSV using pandas
# # #     df = pd.read_csv(csv_path)
    
# # #     # Show confirmation
# # #     st.success(f"Loaded {csv_path} successfully! ✅")
    
# # #     # Display first few rows
# # #     st.subheader("📊 Sample Data Preview:")
# # #     st.dataframe(df.head(10))  # shows first 10 rows
    
# # #     # Optionally show shape and column info
# # #     st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
# # #     st.write("Columns:", list(df.columns))
# # # else:
# # #     st.error("❌ sample_data.csv not found! Please run data/generate_sample.py first.")




# # # #2nd iterface
# # # import streamlit as st
# # # import pandas as pd
# # # import os
# # # import plotly.express as px
# # # from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
# # # from sklearn.preprocessing import StandardScaler, MinMaxScaler
# # # from sklearn.metrics import silhouette_score
# # # import numpy as np

# # # # --------------------- Page Setup ---------------------
# # # st.set_page_config(
# # #     page_title="ClusteringApp 🌱",
# # #     page_icon="🌿",
# # #     layout="wide"
# # # )

# # # # --------------------- Header Section ---------------------
# # # logo_path = os.path.join("assets", "logo.png")
# # # col1, col2 = st.columns([1, 5])
# # # with col1:
# # #     if os.path.exists(logo_path):
# # #         st.image(logo_path, width=100)
# # # with col2:
# # #     st.title("🌱 Clustering Ecosystem Patches")
# # #     st.markdown("#### By Energy-Flow Patterns using Machine Learning")

# # # st.markdown("---")

# # # # --------------------- Load Dataset ---------------------
# # # csv_path = os.path.join("data", "sample_data.csv")

# # # if os.path.exists(csv_path):
# # #     df = pd.read_csv(csv_path)
# # #     st.success(f"✅ Loaded dataset from `{csv_path}`")

# # #     st.markdown("### 📊 Dataset Overview")
# # #     st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
# # #     st.write("**Columns:**", ", ".join(df.columns))

# # #     with st.expander("🔍 Preview Data"):
# # #         st.dataframe(df.head(10))

# # #     # --------------------- Preprocessing Options ---------------------
# # #     st.markdown("### ⚙️ Preprocessing Options")
# # #     numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
# # #     features = st.multiselect("Select features for clustering", numeric_cols, default=numeric_cols[:4])

# # #     scaling_option = st.selectbox("Select scaling method", ["None", "StandardScaler", "MinMaxScaler"])
# # #     if scaling_option == "StandardScaler":
# # #         scaler = StandardScaler()
# # #         df_scaled = scaler.fit_transform(df[features])
# # #     elif scaling_option == "MinMaxScaler":
# # #         scaler = MinMaxScaler()
# # #         df_scaled = scaler.fit_transform(df[features])
# # #     else:
# # #         df_scaled = df[features].values

# # #     # Convert scaled data to DataFrame
# # #     df_scaled = pd.DataFrame(df_scaled, columns=features)

# # #     # --------------------- Handle Missing Values ---------------------
# # #     if df_scaled.isnull().values.any():
# # #         st.warning("⚠️ Missing values detected in the dataset. Imputing with column means...")
# # #         df_scaled = df_scaled.fillna(df_scaled.mean())
# # #     else:
# # #         st.info("✅ No missing values detected in selected features.")

# # #     # --------------------- Choose Clustering Algorithm ---------------------
# # #     st.markdown("### 🧠 Clustering Configuration")
# # #     algo = st.selectbox("Choose Algorithm", ["KMeans", "DBSCAN", "Agglomerative Clustering"])

# # #     if algo == "KMeans":
# # #         n_clusters = st.slider("Number of clusters (k)", 2, 10, 3)
# # #         run = st.button("🚀 Run KMeans Clustering")
# # #         if run:
# # #             model = KMeans(n_clusters=n_clusters, random_state=42)
# # #             labels = model.fit_predict(df_scaled)
# # #             df["Cluster"] = labels
# # #             st.success("✅ KMeans clustering completed!")

# # #     elif algo == "DBSCAN":
# # #         eps = st.slider("EPS (neighborhood size)", 0.1, 5.0, 0.5, step=0.1)
# # #         min_samples = st.slider("Min Samples", 1, 20, 5)
# # #         run = st.button("🚀 Run DBSCAN Clustering")
# # #         if run:
# # #             model = DBSCAN(eps=eps, min_samples=min_samples)
# # #             labels = model.fit_predict(df_scaled)
# # #             df["Cluster"] = labels
# # #             st.success("✅ DBSCAN clustering completed!")

# # #     else:  # Agglomerative Clustering
# # #         n_clusters = st.slider("Number of clusters", 2, 10, 3)
# # #         linkage = st.selectbox("Linkage Method", ["ward", "complete", "average", "single"])
# # #         run = st.button("🚀 Run Agglomerative Clustering")
# # #         if run:
# # #             model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
# # #             labels = model.fit_predict(df_scaled)
# # #             df["Cluster"] = labels
# # #             st.success("✅ Agglomerative clustering completed!")

# # #     # --------------------- Show Results ---------------------
# # #     if "Cluster" in df.columns:
# # #         st.markdown("### 📊 Cluster Results")

# # #         col1, col2 = st.columns(2)
# # #         with col1:
# # #             st.write("**Cluster Counts:**")
# # #             st.dataframe(df["Cluster"].value_counts().rename_axis("Cluster").reset_index(name="Count"))

# # #         with col2:
# # #             try:
# # #                 sil = silhouette_score(df_scaled, df["Cluster"])
# # #                 st.metric("Silhouette Score", f"{sil:.3f}")
# # #             except Exception:
# # #                 st.warning("Silhouette score not available for this configuration.")

# # #         st.markdown("### 🎨 Cluster Visualization")
# # #         if len(features) >= 2:
# # #             x_axis = st.selectbox("X-axis", features, index=0)
# # #             y_axis = st.selectbox("Y-axis", features, index=1)
# # #             fig = px.scatter(df, x=x_axis, y=y_axis, color=df["Cluster"].astype(str),
# # #                              title="2D Cluster Visualization", color_discrete_sequence=px.colors.qualitative.Set2)
# # #             st.plotly_chart(fig, use_container_width=True)
# # #         else:
# # #             st.info("Please select at least 2 features for scatter visualization.")

# # #         # Optional map view
# # #         if {"lat", "lon"}.issubset(df.columns):
# # #             st.markdown("### 🗺️ Cluster Map View")
# # #             fig_map = px.scatter_mapbox(
# # #                 df, lat="lat", lon="lon", color=df["Cluster"].astype(str),
# # #                 hover_data=features, zoom=3, height=400, mapbox_style="open-street-map"
# # #             )
# # #             st.plotly_chart(fig_map, use_container_width=True)

# # #         # Download option
# # #         st.markdown("### 💾 Download Results")
# # #         csv_data = df.to_csv(index=False).encode("utf-8")
# # #         st.download_button("Download Clustered Data (CSV)", csv_data, "clustered_results.csv", "text/csv")

# # # else:
# # #     st.error("❌ `data/sample_data.csv` not found. Please run `python data/generate_sample.py` first.")

# # #-----------------------------------------------------------------------------------------------------------------

# # #3rd interface
# # # import streamlit as st
# # # import pandas as pd
# # # import os
# # # import plotly.express as px
# # # from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
# # # from sklearn.preprocessing import StandardScaler, MinMaxScaler
# # # from sklearn.metrics import silhouette_score
# # # import numpy as np

# # # # --------------------- Page Setup ---------------------
# # # st.set_page_config(
# # #     page_title="ClusteringApp 🌱",
# # #     page_icon="🌿",
# # #     layout="wide"
# # # )

# # # # --------------------- Header Section ---------------------
# # # logo_path = os.path.join("assets", "logo.png")
# # # col1, col2 = st.columns([1, 5])
# # # with col1:
# # #     if os.path.exists(logo_path):
# # #         st.image(logo_path, width=100)
# # # with col2:
# # #     st.title("🌱 Clustering Ecosystem Patches")
# # #     st.markdown("#### By Energy-Flow Patterns using Machine Learning")

# # # st.markdown("---")

# # # # --------------------- Load Dataset ---------------------
# # # csv_path = os.path.join("data", "sample_data.csv")

# # # if os.path.exists(csv_path):
# # #     df = pd.read_csv(csv_path)
# # #     st.success(f"✅ Loaded dataset from `{csv_path}`")

# # #     st.markdown("### 📊 Dataset Overview")
# # #     st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
# # #     st.write("**Columns:**", ", ".join(df.columns))

# # #     with st.expander("🔍 Preview Data"):
# # #         st.dataframe(df.head(10))

# # #     # --------------------- Preprocessing Options ---------------------
# # #     st.markdown("### ⚙️ Preprocessing Options")
# # #     numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
# # #     features = st.multiselect("Select features for clustering", numeric_cols, default=numeric_cols[:4])

# # #     scaling_option = st.selectbox("Select scaling method", ["None", "StandardScaler", "MinMaxScaler"])
# # #     if scaling_option == "StandardScaler":
# # #         scaler = StandardScaler()
# # #         df_scaled = scaler.fit_transform(df[features])
# # #     elif scaling_option == "MinMaxScaler":
# # #         scaler = MinMaxScaler()
# # #         df_scaled = scaler.fit_transform(df[features])
# # #     else:
# # #         df_scaled = df[features].values

# # #     # Convert scaled data to DataFrame
# # #     df_scaled = pd.DataFrame(df_scaled, columns=features)

# # #     # --------------------- Handle Missing Values ---------------------
# # #     st.markdown("### 🧹 Data Cleaning")

# # #     # Drop rows with all NaNs
# # #     before_rows = len(df_scaled)
# # #     df_scaled.dropna(how="all", inplace=True)
# # #     dropped_rows = before_rows - len(df_scaled)

# # #     if dropped_rows > 0:
# # #         st.warning(f"🧽 Dropped {dropped_rows} completely empty rows.")

# # #     # Fill remaining NaN values with column means
# # #     if df_scaled.isnull().values.any():
# # #         st.warning("⚠️ Missing values detected — imputing with column means.")
# # #         df_scaled = df_scaled.fillna(df_scaled.mean())
# # #     else:
# # #         st.info("✅ No missing values detected in selected features.")

# # #     # --------------------- Choose Clustering Algorithm ---------------------
# # #     st.markdown("### 🧠 Clustering Configuration")
# # #     algo = st.selectbox("Choose Algorithm", ["KMeans", "DBSCAN", "Agglomerative Clustering"])

# # #     # --------------------- Run KMeans ---------------------
# # #     if algo == "KMeans":
# # #         n_clusters = st.slider("Number of clusters (k)", 2, 10, 3)
# # #         run = st.button("🚀 Run KMeans Clustering")
# # #         if run:
# # #             model = KMeans(n_clusters=n_clusters, random_state=42)
# # #             labels = model.fit_predict(df_scaled)
# # #             df["Cluster"] = labels
# # #             st.success("✅ KMeans clustering completed!")

# # #     # --------------------- Run DBSCAN ---------------------
# # #     elif algo == "DBSCAN":
# # #         eps = st.slider("EPS (neighborhood size)", 0.1, 5.0, 0.5, step=0.1)
# # #         min_samples = st.slider("Min Samples", 1, 20, 5)
# # #         run = st.button("🚀 Run DBSCAN Clustering")
# # #         if run:
# # #             model = DBSCAN(eps=eps, min_samples=min_samples)
# # #             labels = model.fit_predict(df_scaled)
# # #             df["Cluster"] = labels
# # #             st.success("✅ DBSCAN clustering completed!")

# # #     # --------------------- Run Agglomerative Clustering ---------------------
# # #     else:
# # #         n_clusters = st.slider("Number of clusters", 2, 10, 3)
# # #         linkage = st.selectbox("Linkage Method", ["ward", "complete", "average", "single"])
# # #         run = st.button("🚀 Run Agglomerative Clustering")
# # #         if run:
# # #             model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
# # #             labels = model.fit_predict(df_scaled)
# # #             df["Cluster"] = labels
# # #             st.success("✅ Agglomerative clustering completed!")

# # #     # --------------------- Show Results ---------------------
# # #     if "Cluster" in df.columns:
# # #         st.markdown("### 📊 Cluster Results")

# # #         col1, col2 = st.columns(2)
# # #         with col1:
# # #             st.write("**Cluster Counts:**")
# # #             st.dataframe(df["Cluster"].value_counts().rename_axis("Cluster").reset_index(name="Count"))

# # #         with col2:
# # #             try:
# # #                 sil = silhouette_score(df_scaled, df["Cluster"])
# # #                 st.metric("Silhouette Score", f"{sil:.3f}")
# # #             except Exception:
# # #                 st.warning("Silhouette score not available for this configuration.")

# # #         # --------------------- Visualization ---------------------
# # #         st.markdown("### 🎨 Cluster Visualization")
# # #         if len(features) >= 2:
# # #             x_axis = st.selectbox("X-axis", features, index=0)
# # #             y_axis = st.selectbox("Y-axis", features, index=1)
# # #             fig = px.scatter(df, x=x_axis, y=y_axis, color=df["Cluster"].astype(str),
# # #                              title="2D Cluster Visualization", color_discrete_sequence=px.colors.qualitative.Set2)
# # #             st.plotly_chart(fig, use_container_width=True)
# # #         else:
# # #             st.info("Please select at least 2 features for scatter visualization.")

# # #         # Optional map
# # #         if {"lat", "lon"}.issubset(df.columns):
# # #             st.markdown("### 🗺️ Cluster Map View")
# # #             fig_map = px.scatter_mapbox(
# # #                 df, lat="lat", lon="lon", color=df["Cluster"].astype(str),
# # #                 hover_data=features, zoom=3, height=400, mapbox_style="open-street-map"
# # #             )
# # #             st.plotly_chart(fig_map, use_container_width=True)

# # #         # --------------------- Download Results ---------------------
# # #         st.markdown("### 💾 Download Results")
# # #         csv_data = df.to_csv(index=False).encode("utf-8")
# # #         st.download_button("Download Clustered Data (CSV)", csv_data, "clustered_results.csv", "text/csv")

# # # else:
# # #     st.error("❌ `data/sample_data.csv` not found. Please run `python data/generate_sample.py` first.")

# # #-------------------------------------------------------------------------------------

# # # 4th interface

# # import streamlit as st
# # import pandas as pd
# # import os
# # import plotly.express as px
# # import plotly.graph_objects as go
# # from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
# # from sklearn.preprocessing import StandardScaler, MinMaxScaler
# # from sklearn.metrics import silhouette_score, silhouette_samples
# # from sklearn.decomposition import PCA
# # import numpy as np

# # # PDF report libraries
# # from reportlab.lib.pagesizes import letter
# # from reportlab.lib import colors
# # from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
# # from reportlab.lib.styles import getSampleStyleSheet
# # import io

# # # --------------------- PDF Report Function ---------------------
# # def generate_pdf_report(df, features, algo, sil_score=None):
# #     buffer = io.BytesIO()
# #     doc = SimpleDocTemplate(buffer, pagesize=letter)
# #     styles = getSampleStyleSheet()
# #     elements = []

# #     elements.append(Paragraph("<b>Clustering Ecosystem Patches Report</b>", styles['Title']))
# #     elements.append(Spacer(1, 12))

# #     info = f"""
# #     <b>Algorithm Used:</b> {algo}<br/>
# #     <b>Number of Records:</b> {len(df)}<br/>
# #     <b>Features Used:</b> {', '.join(features)}<br/>
# #     """
# #     if sil_score is not None:
# #         info += f"<b>Silhouette Score:</b> {sil_score:.3f}<br/>"
# #     elements.append(Paragraph(info, styles['Normal']))
# #     elements.append(Spacer(1, 12))

# #     elements.append(Paragraph("<b>Cluster Summary (Mean Feature Values)</b>", styles['Heading3']))
# #     cluster_summary = df.groupby("Cluster")[features].mean().round(3).reset_index()
# #     data = [cluster_summary.columns.tolist()] + cluster_summary.values.tolist()
# #     table = Table(data)
# #     table.setStyle(TableStyle([
# #         ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
# #         ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
# #         ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
# #         ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
# #         ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
# #         ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
# #         ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
# #     ]))
# #     elements.append(table)
# #     elements.append(Spacer(1, 12))
# #     elements.append(Paragraph("Generated by <b>ClusteringApp</b> — Ecosystem Energy Flow Analysis", styles['Italic']))

# #     doc.build(elements)
# #     pdf = buffer.getvalue()
# #     buffer.close()
# #     return pdf


# # # --------------------- Streamlit Page Setup ---------------------
# # st.set_page_config(page_title="ClusteringApp 🌱", page_icon="🌿", layout="wide")

# # # --------------------- Custom Styling ---------------------
# # st.markdown("""
# #     <style>
# #         [data-testid="stAppViewContainer"] {
# #             background-color: #f5fff7;
# #         }
# #         .main {
# #             background-color: #ffffff;
# #             border-radius: 15px;
# #             padding: 20px 40px;
# #             box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
# #         }
# #         h1, h2, h3 {
# #             color: #007f5f;
# #             font-family: 'Poppins', sans-serif;
# #         }
# #         .stButton>button {
# #             background-color: #00a86b;
# #             color: white;
# #             border-radius: 10px;
# #             border: none;
# #             padding: 10px 20px;
# #             font-size: 16px;
# #         }
# #         .stButton>button:hover {
# #             background-color: #00945a;
# #             color: #f0fff4;
# #         }
# #     </style>
# # """, unsafe_allow_html=True)

# # # --------------------- Header Section ---------------------
# # logo_path = os.path.join("assets", "logo.png")
# # col1, col2 = st.columns([1, 5])
# # with col1:
# #     if os.path.exists(logo_path):
# #         st.image(logo_path, width=100)
# # with col2:
# #     st.title("🌱 Clustering Ecosystem Patches")
# #     st.markdown("#### Using Machine Learning for Energy-Flow Pattern Analysis")
# #     st.markdown("##### Project by Bhumi — Guided by Dr. Nupur 🌿")

# # st.markdown("---")

# # # --------------------- Load Dataset ---------------------
# # csv_path = os.path.join("data", "sample_data.csv")

# # if os.path.exists(csv_path):
# #     df = pd.read_csv(csv_path)
# #     st.success(f"✅ Loaded dataset from `{csv_path}`")

# #     st.markdown("### 📊 Dataset Overview")
# #     st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
# #     with st.expander("🔍 Preview Data"):
# #         st.dataframe(df.head(10))

# #     numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# #     # --------------------- Preprocessing ---------------------
# #     st.markdown("### ⚙️ Preprocessing Options")
# #     features = st.multiselect("Select features for clustering", numeric_cols, default=numeric_cols[:4])
# #     scaling_option = st.selectbox("Select scaling method", ["None", "StandardScaler", "MinMaxScaler"])

# #     if scaling_option == "StandardScaler":
# #         scaler = StandardScaler()
# #         df_scaled = scaler.fit_transform(df[features])
# #     elif scaling_option == "MinMaxScaler":
# #         scaler = MinMaxScaler()
# #         df_scaled = scaler.fit_transform(df[features])
# #     else:
# #         df_scaled = df[features].values

# #     df_scaled = pd.DataFrame(df_scaled, columns=features)
# #     df_scaled = df_scaled.fillna(df_scaled.mean())

# #     # --------------------- Algorithm Selection ---------------------
# #     st.markdown("### 🧠 Clustering Configuration")
# #     algo = st.selectbox("Choose Algorithm", ["KMeans", "DBSCAN", "Agglomerative Clustering"])

# #     # Automatically run clustering when settings change
# #     if algo == "KMeans":
# #         n_clusters = st.slider("Number of clusters (k)", 2, 10, 3, key="kmeans_clusters")
# #         st.caption("🔄 Clustering auto-updates when you change this slider.")
# #         model = KMeans(n_clusters=n_clusters, random_state=42)
# #         labels = model.fit_predict(df_scaled)
# #         df["Cluster"] = labels
# #         st.success(f"✅ KMeans clustering updated with {n_clusters} clusters!")

# #         # ---- Elbow Method ----
# #         distortions = []
# #         K = range(2, 11)
# #         for k in K:
# #             km = KMeans(n_clusters=k, random_state=42)
# #             km.fit(df_scaled)
# #             distortions.append(km.inertia_)
# #         fig_elbow = go.Figure()
# #         fig_elbow.add_trace(go.Scatter(x=list(K), y=distortions, mode='lines+markers', line=dict(color='#007f5f')))
# #         fig_elbow.update_layout(title='Elbow Curve', xaxis_title='Number of Clusters (k)', yaxis_title='Inertia')
# #         st.plotly_chart(fig_elbow, use_container_width=True)

# #     elif algo == "DBSCAN":
# #         eps = st.slider("EPS (neighborhood size)", 0.1, 5.0, 0.5, step=0.1, key="dbscan_eps")
# #         min_samples = st.slider("Min Samples", 1, 20, 5, key="dbscan_min")
# #         st.caption("🔄 Clustering auto-updates when you change values.")
# #         model = DBSCAN(eps=eps, min_samples=min_samples)
# #         labels = model.fit_predict(df_scaled)
# #         df["Cluster"] = labels
# #         st.success(f"✅ DBSCAN clustering updated (eps={eps}, min_samples={min_samples})!")

# #     else:
# #         n_clusters = st.slider("Number of clusters", 2, 10, 3, key="agg_clusters")
# #         linkage = st.selectbox("Linkage Method", ["ward", "complete", "average", "single"], key="agg_link")
# #         st.caption("🔄 Clustering auto-updates when you change settings.")
# #         model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
# #         labels = model.fit_predict(df_scaled)
# #         df["Cluster"] = labels
# #         st.success(f"✅ Agglomerative clustering updated with {n_clusters} clusters!")

# #     # --------------------- Results ---------------------
# #     st.markdown("### 📊 Cluster Results")
# #     col1, col2 = st.columns(2)
# #     with col1:
# #         st.write("**Cluster Counts:**")
# #         st.dataframe(df["Cluster"].value_counts().rename_axis("Cluster").reset_index(name="Count"))
# #     with col2:
# #         try:
# #             sil = silhouette_score(df_scaled, df["Cluster"])
# #             st.metric("Silhouette Score", f"{sil:.3f}")
# #         except Exception:
# #             st.warning("Silhouette score not available for this configuration.")

# #     # ---- Cluster Summary ----
# #     st.markdown("### 📋 Cluster Summary (Mean Feature Values)")
# #     st.dataframe(df.groupby("Cluster")[features].mean().style.background_gradient(cmap="Greens"))

# #     # ---- Silhouette Plot ----
# #     st.markdown("### 📈 Silhouette Plot")
# #     try:
# #         sil_samples = silhouette_samples(df_scaled, df["Cluster"])
# #         fig_sil = go.Figure()
# #         y_lower = 10
# #         for i in range(len(np.unique(labels))):
# #             ith_cluster_silhouette_values = sil_samples[df["Cluster"] == i]
# #             ith_cluster_silhouette_values.sort()
# #             size_cluster_i = ith_cluster_silhouette_values.shape[0]
# #             y_upper = y_lower + size_cluster_i
# #             color = px.colors.qualitative.Set2[i % len(px.colors.qualitative.Set2)]
# #             fig_sil.add_trace(go.Scatter(y=np.arange(y_lower, y_upper),
# #                                          x=ith_cluster_silhouette_values,
# #                                          mode='lines',
# #                                          line=dict(color=color),
# #                                          name=f"Cluster {i}"))
# #             y_lower = y_upper + 10
# #         fig_sil.update_layout(title='Silhouette Plot', xaxis_title='Silhouette Coefficient', yaxis_title='Samples')
# #         st.plotly_chart(fig_sil, use_container_width=True)
# #     except Exception:
# #         st.warning("Silhouette plot not available for this configuration.")

# #     # ---- PCA Visualization ----
# #     st.markdown("### 🌈 PCA Visualization (2D Projection)")
# #     pca = PCA(n_components=2)
# #     pca_result = pca.fit_transform(df_scaled)
# #     df_pca = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
# #     df_pca['Cluster'] = df['Cluster'].astype(str)
# #     fig_pca = px.scatter(df_pca, x='PCA1', y='PCA2', color='Cluster',
# #                          title='PCA-based Cluster Visualization',
# #                          color_discrete_sequence=px.colors.qualitative.Pastel)
# #     st.plotly_chart(fig_pca, use_container_width=True)

# #     # ---- Map View ----
# #     if {"lat", "lon"}.issubset(df.columns):
# #         st.markdown("### 🗺️ Cluster Map View")
# #         fig_map = px.scatter_mapbox(
# #             df, lat="lat", lon="lon", color=df["Cluster"].astype(str),
# #             hover_data=features, zoom=3, height=400, mapbox_style="open-street-map"
# #         )
# #         st.plotly_chart(fig_map, use_container_width=True)

# #     # ---- Download Results ----
# #     st.markdown("### 💾 Download Results")
# #     csv_data = df.to_csv(index=False).encode("utf-8")
# #     st.download_button("Download Clustered Data (CSV)", csv_data, "clustered_results.csv", "text/csv")

# #     # ---- PDF Report Export ----
# #     st.markdown("### 🧾 Export to PDF Report")
# #     if st.button("📄 Generate PDF Report"):
# #         try:
# #             sil_score_val = None
# #             try:
# #                 sil_score_val = silhouette_score(df_scaled, df["Cluster"])
# #             except Exception:
# #                 pass

# #             pdf_bytes = generate_pdf_report(df, features, algo, sil_score_val)
# #             st.download_button(
# #                 label="⬇️ Download PDF Report",
# #                 data=pdf_bytes,
# #                 file_name="Clustering_Report.pdf",
# #                 mime="application/pdf"
# #             )
# #         except Exception as e:
# #             st.error(f"Failed to generate PDF: {e}")

# # else:
# #     st.error("❌ `data/sample_data.csv` not found. Please run `python data/generate_sample.py` first.")


# #5th interface
# import streamlit as st
# import pandas as pd
# import os
# import numpy as np
# import plotly.express as px
# from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.metrics import silhouette_score
# from sklearn.decomposition import PCA

# # ---------- Streamlit Page Setup ----------
# st.set_page_config(
#     page_title="🌿 Clustering Ecosystem Patches",
#     page_icon="🌍",
#     layout="wide"
# )

# # ---------- Light Green Theme Styling ----------
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

# [data-testid="stAppViewContainer"] {
#     background: linear-gradient(to right, #f7fff9, #e9f7ef);
#     color: #0a3d2e;
#     font-family: 'Poppins', sans-serif;
# }

# /* Sidebar */
# [data-testid="stSidebar"] {
#     background-color: #f1faee;
#     border-right: 3px solid #2a9d8f;
# }
# [data-testid="stSidebar"] h2 {
#     color: #2a9d8f;
# }

# /* Buttons */
# .stButton>button {
#     background-color: #2a9d8f;
#     color: white;
#     border: none;
#     border-radius: 8px;
#     font-weight: 600;
#     transition: all 0.3s ease;
# }
# .stButton>button:hover {
#     background-color: #1f776b;
#     transform: scale(1.03);
# }

# /* Titles */
# h1, h2, h3 {
#     color: #1b4332;
#     font-weight: 600;
# }

# /* Dataframe style */
# .stDataFrame {
#     border-radius: 10px;
#     background-color: #ffffff;
#     box-shadow: 0 0 10px rgba(0,0,0,0.1);
# }
# </style>
# """, unsafe_allow_html=True)

# # ---------- Header Section ----------
# col1, col2 = st.columns([1, 6])
# logo_path = os.path.join("assets", "logo.png")
# with col1:
#     if os.path.exists(logo_path):
#         st.image(logo_path, width=100)
# with col2:
#     st.markdown("<h1>🌿 Clustering Ecosystem Patches</h1>", unsafe_allow_html=True)
#     st.markdown("<h4>Analyzing Energy Flow Patterns Using Unsupervised Learning</h4>", unsafe_allow_html=True)

# st.markdown("---")

# # ---------- Load Data ----------
# csv_path = os.path.join("data", "sample_data.csv")

# if os.path.exists(csv_path):
#     df = pd.read_csv(csv_path)
#     st.success("✅ Dataset loaded successfully.")
# else:
#     st.error("❌ No dataset found. Please place `sample_data.csv` inside `/data` folder.")
#     st.stop()

# # ---------- Sidebar Controls ----------
# st.sidebar.header("⚙️ Preprocessing & Clustering Settings")

# numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# features = st.sidebar.multiselect("Select features for clustering", numeric_cols, default=numeric_cols[:4])
# scaling_option = st.sidebar.selectbox("Scaling Method", ["None", "StandardScaler", "MinMaxScaler"])

# # Scaling
# if scaling_option == "StandardScaler":
#     scaler = StandardScaler()
#     df_scaled = scaler.fit_transform(df[features])
# elif scaling_option == "MinMaxScaler":
#     scaler = MinMaxScaler()
#     df_scaled = scaler.fit_transform(df[features])
# else:
#     df_scaled = df[features].values

# df_scaled = pd.DataFrame(df_scaled, columns=features)
# if df_scaled.isnull().values.any():
#     df_scaled = df_scaled.fillna(df_scaled.mean())

# algo = st.sidebar.selectbox("Select Clustering Algorithm", ["KMeans", "DBSCAN", "Agglomerative Clustering"])

# # ---------- Main Dashboard Layout ----------
# col_data, col_viz = st.columns([1.2, 2.3])

# # --- Left: Data Preview ---
# with col_data:
#     st.subheader("📋 Data Overview")
#     st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
#     st.dataframe(df.head(8), use_container_width=True)
#     st.bar_chart(df[features].head(20))

# # --- Right: Visualization / Clustering Output ---
# with col_viz:
#     st.subheader("🌍 Cluster Analysis")

#     if algo == "KMeans":
#         n_clusters = st.slider("Number of Clusters (k)", 2, 10, 3)
#         model = KMeans(n_clusters=n_clusters, random_state=42)
#         labels = model.fit_predict(df_scaled)
#     elif algo == "DBSCAN":
#         eps = st.slider("Neighborhood Size (EPS)", 0.1, 5.0, 0.5, step=0.1)
#         min_samples = st.slider("Minimum Samples", 1, 20, 5)
#         model = DBSCAN(eps=eps, min_samples=min_samples)
#         labels = model.fit_predict(df_scaled)
#     else:
#         n_clusters = st.slider("Number of Clusters", 2, 10, 3)
#         linkage = st.selectbox("Linkage Method", ["ward", "complete", "average", "single"])
#         model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
#         labels = model.fit_predict(df_scaled)

#     df["Cluster"] = labels

#     try:
#         sil = silhouette_score(df_scaled, labels)
#         st.metric("🌿 Silhouette Score", f"{sil:.3f}")
#     except:
#         st.warning("Silhouette score unavailable for single cluster or noise labels.")

#     # ---------- PCA 2D Visualization ----------
#     pca = PCA(n_components=2)
#     df_pca = pd.DataFrame(pca.fit_transform(df_scaled), columns=["PCA1", "PCA2"])
#     df_pca["Cluster"] = labels
#     fig_2d = px.scatter(
#         df_pca,
#         x="PCA1", y="PCA2",
#         color=df_pca["Cluster"].astype(str),
#         color_discrete_sequence=px.colors.qualitative.G10,
#         title="2D Cluster Visualization (PCA Projection)",
#         template="plotly_white"
#     )
#     st.plotly_chart(fig_2d, use_container_width=True)

#     # ---------- Map View (if lat/lon exist) ----------
#     if {"lat", "lon"}.issubset(df.columns):
#         st.subheader("🗺️ Map View")
#         fig_map = px.scatter_mapbox(
#             df, lat="lat", lon="lon",
#             color=df["Cluster"].astype(str),
#             hover_data=features,
#             zoom=3,
#             mapbox_style="open-street-map",
#             height=400,
#             title="Cluster Distribution on Map"
#         )
#         st.plotly_chart(fig_map, use_container_width=True)

# # ---------- Download Clustered Results ----------
# st.markdown("---")
# csv_data = df.to_csv(index=False).encode("utf-8")
# st.download_button("⬇️ Download Clustered Data (CSV)", csv_data, "clustered_results.csv", "text/csv")

# st.success("🌳 Analysis Complete! Adjust sidebar settings to explore different models.")

#6th----------------------
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

 