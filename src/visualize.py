# # src/visualize.py
# import plotly.express as px
# import pandas as pd
# import numpy as np
# from sklearn.metrics import silhouette_samples
# from sklearn.cluster import AgglomerativeClustering
# from scipy.cluster import hierarchy
# import plotly.figure_factory as ff
# import plotly.graph_objects as go

# def scatter_plot_2d(df: pd.DataFrame, x_col: str, y_col: str, label_column="cluster"):
#     df_plot = df.copy()
#     df_plot[label_column] = df_plot[label_column].astype(str)
#     fig = px.scatter(df_plot, x=x_col, y=y_col, color=label_column, hover_data=df_plot.columns)
#     fig.update_layout(title=f"{y_col} vs {x_col} (colored by {label_column})")
#     return fig

# def plot_silhouette_scores(X, labels):
#     # compute silhouette samples and plot a horizontal bar chart
#     try:
#         samples = silhouette_samples(X, labels)
#         df = pd.DataFrame({"label": labels, "silhouette": samples})
#         df = df.sort_values(["label", "silhouette"], ascending=[True, False])
#         fig = px.bar(df, x="silhouette", y=df.index.astype(str), color="label", orientation="h", title="Silhouette values per sample")
#         fig.update_layout(height=600)
#         return fig
#     except Exception as e:
#         fig = go.Figure()
#         fig.add_annotation(text=f"Silhouette plot not available: {e}", showarrow=False)
#         return fig

# def plot_dendrogram_plotly(X):
#     # produce linkage and a dendrogram via scipy
#     Z = hierarchy.linkage(X, method="ward")
#     fig = ff.create_dendrogram(X, linkagefun=lambda x: Z)
#     fig.update_layout(width=800, height=400)
#     return fig

#2nd code-------------------------------------------------------------------------------------

# src/visualize.py
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_samples
from scipy.cluster import hierarchy
import plotly.figure_factory as ff
import plotly.graph_objects as go

# ==========================
# 🔮 Futuristic Visualization Module
# ==========================

# Neon-style color palette
NEON_COLORS = [
    "#00FFFF",  # cyan
    "#FF00FF",  # magenta
    "#39FF14",  # neon green
    "#FFAA00",  # amber
    "#00BFFF",  # bright blue
    "#FF4D4D",  # red
    "#9D00FF",  # violet
    "#00FFB3",  # mint
]

# Common layout style (dark futuristic look)
def _apply_sci_fi_layout(fig, title):
    fig.update_layout(
        title=dict(
            text=f"🧬 {title}",
            x=0.5,
            font=dict(size=20, color="#00FFFF", family="Orbitron, monospace"),
        ),
        plot_bgcolor="#0D1117",
        paper_bgcolor="#0D1117",
        font=dict(color="#E6EDF3", family="Exo 2, sans-serif"),
        hoverlabel=dict(bgcolor="#111827", font_size=12, font_family="Consolas"),
        legend=dict(
            bgcolor="rgba(13,17,23,0.5)",
            bordercolor="#00FFFF",
            borderwidth=1,
            title="Clusters",
            font=dict(size=12, color="#00FFFF"),
        ),
        margin=dict(l=60, r=60, t=80, b=60),
    )
    return fig


# ==========================================
# 1️⃣ 2D Scatter Plot (Clusters)
# ==========================================
def scatter_plot_2d(df: pd.DataFrame, x_col: str, y_col: str, label_column="Cluster"):
    """Futuristic 2D scatter visualization for cluster analysis."""
    df_plot = df.copy()
    df_plot[label_column] = df_plot[label_column].astype(str)

    fig = px.scatter(
        df_plot,
        x=x_col,
        y=y_col,
        color=label_column,
        hover_data=df_plot.columns,
        color_discrete_sequence=NEON_COLORS,
        symbol_sequence=["circle", "square", "diamond", "cross"],
    )

    # Add glowing effect (via marker outlines)
    fig.update_traces(
        marker=dict(size=10, line=dict(width=2, color="#0D1117")),
        selector=dict(mode="markers"),
    )

    fig = _apply_sci_fi_layout(fig, f"{y_col} vs {x_col} — Cluster View")
    return fig


# ==========================================
# 2️⃣ Silhouette Plot (Cluster Evaluation)
# ==========================================
def plot_silhouette_scores(X, labels):
    """Visualize silhouette score distribution for each data point."""
    try:
        samples = silhouette_samples(X, labels)
        df = pd.DataFrame({"label": labels, "silhouette": samples})
        df = df.sort_values(["label", "silhouette"], ascending=[True, False])

        fig = px.bar(
            df,
            x="silhouette",
            y=df.index.astype(str),
            color="label",
            orientation="h",
            color_discrete_sequence=NEON_COLORS,
            title="Silhouette Score Distribution",
        )

        fig.update_traces(marker_line_color="#111", marker_line_width=0.5)
        fig = _apply_sci_fi_layout(fig, "Silhouette Values per Sample")
        fig.update_layout(height=600, xaxis_title="Silhouette Coefficient", yaxis_title="Samples")

        return fig

    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"⚠️ Silhouette plot unavailable: {e}",
            showarrow=False,
            font=dict(color="#FF4D4D", size=16),
        )
        fig = _apply_sci_fi_layout(fig, "Silhouette Plot Error")
        return fig


# ==========================================
# 3️⃣ Dendrogram (Hierarchical Visualization)
# ==========================================
def plot_dendrogram_plotly(X):
    """Create a high-tech dendrogram for hierarchical clustering."""
    Z = hierarchy.linkage(X, method="ward")

    fig = ff.create_dendrogram(X, linkagefun=lambda x: Z, color_threshold=0)
    fig.update_traces(line=dict(width=1.5, color="#00FFFF"))

    fig = _apply_sci_fi_layout(fig, "Hierarchical Dendrogram — Energy Linkage")
    fig.update_layout(width=1000, height=500, showlegend=False)
    return fig


# ==========================================
# 4️⃣ PCA 3D Cluster Visualization (Extra)
# ==========================================
def plot_3d_clusters(df_pca, label_column="Cluster"):
    """Optional futuristic 3D PCA plot for cluster visualization."""
    df_pca[label_column] = df_pca[label_column].astype(str)

    fig = px.scatter_3d(
        df_pca,
        x="PCA1",
        y="PCA2",
        z="PCA3" if "PCA3" in df_pca.columns else "PCA2",
        color=label_column,
        color_discrete_sequence=NEON_COLORS,
        symbol=label_column,
    )

    fig.update_traces(marker=dict(size=6, line=dict(width=1, color="#000000")))
    fig = _apply_sci_fi_layout(fig, "3D Cluster Map — PCA Projection")
    fig.update_layout(scene=dict(
        xaxis=dict(backgroundcolor="#0A0F1C", color="#00FFFF"),
        yaxis=dict(backgroundcolor="#0A0F1C", color="#00FFFF"),
        zaxis=dict(backgroundcolor="#0A0F1C", color="#00FFFF"),
    ))
    return fig

