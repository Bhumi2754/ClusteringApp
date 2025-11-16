import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import plotly.express as px
import plotly.graph_objects as go
from reportlab.lib.pagesizes import A4, portrait
from reportlab.pdfgen import canvas
import time

# ---------------- Page config & Theme ----------------
st.set_page_config(page_title="🌿 Clustering Explorer", page_icon="🌍", layout="wide")
st.markdown("""
<style>
* { font-family: 'Poppins', sans-serif; }
[data-testid="stAppViewContainer"] { background: #f3f9ff; color: #003566; padding: 12px; }
[data-testid="stSidebar"] { background: #e3f1ff; border-right: 3px solid #0077b6; }
.stButton>button { background: #0077b6; color:white; border-radius:10px; padding:8px 16px; font-weight:600; }
.stButton>button:hover { background:#005a8c; transform:scale(1.02); }
.card { background: white; padding: 20px; border-radius:12px; box-shadow: 0 10px 30px rgba(0,0,0,0.15); margin-bottom:20px; }
[data-testid="stDataFrame"] { border-radius:10px; box-shadow: 0 6px 18px rgba(0,0,0,0.08); background:white; }
.mapboxgl-map { border-radius:10px !important; }
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.title("🌿 Clustering Explorer Dashboard")
st.subheader("Analyzing Energy Flow Patterns, clusters, and trends")

# ---------------- Load dataset ----------------
uploaded = st.sidebar.file_uploader("Upload CSV file (optional)", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    source = "Uploaded CSV"
else:
    CSV_PATH = "data/large_sample_data.csv"
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        source = CSV_PATH
    else:
        st.error("No dataset found. Please upload a CSV or place a file at 'data/large_sample_data.csv'")
        st.stop()
st.sidebar.caption(f"Data source: {source}")

# ---------------- Sidebar: Features & Preprocessing ----------------
st.sidebar.header("Data & Preprocessing")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found in dataset.")
    st.stop()

features = st.sidebar.multiselect("Select columns to use for clustering", numeric_cols, default=numeric_cols[:4])
if not features:
    st.error("Select at least one column to continue.")
    st.stop()

scaling_opt = st.sidebar.selectbox("Scaling method", ["None", "StandardScaler", "MinMaxScaler"],
                                   help="Scale data so all features are comparable. StandardScaler = mean 0, std 1; MinMaxScaler = range 0-1")
impute_opt = st.sidebar.selectbox("Missing value handling", ["mean", "median", "drop"],
                                  help="Fill missing values with mean, median, or drop rows with missing values")
remove_outliers = st.sidebar.checkbox("Remove extreme values (|z|>3)", help="Removes rows with very high or low values in selected columns")

# ---------------- Sidebar: Clustering ----------------
st.sidebar.header("Clustering Settings")
algos_to_compare = st.sidebar.multiselect(
    "Clustering method", ["KMeans","DBSCAN","Agglomerative"], default=["KMeans","DBSCAN"],
    help="Choose one or more methods to group your data"
)
max_k = st.sidebar.slider("Maximum clusters to test", min_value=4, max_value=20, value=8,
                          help="Dashboard will test up to this number of clusters and pick the best automatically")

# ---------------- Sidebar: Time periods / Simulation ----------------
st.sidebar.header("Time slices / Simulation")
n_periods = st.sidebar.slider("Number of dataset periods", 2, 12, 6,
                              help="Split your data into periods to see how clusters change over time")
simulate = st.sidebar.checkbox("Enable step-by-step simulation")
sim_rows = st.sidebar.slider("Rows to simulate", 50, min(5000,len(df)), value=min(500,len(df))) if simulate else len(df)

# ---------------- Preprocessing & clustering functions ----------------
@st.cache_data
def preprocess_df(df_sub, features, impute, scale, outliers):
    tmp = df_sub[features].copy().astype(float)
    if impute == "drop": tmp = tmp.dropna()
    elif impute == "mean": tmp = tmp.fillna(tmp.mean())
    else: tmp = tmp.fillna(tmp.median())
    if outliers:
        arr = tmp.to_numpy()
        z = (arr - arr.mean(axis=0)) / (arr.std(axis=0) + 1e-9)
        mask = np.all(np.abs(z) <= 3, axis=1)
        tmp = tmp.iloc[mask]
    if scale == "StandardScaler":
        tmp[:] = StandardScaler().fit_transform(tmp)
    elif scale == "MinMaxScaler":
        tmp[:] = MinMaxScaler().fit_transform(tmp)
    return tmp

@st.cache_data
def choose_best_k(X, min_k=2, max_k=8):
    best_k, best_score = min_k, -1
    for k in range(min_k, max_k+1):
        if len(X) < k: continue
        km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=256).fit(X)
        labs = km.labels_
        if len(set(labs))<2: continue
        sc = silhouette_score(X, labs)
        if sc > best_score:
            best_score = sc
            best_k = k
    return best_k

def metrics_from_labels(X, labels):
    out = {}
    try: out['silhouette'] = silhouette_score(X, labels) if len(set(labels))>=2 else np.nan
    except: out['silhouette']=np.nan
    try: out['davies_bouldin']=davies_bouldin_score(X, labels) if len(set(labels))>=2 else np.nan
    except: out['davies_bouldin']=np.nan
    try: out['calinski_harabasz']=calinski_harabasz_score(X, labels) if len(set(labels))>=2 else np.nan
    except: out['calinski_harabasz']=np.nan
    out['n_clusters']=len([l for l in set(labels) if l!=-1])
    out['counts']=dict(pd.Series(labels).value_counts().to_dict())
    return out

def centroid_insight_text(centroid_series, overall_mean):
    parts=[]
    for feat,val in centroid_series.items():
        mean = overall_mean.get(feat,0)
        pct=(val-mean)/(abs(mean)+1e-9)
        if pct>0.25: parts.append(f"{feat} high (+{pct*100:.0f}%)")
        elif pct<-0.25: parts.append(f"{feat} low ({pct*100:.0f}%)")
        else: parts.append(f"{feat} normal")
    return "; ".join(parts)

# ---------------- Prepare dataset periods ----------------
df_work = df.iloc[:sim_rows].copy().reset_index(drop=True)
indices_split = np.array_split(df_work.index.to_numpy(), n_periods)
period_names = [f"Period {i+1}" for i in range(len(indices_split))]
period_col = np.empty(len(df_work),dtype=object)
for pname, idxs in zip(period_names, indices_split):
    period_col[idxs]=pname
df_work["_period"]=period_col
periods = list(sorted(df_work["_period"].unique(), key=lambda x:int(x.split(" ")[1])))
overall_means_raw = df[features].mean().to_dict()

# ---------------- Cluster per period ----------------
@st.cache_data
def process_periods(df_work, periods, features, algos, impute, scale, outliers, max_k):
    period_results = {}
    for p in periods:
        sub = df_work[df_work["_period"]==p]
        proc = preprocess_df(sub, features, impute, scale, outliers)
        if proc.shape[0]==0:
            period_results[p] = {"error":"no rows after preprocessing"}
            continue
        X = proc.values
        best_k = choose_best_k(X, min_k=2, max_k=min(max_k,int(np.sqrt(len(X)))))
        algos_info={}
        for algo_name in algos:
            try:
                if algo_name=="KMeans": model = MiniBatchKMeans(n_clusters=best_k, random_state=42, batch_size=256).fit(X); labels=model.labels_
                elif algo_name=="DBSCAN": model = DBSCAN(eps=0.5, min_samples=5).fit(X); labels=model.labels_
                else: model = AgglomerativeClustering(n_clusters=best_k).fit(X); labels=model.labels_
                metrics = metrics_from_labels(X, labels)
                centroids = pd.DataFrame(columns=features)
                for lab in sorted(set(labels)):
                    if lab==-1: continue
                    members = proc.iloc[np.where(labels==lab)[0]]
                    if len(members)>0: centroids.loc[lab] = members.mean().values
                algos_info[algo_name] = {"model":model, "labels":labels, "metrics":metrics, "centroids":centroids}
            except Exception as e:
                algos_info[algo_name] = {"error":str(e)}
        period_results[p]={"proc":proc,"X":X,"best_k":best_k,"algos":algos_info}
    return period_results

period_results = process_periods(df_work, periods, features, algos_to_compare, impute_opt, scaling_opt, remove_outliers, max_k)

# ---------------- Dataset Overview ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Overview")
st.write(f"Rows: {len(df_work)}, Numeric columns: {len(features)}, Dataset periods: {len(periods)}")
st.write("Periods:", ", ".join(periods))
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Centroid Timeline ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Cluster Center Timeline")
algo_timeline = st.selectbox("Select method for timeline", algos_to_compare)
feature_timeline = st.selectbox("Select feature to see cluster changes", features)
all_cluster_ids=set()
centroids_by_period={}
for p in periods:
    info = period_results[p]["algos"].get(algo_timeline, {})
    cent = info.get("centroids") if info else pd.DataFrame(columns=features)
    centroids_by_period[p]=cent
    if isinstance(cent, pd.DataFrame) and not cent.empty:
        all_cluster_ids.update(cent.index.tolist())
if all_cluster_ids:
    fig = go.Figure()
    for cid in sorted(all_cluster_ids):
        yvals = [float(centroids_by_period[p].loc[cid,feature_timeline]) if cid in centroids_by_period[p].index else np.nan for p in periods]
        fig.add_trace(go.Scatter(x=periods, y=yvals, mode="lines+markers", name=f"Cluster {cid}"))
    fig.update_layout(title=f"Cluster center changes for {feature_timeline} ({algo_timeline})", xaxis_title="Dataset Period", yaxis_title=feature_timeline, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No cluster centers to display.")
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Radar Chart ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Cluster Feature Radar Chart")
rad_period = st.selectbox("Select period for radar chart", periods)
rad_algo = st.selectbox("Select method for radar chart", algos_to_compare)
rad_info = period_results[rad_period]["algos"].get(rad_algo, {})
rad_centroids = rad_info.get("centroids")
if rad_centroids is None or rad_centroids.empty:
    st.info("No cluster centers available for radar chart.")
else:
    dfc = rad_centroids.astype(float)
    minv, maxv = dfc.min(), dfc.max()
    dfn = (dfc - minv)/(maxv - minv + 1e-9)
    categories = list(dfn.columns)
    fig=go.Figure()
    for idx,row in dfn.iterrows():
        vals=row.tolist()
        fig.add_trace(go.Scatterpolar(r=vals+[vals[0]],theta=categories+[categories[0]],fill='toself',name=f"Cluster {idx}"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,1])),showlegend=True,template="plotly_white")
    st.plotly_chart(fig,use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PCA 2D Scatter ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("2D PCA View & Map")
vis_period = st.selectbox("Select period for PCA", periods)
vis_algo = st.selectbox("Select method for PCA", algos_to_compare)
vis_info = period_results[vis_period]["algos"].get(vis_algo, {})
if "labels" in vis_info:
    labels = vis_info["labels"]
    proc = period_results[vis_period]["proc"].copy()
    proc["Cluster"]=labels.astype(str)
    pca2 = PCA(n_components=2).fit_transform(proc[features])
    proc["PCA1"], proc["PCA2"] = pca2[:,0], pca2[:,1]
    fig2 = px.scatter(proc, x="PCA1",y="PCA2",color="Cluster",hover_data=features,title=f"PCA 2D — {vis_period} ({vis_algo})",template="plotly_white")
    st.plotly_chart(fig2,use_container_width=True)
    if not {"lat","lon"}.issubset(proc.columns):
        proc["lat"]=np.interp(pca2[:,0],(pca2[:,0].min(),pca2[:,0].max()),(-20,55))
        proc["lon"]=np.interp(pca2[:,1],(pca2[:,1].min(),pca2[:,1].max()),(-140,140))
    figmap = px.scatter_mapbox(proc,lat="lat",lon="lon",color="Cluster",hover_data=features,zoom=1.5,mapbox_style="carto-positron",title=f"Map — {vis_period} ({vis_algo})")
    st.plotly_chart(figmap,use_container_width=True)
else:
    st.info("No visualization available.")
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Cluster Insights ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Cluster Insights")
ins_period = st.selectbox("Select period for insights", periods)
ins_algo = st.selectbox("Select method for insights", algos_to_compare)
ins_info = period_results[ins_period]["algos"].get(ins_algo,{})
ins_centroids = ins_info.get("centroids")
if ins_centroids is not None and not ins_centroids.empty:
    for cid in ins_centroids.index:
        txt = centroid_insight_text(ins_centroids.loc[cid].to_dict(), overall_means_raw)
        st.markdown(f"**Cluster {cid}:** {txt}")
else:
    st.info("No centroid data for insights.")
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PDF Report ----------------
def make_pdf(period,algo,centroids,metrics):
    buf=io.BytesIO()
    c=canvas.Canvas(buf,pagesize=portrait(A4))
    w,h=portrait(A4)
    c.setFont("Helvetica-Bold",16)
    c.drawString(40,h-50,"Clustering Analysis Report")
    c.setFont("Helvetica",10)
    c.drawString(40,h-70,f"Period: {period}  Method: {algo}  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y=h-100
    c.setFont("Helvetica-Bold",12)
    c.drawString(40,y,"Metrics:")
    y-=16
    for k,v in (metrics or {}).items():
        c.setFont("Helvetica",10)
        c.drawString(48,y,f"{k}: {v}")
        y-=12
        if y<80: c.showPage(); y=h-50
    y-=8
    c.setFont("Helvetica-Bold",12)
    c.drawString(40,y,"Cluster Centers:")
    y-=14
    c.setFont("Helvetica",10)
    if centroids is not None and not centroids.empty:
        for idx,row in centroids.iterrows():
            line=f"Cluster {idx}: "+", ".join([f"{f}={row[f]:.3f}" for f in centroids.columns])
            c.drawString(48,y,line)
            y-=12
            if y<80: c.showPage(); y=h-50
    else:
        c.drawString(48,y,"No cluster centers available.")
    c.showPage(); c.save()
    buf.seek(0)
    return buf.read()

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Download PDF Report")
pdf_period = st.selectbox("Select period for PDF", periods)
pdf_algo = st.selectbox("Select method for PDF", algos_to_compare)
pdf_info = period_results[pdf_period]["algos"].get(pdf_algo,{})
pdf_centroids = pdf_info.get("centroids")
pdf_metrics = pdf_info.get("metrics")
if st.button("Generate PDF"):
    pdf_bytes = make_pdf(pdf_period,pdf_algo,pdf_centroids,pdf_metrics)
    st.download_button("Download PDF", data=pdf_bytes, file_name=f"Clustering_{pdf_period}_{pdf_algo}.pdf", mime="application/pdf")
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Help / User Guide ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📘 How to use this Dashboard")
with st.expander("Click to see instructions"):
    st.markdown("""
1. **Upload CSV**: Use numeric columns only for clustering.
2. **Select Features**: Pick 2–6 columns for clustering analysis.
3. **Scaling & Missing Values**: StandardScaler / MinMaxScaler recommended.
4. **Clustering Method**: Compare multiple methods (KMeans, DBSCAN, Agglomerative).
5. **Simulation / Periods**: Split your dataset to see clusters evolve.
6. **Visualization**:
   - Cluster Center Timeline: How clusters change for a feature.
   - Radar Chart: Compare feature distribution in clusters.
   - PCA 2D & Map: See cluster patterns in 2D or pseudo-map.
7. **Insights**: Read cluster descriptions automatically generated.
8. **PDF Report**: Download full clustering report for any period & method.
""")
st.markdown("</div>", unsafe_allow_html=True)

