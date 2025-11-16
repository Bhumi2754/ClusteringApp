# # src/clustering.py
# import numpy as np
# from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
# from sklearn import metrics

# def run_kmeans(X, n_clusters=3, random_state=42):
#     model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
#     labels = model.fit_predict(X)
#     return labels, model

# def run_dbscan(X, eps=0.5, min_samples=5):
#     model = DBSCAN(eps=eps, min_samples=min_samples)
#     labels = model.fit_predict(X)
#     return labels, model

# def run_agglomerative(X, n_clusters=3, linkage="ward"):
#     model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
#     labels = model.fit_predict(X)
#     return labels, model

# def compute_cluster_metrics(X, labels):
#     metrics_dict = {}
#     # number of clusters (-1 is noise for DBSCAN)
#     unique_labels = set(labels)
#     n_clusters = len([l for l in unique_labels if l != -1])
#     metrics_dict["n_clusters"] = n_clusters
#     # silhouette (valid only when n_clusters >= 2 and no single-cluster)
#     try:
#         if n_clusters >= 2:
#             metrics_dict["silhouette_score"] = metrics.silhouette_score(X, labels)
#         else:
#             metrics_dict["silhouette_score"] = float("nan")
#     except Exception:
#         metrics_dict["silhouette_score"] = float("nan")

#     try:
#         metrics_dict["davies_bouldin"] = metrics.davies_bouldin_score(X, labels)
#     except Exception:
#         metrics_dict["davies_bouldin"] = float("nan")

#     try:
#         metrics_dict["calinski_harabasz"] = metrics.calinski_harabasz_score(X, labels)
#     except Exception:
#         metrics_dict["calinski_harabasz"] = float("nan")

#     return metrics_dict

#2nd code
# src/clustering.py
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn import metrics
from typing import Tuple, Dict, Any, List

def run_kmeans(X: np.ndarray, n_clusters: int = 3, random_state: int = 42) -> Tuple[np.ndarray, KMeans]:
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    return labels, model

def elbow_curve(X: np.ndarray, k_min: int = 2, k_max: int = 10) -> Tuple[List[int], List[float]]:
    inertias = []
    Ks = list(range(k_min, k_max + 1))
    for k in Ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        km.fit(X)
        inertias.append(km.inertia_)
    return Ks, inertias

def run_dbscan(X: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> Tuple[np.ndarray, DBSCAN]:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return labels, model

def run_agglomerative(X: np.ndarray, n_clusters: int = 3, linkage: str = "ward") -> Tuple[np.ndarray, AgglomerativeClustering]:
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)
    return labels, model

def compute_cluster_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    metrics_dict = {}
    unique_labels = set(labels)
    # count clusters excluding DBSCAN noise label (-1)
    n_clusters = len([l for l in unique_labels if l != -1])
    metrics_dict["n_clusters"] = n_clusters
    try:
        if n_clusters >= 2:
            metrics_dict["silhouette_score"] = float(metrics.silhouette_score(X, labels))
        else:
            metrics_dict["silhouette_score"] = float("nan")
    except Exception:
        metrics_dict["silhouette_score"] = float("nan")

    try:
        metrics_dict["davies_bouldin"] = float(metrics.davies_bouldin_score(X, labels))
    except Exception:
        metrics_dict["davies_bouldin"] = float("nan")

    try:
        metrics_dict["calinski_harabasz"] = float(metrics.calinski_harabasz_score(X, labels))
    except Exception:
        metrics_dict["calinski_harabasz"] = float("nan")

    # counts per cluster
    counts = {}
    for lab in sorted(unique_labels):
        counts[int(lab)] = int((labels == lab).sum())
    metrics_dict["counts"] = counts

    return metrics_dict
