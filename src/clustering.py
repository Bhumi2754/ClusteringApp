# src/clustering.py
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn import metrics

def run_kmeans(X, n_clusters=3, random_state=42):
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    return labels, model

def run_dbscan(X, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return labels, model

def run_agglomerative(X, n_clusters=3, linkage="ward"):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)
    return labels, model

def compute_cluster_metrics(X, labels):
    metrics_dict = {}
    # number of clusters (-1 is noise for DBSCAN)
    unique_labels = set(labels)
    n_clusters = len([l for l in unique_labels if l != -1])
    metrics_dict["n_clusters"] = n_clusters
    # silhouette (valid only when n_clusters >= 2 and no single-cluster)
    try:
        if n_clusters >= 2:
            metrics_dict["silhouette_score"] = metrics.silhouette_score(X, labels)
        else:
            metrics_dict["silhouette_score"] = float("nan")
    except Exception:
        metrics_dict["silhouette_score"] = float("nan")

    try:
        metrics_dict["davies_bouldin"] = metrics.davies_bouldin_score(X, labels)
    except Exception:
        metrics_dict["davies_bouldin"] = float("nan")

    try:
        metrics_dict["calinski_harabasz"] = metrics.calinski_harabasz_score(X, labels)
    except Exception:
        metrics_dict["calinski_harabasz"] = float("nan")

    return metrics_dict
