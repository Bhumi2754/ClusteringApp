# src/preprocess.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple

def preprocess_dataframe(df: pd.DataFrame, impute_method="mean", scaling="StandardScaler (z-score)"):
    """
    Return numpy array X and processed dataframe (with same columns).
    impute_method: "mean", "median", "drop"
    scaling: "None", "StandardScaler (z-score)", "MinMaxScaler"
    """
    proc = df.copy()
    # Ensure numeric
    proc = proc.apply(pd.to_numeric, errors="coerce")
    if impute_method == "drop":
        proc = proc.dropna()
    else:
        strategy = "mean" if impute_method == "mean" else "median"
        imputer = SimpleImputer(strategy=strategy)
        proc[:] = imputer.fit_transform(proc)

    # scaling
    if scaling == "StandardScaler (z-score)":
        scaler = StandardScaler()
        proc[:] = scaler.fit_transform(proc)
    elif scaling == "MinMaxScaler":
        scaler = MinMaxScaler()
        proc[:] = scaler.fit_transform(proc)
    # else no scaling

    X = proc.values
    return X, proc

def generate_feature_summary(df: pd.DataFrame):
    """Return simple summary statistics for the UI."""
    return df.describe().T
