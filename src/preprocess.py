# # src/preprocess.py
# import pandas as pd
# import numpy as np
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from typing import Tuple

# def preprocess_dataframe(df: pd.DataFrame, impute_method="mean", scaling="StandardScaler (z-score)"):
#     """
#     Return numpy array X and processed dataframe (with same columns).
#     impute_method: "mean", "median", "drop"
#     scaling: "None", "StandardScaler (z-score)", "MinMaxScaler"
#     """
#     proc = df.copy()
#     # Ensure numeric
#     proc = proc.apply(pd.to_numeric, errors="coerce")
#     if impute_method == "drop":
#         proc = proc.dropna()
#     else:
#         strategy = "mean" if impute_method == "mean" else "median"
#         imputer = SimpleImputer(strategy=strategy)
#         proc[:] = imputer.fit_transform(proc)

#     # scaling
#     if scaling == "StandardScaler (z-score)":
#         scaler = StandardScaler()
#         proc[:] = scaler.fit_transform(proc)
#     elif scaling == "MinMaxScaler":
#         scaler = MinMaxScaler()
#         proc[:] = scaler.fit_transform(proc)
#     # else no scaling

#     X = proc.values
#     return X, proc

# def generate_feature_summary(df: pd.DataFrame):
#     """Return simple summary statistics for the UI."""
#     return df.describe().T


#2nd code
# src/preprocess.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple

def preprocess_dataframe(df: pd.DataFrame,
                         features: list,
                         impute_method: str = "mean",
                         scaling: str = "StandardScaler (z-score)",
                         remove_outliers: bool = False,
                         z_thresh: float = 3.0) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Preprocess dataframe and return numpy array X and processed dataframe (only selected features).
    - features: list of column names to keep (order preserved)
    - impute_method: "mean", "median", "drop"
    - scaling: "None", "StandardScaler (z-score)", "MinMaxScaler"
    - remove_outliers: if True, remove rows with z-score > z_thresh on any feature
    """
    proc = df[features].copy()

    # Convert everything to numeric (coerce errors to NaN)
    proc = proc.apply(pd.to_numeric, errors="coerce")

    # Optional outlier removal (z-score)
    if remove_outliers:
        z = (proc - proc.mean()) / proc.std(ddof=0)
        mask = (z.abs() <= z_thresh).all(axis=1)
        proc = proc[mask].reset_index(drop=True)

    # Imputation
    if impute_method == "drop":
        proc = proc.dropna().reset_index(drop=True)
    else:
        strategy = "mean" if impute_method == "mean" else "median"
        imputer = SimpleImputer(strategy=strategy)
        proc[:] = imputer.fit_transform(proc)

    # Scaling
    if scaling == "StandardScaler (z-score)":
        scaler = StandardScaler()
        proc[:] = scaler.fit_transform(proc)
    elif scaling == "MinMaxScaler":
        scaler = MinMaxScaler()
        proc[:] = scaler.fit_transform(proc)
    # else: no scaling

    X = proc.values
    return X, proc

def generate_feature_summary(df: pd.DataFrame, features: list = None):
    """Return simple summary statistics for the UI (as dataframe)."""
    if features is None:
        features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    return df[features].describe().T.round(3)
