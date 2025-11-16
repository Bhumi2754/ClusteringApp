# # src/data_loader.py
# import pandas as pd
# import streamlit as st
# from typing import Optional
# import io

# def load_csv(uploaded_file) -> Optional[pd.DataFrame]:
#     """Load uploaded csv or return None (Streamlit will handle UI)."""
#     if uploaded_file is None:
#         # Caller should use sample
#         return None
#     try:
#         df = pd.read_csv(uploaded_file)
#     except Exception as e:
#         st.error(f"Error reading CSV: {e}")
#         return None

#     if df.shape[0] == 0:
#         st.error("CSV appears empty.")
#         return None
#     return df

#2nd code

# src/data_loader.py
import pandas as pd
from typing import Optional, Tuple
import os

SAMPLE_PATH = os.path.join("data", "sample_data.csv")

def load_csv_from_streamlit(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load a CSV uploaded via Streamlit file_uploader.
    Returns DataFrame or None.
    """
    if uploaded_file is None:
        return None
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        # Let caller show the error via Streamlit
        raise ValueError(f"Error reading CSV: {e}")
    if df.shape[0] == 0:
        raise ValueError("CSV appears empty.")
    return df

def load_sample_if_none(df: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, str]:
    """
    If df is None, attempt to load sample CSV from data/sample_data.csv.
    Returns (df, source) where source is 'uploaded' or 'sample'
    """
    if df is not None:
        return df, "uploaded"
    if os.path.exists(SAMPLE_PATH):
        sample_df = pd.read_csv(SAMPLE_PATH)
        return sample_df, "sample"
    # fallback: create a very small synthetic sample
    sample_df = pd.DataFrame({
        "NPP": [40, 60, 55],
        "Biomass": [100, 120, 80],
        "Temperature": [22.5, 24.0, 19.5],
        "Trophic_Efficiency": [0.12, 0.15, 0.10],
        "Energy_Flow": [5.2, 6.1, 4.8],
        "lat": [12.9, 13.1, 12.8],
        "lon": [77.5, 77.6, 77.4]
    })
    return sample_df, "generated_sample"
