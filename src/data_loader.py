# src/data_loader.py
import pandas as pd
import streamlit as st
from typing import Optional
import io

def load_csv(uploaded_file) -> Optional[pd.DataFrame]:
    """Load uploaded csv or return None (Streamlit will handle UI)."""
    if uploaded_file is None:
        # Caller should use sample
        return None
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None

    if df.shape[0] == 0:
        st.error("CSV appears empty.")
        return None
    return df
