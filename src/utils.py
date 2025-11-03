# src/utils.py
import pandas as pd
import io
import os

def df_to_csv_download(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")

def ensure_data_folder():
    if not os.path.exists("data"):
        os.makedirs("data")
