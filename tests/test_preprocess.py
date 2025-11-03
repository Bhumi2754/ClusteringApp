# tests/test_preprocess.py
import pandas as pd
from src.preprocess import preprocess_dataframe

def test_preprocess_basic():
    df = pd.DataFrame({
        "a":[1,2,3],
        "b":[4,5,6]
    })
    X, proc = preprocess_dataframe(df, impute_method="mean", scaling="StandardScaler (z-score)")
    assert X.shape == (3,2)
