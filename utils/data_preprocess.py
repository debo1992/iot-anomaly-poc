# utils/feature_engineering.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def feature_engineer(df):
    """
    Adds lag feature for front_door_open.
    df: DataFrame with 'front_door_open' column
    """
    df = df.copy()
    # 1-step lag (previous hour)
    df['front_door_open_lag1'] = df['front_door_open'].shift(1).fillna(0)
    # 2-step lag if needed
    df['front_door_open_lag2'] = df['front_door_open'].shift(2).fillna(0)
    return df