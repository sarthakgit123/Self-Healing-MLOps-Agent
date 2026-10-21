"""Data cleaning, deterministic chronological splitting, and taxi features."""
from __future__ import annotations

import numpy as np
import pandas as pd
from src.utils import (FARE_BOUNDS, NYC_LAT_BOUNDS, NYC_LON_BOUNDS,
                       PASSENGER_BOUNDS, VALIDATION_FRACTION)

AIRPORTS = {"JFK": (40.6413, -73.7781), "LGA": (40.7769, -73.8740), "EWR": (40.6895, -74.1745)}
FEATURE_COLUMNS = [
    "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
    "passenger_count", "pickup_hour", "pickup_day_of_week", "pickup_month", "pickup_year",
    "is_weekend", "abs_lat_diff", "abs_lon_diff", "haversine_distance_km",
    "manhattan_distance_km", "pickup_dist_jfk", "dropoff_dist_jfk", "pickup_dist_lga",
    "dropoff_dist_lga", "pickup_dist_ewr", "dropoff_dist_ewr",
]


def haversine_vectorized(lon1, lat1, lon2, lat2):
    """Great-circle distance in kilometers; inputs may be scalars or arrays."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def clean_taxi_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the notebook's original domain filters without mutating input."""
    required = {"fare_amount", "pickup_datetime", "pickup_longitude", "pickup_latitude",
                "dropoff_longitude", "dropoff_latitude", "passenger_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = df.dropna().copy()
    data = data[data["fare_amount"].between(*FARE_BOUNDS)]
    data = data[data["passenger_count"].between(*PASSENGER_BOUNDS)]
    lon_ok = data["pickup_longitude"].between(*NYC_LON_BOUNDS) & data["dropoff_longitude"].between(*NYC_LON_BOUNDS)
    lat_ok = data["pickup_latitude"].between(*NYC_LAT_BOUNDS) & data["dropoff_latitude"].between(*NYC_LAT_BOUNDS)
    data = data[lon_ok & lat_ok].copy()
    data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"], format="%Y-%m-%d %H:%M:%S UTC", errors="raise")
    stationary = data["pickup_longitude"].eq(data["dropoff_longitude"]) & data["pickup_latitude"].eq(data["dropoff_latitude"])
    return data.loc[~stationary].copy()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate the original 20 prediction features and remove trips <= 50 m."""
    data = df.copy()
    dt = pd.to_datetime(data["pickup_datetime"])
    data["pickup_hour"] = dt.dt.hour
    data["pickup_day_of_week"] = dt.dt.dayofweek
    data["pickup_month"] = dt.dt.month
    data["pickup_year"] = dt.dt.year
    data["is_weekend"] = data["pickup_day_of_week"].isin([5, 6]).astype(int)
    data["abs_lat_diff"] = (data["dropoff_latitude"] - data["pickup_latitude"]).abs()
    data["abs_lon_diff"] = (data["dropoff_longitude"] - data["pickup_longitude"]).abs()
    data["haversine_distance_km"] = haversine_vectorized(data["pickup_longitude"], data["pickup_latitude"], data["dropoff_longitude"], data["dropoff_latitude"])
    data["manhattan_distance_km"] = data["abs_lat_diff"] * 111.0 + data["abs_lon_diff"] * 84.0
    for code, (lat, lon) in AIRPORTS.items():
        data[f"pickup_dist_{code.lower()}"] = haversine_vectorized(data["pickup_longitude"], data["pickup_latitude"], lon, lat)
        data[f"dropoff_dist_{code.lower()}"] = haversine_vectorized(data["dropoff_longitude"], data["dropoff_latitude"], lon, lat)
    return data.loc[data["haversine_distance_km"] > 0.05].copy()


def chronological_split(features: pd.DataFrame, validation_fraction: float = VALIDATION_FRACTION):
    """Sort by timestamp and take the latest fraction as held-out validation."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    ordered = features.sort_values("pickup_datetime").reset_index(drop=True)
    split_idx = int(len(ordered) * (1 - validation_fraction))
    return ordered.iloc[:split_idx].copy(), ordered.iloc[split_idx:].copy()
