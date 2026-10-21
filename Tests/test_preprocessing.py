import pandas as pd
from src.preprocessing import FEATURE_COLUMNS, clean_taxi_data, engineer_features, chronological_split


def sample_rows():
    return pd.DataFrame({
        "fare_amount": [10.0, 12.0, 8.0, -1.0, 9.0],
        "pickup_datetime": ["2015-01-01 00:00:00 UTC", "2015-01-03 12:00:00 UTC", "2015-01-05 12:00:00 UTC", "2015-01-06 12:00:00 UTC", "2015-01-07 12:00:00 UTC"],
        "pickup_longitude": [-73.99, -73.98, -73.97, -73.96, -73.95],
        "pickup_latitude": [40.75, 40.75, 40.75, 40.75, 40.75],
        "dropoff_longitude": [-73.98, -73.96, -73.95, -73.95, -73.94],
        "dropoff_latitude": [40.76, 40.76, 40.76, 40.76, 40.76],
        "passenger_count": [1, 2, 1, 1, 1],
    })


def test_cleaning_and_feature_columns():
    clean = clean_taxi_data(sample_rows())
    assert len(clean) == 4
    features = engineer_features(clean)
    assert list(set(FEATURE_COLUMNS) - set(features.columns)) == []
    assert (features.haversine_distance_km > 0.05).all()
    assert features.iloc[0].pickup_day_of_week == 3
    assert features.iloc[0].is_weekend == 0


def test_chronological_split_keeps_latest_rows_for_validation():
    features = engineer_features(clean_taxi_data(sample_rows()))
    train, validation = chronological_split(features, validation_fraction=0.25)
    assert len(train) == 3 and len(validation) == 1
    assert train.pickup_datetime.max() < validation.pickup_datetime.min()
