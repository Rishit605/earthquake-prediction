from datetime import timezone

import pandas as pd

from eq_prediction.pipeline.clean import clean_raw_events, deduplicate_events
from eq_prediction.pipeline.features import make_model_ready
from eq_prediction.pipeline.normalize import (
    normalize_legacy_dataframe,
    normalize_usgs_features,
)
from eq_prediction.pipeline.split import chronological_split


def test_local_csv_normalization_parses_geo_wkt():
    raw = pd.DataFrame(
        [
            {
                "geo": "POINT Z (-157.5767 52.8086 10)",
                "code": "abc123",
                "time": 1656546654873,
                "updated": 1662238999040,
                "mag": 3.4,
                "magType": "ml",
                "type": "earthquake",
            }
        ]
    )

    result = normalize_legacy_dataframe(raw)

    assert result.loc[0, "event_id"] == "abc123"
    assert result.loc[0, "longitude"] == -157.5767
    assert result.loc[0, "latitude"] == 52.8086
    assert result.loc[0, "depth_km"] == 10
    assert result.loc[0, "time"].tzinfo == timezone.utc


def test_usgs_geojson_normalization():
    features = [
        {
            "id": "us1",
            "geometry": {"coordinates": [77.1, 28.6, 12.0]},
            "properties": {
                "time": 1700000000000,
                "updated": 1700000001000,
                "mag": 4.2,
                "magType": "mb",
                "type": "earthquake",
                "status": "reviewed",
            },
        }
    ]

    result = normalize_usgs_features(features)

    assert result.loc[0, "event_id"] == "us1"
    assert result.loc[0, "longitude"] == 77.1
    assert result.loc[0, "latitude"] == 28.6
    assert result.loc[0, "depth_km"] == 12.0


def test_invalid_coordinates_are_rejected():
    raw = pd.DataFrame(
        [
            {
                "event_id": "bad",
                "time": "2024-01-01T00:00:00Z",
                "updated": "2024-01-01T00:00:01Z",
                "longitude": 999,
                "latitude": 20,
                "mag": 3.0,
                "type": "earthquake",
            },
            {
                "event_id": "good",
                "time": "2024-01-01T00:00:00Z",
                "updated": "2024-01-01T00:00:01Z",
                "longitude": 10,
                "latitude": 20,
                "mag": 3.0,
                "type": "earthquake",
            },
        ]
    )

    clean, rejected = clean_raw_events(raw)

    assert len(clean) == 1
    assert clean.loc[0, "event_id"] == "good"
    assert len(rejected) == 1


def test_deduplicate_keeps_newest_updated():
    raw = pd.DataFrame(
        [
            {
                "event_id": "same",
                "time": "2024-01-01T00:00:00Z",
                "updated": "2024-01-01T00:00:01Z",
                "longitude": 10,
                "latitude": 20,
                "mag": 3.0,
            },
            {
                "event_id": "same",
                "time": "2024-01-01T00:00:00Z",
                "updated": "2024-01-01T00:00:02Z",
                "longitude": 10,
                "latitude": 20,
                "mag": 3.5,
            },
        ]
    )

    result = deduplicate_events(raw)

    assert len(result) == 1
    assert result.loc[0, "mag"] == 3.5


def test_feature_creation_and_chronological_split():
    raw = pd.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(10)],
            "time": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
            "mag": [3.0 + i / 10 for i in range(10)],
            "longitude": [10.0] * 10,
            "latitude": [20.0] * 10,
            "depth_km": [5.0] * 10,
            "dmin": [0.2] * 10,
            "rms": [0.5] * 10,
            "gap": [90.0] * 10,
            "nst": [20.0] * 10,
            "magType": ["ml"] * 10,
        }
    )

    features = make_model_ready(raw, timeseries=True)
    splits = chronological_split(features)

    assert "dmin_km" in features.columns
    assert "hour_sin" in features.columns
    assert "eq_count_last_7d" in features.columns
    assert len(splits.train) == 7
    assert len(splits.validation) == 1
    assert len(splits.test) == 2


def test_normal_features_exclude_timeseries_features():
    raw = pd.DataFrame(
        {
            "event_id": ["e0"],
            "time": [pd.Timestamp("2024-01-01", tz="UTC")],
            "mag": [3.0],
            "longitude": [10.0],
            "latitude": [20.0],
            "depth_km": [5.0],
            "dmin": [0.2],
            "rms": [0.5],
            "gap": [90.0],
            "nst": [20.0],
            "magType": ["ml"],
        }
    )

    features = make_model_ready(raw)

    assert "hour_sin" not in features.columns
    assert "eq_count_last_7d" not in features.columns
