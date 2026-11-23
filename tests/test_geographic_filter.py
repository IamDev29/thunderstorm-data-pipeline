"""Unit tests for Geographic and Event-Type Filter."""

import numpy as np
import pandas as pd
import pytest

from src.geographic_filter import BoundingBox, GeographicFilter


@pytest.fixture
def sample_config():
    return {
        "event_types": [
            "Thunderstorm Wind",
            "Hail",
            "Lightning",
            "Heavy Rain",
            "Flash Flood",
            "Tornado",
            "Tropical Storm",
            "Hurricane (Typhoon)",
        ],
        "geographic_filters": {
            "target_states": {
                "florida": "FLORIDA",
                "texas": "TEXAS",
                "louisiana": "LOUISIANA",
                "mississippi": "MISSISSIPPI",
            },
            "regions": {
                "florida": {
                    "bounding_box": {"lat_min": 24.5, "lat_max": 31.0, "lon_min": -88.0, "lon_max": -80.0}
                },
                "texas": {
                    "coastal_focus": {"lat_min": 25.8, "lat_max": 31.5, "lon_min": -98.5, "lon_max": -93.5}
                },
            },
            "combined_gulf_coast": {
                "states": ["FLORIDA", "TEXAS", "LOUISIANA", "MISSISSIPPI"],
                "bounding_box": {"lat_min": 24.5, "lat_max": 32.0, "lon_min": -98.5, "lon_max": -80.0},
            },
        },
    }


def test_bounding_box_contains():
    bbox = BoundingBox(lat_min=24.5, lat_max=31.0, lon_min=-88.0, lon_max=-80.0)
    assert bbox.contains(28.0, -82.0) is True
    assert bbox.contains(35.0, -82.0) is False  # Too far north
    assert bbox.contains(28.0, -95.0) is False  # Too far west


def test_filter_by_state(sample_config):
    geo_filter = GeographicFilter(sample_config)
    df = pd.DataFrame(
        {
            "event_id": ["1", "2", "3", "4"],
            "state": ["FLORIDA", "TEXAS", "CALIFORNIA", "florida "],
        }
    )
    fl_df = geo_filter.filter_by_state(df, "FLORIDA")
    assert len(fl_df) == 2
    assert set(fl_df["event_id"]) == {"1", "4"}


def test_missing_coordinates_are_retained_with_flag(sample_config):
    geo_filter = GeographicFilter(sample_config)
    bbox = BoundingBox(lat_min=24.5, lat_max=31.0, lon_min=-88.0, lon_max=-80.0)

    df = pd.DataFrame(
        {
            "event_id": ["E1", "E2", "E3"],
            "state": ["FLORIDA", "FLORIDA", "FLORIDA"],
            "begin_lat_num": [28.0, np.nan, 35.0],  # E1 in bbox, E2 missing, E3 out of bbox
            "begin_lon_num": [-82.0, np.nan, -82.0],
            "coord_status": ["VALID", "MISSING", "VALID"],
        }
    )

    filtered = geo_filter.apply_bounding_box_filter(df, bbox, include_missing_coords=True)

    # E1 (in bbox) and E2 (missing coords) should be retained!
    # E3 (out of bbox) should be excluded!
    assert len(filtered) == 2
    assert set(filtered["event_id"]) == {"E1", "E2"}

    # Verify insufficient_coords flag
    e1_row = filtered[filtered["event_id"] == "E1"].iloc[0]
    e2_row = filtered[filtered["event_id"] == "E2"].iloc[0]
    assert e1_row["insufficient_coords"] == False
    assert e2_row["insufficient_coords"] == True


def test_event_type_filtering_preserves_original_values(sample_config):
    geo_filter = GeographicFilter(sample_config)
    df = pd.DataFrame(
        {
            "event_id": ["1", "2", "3", "4"],
            "event_type": [
                "Thunderstorm Wind",
                "HAIL",
                "Winter Storm",
                "Hurricane (Typhoon)",
            ],
        }
    )

    filtered = geo_filter.filter_by_event_types(df)
    # Winter Storm excluded; others kept
    assert len(filtered) == 3
    # Check that original values/casing are preserved
    assert "HAIL" in filtered["event_type"].values
    assert "Thunderstorm Wind" in filtered["event_type"].values
    assert "Hurricane (Typhoon)" in filtered["event_type"].values


def test_filter_associated_locations_and_fatalities():
    locations_df = pd.DataFrame(
        {
            "event_id": ["E1", "E1", "E2", "E3"],
            "location_index": [1, 2, 1, 1],
            "location": ["Town A", "Town B", "Town C", "Town D"],
        }
    )
    fatalities_df = pd.DataFrame(
        {
            "event_id": ["E1", "E3"],
            "fatality_id": ["F1", "F2"],
            "fatality_type": ["D", "I"],
        }
    )

    selected_event_ids = {"E1", "E2"}
    filtered_locs = GeographicFilter.filter_associated_locations(locations_df, selected_event_ids)
    filtered_fats = GeographicFilter.filter_associated_fatalities(fatalities_df, selected_event_ids)

    # E1 has 2 locations, E2 has 1 location -> 3 total locations retained
    assert len(filtered_locs) == 3
    assert set(filtered_locs["event_id"]) == {"E1", "E2"}

    # Fatalities: only E1 retained
    assert len(filtered_fats) == 1
    assert filtered_fats.iloc[0]["event_id"] == "E1"
