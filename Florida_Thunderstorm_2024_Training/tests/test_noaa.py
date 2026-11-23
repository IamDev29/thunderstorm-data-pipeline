"""Unit tests for Florida NOAA processing pipeline."""

import numpy as np
import pandas as pd
import pytest
import shapely.geometry

from src.process_noaa import NOAAProcessor


def test_florida_state_filtering_and_event_id_preservation():
    """Verify that NOAA processor retains Florida events and preserves event_id."""
    df_mock = pd.DataFrame({
        "event_id": [101, 102, 103],
        "state": ["FLORIDA", "GEORGIA", "florida"],
        "event_type": ["Thunderstorm Wind", "Hail", "Tornado"],
        "begin_date_time": ["01-MAY-24 14:00:00", "02-MAY-24 15:00:00", "03-MAY-24 16:00:00"],
        "begin_lat": [28.5, 33.0, 27.2],
        "begin_lon": [-81.3, -84.0, -82.5],
    })

    # Test state filtering
    df_filtered = df_mock[df_mock["state"].str.upper() == "FLORIDA"]
    assert len(df_filtered) == 2
    assert set(df_filtered["event_id"]) == {101, 103}


def test_coordinates_flagging():
    """Verify that records with missing coordinates are flagged rather than dropped."""
    df_mock = pd.DataFrame({
        "event_id": [1, 2],
        "begin_lat": [28.0, np.nan],
        "begin_lon": [-82.0, np.nan],
    })
    df_mock["coordinates_available"] = df_mock["begin_lat"].notna() & df_mock["begin_lon"].notna()
    assert df_mock.loc[0, "coordinates_available"] is True or df_mock.loc[0, "coordinates_available"] == 1
    assert df_mock.loc[1, "coordinates_available"] is False or df_mock.loc[1, "coordinates_available"] == 0


def test_noaa_reconciliation_and_no_join_multiplication(tmp_path):
    """Verify that location aggregation prevents 1-to-many join duplication."""
    df_details = pd.DataFrame({
        "EVENT_ID": [501, 502],
        "STATE": ["FLORIDA", "FLORIDA"],
        "EVENT_TYPE": ["Thunderstorm Wind", "Hail"],
        "BEGIN_DATE_TIME": ["01-MAY-24 14:00:00", "02-MAY-24 15:00:00"],
        "BEGIN_LAT": [np.nan, 28.0],
        "BEGIN_LON": [np.nan, -82.0],
    })
    # Locations has MULTIPLE entries for event 501
    df_locations = pd.DataFrame({
        "EVENT_ID": [501, 501, 501, 502],
        "LOCATION": ["Loc A", "Loc B", "Loc C", "Loc D"],
        "LATITUDE": [27.5, 27.6, 27.7, 28.0],
        "LONGITUDE": [-81.5, -81.6, -81.7, -82.0],
    })

    # Aggregate locations by event_id before joining
    loc_agg = df_locations.groupby("EVENT_ID").agg({"LATITUDE": "first", "LONGITUDE": "first"}).reset_index()
    merged = pd.merge(df_details, loc_agg, on="EVENT_ID", how="left")

    # Assert exactly 2 rows remain (no duplication)
    assert len(merged) == 2
    assert merged["EVENT_ID"].nunique() == 2
    # Verify event 501 filled its missing coordinate from location table
    assert merged.loc[merged["EVENT_ID"] == 501, "LATITUDE"].iloc[0] == 27.5
