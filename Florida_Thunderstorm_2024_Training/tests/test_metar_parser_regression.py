"""
test_metar_parser_regression.py
================================
Regression test suite for the corrected METAR thunderstorm parser.

Validates:
1. Station identifier KDTS with no thunderstorm weather group evaluates to False.
2. Station identifier KTTS with no thunderstorm weather group evaluates to False.
3. Ordinary stations with explicit TS, +TSRA, -TSRA, TSSN, TSRASN, TSGR evaluate to True.
4. Vicinity thunderstorm (VCTS) is recognized and separated from in-station TS.
5. Genuine lightning remarks (LTG, TSB, TSE) are parsed and separated.
6. Cumulonimbus (CB) alone in clouds or remarks is detected but STRICTLY EXCLUDED from active TS.
7. Unrelated METAR remarks containing "TS" (e.g. T-groups, POSTS, TESTS, MADISHF) evaluate to False.
8. Missing, null, and malformed METAR strings are handled gracefully without exceptions.
9. Mixed present-weather groups (e.g. -TSRA BR HZ) are correctly parsed.
10. Station identifiers and report metadata (METAR, SPECI, AUTO, COR) are not interpreted as weather.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from Florida_Thunderstorm_2024_Training.src.parse_metar_robust import parse_metar_thunderstorm_robust


def test_01_kdts_no_thunderstorm():
    """KDTS clear sky report must NEVER evaluate to thunderstorm."""
    raw = "KDTS 010000Z AUTO 00000KT 10SM CLR 10/07 A3019 RMK T01000070 MADISHF"
    res = parse_metar_thunderstorm_robust(raw, wxcodes_text="", station_id="DTS")
    assert res["has_thunderstorm_ts"] is False
    assert res["has_vicinity_thunderstorm_vcts"] is False
    assert res["has_genuine_lightning_remark"] is False
    assert res["strict_thunderstorm_reported"] is False
    assert res["cb_alone"] is False


def test_02_ktts_no_thunderstorm():
    """KTTS clear sky report must NEVER evaluate to thunderstorm."""
    raw = "KTTS 010055Z AUTO 00000KT 10SM CLR 10/09 A3020 RMK AO2 SLP231 T01000086 $"
    res = parse_metar_thunderstorm_robust(raw, wxcodes_text="", station_id="TTS")
    assert res["has_thunderstorm_ts"] is False
    assert res["has_vicinity_thunderstorm_vcts"] is False
    assert res["has_genuine_lightning_remark"] is False
    assert res["strict_thunderstorm_reported"] is False
    assert res["cb_alone"] is False


def test_03_prefixed_metar_kdts_ktts():
    """Reports starting with report-type prefix 'METAR KDTS' or 'SPECI KTTS'."""
    raw_dts = "METAR KDTS 010000Z AUTO 00000KT 10SM CLR 10/07 A3019 RMK T01000070 MADISHF"
    res_dts = parse_metar_thunderstorm_robust(raw_dts, station_id="DTS")
    assert res_dts["strict_thunderstorm_reported"] is False

    raw_tts = "SPECI KTTS 010055Z AUTO 00000KT 10SM CLR 10/09 A3020 RMK AO2"
    res_tts = parse_metar_thunderstorm_robust(raw_tts, station_id="TTS")
    assert res_tts["strict_thunderstorm_reported"] is False


def test_04_genuine_ts_at_dts_and_tts():
    """When KDTS and KTTS actually observe thunderstorms, they MUST be detected."""
    # DTS with -TSRA and TSB22
    raw_dts = "KDTS 090728Z AUTO 15020G38KT 3SM -TSRA BR OVC004 18/16 A2964 RMK AO2 PK WND 15039/0713 RAB24 TSB22 PRESFR"
    res_dts = parse_metar_thunderstorm_robust(raw_dts, wxcodes_text="-TSRA BR", station_id="DTS")
    assert res_dts["has_thunderstorm_ts"] is True
    assert res_dts["has_genuine_lightning_remark"] is True
    assert res_dts["strict_thunderstorm_reported"] is True

    # TTS with TSRA
    raw_tts = "KTTS 050209Z AUTO 24010G26KT 3SM TSRA SCT030 BKN055 17/13 A2960 RMK AO2 TSB09"
    res_tts = parse_metar_thunderstorm_robust(raw_tts, wxcodes_text="TSRA", station_id="TTS")
    assert res_tts["has_thunderstorm_ts"] is True
    assert res_tts["has_genuine_lightning_remark"] is True
    assert res_tts["strict_thunderstorm_reported"] is True


def test_05_explicit_ts_combinations():
    """Verify various standard thunderstorm present-weather groups."""
    valid_groups = [
        "TS", "-TS", "+TS", "TSRA", "-TSRA", "+TSRA",
        "TSSN", "-TSSN", "+TSSN", "TSRASN", "-TSRASN",
        "TSGR", "+TSGR", "TSGS", "-TSGS", "TSPL"
    ]
    for grp in valid_groups:
        raw = f"KMIA 151853Z 18012KT 5SM {grp} BKN030 25/20 A2995"
        res = parse_metar_thunderstorm_robust(raw, wxcodes_text=grp, station_id="MIA")
        assert res["has_thunderstorm_ts"] is True, f"Failed on group: {grp}"
        assert res["strict_thunderstorm_reported"] is True, f"Failed strict TS on group: {grp}"


def test_06_vicinity_vcts():
    """VCTS must be marked as vicinity TS and strict TS, but NOT explicit TS."""
    raw = "KMCO 151853Z 18012KT 10SM VCTS SCT035 28/21 A2998 RMK AO2"
    res = parse_metar_thunderstorm_robust(raw, wxcodes_text="VCTS", station_id="MCO")
    assert res["has_thunderstorm_ts"] is False
    assert res["has_vicinity_thunderstorm_vcts"] is True
    assert res["strict_thunderstorm_reported"] is True

    raw_vctsra = "KJAX 151853Z 18012KT 6SM VCTSRA SCT035 28/21 A2998"
    res_vctsra = parse_metar_thunderstorm_robust(raw_vctsra, wxcodes_text="VCTSRA", station_id="JAX")
    assert res_vctsra["has_thunderstorm_ts"] is False
    assert res_vctsra["has_vicinity_thunderstorm_vcts"] is True
    assert res_vctsra["strict_thunderstorm_reported"] is True


def test_07_lightning_remarks():
    """Genuine lightning in remarks without TS in body."""
    raw_ltg = "KTLH 151853Z 18012KT 10SM CLR 25/20 A2992 RMK AO2 LTGICCG OHD"
    res = parse_metar_thunderstorm_robust(raw_ltg, station_id="TLH")
    assert res["has_thunderstorm_ts"] is False
    assert res["has_genuine_lightning_remark"] is True
    assert res["strict_thunderstorm_reported"] is True
    assert "LTGICCG OHD" in res["lightning_remarks"]

    raw_frq = "KTLH 151853Z 18012KT 10SM CLR 25/20 A2992 RMK AO2 FRQ LTG DSNT NW"
    res_frq = parse_metar_thunderstorm_robust(raw_frq, station_id="TLH")
    assert res_frq["has_genuine_lightning_remark"] is True
    assert res_frq["strict_thunderstorm_reported"] is True

    raw_tsb = "KPNS 151853Z 18012KT 10SM -RA 22/20 A2992 RMK AO2 RAB1810 TSB1845"
    res_tsb = parse_metar_thunderstorm_robust(raw_tsb, wxcodes_text="-RA", station_id="PNS")
    assert res_tsb["has_thunderstorm_ts"] is False
    assert res_tsb["has_genuine_lightning_remark"] is True
    assert res_tsb["strict_thunderstorm_reported"] is True


def test_08_cumulonimbus_cb_alone():
    """CB alone must be detected but STRICTLY EXCLUDED from active TS."""
    raw_cb_cloud = "KORL 151853Z 18012KT 10SM FEW040CB 28/20 A2995 RMK AO2"
    res_cloud = parse_metar_thunderstorm_robust(raw_cb_cloud, station_id="ORL")
    assert res_cloud["has_thunderstorm_ts"] is False
    assert res_cloud["has_vicinity_thunderstorm_vcts"] is False
    assert res_cloud["has_genuine_lightning_remark"] is False
    assert res_cloud["has_cumulonimbus_cb"] is True
    assert res_cloud["cb_alone"] is True
    assert res_cloud["strict_thunderstorm_reported"] is False

    raw_cb_rmk = "KORL 151853Z 18012KT 10SM SCT040 28/20 A2995 RMK AO2 CB DSNT SE"
    res_rmk = parse_metar_thunderstorm_robust(raw_cb_rmk, station_id="ORL")
    assert res_rmk["has_cumulonimbus_cb"] is True
    assert res_rmk["cb_alone"] is True
    assert res_rmk["strict_thunderstorm_reported"] is False


def test_09_unrelated_tokens_containing_ts():
    """Unrelated tokens containing substring 'TS' must not trigger false positives."""
    raw = "KMIA 151853Z 18012KT 10SM CLR 18/12 A3012 RMK AO2 TESTS RESULTS POSTS T01800120 MADISHF"
    res = parse_metar_thunderstorm_robust(raw, station_id="MIA")
    assert res["has_thunderstorm_ts"] is False
    assert res["has_vicinity_thunderstorm_vcts"] is False
    assert res["has_genuine_lightning_remark"] is False
    assert res["strict_thunderstorm_reported"] is False


def test_10_missing_and_malformed_reports():
    """Handle null, empty, or weirdly formatted METARs gracefully."""
    # None / empty
    res_none = parse_metar_thunderstorm_robust(None, None)
    assert res_none["strict_thunderstorm_reported"] is False

    res_empty = parse_metar_thunderstorm_robust("", "")
    assert res_empty["strict_thunderstorm_reported"] is False

    # Only wxcodes available
    res_wx = parse_metar_thunderstorm_robust(None, "TSRA")
    assert res_wx["has_thunderstorm_ts"] is True
    assert res_wx["strict_thunderstorm_reported"] is True

    # Malformed whitespace and lowercase
    res_malformed = parse_metar_thunderstorm_robust("   metar   kdts   010000z   auto  tsra  10sm   rmk   ao2  ", station_id="DTS")
    assert res_malformed["has_thunderstorm_ts"] is True
    assert res_malformed["strict_thunderstorm_reported"] is True


def test_11_mixed_present_weather():
    """Mixed weather groups like '-TSRA BR HZ'."""
    raw = "KJAX 151853Z 18012KT 4SM -TSRA BR HZ OVC025 22/20 A2995 RMK AO2"
    res = parse_metar_thunderstorm_robust(raw, wxcodes_text="-TSRA BR HZ", station_id="JAX")
    assert res["has_thunderstorm_ts"] is True
    assert res["strict_thunderstorm_reported"] is True
