import sys
from pathlib import Path
sys.path.insert(0, str(Path("Florida_Thunderstorm_2024_Training").resolve()))
sys.path.insert(0, str(Path(".").resolve()))
import pandas as pd
from Florida_Thunderstorm_2024_Training.src.parse_metar_robust import parse_metar_thunderstorm_robust

OUT_DIR = Path("data/processed/asos_parser_fixed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

test_cases = [
    {
        "test_id": "TC01",
        "description": "KDTS clear sky report (no weather)",
        "station_id": "DTS",
        "raw_metar": "KDTS 010000Z AUTO 00000KT 10SM CLR 10/07 A3019 RMK T01000070 MADISHF",
        "wxcodes": "",
        "expected_ts": False,
        "expected_vcts": False,
        "expected_strict": False,
        "expected_cb_alone": False,
    },
    {
        "test_id": "TC02",
        "description": "KTTS clear sky report (no weather)",
        "station_id": "TTS",
        "raw_metar": "KTTS 010055Z AUTO 00000KT 10SM CLR 10/09 A3020 RMK AO2 SLP231 T01000086 $",
        "wxcodes": "",
        "expected_ts": False,
        "expected_vcts": False,
        "expected_strict": False,
        "expected_cb_alone": False,
    },
    {
        "test_id": "TC03",
        "description": "Prefixed METAR KDTS clear sky",
        "station_id": "DTS",
        "raw_metar": "METAR KDTS 010000Z AUTO 00000KT 10SM CLR 10/07 A3019 RMK T01000070 MADISHF",
        "wxcodes": "",
        "expected_ts": False,
        "expected_vcts": False,
        "expected_strict": False,
        "expected_cb_alone": False,
    },
    {
        "test_id": "TC04",
        "description": "Prefixed SPECI KTTS clear sky",
        "station_id": "TTS",
        "raw_metar": "SPECI KTTS 010055Z AUTO 00000KT 10SM CLR 10/09 A3020 RMK AO2",
        "wxcodes": "",
        "expected_ts": False,
        "expected_vcts": False,
        "expected_strict": False,
        "expected_cb_alone": False,
    },
    {
        "test_id": "TC05",
        "description": "KDTS genuine active thunderstorm (-TSRA BR TSB22)",
        "station_id": "DTS",
        "raw_metar": "KDTS 090728Z AUTO 15020G38KT 3SM -TSRA BR OVC004 18/16 A2964 RMK AO2 PK WND 15039/0713 RAB24 TSB22 PRESFR",
        "wxcodes": "-TSRA BR",
        "expected_ts": True,
        "expected_vcts": False,
        "expected_strict": True,
        "expected_cb_alone": False,
    },
    {
        "test_id": "TC06",
        "description": "KTTS genuine active thunderstorm (TSRA TSB09)",
        "station_id": "TTS",
        "raw_metar": "KTTS 050209Z AUTO 24010G26KT 3SM TSRA SCT030 BKN055 17/13 A2960 RMK AO2 TSB09",
        "wxcodes": "TSRA",
        "expected_ts": True,
        "expected_vcts": False,
        "expected_strict": True,
        "expected_cb_alone": False,
    },
    {
        "test_id": "TC07",
        "description": "KDTS vicinity thunderstorm with distant lightning (VCTS LTG DSNT)",
        "station_id": "DTS",
        "raw_metar": "KDTS 081953Z 11013G21KT 10SM VCTS OVC075 14/07 A3002 RMK AO2 LTG DSNT N RAB1858E21 SLP166 P0001",
        "wxcodes": "VCTS",
        "expected_ts": False,
        "expected_vcts": True,
        "expected_strict": True,
        "expected_cb_alone": False,
    },
    {
        "test_id": "TC08",
        "description": "Standard explicit TS at KMIA",
        "station_id": "MIA",
        "raw_metar": "KMIA 151853Z 18012KT 5SM TS BKN030 25/20 A2995",
        "wxcodes": "TS",
        "expected_ts": True,
        "expected_vcts": False,
        "expected_strict": True,
        "expected_cb_alone": False,
    },
    {
        "test_id": "TC09",
        "description": "Heavy thunderstorm with rain (+TSRA)",
        "station_id": "JAX",
        "raw_metar": "KJAX 151853Z 18012KT 2SM +TSRA OVC020 24/22 A2992",
        "wxcodes": "+TSRA",
        "expected_ts": True,
        "expected_vcts": False,
        "expected_strict": True,
        "expected_cb_alone": False,
    },
    {
        "test_id": "TC10",
        "description": "Thunderstorm with snow and rain (TSRASN)",
        "station_id": "PNS",
        "raw_metar": "KPNS 151853Z 18012KT 4SM TSRASN OVC015 01/00 A2990",
        "wxcodes": "TSRASN",
        "expected_ts": True,
        "expected_vcts": False,
        "expected_strict": True,
        "expected_cb_alone": False,
    },
    {
        "test_id": "TC11",
        "description": "Thunderstorm with hail (+TSGR)",
        "station_id": "TLH",
        "raw_metar": "KTLH 151853Z 18012KT 3SM +TSGR BKN030 22/18 A2994",
        "wxcodes": "+TSGR",
        "expected_ts": True,
        "expected_vcts": False,
        "expected_strict": True,
        "expected_cb_alone": False,
    },
    {
        "test_id": "TC12",
        "description": "Vicinity thunderstorm without in-station TS (VCTS)",
        "station_id": "MCO",
        "raw_metar": "KMCO 151853Z 18012KT 10SM VCTS SCT035 28/21 A2998 RMK AO2",
        "wxcodes": "VCTS",
        "expected_ts": False,
        "expected_vcts": True,
        "expected_strict": True,
        "expected_cb_alone": False,
    },
    {
        "test_id": "TC13",
        "description": "Lightning remarks in RMK section without TS in body (LTGICCG OHD)",
        "station_id": "TLH",
        "raw_metar": "KTLH 151853Z 18012KT 10SM CLR 25/20 A2992 RMK AO2 LTGICCG OHD",
        "wxcodes": "",
        "expected_ts": False,
        "expected_vcts": False,
        "expected_strict": True,
        "expected_cb_alone": False,
    },
    {
        "test_id": "TC14",
        "description": "Cumulonimbus (CB) cloud alone in body without TS",
        "station_id": "ORL",
        "raw_metar": "KORL 151853Z 18012KT 10SM FEW040CB 28/20 A2995 RMK AO2",
        "wxcodes": "",
        "expected_ts": False,
        "expected_vcts": False,
        "expected_strict": False,
        "expected_cb_alone": True,
    },
    {
        "test_id": "TC15",
        "description": "Cumulonimbus remark alone in RMK without TS",
        "station_id": "ORL",
        "raw_metar": "KORL 151853Z 18012KT 10SM SCT040 28/20 A2995 RMK AO2 CB DSNT SE",
        "wxcodes": "",
        "expected_ts": False,
        "expected_vcts": False,
        "expected_strict": False,
        "expected_cb_alone": True,
    },
    {
        "test_id": "TC16",
        "description": "Unrelated tokens containing substring 'TS' (TESTS, POSTS, T-group)",
        "station_id": "MIA",
        "raw_metar": "KMIA 151853Z 18012KT 10SM CLR 18/12 A3012 RMK AO2 TESTS RESULTS POSTS T01800120 MADISHF",
        "wxcodes": "",
        "expected_ts": False,
        "expected_vcts": False,
        "expected_strict": False,
        "expected_cb_alone": False,
    },
    {
        "test_id": "TC17",
        "description": "Mixed present weather (-TSRA BR HZ)",
        "station_id": "JAX",
        "raw_metar": "KJAX 151853Z 18012KT 4SM -TSRA BR HZ OVC025 22/20 A2995 RMK AO2",
        "wxcodes": "-TSRA BR HZ",
        "expected_ts": True,
        "expected_vcts": False,
        "expected_strict": True,
        "expected_cb_alone": False,
    },
    {
        "test_id": "TC18",
        "description": "Null / None METAR string",
        "station_id": "MIA",
        "raw_metar": None,
        "wxcodes": None,
        "expected_ts": False,
        "expected_vcts": False,
        "expected_strict": False,
        "expected_cb_alone": False,
    },
    {
        "test_id": "TC19",
        "description": "Empty string METAR",
        "station_id": "MIA",
        "raw_metar": "",
        "wxcodes": "",
        "expected_ts": False,
        "expected_vcts": False,
        "expected_strict": False,
        "expected_cb_alone": False,
    },
    {
        "test_id": "TC20",
        "description": "Malformed whitespace and case",
        "station_id": "DTS",
        "raw_metar": "   metar   kdts   010000z   auto  tsra  10sm   rmk   ao2  ",
        "wxcodes": "",
        "expected_ts": True,
        "expected_vcts": False,
        "expected_strict": True,
        "expected_cb_alone": False,
    },
]

rows = []
for tc in test_cases:
    res = parse_metar_thunderstorm_robust(tc["raw_metar"], tc["wxcodes"], tc["station_id"])
    passed = (
        res["has_thunderstorm_ts"] == tc["expected_ts"]
        and res["has_vicinity_thunderstorm_vcts"] == tc["expected_vcts"]
        and res["strict_thunderstorm_reported"] == tc["expected_strict"]
        and res["cb_alone"] == tc["expected_cb_alone"]
    )
    rows.append({
        "test_id": tc["test_id"],
        "description": tc["description"],
        "station_id": tc["station_id"],
        "raw_metar": tc["raw_metar"],
        "wxcodes": tc["wxcodes"],
        "expected_ts": tc["expected_ts"],
        "actual_ts": res["has_thunderstorm_ts"],
        "expected_vcts": tc["expected_vcts"],
        "actual_vcts": res["has_vicinity_thunderstorm_vcts"],
        "expected_strict": tc["expected_strict"],
        "actual_strict": res["strict_thunderstorm_reported"],
        "expected_cb_alone": tc["expected_cb_alone"],
        "actual_cb_alone": res["cb_alone"],
        "lightning_remarks": res["lightning_remarks"],
        "status": "PASS" if passed else "FAIL"
    })

df_tests = pd.DataFrame(rows)
df_tests.to_csv(OUT_DIR / "parser_regression_tests.csv", index=False)
subproject_out = Path("Florida_Thunderstorm_2024_Training/data/processed/asos_parser_fixed")
subproject_out.mkdir(parents=True, exist_ok=True)
df_tests.to_csv(subproject_out / "parser_regression_tests.csv", index=False)

print(f"Exported parser regression tests ({len(df_tests)} cases):")
print(df_tests["status"].value_counts().to_dict())
assert (df_tests["status"] == "PASS").all(), "Some regression tests failed!"
