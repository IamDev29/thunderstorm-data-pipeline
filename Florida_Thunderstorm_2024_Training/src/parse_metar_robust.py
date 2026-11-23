import re
from typing import Dict, Any, List, Optional
import pandas as pd

# Regex patterns for present-weather thunderstorm groups
# Explicit TS: TS, +TSRA, -TSRA, TSRA, TSSN, TSRASN, TSGR, TSGS, TSPL, ?TSRA, etc.
# Must NOT match station callsigns like KDTS, KTTS, or remarks like T01000070
EXPLICIT_TS_REGEX = re.compile(
    r"^(?:[+-]|\?)?TS(?:RA|DZ|SN|SG|PL|GS|GR|IC|UP)*$",
    re.IGNORECASE
)

# Vicinity TS: VCTS, VCTSRA, VCTSSN, etc.
VICINITY_TS_REGEX = re.compile(
    r"^VC\??TS(?:RA|DZ|SN|SG|PL|GS|GR|IC|UP)*$",
    re.IGNORECASE
)

# Known METAR body metadata tokens to skip
METAR_METADATA_TOKENS = {
    "METAR", "SPECI", "TESTM", "COR", "AUTO", "RTD", "FINO", "NIL", "CCA", "CCB"
}

# Date-time token regex (e.g., 010000Z, 251853Z)
DATETIME_TOKEN_REGEX = re.compile(r"^\d{4,6}Z$", re.IGNORECASE)

# Genuine lightning remark regex (LTG, LTGIC, LTGCG, OCNL LTG, FRQ LTG, etc., or TSB/TSE)
GENUINE_LTG_REGEX = re.compile(
    r"\b(?:OCNL|FRQ|CONS)?\s*LTG(?:IC|CC|CG|CA)*(?:\s+(?:OHD|VC|DSNT|[NSEW]{1,2}|ALQDS|AND|[A-Z0-9\-\/]+)){0,4}\b",
    re.IGNORECASE
)
TS_BEGIN_END_REGEX = re.compile(r"\b(?:TSB|TSE)\d{2,4}\b", re.IGNORECASE)

# Cumulonimbus regex (e.g. CB, CB DSNT, FEW040CB, SCT030CB)
CB_REGEX = re.compile(r"\bCB\b|\bCB\s+[A-Z0-9\-\/]+|\d{3}CB\b", re.IGNORECASE)


def parse_metar_thunderstorm_robust(
    metar_text: Optional[str],
    wxcodes_text: Optional[str] = "",
    station_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Robustly parse METAR text and wxcodes for thunderstorm phenomena.
    
    Guarantees:
    - Never matches station callsigns (e.g. KDTS, KTTS).
    - Accurately separates body from remarks.
    - Accurately classifies explicit TS (+TSRA, -TS, TSSN, etc.) vs vicinity VCTS.
    - Accurately captures genuine lightning remarks (LTG, TSB, TSE).
    - Accurately detects Cumulonimbus (CB) without treating CB alone as TS.
    - Handles missing, malformed, or ambiguous METAR strings gracefully.
    """
    metar = str(metar_text).strip() if pd.notna(metar_text) else ""
    wx = str(wxcodes_text).strip() if pd.notna(wxcodes_text) else ""

    metar_upper = metar.upper()
    wx_upper = wx.upper()

    # Fast pre-filter: if none of the convective substrings exist, return all False immediately
    if ("TS" not in metar_upper and "LTG" not in metar_upper and "CB" not in metar_upper and "TS" not in wx_upper):
        return {
            "has_thunderstorm_ts": False,
            "has_vicinity_thunderstorm_vcts": False,
            "has_genuine_lightning_remark": False,
            "has_cumulonimbus_cb": False,
            "strict_thunderstorm_reported": False,
            "cb_alone": False,
            "lightning_remarks": None,
        }

    has_ts = False
    has_vcts = False
    has_genuine_ltg = False
    has_cb = False
    ltg_matches: List[str] = []

    # 1. Parse IEM wxcodes field first (if available)
    if wx and wx.lower() not in ["none", "nan", ""]:
        for code in re.split(r"[\s,]+", wx):
            code_clean = code.strip().upper()
            if not code_clean:
                continue
            if VICINITY_TS_REGEX.match(code_clean):
                has_vcts = True
            elif EXPLICIT_TS_REGEX.match(code_clean):
                has_ts = True

    # 2. Parse METAR body and remarks (if metar text available)
    if metar:
        parts = re.split(r"\s+RMK(?:\s+|$)", metar, maxsplit=1)
        body = parts[0].strip()
        remarks = parts[1].strip() if len(parts) > 1 else ""

        # Tokenize body and strip station callsign and report metadata
        body_tokens = body.split()
        if body_tokens:
            start_idx = 0
            # If first token is report type (METAR, SPECI, etc.), advance
            if body_tokens[start_idx].upper() in METAR_METADATA_TOKENS:
                start_idx += 1
            
            # The next token is the station identifier (e.g. KDTS, KTTS, KMIA, 1J0)
            if start_idx < len(body_tokens):
                stn_token = body_tokens[start_idx].upper()
                # Check if it looks like a station identifier (3-4 chars)
                # or matches the known station_id
                if station_id and (stn_token == station_id.upper() or stn_token == f"K{station_id.upper()}"):
                    start_idx += 1
                elif len(stn_token) in [3, 4, 5] and re.match(r"^[A-Z0-9]{3,5}$", stn_token):
                    start_idx += 1

            # If next token is datetime group (e.g. 010000Z), advance
            if start_idx < len(body_tokens) and DATETIME_TOKEN_REGEX.match(body_tokens[start_idx]):
                start_idx += 1

            # Now evaluate remaining body tokens
            weather_tokens = body_tokens[start_idx:]
            for token in weather_tokens:
                t_clean = token.upper().strip()
                # Skip known modifier tokens
                if t_clean in METAR_METADATA_TOKENS:
                    continue
                
                # Check for explicit TS
                if EXPLICIT_TS_REGEX.match(t_clean):
                    has_ts = True
                # Check for vicinity TS
                elif VICINITY_TS_REGEX.match(t_clean):
                    has_vcts = True

        # Check for Cumulonimbus (CB) in body clouds or remarks
        if CB_REGEX.search(metar):
            has_cb = True

        # Parse Remarks Section for genuine lightning remarks
        if remarks:
            # Genuine lightning occurrences (LTG, LTGIC, etc.)
            for m in GENUINE_LTG_REGEX.finditer(remarks):
                match_str = m.group().strip()
                if match_str and match_str.upper() != "CB":
                    ltg_matches.append(match_str)
                    has_genuine_ltg = True

            # Thunderstorm began/ended (TSB, TSE)
            for m in TS_BEGIN_END_REGEX.finditer(remarks):
                match_str = m.group().strip()
                ltg_matches.append(match_str)
                has_genuine_ltg = True

    # Strict thunderstorm definition: TS | VCTS | Genuine LTG remark
    # Note: CB alone is strictly excluded
    strict_ts = bool(has_ts or has_vcts or has_genuine_ltg)
    cb_alone = bool(has_cb and not strict_ts)

    return {
        "has_thunderstorm_ts": bool(has_ts),
        "has_vicinity_thunderstorm_vcts": bool(has_vcts),
        "has_genuine_lightning_remark": bool(has_genuine_ltg),
        "has_cumulonimbus_cb": bool(has_cb),
        "strict_thunderstorm_reported": strict_ts,
        "cb_alone": cb_alone,
        "lightning_remarks": "; ".join(ltg_matches) if ltg_matches else None,
    }
