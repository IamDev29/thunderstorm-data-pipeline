"""
test_single_date_query.py
"""
import requests
import io
import time
import pandas as pd

url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
params = {
    "network": "FL_ASOS",
    "data": ["metar", "wxcodes", "tmpf", "dwpf", "sknt", "drct", "gust", "vsby", "p01i", "alti"],
    "year1": "2024", "month1": "1", "day1": "5",
    "year2": "2024", "month2": "1", "day2": "5",
    "hour2": "23", "minute2": "59",
    "tz": "Etc/UTC", "format": "onlycomma", "latlon": "yes", "elev": "yes",
    "missing": "null", "trace": "T", "direct": "yes", "report_type": ["1", "2"]
}
t0 = time.time()
resp = requests.get(url, params=params, headers={"User-Agent": "FloridaThunderstormAudit/1.0"}, timeout=45)
dur = time.time() - t0
print(f"Status: {resp.status_code}, Duration: {dur:.2f}s, Bytes: {len(resp.text):,}")

df = pd.read_csv(io.StringIO(resp.text))
print(f"Rows returned: {len(df):,}")
print(f"Unique stations: {df['station'].nunique()}")
print(f"Min valid: {df['valid'].min()}, Max valid: {df['valid'].max()}")
print(f"Unique dates in returned data: {pd.to_datetime(df['valid']).dt.date.unique()}")
