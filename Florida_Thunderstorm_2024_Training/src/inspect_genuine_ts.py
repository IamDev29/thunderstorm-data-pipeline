import pandas as pd

df = pd.read_parquet("data/processed/asos_repair/florida_asos_observations_repaired_2024.parquet")
# Find rows with actual TS in wxcodes or lightning in remarks
sample_wx_ts = df[df["wxcodes"].fillna("").str.contains(r"\b(?:\+|-)?(?:VC)?TS", regex=True)]
print(f"Total rows with regex TS in wxcodes: {len(sample_wx_ts):,}")
print("Sample raw METAR strings with TS in wxcodes:")
for idx, r in sample_wx_ts.head(10).iterrows():
    w_str = str(r['wxcodes']).encode('ascii', errors='replace').decode('ascii')
    m_str = str(r['metar']).encode('ascii', errors='replace').decode('ascii')
    print(f"[{r['station_id']}] wx='{w_str}' | metar='{m_str}'")

sample_dts_genuine = df[(df["station_id"] == "DTS") & df["wxcodes"].fillna("").str.contains(r"\b(?:\+|-)?(?:VC)?TS", regex=True)]
print(f"\nStation DTS rows with genuine TS in wxcodes: {len(sample_dts_genuine)}")
for idx, r in sample_dts_genuine.head(5).iterrows():
    w_str = str(r['wxcodes']).encode('ascii', errors='replace').decode('ascii')
    m_str = str(r['metar']).encode('ascii', errors='replace').decode('ascii')
    print(f"[DTS] wx='{w_str}' | metar='{m_str}'")

sample_tts_genuine = df[(df["station_id"] == "TTS") & df["wxcodes"].fillna("").str.contains(r"\b(?:\+|-)?(?:VC)?TS", regex=True)]
print(f"\nStation TTS rows with genuine TS in wxcodes: {len(sample_tts_genuine)}")
for idx, r in sample_tts_genuine.head(5).iterrows():
    w_str = str(r['wxcodes']).encode('ascii', errors='replace').decode('ascii')
    m_str = str(r['metar']).encode('ascii', errors='replace').decode('ascii')
    print(f"[TTS] wx='{w_str}' | metar='{m_str}'")
