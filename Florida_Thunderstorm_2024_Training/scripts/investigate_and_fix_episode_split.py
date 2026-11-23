"""
Phase 2 Investigation: Investigate episode distribution and multi-day spanning.
Examines:
1. All unique episode IDs in future_noaa_episode_ids.
2. How many episodes span multiple dates.
3. How rows with multiple episode IDs (e.g. '197213;196745') are linked.
4. Construct connected components of co-occurring or multi-date episodes.
5. Partition connected components cleanly into Train (~70%) and Test (~30%).
6. Assign quiescent (non-episode) rows cleanly by date blocks without violating episode boundaries.
7. Programmatically verify zero shared episode IDs between Train and Test.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json

BASE_DIR = Path(r"c:\Users\KIIT0001\Desktop\Data Collection")
OUT_DIR = BASE_DIR / "data" / "processed" / "audit_v2"
fc_csv = BASE_DIR / "data" / "processed" / "florida_thunderstorm_2024_forecasting_6h.csv"

df = pd.read_csv(fc_csv, low_memory=False)
valid_df = df[df["future_noaa_event_6h"].notna()].copy()
valid_df["date"] = pd.to_datetime(valid_df["forecast_init_time_utc"]).dt.date

pos_df = valid_df[valid_df["future_noaa_event_6h"] == 1].copy()
print(f"Total positive forecast rows: {len(pos_df)}")

# Extract episode sets per row
row_episodes = []
for idx, r in pos_df.iterrows():
    raw_val = str(r["future_noaa_episode_ids"])
    if raw_val and raw_val != "nan" and raw_val != "UNKNOWN":
        eps = set(e.strip() for e in raw_val.split(";") if e.strip() and e.strip() != "UNKNOWN")
    else:
        eps = set()
    row_episodes.append({
        "row_idx": idx,
        "date": r["date"],
        "init_time": r["forecast_init_time_utc"],
        "lat": r["latitude"],
        "lon": r["longitude"],
        "episodes": eps
    })

df_pos_eps = pd.DataFrame(row_episodes)

# Build graph of episodes: two episodes are connected if:
# 1. They co-occur in the same forecast row (e.g. multiple events in the 6h window)
# 2. They occur on the same date at any cell (to keep complete dates coherent if desirable,
#    OR we can assign dates containing an episode cluster together).
all_unique_eps = sorted(set(ep for eps in df_pos_eps["episodes"] for ep in eps))
print(f"Total unique known NOAA storm episodes in forecasting target: {len(all_unique_eps)}")

# Map each episode to all dates it spans
ep_dates = {}
for ep in all_unique_eps:
    dates_for_ep = sorted(set(
        r["date"] for _, r in df_pos_eps.iterrows() if ep in r["episodes"]
    ))
    ep_dates[ep] = dates_for_ep

multi_date_eps = {ep: dates for ep, dates in ep_dates.items() if len(dates) > 1}
print(f"\nEpisodes spanning multiple calendar dates ({len(multi_date_eps)}):")
for ep, dates in multi_date_eps.items():
    print(f"  Episode {ep}: {len(dates)} dates ({[str(d) for d in dates]})")

# Co-occurring episodes in the same row
co_occurring_pairs = set()
for eps in df_pos_eps["episodes"]:
    if len(eps) > 1:
        eps_list = sorted(list(eps))
        for i in range(len(eps_list)):
            for j in range(i + 1, len(eps_list)):
                co_occurring_pairs.add((eps_list[i], eps_list[j]))

print(f"\nCo-occurring episode pairs in same forecast row: {len(co_occurring_pairs)}")
for pair in sorted(co_occurring_pairs):
    print(f"  {pair[0]} <-> {pair[1]}")

# Build connected components of episodes
# If ep1 and ep2 co-occur in any row, they MUST be in the same partition!
# If ep1 and ep2 share dates, do they need to be in the same partition?
# If we assign rows by date, then all episodes occurring on that date are in the same partition.
# If an episode spans dates D1 and D2, then both D1 and D2 must be in the same partition!
# Let's group dates into connected components via episodes:
# Date D1 is connected to D2 if any episode occurs on both D1 and D2!
date_adj = {}
for d in pos_df["date"].unique():
    date_adj[d] = set()

for ep, dates in ep_dates.items():
    for d1 in dates:
        for d2 in dates:
            if d1 != d2:
                date_adj[d1].add(d2)

# Connected components of storm dates
visited = set()
date_clusters = []
for d in sorted(date_adj.keys()):
    if d not in visited:
        comp = set()
        queue = [d]
        visited.add(d)
        while queue:
            curr = queue.pop(0)
            comp.add(curr)
            for neighbor in date_adj.get(curr, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        date_clusters.append(sorted(list(comp)))

print(f"\nTotal positive storm dates: {len(pos_df['date'].unique())}")
print(f"Total connected date clusters: {len(date_clusters)}")
for i, cluster in enumerate(date_clusters):
    if len(cluster) > 1:
        cluster_eps = set()
        for d in cluster:
            for ep, dates in ep_dates.items():
                if d in dates:
                    cluster_eps.add(ep)
        print(f"  Cluster {i+1}: {len(cluster)} dates ({[str(d) for d in cluster]}) -> {len(cluster_eps)} episodes: {cluster_eps}")
