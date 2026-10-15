"""
Member 2 - Dataset Validation and Temporal Profiler
Inspects fire_archive_SV-C2_817450.csv to extract exact dataset metrics,
schema, temporal coverage, nulls, and date/time validity.
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np

DATA_PATH = os.path.abspath("fire_archive_SV-C2_817450.csv")
if not os.path.exists(DATA_PATH):
    # Try looking in parent directory
    DATA_PATH = os.path.abspath("../fire_archive_SV-C2_817450.csv")

print(f"Inspecting dataset at: {DATA_PATH}")
file_size_bytes = os.path.getsize(DATA_PATH)
file_size_mb = file_size_bytes / (1024 * 1024)
print(f"File Size: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")

# Read chunks to compute exact counts and avoid excessive RAM usage
chunk_size = 500000
total_rows = 0
null_counts = {}
date_counts = {}
hour_counts = {h: 0 for h in range(24)}
invalid_times = 0
invalid_dates = 0
daynight_counts = {}
frp_sum = 0.0
frp_count = 0
frp_min = float('inf')
frp_max = float('-inf')
type_counts = {}

t0 = time.time()
print("\nProcessing full dataset in chunks of 500,000 rows...")

for chunk_idx, chunk in enumerate(pd.read_csv(DATA_PATH, chunksize=chunk_size, low_memory=False)):
    total_rows += len(chunk)
    
    # Null counts
    for col in chunk.columns:
        null_counts[col] = null_counts.get(col, 0) + int(chunk[col].isna().sum())
    
    # Dates
    dates = chunk['acq_date'].dropna().astype(str)
    for d, c in dates.value_counts().items():
        date_counts[d] = date_counts.get(d, 0) + c
        
    # Times (acq_time)
    times = chunk['acq_time'].dropna()
    for t in times:
        try:
            val = int(t)
            hour = val // 100
            minute = val % 100
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                hour_counts[hour] += 1
            else:
                invalid_times += 1
        except Exception:
            invalid_times += 1
            
    # Day/Night
    for dn, c in chunk['daynight'].dropna().value_counts().items():
        daynight_counts[dn] = daynight_counts.get(dn, 0) + c
        
    # Fire Type
    if 'type' in chunk.columns:
        for tp, c in chunk['type'].dropna().value_counts().items():
            type_counts[int(tp)] = type_counts.get(int(tp), 0) + c
            
    # FRP
    valid_frp = chunk['frp'].dropna()
    if len(valid_frp) > 0:
        frp_sum += float(valid_frp.sum())
        frp_count += len(valid_frp)
        frp_min = min(frp_min, float(valid_frp.min()))
        frp_max = max(frp_max, float(valid_frp.max()))
        
    print(f"  Processed chunk {chunk_idx + 1}: {total_rows:,} rows so far... ({time.time() - t0:.1f}s)")

elapsed = time.time() - t0
print(f"\nCompleted in {elapsed:.2f}s!")
print(f"Total Rows: {total_rows:,}")
print(f"Null counts per column:\n{json.dumps(null_counts, indent=2)}")

all_dates = sorted(date_counts.keys())
print(f"Date Range: {all_dates[0]} to {all_dates[-1]}")
print(f"Total Unique Dates: {len(all_dates)}")

# Check for date gaps
start_date = pd.to_datetime(all_dates[0])
end_date = pd.to_datetime(all_dates[-1])
expected_days = (end_date - start_date).days + 1
missing_dates = []
date_set = set(all_dates)
for d in pd.date_range(start_date, end_date):
    d_str = d.strftime("%Y-%m-%d")
    if d_str not in date_set:
        missing_dates.append(d_str)

print(f"Expected Calendar Days: {expected_days}")
print(f"Observed Calendar Days: {len(all_dates)}")
print(f"Missing Calendar Days Count: {len(missing_dates)}")
if missing_dates:
    print(f"First 10 Missing Dates: {missing_dates[:10]}")

# Top 10 Busiest Dates
top_dates = sorted(date_counts.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 Busiest Dates:")
for d, c in top_dates:
    print(f"  {d}: {c:,} hotspots")

# Monthly Aggregation
monthly_counts = {}
for d, c in date_counts.items():
    ym = d[:7]
    monthly_counts[ym] = monthly_counts.get(ym, 0) + c

sorted_months = sorted(monthly_counts.items(), key=lambda x: x[0])
print("\nMonthly Distribution:")
for ym, c in sorted_months:
    print(f"  {ym}: {c:,} hotspots")

# Hourly Distribution
print("\nHourly UTC Distribution:")
for h in range(24):
    print(f"  Hour {h:02d}: {hour_counts[h]:,} hotspots")

print(f"\nInvalid Times Count: {invalid_times}")
print(f"Day/Night Counts: {daynight_counts}")
print(f"Type Counts: {type_counts}")
print(f"FRP Stats: Mean={frp_sum/frp_count:.2f} MW, Min={frp_min:.2f} MW, Max={frp_max:.2f} MW, Count={frp_count:,}")

# Save profiling report
validation_summary = {
    "dataset_path": DATA_PATH,
    "file_size_mb": round(file_size_mb, 2),
    "total_rows": total_rows,
    "null_counts": null_counts,
    "min_date": all_dates[0],
    "max_date": all_dates[-1],
    "total_unique_dates": len(all_dates),
    "expected_calendar_days": expected_days,
    "missing_calendar_days": missing_dates,
    "top_10_dates": top_dates,
    "monthly_counts": monthly_counts,
    "hourly_counts": hour_counts,
    "invalid_times": invalid_times,
    "daynight_counts": daynight_counts,
    "type_counts": type_counts,
    "frp_mean": round(frp_sum / frp_count, 2) if frp_count > 0 else 0,
    "frp_min": frp_min,
    "frp_max": frp_max,
    "profiling_duration_sec": round(elapsed, 2)
}

os.makedirs("outputs/data", exist_ok=True)
with open("outputs/data/dataset_validation_member2.json", "w") as f:
    json.dump(validation_summary, f, indent=2)

print("\nSaved validation summary to outputs/data/dataset_validation_member2.json")
