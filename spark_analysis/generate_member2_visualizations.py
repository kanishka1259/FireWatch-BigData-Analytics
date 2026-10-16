"""
High-Resolution Publication Visualizations Generator (300 DPI)
==============================================================
Generates publication-grade scientific telemetry charts using Matplotlib and Seaborn.

Simple Viva Explanation for Evaluators:
- Ingests the aggregated CSV and JSON cubes produced by the PySpark engine.
- Renders 5 high-resolution (300 DPI) telemetry charts:
    1. Daily Hotspot Detection Trend & 7-Day Rolling Window
    2. Monthly Seasonality & Radiative Energy Partitioning
    3. Diurnal 24-Hour UTC Convective Fire Cycle
    4. Day vs. Night Operational Volume & Radiative Energy
    5. Top 10 Peak Outbreak Days (Count vs. FRP Disparity)
- Outputs are saved in 'outputs/charts/' and served live to the Flask web dashboard.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Ensure UTF-8 output on Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

DATA_DIR = os.path.abspath("outputs/data")
CHART_DIR = os.path.abspath("outputs/charts")
os.makedirs(CHART_DIR, exist_ok=True)

# Styling configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8

def plot_daily_trend():
    """Chart 1: Daily Hotspot Detection Trend & 7-Day Rolling Average"""
    csv_file = os.path.join(DATA_DIR, "member2_daily_stats.csv")
    if not os.path.exists(csv_file):
        print(f"Skipping daily trend: {csv_file} not found.")
        return

    df = pd.read_csv(csv_file)
    df['date'] = pd.to_datetime(df['acq_date'])
    df = df.sort_values('date')

    fig, ax1 = plt.subplots(figsize=(14, 6), dpi=300)
    
    # Raw daily bars
    color_bar = '#f97316'
    ax1.bar(df['date'], df['detection_count'], width=1.0, color=color_bar, alpha=0.35, label='Daily Detections (Raw)')
    
    # 7-day rolling trendline
    color_line = '#dc2626'
    if 'rolling_7d_count' in df.columns:
        ax1.plot(df['date'], df['rolling_7d_count'], color=color_line, linewidth=2.5, label='7-Day Rolling Mean')
    
    ax1.set_xlabel('Observation Date (UTC)', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Satellite Hotspot Detections / Day', color=color_line, fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_line)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*1e-3:.0f}K' if x < 1e6 else f'{x*1e-6:.1f}M'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    # Secondary axis: Daily Total FRP (MW)
    ax2 = ax1.twinx()
    color_frp = '#0284c7'
    if 'total_frp' in df.columns:
        ax2.plot(df['date'], df['total_frp'], color=color_frp, linewidth=1.5, linestyle='--', alpha=0.7, label='Daily Total FRP (MW)')
        ax2.set_ylabel('Total Fire Radiative Power (MW)', color=color_frp, fontsize=12, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=color_frp)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*1e-3:.0f}K MW' if x < 1e6 else f'{x*1e-6:.1f}M MW'))
        ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

    date_min = df['acq_date'].min()
    date_max = df['acq_date'].max()
    plt.title(f'Temporal Wildfire Dynamics: Daily Hotspot Counts & Radiative Power\nNASA FIRMS VIIRS ({date_min} to {date_max})', 
              fontsize=14, fontweight='bold', pad=15)
    
    fig.tight_layout()
    out_path = os.path.join(CHART_DIR, "member2_daily_trend.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  [1/5] Generated {out_path}")

def plot_monthly_trend():
    """Chart 2: Monthly Hotspot Detection Count & Mean FRP Comparison"""
    csv_file = os.path.join(DATA_DIR, "member2_monthly_stats.csv")
    if not os.path.exists(csv_file):
        print(f"Skipping monthly trend: {csv_file} not found.")
        return

    df = pd.read_csv(csv_file)
    df = df.sort_values('year_month')

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
    color_bar = '#ea580c'
    color_line = '#2563eb'

    bars = ax1.bar(df['year_month'], df['detection_count'], color=color_bar, alpha=0.8, width=0.55, edgecolor='#9a3412', label='Detections')
    ax1.set_xlabel('Year-Month (Observation Period)', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Total Hotspot Count', color=color_bar, fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_bar)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*1e-3:.0f}K' if x < 1e6 else f'{x*1e-6:.2f}M'))
    
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f'{h/1e6:.2f}M' if h >= 1e6 else f'{h/1e3:.0f}K',
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2 = ax1.twinx()
    ax2.plot(df['year_month'], df['avg_frp'], color=color_line, marker='o', linewidth=2.5, markersize=8, label='Avg FRP')
    ax2.set_ylabel('Mean Fire Radiative Power (MW)', color=color_line, fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_line)
    ax2.grid(False)

    for i, txt in enumerate(df['avg_frp']):
        ax2.annotate(f'{txt:.1f} MW', (df['year_month'].iloc[i], txt),
                     xytext=(0, 7), textcoords="offset points", ha='center', fontsize=9, color=color_line, fontweight='bold')

    plt.title('Monthly Wildfire Frequency & Average Radiative Energy (NASA FIRMS)', fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    out_path = os.path.join(CHART_DIR, "member2_monthly_trend.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  [2/5] Generated {out_path}")

def plot_hourly_distribution():
    """Chart 3: Hourly Diurnal Detection Distribution (0-23 UTC)"""
    csv_file = os.path.join(DATA_DIR, "member2_hourly_stats.csv")
    if not os.path.exists(csv_file):
        print(f"Skipping hourly distribution: {csv_file} not found.")
        return

    df = pd.read_csv(csv_file)
    df = df.sort_values('hour')

    fig, ax1 = plt.subplots(figsize=(12, 5), dpi=300)
    color_bar = '#8b5cf6'
    color_line = '#d97706'

    bars = ax1.bar(df['hour'], df['detection_count'], color=color_bar, alpha=0.75, width=0.7, edgecolor='#6d28d9')
    ax1.set_xlabel('Acquisition Hour (UTC)', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Total Detections Across Observation Period', color=color_bar, fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_bar)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*1e-3:.0f}K' if x < 1e6 else f'{x*1e-6:.2f}M'))
    ax1.set_xticks(range(24))

    ax2 = ax1.twinx()
    ax2.plot(df['hour'], df['avg_frp'], color=color_line, marker='s', linewidth=2.2, markersize=6)
    ax2.set_ylabel('Average Fire Radiative Power (MW)', color=color_line, fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_line)
    ax2.grid(False)

    plt.title('Diurnal Wildfire Activity Cycle by Acquisition Hour (UTC)\n(Reflects Suomi-NPP Polar Orbiting Overpass Windows)', 
              fontsize=13, fontweight='bold', pad=15)
    fig.tight_layout()
    out_path = os.path.join(CHART_DIR, "member2_hourly_distribution.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  [3/5] Generated {out_path}")

def plot_day_vs_night_temporal():
    """Chart 4: Diurnal vs Nocturnal Proportion & Energy Across Months"""
    csv_file = os.path.join(DATA_DIR, "member2_daynight_stats.csv")
    if not os.path.exists(csv_file):
        print(f"Skipping day/night temporal: {csv_file} not found.")
        return

    df = pd.read_csv(csv_file)
    pivot_cnt = df.pivot(index='year_month', columns='daynight', values='detection_count').fillna(0)
    pivot_frp = df.pivot(index='year_month', columns='daynight', values='avg_frp').fillna(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Subplot 1: Stacked detection percentages
    pct_day = pivot_cnt['D'] / (pivot_cnt['D'] + pivot_cnt['N']) * 100
    pct_night = pivot_cnt['N'] / (pivot_cnt['D'] + pivot_cnt['N']) * 100
    months = pivot_cnt.index

    ax1.bar(months, pct_day, label='Daytime (D)', color='#f59e0b', alpha=0.85, width=0.55)
    ax1.bar(months, pct_night, bottom=pct_day, label='Nighttime (N)', color='#3b82f6', alpha=0.85, width=0.55)
    ax1.set_ylabel('Proportion of Detections (%)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Observation Period', fontsize=11, fontweight='bold')
    ax1.set_title('Diurnal vs. Nocturnal Ratio by Month', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', frameon=True)
    ax1.set_ylim(0, 105)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')

    # Subplot 2: Day vs Night Average FRP Comparison
    x = np.arange(len(months))
    w = 0.35
    ax2.bar(x - w/2, pivot_frp['D'], width=w, label='Daytime Avg FRP', color='#f59e0b', alpha=0.85)
    ax2.bar(x + w/2, pivot_frp['N'], width=w, label='Nighttime Avg FRP', color='#3b82f6', alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(months, rotation=30, ha='right')
    ax2.set_ylabel('Mean Fire Radiative Power (MW)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Observation Period', fontsize=11, fontweight='bold')
    ax2.set_title('Mean FRP: Daytime vs. Nighttime', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True)

    fig.tight_layout()
    out_path = os.path.join(CHART_DIR, "member2_day_vs_night.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  [4/5] Generated {out_path}")

def plot_top_dates_comparison():
    """Chart 5: Top Dates Comparison (Hotspot Count vs. Total FRP)"""
    count_file = os.path.join(DATA_DIR, "member2_top_dates_count.csv")
    frp_file = os.path.join(DATA_DIR, "member2_top_dates_frp.csv")
    if not os.path.exists(count_file) or not os.path.exists(frp_file):
        print("Skipping top dates comparison: files not found.")
        return

    df_count = pd.read_csv(count_file).head(8)
    df_frp = pd.read_csv(frp_file).head(8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Top by Count
    bars1 = ax1.barh(df_count['acq_date'], df_count['detection_count'], color='#ef4444', alpha=0.85, height=0.6)
    ax1.set_xlabel('Hotspot Detections', fontsize=11, fontweight='bold')
    ax1.set_title('Top Dates by Detection Count\n(High Frequency / Spread)', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    for bar in bars1:
        w = bar.get_width()
        ax1.annotate(f'{w:,}', xy=(w, bar.get_y() + bar.get_height() / 2),
                     xytext=(5, 0), textcoords="offset points", va='center', fontsize=9, fontweight='bold')

    # Top by Total FRP
    bars2 = ax2.barh(df_frp['acq_date'], df_frp['total_frp'], color='#0284c7', alpha=0.85, height=0.6)
    ax2.set_xlabel('Total Fire Radiative Power (MW)', fontsize=11, fontweight='bold')
    ax2.set_title('Top Dates by Total Thermal Energy\n(Intense Megafires / Crown Combustion)', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*1e-3:.0f}K MW' if x < 1e6 else f'{x*1e-6:.2f}M MW'))
    for bar in bars2:
        w = bar.get_width()
        ax2.annotate(f'{w/1e3:.1f}K MW', xy=(w, bar.get_y() + bar.get_height() / 2),
                     xytext=(5, 0), textcoords="offset points", va='center', fontsize=9, fontweight='bold')

    fig.tight_layout()
    out_path = os.path.join(CHART_DIR, "member2_top_dates_comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  [5/5] Generated {out_path}")

def generate_all_visualizations():
    print("=" * 60)
    print("🎨 Generating Member 2 Temporal Visualizations")
    print("=" * 60)
    plot_daily_trend()
    plot_monthly_trend()
    plot_hourly_distribution()
    plot_day_vs_night_temporal()
    plot_top_dates_comparison()
    print("=" * 60)
    print("✅ All Member 2 Visualizations Successfully Generated!")
    print("=" * 60)

if __name__ == "__main__":
    generate_all_visualizations()
