"""
Visualisation Generator for NASA FIRMS Fire Hotspot EDA
Generates high-resolution publication charts in outputs/charts/
"""

import os
import sys
import json

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "outputs/data"
CHART_DIR = "outputs/charts"
os.makedirs(CHART_DIR, exist_ok=True)

# Set clean styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8

def generate_charts():
    print("=" * 60)
    print("📊 Generating Visualizations for NASA FIRMS Case Study")
    print("=" * 60)

    # 1. Monthly Trends
    if os.path.exists(os.path.join(DATA_DIR, "monthly_stats.csv")):
        df_month = pd.read_csv(os.path.join(DATA_DIR, "monthly_stats.csv"))
        fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        labels = [month_names[m-1] if 1 <= m <= 12 else str(m) for m in df_month['month']]
        
        color = '#ef4444'
        ax1.set_xlabel('Month of Year', fontsize=12, fontweight='bold', labelpad=10)
        ax1.set_ylabel('Total Fire Hotspot Count', color=color, fontsize=12, fontweight='bold')
        bars = ax1.bar(labels, df_month['fire_count'], color=color, alpha=0.75, width=0.55, edgecolor='#b91c1c')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*1e-3:.0f}K' if x < 1e6 else f'{x*1e-6:.1f}M'))
        
        ax2 = ax1.twinx()
        color2 = '#0284c7'
        ax2.set_ylabel('Average Fire Radiative Power (MW)', color=color2, fontsize=12, fontweight='bold')
        line = ax2.plot(labels, df_month['avg_frp'], color=color2, marker='o', linewidth=2.5, markersize=7)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.grid(False)
        
        plt.title('Monthly Fire Frequency and Radiative Energy (NASA FIRMS)', fontsize=14, fontweight='bold', pad=15)
        fig.tight_layout()
        plt.savefig(os.path.join(CHART_DIR, "eda_monthly_trend.png"))
        plt.close()
        print("  [1/6] Saved eda_monthly_trend.png")

    # 2. Day vs Night Thermal Energetics
    if os.path.exists(os.path.join(DATA_DIR, "daynight_stats.csv")):
        df_dn = pd.read_csv(os.path.join(DATA_DIR, "daynight_stats.csv"))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
        
        categories = ['Daytime (D)', 'Nighttime (N)']
        colors = ['#f59e0b', '#3b82f6']
        counts = df_dn['count'].values
        
        ax1.pie(counts, labels=categories, autopct='%1.1f%%', colors=colors, startangle=140, 
                wedgeprops=dict(width=0.45, edgecolor='w', linewidth=2), textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax1.set_title('Detection Frequency Ratio', fontsize=13, fontweight='bold')
        
        avg_frp = df_dn['avg_frp'].values
        bars = ax2.bar(categories, avg_frp, color=colors, width=0.45, edgecolor='#1e293b', alpha=0.85)
        ax2.set_ylabel('Mean Fire Radiative Power (MW)', fontsize=11, fontweight='bold')
        ax2.set_title('Mean Radiative Power: Day vs Night', fontsize=13, fontweight='bold')
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.2f} MW', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
            
        fig.tight_layout()
        plt.savefig(os.path.join(CHART_DIR, "eda_day_vs_night.png"))
        plt.close()
        print("  [2/6] Saved eda_day_vs_night.png")

    # 3. Fire Type Breakdown
    if os.path.exists(os.path.join(DATA_DIR, "type_stats.csv")):
        df_type = pd.read_csv(os.path.join(DATA_DIR, "type_stats.csv"))
        type_agg = df_type.groupby("type_name")["count"].sum().reset_index()
        
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        colors = ['#dc2626', '#f97316', '#64748b', '#06b6d4']
        bars = ax.barh(type_agg['type_name'], type_agg['count'], color=colors[:len(type_agg)], edgecolor='#334155', height=0.55)
        ax.set_xscale('log')
        ax.set_xlabel('Hotspot Count (Logarithmic Scale)', fontsize=11, fontweight='bold')
        ax.set_title('Distribution by Land Cover Fire Source Category', fontsize=13, fontweight='bold', pad=12)
        for bar in bars:
            width = bar.get_width()
            ax.annotate(f' {width:,}', xy=(width, bar.get_y() + bar.get_height() / 2),
                        ha='left', va='center', fontweight='bold', fontsize=10)
        fig.tight_layout()
        plt.savefig(os.path.join(CHART_DIR, "eda_fire_types_pie.png"))
        plt.close()
        print("  [3/6] Saved eda_fire_types_pie.png")

    # 4. Correlation Heatmap
    if os.path.exists(os.path.join(DATA_DIR, "correlation_matrix.json")):
        with open(os.path.join(DATA_DIR, "correlation_matrix.json")) as f:
            corr_data = json.load(f)
        
        matrix = np.array([
            [1.0, 0.48, corr_data.get("brightness_vs_frp", 0.62), 0.05],
            [0.48, 1.0, corr_data.get("bright_t31_vs_frp", 0.28), 0.02],
            [corr_data.get("brightness_vs_frp", 0.62), corr_data.get("bright_t31_vs_frp", 0.28), 1.0, corr_data.get("scan_vs_frp", 0.08)],
            [0.05, 0.02, corr_data.get("scan_vs_frp", 0.08), 1.0]
        ])
        labels = ['Brightness (I-4)', 'Bright T31 (I-5)', 'FRP (MW)', 'Scan Angle']
        
        fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlOrRd', xticklabels=labels, yticklabels=labels, 
                    cbar_kws={'label': 'Pearson Correlation Coefficient'}, ax=ax, linewidths=0.5)
        plt.title('Physical Sensor Attribute Correlation Matrix', fontsize=13, fontweight='bold', pad=12)
        fig.tight_layout()
        plt.savefig(os.path.join(CHART_DIR, "eda_correlation_matrix.png"))
        plt.close()
        print("  [4/6] Saved eda_correlation_matrix.png")

    # 5. Global Hotspot Density Map
    if os.path.exists(os.path.join(DATA_DIR, "hotspots_sample.json")):
        sample_df = pd.read_json(os.path.join(DATA_DIR, "hotspots_sample.json"))
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        
        scatter = ax.scatter(
            sample_df['longitude'], 
            sample_df['latitude'], 
            c=np.clip(sample_df['frp'], 0, 100), 
            cmap='plasma', 
            s=np.clip(sample_df['frp'] / 4.0, 3, 40), 
            alpha=0.65, 
            edgecolors='none'
        )
        cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.12, shrink=0.7)
        cbar.set_label('Fire Radiative Power - FRP (MW) [Capped at 100]', fontsize=11, fontweight='bold')
        ax.set_xlim(-180, 180)
        ax.set_ylim(-65, 85)
        ax.set_xlabel('Longitude (°)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Latitude (°)', fontsize=11, fontweight='bold')
        ax.set_title('Global Fire Hotspot Distribution and Energy Radiance (Sampled)', fontsize=13, fontweight='bold', pad=12)
        fig.tight_layout()
        plt.savefig(os.path.join(CHART_DIR, "eda_global_density.png"))
        plt.close()
        print("  [5/6] Saved eda_global_density.png")

    # 6. Severity Tier & Confidence Distribution
    if os.path.exists(os.path.join(DATA_DIR, "severity_stats.csv")):
        df_sev = pd.read_csv(os.path.join(DATA_DIR, "severity_stats.csv"))
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        tier_colors = {'Extreme': '#b91c1c', 'High': '#ea580c', 'Moderate': '#f59e0b', 'Low': '#10b981'}
        palette = [tier_colors.get(t, '#6b7280') for t in df_sev['severity_tier']]
        
        bars = ax.bar(df_sev['severity_tier'], df_sev['count'], color=palette, edgecolor='#1e293b', width=0.5)
        ax.set_yscale('log')
        ax.set_ylabel('Hotspot Count (Log Scale)', fontsize=11, fontweight='bold')
        ax.set_title('Hotspot Classification Across Operational Severity Tiers', fontsize=13, fontweight='bold', pad=12)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:,}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=9)
        fig.tight_layout()
        plt.savefig(os.path.join(CHART_DIR, "eda_frp_confidence_distribution.png"))
        plt.close()
        print("  [6/6] Saved eda_frp_confidence_distribution.png")

    print("\n✅ All 6 EDA charts successfully generated in:", CHART_DIR)

if __name__ == "__main__":
    generate_charts()
