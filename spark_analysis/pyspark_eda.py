"""
PySpark Preprocessing and Exploratory Data Analysis (EDA)
NASA FIRMS Global Fire Hotspot Dataset (~9.5 Million Records)
"""

import os
import sys
import time
import json

# Ensure UTF-8 output on Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_date, month, year, floor, when, count, avg, sum as spark_sum,
    max as spark_max, min as spark_min, stddev, corr, round as spark_round
)

DATA_PATH = "fire_archive_SV-C2_817450.csv"
OUTPUT_DIR = "outputs/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_eda():
    start_time = time.time()
    print("=" * 70)
    print("🔥 NASA FIRMS Big Data Analysis - PySpark EDA Engine")
    print("=" * 70)

    # 1. Initialize SparkSession
    print("\n[Step 1/5] Initializing PySpark Session...")
    spark = SparkSession.builder \
        .appName("NASAFIRMS_Fire_Analysis") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    print(f"  PySpark Version: {spark.version}")
    print(f"  Spark Master: {spark.sparkContext.master}")

    # 2. Read Dataset
    print(f"\n[Step 2/5] Ingesting CSV dataset: {DATA_PATH}...")
    load_start = time.time()
    df_raw = spark.read.option("header", "true").option("inferSchema", "true").csv(DATA_PATH)
    total_records = df_raw.count()
    load_duration = time.time() - load_start
    print(f"  Total Raw Records Loaded: {total_records:,} rows in {load_duration:.2f}s")
    print(f"  Schema attributes: {df_raw.columns}")

    # 3. Preprocessing & Feature Engineering
    print("\n[Step 3/5] Cleaning, Casting & Feature Engineering...")
    df_clean = df_raw.select(
        col("latitude").cast("double").alias("latitude"),
        col("longitude").cast("double").alias("longitude"),
        col("brightness").cast("double").alias("brightness"),
        col("scan").cast("double").alias("scan"),
        col("track").cast("double").alias("track"),
        col("acq_date"),
        col("acq_time").cast("int").alias("acq_time"),
        col("satellite"),
        col("confidence"),
        col("version"),
        col("bright_t31").cast("double").alias("bright_t31"),
        col("frp").cast("double").alias("frp"),
        col("daynight"),
        col("type").cast("int").alias("type")
    ).dropna(subset=["latitude", "longitude", "brightness", "bright_t31", "frp"])

    # Engineered columns
    df_feat = df_clean.withColumn("brightness_diff", spark_round(col("brightness") - col("bright_t31"), 2)) \
                      .withColumn("hour", floor(col("acq_time") / 100)) \
                      .withColumn("month", month(to_date(col("acq_date")))) \
                      .withColumn("is_day", when(col("daynight") == "D", 1).otherwise(0)) \
                      .withColumn("lat_bin", floor(col("latitude") / 5.0) * 5.0) \
                      .withColumn("lon_bin", floor(col("longitude") / 5.0) * 5.0) \
                      .withColumn("severity_tier", 
                          when(col("frp") >= 50.0, "Extreme")
                          .when(col("frp") >= 20.0, "High")
                          .when(col("frp") >= 5.0, "Moderate")
                          .otherwise("Low")
                      )

    clean_count = df_feat.count()
    print(f"  Cleaned Valid Hotspots: {clean_count:,} ({((clean_count/total_records)*100):.2f}% retention)")

    # 4. Answering the 5 Analytical Frame Questions
    print("\n[Step 4/5] Computing Analytical Solutions to the 5 Core Questions...")

    # Q1: Spatiotemporal Monthly & Hourly Fire Dynamics
    print("  -> Answering Q1: Spatiotemporal Dynamics (Monthly & Hourly Aggregates)...")
    monthly_stats = df_feat.groupBy("month").agg(
        count("*").alias("fire_count"),
        spark_round(avg("frp"), 2).alias("avg_frp"),
        spark_round(spark_sum("frp"), 2).alias("total_frp"),
        spark_round(avg("brightness"), 2).alias("avg_brightness")
    ).orderBy("month").toPandas()

    hourly_stats = df_feat.groupBy("hour").agg(
        count("*").alias("fire_count"),
        spark_round(avg("frp"), 2).alias("avg_frp")
    ).orderBy("hour").toPandas()

    # Q2: Diurnal vs Nocturnal Fire Energy Dynamics
    print("  -> Answering Q2: Day vs Night Thermal Energetics...")
    daynight_stats = df_feat.groupBy("daynight").agg(
        count("*").alias("count"),
        spark_round(avg("frp"), 2).alias("avg_frp"),
        spark_round(spark_max("frp"), 2).alias("max_frp"),
        spark_round(avg("brightness"), 2).alias("avg_brightness"),
        spark_round(avg("bright_t31"), 2).alias("avg_bright_t31"),
        spark_round(avg("brightness_diff"), 2).alias("avg_thermal_contrast")
    ).toPandas()

    # Q3: Fire Type & Confidence Distribution
    print("  -> Answering Q3: Fire Classification (Wildfire vs Industrial) & Confidence...")
    type_names = {0: "Vegetation/Wildfire", 1: "Active Volcano", 2: "Static Land/Industrial", 3: "Offshore Flare"}
    type_stats = df_feat.groupBy("type", "confidence").agg(
        count("*").alias("count"),
        spark_round(avg("frp"), 2).alias("avg_frp")
    ).orderBy("type", "confidence").toPandas()
    type_stats["type_name"] = type_stats["type"].map(lambda t: type_names.get(t, "Unknown"))

    # Q4: Correlation Analysis of Thermal Attributes
    print("  -> Answering Q4: Physical Correlation Matrix...")
    corr_b_frp = df_feat.stat.corr("brightness", "frp")
    corr_t31_frp = df_feat.stat.corr("bright_t31", "frp")
    corr_diff_frp = df_feat.stat.corr("brightness_diff", "frp")
    corr_scan_frp = df_feat.stat.corr("scan", "frp")
    corr_matrix = {
        "brightness_vs_frp": round(corr_b_frp, 4),
        "bright_t31_vs_frp": round(corr_t31_frp, 4),
        "brightness_diff_vs_frp": round(corr_diff_frp, 4),
        "scan_vs_frp": round(corr_scan_frp, 4)
    }

    # Q5: Geographic Clustering of Extreme Fire Events (FRP >= 50 MW)
    print("  -> Answering Q5: Geographic Clustering of Extreme Hotspots...")
    geo_clusters = df_feat.filter(col("frp") >= 50.0).groupBy("lat_bin", "lon_bin").agg(
        count("*").alias("extreme_fire_count"),
        spark_round(avg("frp"), 2).alias("mean_extreme_frp"),
        spark_round(spark_max("frp"), 2).alias("peak_frp")
    ).orderBy(col("extreme_fire_count").desc()).limit(20).toPandas()

    # Severity distribution
    severity_stats = df_feat.groupBy("severity_tier").agg(
        count("*").alias("count"),
        spark_round(avg("frp"), 2).alias("avg_frp"),
        spark_round(avg("brightness"), 2).alias("avg_brightness")
    ).toPandas()

    # Overall dataset metrics
    summary_metrics = {
        "total_records": int(total_records),
        "valid_records": int(clean_count),
        "mean_frp": float(df_feat.select(avg("frp")).collect()[0][0]),
        "max_frp": float(df_feat.select(spark_max("frp")).collect()[0][0]),
        "mean_brightness": float(df_feat.select(avg("brightness")).collect()[0][0]),
        "mean_bright_t31": float(df_feat.select(avg("bright_t31")).collect()[0][0]),
        "total_extreme_fires": int(df_feat.filter(col("frp") >= 50.0).count()),
        "processing_time_seconds": round(time.time() - start_time, 2)
    }

    # 5. Export results for Visualizations, NoSQL & Web Dashboard
    print("\n[Step 5/5] Exporting Processed Aggregates to JSON & CSV...")
    monthly_stats.to_csv(os.path.join(OUTPUT_DIR, "monthly_stats.csv"), index=False)
    hourly_stats.to_csv(os.path.join(OUTPUT_DIR, "hourly_stats.csv"), index=False)
    daynight_stats.to_csv(os.path.join(OUTPUT_DIR, "daynight_stats.csv"), index=False)
    type_stats.to_csv(os.path.join(OUTPUT_DIR, "type_stats.csv"), index=False)
    geo_clusters.to_csv(os.path.join(OUTPUT_DIR, "geo_clusters.csv"), index=False)
    severity_stats.to_csv(os.path.join(OUTPUT_DIR, "severity_stats.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "summary_metrics.json"), "w") as f:
        json.dump(summary_metrics, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "correlation_matrix.json"), "w") as f:
        json.dump(corr_matrix, f, indent=2)

    # Export a stratified sample of 5,000 points for Leaflet map & NoSQL document store
    print("  -> Exporting stratified geo-hotspot sample for Map & NoSQL...")
    sample_df = df_feat.sample(fraction=0.001, seed=42).limit(5000).toPandas()
    sample_df.to_json(os.path.join(OUTPUT_DIR, "hotspots_sample.json"), orient="records", indent=2)

    spark.stop()
    print("\n" + "=" * 70)
    print(f"✅ PySpark Preprocessing & EDA Completed in {time.time() - start_time:.2f}s!")
    print(f"📁 Processed files saved in: {OUTPUT_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    run_eda()
