"""
PySpark Transformations and Partitioning Engine
================================================
Comprehensive Big Data Analytics Demonstration:
1. Partitioning Mechanics:
   - Inspecting Partitions: getNumPartitions()
   - Repartitioning (Scaling UP with Full Shuffle): df.repartition(n)
   - Coalescing (Scaling DOWN without Full Shuffle): df.coalesce(n)
   - Custom Key Partitioning: df.repartition(n, col("year_month"))
   - Window Partitioning: Window.partitionBy("month").orderBy(...)
2. Narrow Transformations (Zero Network Shuffle - Pipelined in RAM):
   - select(), filter(), withColumn(), drop()
   - RDD: map(), filter(), mapValues()
3. Wide Transformations (Requires Network Shuffle across Cluster Nodes):
   - groupBy(), agg(), orderBy(), distinct()
   - RDD: reduceByKey(), sortByKey(), groupByKey()

Author: FireWatch Big Data Analytics
"""

import os
import sys
import time

# Ensure UTF-8 output on Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure valid Python executable for PySpark workers
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Ensure valid JAVA_HOME for PySpark
java_home = os.environ.get("JAVA_HOME", "")
if not java_home or not os.path.isdir(java_home):
    for candidate in [r"C:\Program Files\Java\jdk-21", r"C:\Program Files\Java\jdk1.8.0_51"]:
        if os.path.isdir(candidate):
            os.environ["JAVA_HOME"] = candidate
            os.environ["PATH"] = os.path.join(candidate, "bin") + ";" + os.environ.get("PATH", "")
            break

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, month, year, concat_ws, lpad, avg, sum as spark_sum, count as spark_count, round as spark_round
from pyspark.sql.window import Window

DATA_FILE = "fire_archive_SV-C2_817450.csv"
for candidate in [
    DATA_FILE,
    os.path.join("..", DATA_FILE),
    os.path.join(os.path.dirname(__file__), "..", DATA_FILE),
    os.path.join(os.path.dirname(__file__), "..", "..", DATA_FILE)
]:
    if os.path.exists(candidate):
        DATA_FILE = os.path.abspath(candidate)
        break

def demonstrate_transformations_and_partitions():
    print("=" * 80)
    print("🔥 PySpark Transformations & Partitioning Engine")
    print("   Big Data Analytics Architecture (Narrow vs Wide, Partitions)")
    print("=" * 80)

    # 1. Initialize SparkSession
    spark = SparkSession.builder \
        .appName("PySpark_Transformations_And_Partitions") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    # Ingest Data
    print(f"\n[Step 1] Ingesting Telemetry Data from: {DATA_FILE}...")
    df_raw = spark.read.option("header", "true").option("inferSchema", "true").csv(DATA_FILE)
    
    # Clean DataFrame (Narrow Transformations: select, filter, withColumn)
    df_clean = df_raw.select(
        col("frp").cast("double"),
        col("brightness").cast("double"),
        col("bright_t31").cast("double"),
        col("acq_date"),
        col("acq_time").cast("int"),
        col("daynight")
    ).filter(
        col("acq_date").isNotNull() & col("frp").isNotNull() & col("daynight").isin(["D", "N"])
    ).withColumn("year", year(to_date(col("acq_date"), "yyyy-MM-dd"))) \
     .withColumn("month", month(to_date(col("acq_date"), "yyyy-MM-dd"))) \
     .withColumn("year_month", concat_ws("-", col("year"), lpad(col("month"), 2, "0")))

    # Cache preprocessed DataFrame
    df_clean.cache()
    total_count = df_clean.count()
    print(f"  Loaded clean dataset: {total_count:,} records.")

    # =========================================================================
    # SECTION A: PARTITION MECHANICS & CLUSTER DISTRIBUTION
    # =========================================================================
    print("\n" + "=" * 80)
    print("📦 SECTION A: PARTITION MECHANICS & CLUSTER DISTRIBUTION")
    print("=" * 80)
    print("Viva Concept: Partitions are the physical chunks into which Big Data is divided")
    print("across cluster worker nodes to enable parallel computation.")

    # A1. Default Partition Count
    initial_partitions = df_clean.rdd.getNumPartitions()
    print(f"\n[1] Default Partition Count: df.rdd.getNumPartitions() = {initial_partitions} partitions")
    print("    • Why: PySpark automatically partitions input CSV files based on file size (64MB/128MB splits).")

    # A2. Repartition (Scaling UP / Full Network Shuffle)
    print("\n[2] Repartitioning: df.repartition(12)")
    t0 = time.time()
    df_repartitioned = df_clean.repartition(12)
    rep_partitions = df_repartitioned.rdd.getNumPartitions()
    print(f"    • Partition count after repartition(12): {rep_partitions} partitions (took {time.time() - t0:.4f}s)")
    print("    • Viva Rule: Use repartition() when INCREASING partitions or rebalancing skewed data.")
    print("      Drawback: Performs a FULL network shuffle across all cluster nodes.")

    # A3. Coalesce (Scaling DOWN / No Full Network Shuffle)
    print("\n[3] Coalescing: df.coalesce(4)")
    t0 = time.time()
    df_coalesced = df_repartitioned.coalesce(4)
    coa_partitions = df_coalesced.rdd.getNumPartitions()
    print(f"    • Partition count after coalesce(4): {coa_partitions} partitions (took {time.time() - t0:.4f}s)")
    print("    • Viva Rule: Use coalesce() when DECREASING partitions (e.g., before saving output).")
    print("      Advantage: Merges adjacent partitions locally in memory WITHOUT full network shuffle (Fast & Low I/O)!")

    # A4. Hash Partitioning by Key (col("year_month"))
    print("\n[4] Key Partitioning: df.repartition(6, col('year_month'))")
    t0 = time.time()
    df_keyed = df_clean.repartition(6, col("year_month"))
    key_partitions = df_keyed.rdd.getNumPartitions()
    print(f"    • Partition count: {key_partitions} partitions (took {time.time() - t0:.4f}s)")
    print("    • Viva Rule: Partitions data by the hash of 'year_month'.")
    print("      Ensures that all fire records from the SAME month live on the SAME physical node,")
    print("      avoiding repetitive shuffles during downstream groupBy('year_month') queries!")

    # =========================================================================
    # SECTION B: NARROW TRANSFORMATIONS (ZERO NETWORK SHUFFLE)
    # =========================================================================
    print("\n" + "=" * 80)
    print("⚡ SECTION B: NARROW TRANSFORMATIONS (No Network Shuffle - Pipelined in RAM)")
    print("=" * 80)
    print("Viva Concept: Each input partition contributes to at most ONE output partition.")
    print("Spark executes these in a single stage without sending data over the network cable.")

    # B1. select() -> Column projection
    print("\n[1] Narrow: df.select('year_month', 'frp', 'brightness')")
    df_narrow_sel = df_clean.select("year_month", "frp", "brightness")
    print(f"    • Result Schema: {df_narrow_sel.columns}")

    # B2. filter() -> Row filtering (Predicate Pushdown)
    print("\n[2] Narrow: df.filter((col('daynight') == 'N') & (col('frp') >= 50.0))")
    t0 = time.time()
    df_narrow_flt = df_clean.filter((col("daynight") == "N") & (col("frp") >= 50.0))
    severe_night_fires = df_narrow_flt.count()
    print(f"    • Filtered {severe_night_fires:,} extreme night fires (>=50 MW) in {time.time() - t0:.4f}s")

    # B3. withColumn() -> Element-wise column derived transformation
    print("\n[3] Narrow: df.withColumn('frp_kw', col('frp') * 1000)")
    df_narrow_col = df_clean.withColumn("frp_kw", spark_round(col("frp") * 1000, 1))
    sample_row = df_narrow_col.select("acq_date", "frp", "frp_kw").first()
    print(f"    • Sample computation: FRP={sample_row['frp']} MW -> {sample_row['frp_kw']} kW")

    # =========================================================================
    # SECTION C: WIDE TRANSFORMATIONS (REQUIRES NETWORK SHUFFLE)
    # =========================================================================
    print("\n" + "=" * 80)
    print("🔄 SECTION C: WIDE TRANSFORMATIONS (Requires Network Shuffle Across Cluster)")
    print("=" * 80)
    print("Viva Concept: Multiple input partitions contribute to output partitions.")
    print("Spark MUST write shuffle files to disk and send records across worker nodes.")

    # C1. groupBy().agg() -> Distributed Key Aggregation
    print("\n[1] Wide: df.groupBy('year_month').agg(...)")
    t0 = time.time()
    df_wide_agg = df_clean.groupBy("year_month").agg(
        spark_count("*").alias("fire_count"),
        spark_round(spark_sum("frp"), 2).alias("total_frp_mw"),
        spark_round(avg("frp"), 2).alias("avg_frp_mw")
    )
    top_months = df_wide_agg.orderBy(col("fire_count").desc()).limit(3).collect()
    print(f"    • GroupBy Aggregation executed in {time.time() - t0:.4f}s")
    for r in top_months:
        print(f"      - Month {r['year_month']}: {r['fire_count']:,} fires | Total FRP: {r['total_frp_mw']:,} MW | Avg: {r['avg_frp_mw']} MW")

    # C2. orderBy() -> Global Distributed Sort
    print("\n[2] Wide: df.orderBy(col('frp').desc())")
    t0 = time.time()
    top_wildfire = df_clean.orderBy(col("frp").desc()).select("acq_date", "frp", "brightness").first()
    print(f"    • Global Sort executed in {time.time() - t0:.4f}s")
    print(f"    • Highest Recorded Wildfire: Date {top_wildfire['acq_date']} with {top_wildfire['frp']} MW Radiative Power!")

    # C3. Window.partitionBy() -> Distributed Analytical Windowing
    print("\n[3] Wide: Window.partitionBy('year').orderBy('frp')")
    window_spec = Window.partitionBy("year").orderBy(col("frp").desc())
    from pyspark.sql.functions import row_number
    df_windowed = df_clean.withColumn("rank_in_year", row_number().over(window_spec))
    print("    • Window partitioning distributes data by 'year' across workers to compute intra-year rankings!")

    # =========================================================================
    # SECTION D: SUMMARY FOR EVALUATORS / MA'AM
    # =========================================================================
    print("\n" + "=" * 80)
    print("🎓 SUMMARY CHEAT SHEET FOR PROFESSOR / MA'AM VIVA:")
    print("=" * 80)
    print("1. What is a Transformation?")
    print("   • Operations that transform one distributed dataset into another.")
    print("   • Lazy Evaluation: Transformations do not execute until an Action (count, collect) is called.")
    print("2. Narrow Transformations (Zero Shuffle):")
    print("   • map, filter, select, withColumn.")
    print("   • Output partition depends on only ONE input partition. Highly efficient and pipelined in RAM.")
    print("3. Wide Transformations (Network Shuffle):")
    print("   • groupBy, agg, orderBy, distinct, reduceByKey, join.")
    print("   • Output partition depends on MULTIPLE input partitions. Requires network shuffle.")
    print("4. Partitioning (repartition vs coalesce):")
    print("   • getNumPartitions(): Inspects partition count.")
    print("   • repartition(n): Increases partitions or redistributes data. Requires FULL network shuffle.")
    print("   • coalesce(n): Decreases partitions. Merges partitions WITHOUT full shuffle (saves I/O).")
    print("   • repartition(n, col): Hash-partitions data by column key for fast subsequent grouping.")
    print("=" * 80)

    spark.stop()
    print("\n✅ PySpark Transformations and Partitioning Engine Completed Successfully!")

if __name__ == "__main__":
    demonstrate_transformations_and_partitions()
