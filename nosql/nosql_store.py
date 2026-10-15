"""
NoSQL Document Store Engine for NASA FIRMS Fire Hotspots
Supports fast JSON document storage, bounding box geospatial queries,
severity filtering, and optional MongoDB integration.
"""

import os
import sys
import json
import sqlite3
import time

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

DB_PATH = "outputs/data/fire_nosql_store.db"
SAMPLE_JSON = "outputs/data/hotspots_sample.json"

class FireNoSQLStore:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.mongo_client = None
        self._init_mongo()
        self._init_sqlite_doc_store()

    def _init_mongo(self):
        try:
            import pymongo
            client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=1000)
            client.admin.command('ping')
            self.mongo_client = client["firms_db"]["hotspots"]
            print("  [NoSQL] Connected to live MongoDB server on localhost:27017")
        except Exception:
            self.mongo_client = None
            print("  [NoSQL] Using Embedded High-Speed Document Store (SQLite JSONB)")

    def _init_sqlite_doc_store(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fire_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT UNIQUE,
                    latitude REAL,
                    longitude REAL,
                    frp REAL,
                    severity_tier TEXT,
                    acq_date TEXT,
                    daynight TEXT,
                    document JSON
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_geo ON fire_documents(latitude, longitude)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_frp ON fire_documents(frp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tier ON fire_documents(severity_tier)")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS temporal_aggregates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agg_type TEXT,
                    time_key TEXT UNIQUE,
                    detection_count INTEGER,
                    avg_frp REAL,
                    total_frp REAL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_time_key ON temporal_aggregates(time_key)")
            conn.commit()

    def insert_documents(self, documents):
        t0 = time.time()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            rows = []
            for doc in documents:
                doc_id = f"{doc.get('latitude')}_{doc.get('longitude')}_{doc.get('acq_date')}_{doc.get('acq_time')}"
                rows.append((
                    doc_id,
                    float(doc.get("latitude", 0)),
                    float(doc.get("longitude", 0)),
                    float(doc.get("frp", 0)),
                    str(doc.get("severity_tier", "Low")),
                    str(doc.get("acq_date", "")),
                    str(doc.get("daynight", "D")),
                    json.dumps(doc)
                ))
            cursor.executemany("""
                INSERT OR REPLACE INTO fire_documents 
                (doc_id, latitude, longitude, frp, severity_tier, acq_date, daynight, document)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()
            
        if self.mongo_client is not None:
            try:
                self.mongo_client.insert_many(documents, ordered=False)
            except Exception:
                pass
                
        duration = time.time() - t0
        print(f"  [NoSQL] Ingested {len(documents):,} documents in {duration:.2f}s")

    def query_bbox(self, min_lat, max_lat, min_lon, max_lon, min_frp=0.0, limit=200):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT document FROM fire_documents
                WHERE latitude BETWEEN ? AND ?
                  AND longitude BETWEEN ? AND ?
                  AND frp >= ?
                ORDER BY frp DESC
                LIMIT ?
            """, (min_lat, max_lat, min_lon, max_lon, min_frp, limit))
            results = [json.loads(row[0]) for row in cursor.fetchall()]
            return results

    def query_severity(self, tier="Extreme", limit=100):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT document FROM fire_documents
                WHERE severity_tier = ?
                ORDER BY frp DESC
                LIMIT ?
            """, (tier, limit))
            return [json.loads(row[0]) for row in cursor.fetchall()]

    def get_stats(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), AVG(frp), MAX(frp) FROM fire_documents")
            count, avg_frp, max_frp = cursor.fetchone()
            return {
                "total_stored_hotspots": count or 0,
                "avg_frp": round(avg_frp or 0, 2),
                "max_frp": round(max_frp or 0, 2),
                "backend": "MongoDB (localhost:27017)" if self.mongo_client is not None else "Embedded NoSQL Document Store"
            }

    def store_temporal_aggregates(self, daily_csv=None, monthly_csv=None):
        """Persists Member 2 temporal aggregates into NoSQL document store."""
        daily_csv = daily_csv or "outputs/data/member2_daily_stats.csv"
        monthly_csv = monthly_csv or "outputs/data/member2_monthly_stats.csv"

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            inserted_count = 0

            # Store monthly aggregates
            if os.path.exists(monthly_csv):
                import pandas as pd
                df_m = pd.read_csv(monthly_csv)
                for _, row in df_m.iterrows():
                    cursor.execute("""
                        INSERT OR REPLACE INTO temporal_aggregates
                        (agg_type, time_key, detection_count, avg_frp, total_frp)
                        VALUES (?, ?, ?, ?, ?)
                    """, ("monthly", str(row.get("year_month", row.get("month"))), int(row["detection_count"]), float(row["avg_frp"]), float(row["total_frp"])))
                    inserted_count += 1

            # Store top daily aggregates
            if os.path.exists(daily_csv):
                import pandas as pd
                df_d = pd.read_csv(daily_csv)
                for _, row in df_d.head(30).iterrows():
                    cursor.execute("""
                        INSERT OR REPLACE INTO temporal_aggregates
                        (agg_type, time_key, detection_count, avg_frp, total_frp)
                        VALUES (?, ?, ?, ?, ?)
                    """, ("daily", str(row["acq_date"]), int(row["detection_count"]), float(row["avg_frp"]), float(row["total_frp"])))
                    inserted_count += 1

            conn.commit()
            print(f"  [NoSQL] Ingested {inserted_count} temporal aggregate records.")

    def get_temporal_aggregates(self, agg_type="monthly"):
        """Retrieves temporal aggregates from NoSQL document store."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT time_key, detection_count, avg_frp, total_frp
                FROM temporal_aggregates
                WHERE agg_type = ?
                ORDER BY time_key ASC
            """, (agg_type,))
            rows = cursor.fetchall()
            return [{"time_key": r[0], "detection_count": r[1], "avg_frp": r[2], "total_frp": r[3]} for r in rows]

def populate_nosql():
    print("=" * 60)
    print("💾 Initializing NoSQL Backend Document Store")
    print("=" * 60)
    store = FireNoSQLStore()
    
    if os.path.exists(SAMPLE_JSON):
        with open(SAMPLE_JSON) as f:
            docs = json.load(f)
        store.insert_documents(docs)
        print("  Stats:", store.get_stats())
        print("  Sample Query (Extreme fires):", len(store.query_severity("Extreme")), "hotspots retrieved.")
    else:
        print("  No sample JSON found, please run PySpark EDA first.")

    # Populate Member 2 temporal aggregates
    store.store_temporal_aggregates()

if __name__ == "__main__":
    populate_nosql()
