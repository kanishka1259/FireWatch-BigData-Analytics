import os, stat

HOME = "/home/kanishka"
B = HOME + "/bigdata_project"

def write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    os.chmod(path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP)
    print("Written:", path)

write(B + "/run_all.sh", """#!/bin/bash
export HADOOP_HOME=$HOME/hadoop
export HIVE_HOME=$HOME/hive
export HBASE_HOME=/opt/hbase
P=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$HIVE_HOME/bin:$HBASE_HOME/bin
export PATH=$P
VENV=$HOME/bigdata_project/venv/bin/python3
PROJECT=$HOME/bigdata_project
mkdir -p $PROJECT/logs
echo "======================================================================"
echo "  NASA FIRMS Fire Hotspot - Big Data Pipeline"
echo "  Stack: PySpark + HDFS + Hive + HBase + Flask"
echo "======================================================================"
echo "[1/6] Starting HDFS..."
$HADOOP_HOME/sbin/start-dfs.sh 2>/dev/null || true
sleep 5
hdfs dfs -mkdir -p /user/kanishka/fire_data/raw 2>/dev/null || true
hdfs dfs -put -f $PROJECT/fire_archive_SV-C2_817450.csv /user/kanishka/fire_data/raw/ 2>/dev/null || true
echo "[2/6] PySpark EDA (this takes 10-15 min on 9.5M records)..."
$VENV $PROJECT/spark_analysis/01_preprocessing_eda.py 2>&1 | tee $PROJECT/logs/eda.log
echo "[3/6] Hive ORC tables..."
$HIVE_HOME/bin/hive -f $PROJECT/hive_scripts/create_tables.hql 2>&1 | tee $PROJECT/logs/hive.log
echo "[4/6] HBase NoSQL backend..."
$VENV $PROJECT/nosql/hbase_loader.py 2>&1 | tee $PROJECT/logs/hbase.log
echo "[5/6] ML Pipeline (RF + GBT + Regressor)..."
$VENV $PROJECT/ml_model/02_ml_pipeline.py 2>&1 | tee $PROJECT/logs/ml.log
echo "[6/6] Starting Flask Dashboard at http://localhost:5000"
$VENV $PROJECT/webapp/app.py
""")

write(B + "/run_eda_only.sh", """#!/bin/bash
export HADOOP_HOME=$HOME/hadoop
P=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin
export PATH=$P
VENV=$HOME/bigdata_project/venv/bin/python3
PROJECT=$HOME/bigdata_project
mkdir -p $PROJECT/logs
echo "Running PySpark EDA on ~9.5M fire records..."
$VENV $PROJECT/spark_analysis/01_preprocessing_eda.py 2>&1 | tee $PROJECT/logs/eda.log
echo "Done! Charts: $PROJECT/outputs/charts/"
""")

write(B + "/run_ml_only.sh", """#!/bin/bash
VENV=$HOME/bigdata_project/venv/bin/python3
PROJECT=$HOME/bigdata_project
mkdir -p $PROJECT/logs
echo "Running ML Pipeline (Random Forest + GBT)..."
$VENV $PROJECT/ml_model/02_ml_pipeline.py 2>&1 | tee $PROJECT/logs/ml.log
echo "ML Done!"
""")

write(B + "/run_webapp.sh", """#!/bin/bash
VENV=$HOME/bigdata_project/venv/bin/python3
PROJECT=$HOME/bigdata_project
echo "Starting FireWatch Dashboard: http://localhost:5000"
$VENV $PROJECT/webapp/app.py
""")

print("\nAll run scripts created!")
print("To run in WSL Ubuntu terminal:")
print("  bash ~/bigdata_project/run_eda_only.sh")
print("  bash ~/bigdata_project/run_all.sh")
