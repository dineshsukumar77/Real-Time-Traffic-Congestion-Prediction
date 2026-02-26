# src/streaming/spark_job.py
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_timestamp, lit
from pyspark.sql.types import StructType, StringType, DoubleType, IntegerType
from pyspark.sql.functions import udf
from pyspark.sql import DataFrame
from pyspark.sql.functions import expr
from pyspark.sql.functions import from_utc_timestamp

# Windows Hadoop (update if needed)
os.environ.setdefault("HADOOP_HOME", "C:\\hadoop")
os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + os.path.join(os.environ.get("HADOOP_HOME","C:\\hadoop"), "bin")

SILVER_PATH = "data/silver"
CHECKPOINT_PATH = os.path.join(SILVER_PATH, "_checkpoints")

schema = StructType() \
    .add("segment_id", StringType()) \
    .add("timestamp_utc", StringType()) \
    .add("lat", DoubleType()) \
    .add("lon", DoubleType()) \
    .add("speed_current_kmph", DoubleType()) \
    .add("speed_freeflow_kmph", DoubleType()) \
    .add("travel_time_sec", IntegerType()) \
    .add("congestion_index", DoubleType()) \
    .add("source", StringType())

spark = SparkSession.builder \
    .appName("TrafficKafkaToParquet") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.0") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Create sink dirs if not exist
os.makedirs(SILVER_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", os.getenv("KAFKA_BROKER", "localhost:9092")) \
    .option("subscribe", os.getenv("RAW_TOPIC", "traffic.raw")) \
    .option("startingOffsets", "latest") \
    .option("failOnDataLoss", "false") \
    .load()

# Parse JSON value
parsed = kafka_df.selectExpr("CAST(value AS STRING) as json_str") \
    .select(from_json(col("json_str"), schema).alias("data")) \
    .select("data.*")

# Convert timestamp_utc string to timestamp type and add IST timestamp
parsed2 = parsed.withColumn("timestamp_utc_ts", to_timestamp(col("timestamp_utc"))) \
    .withColumn("timestamp_ist_ts", from_utc_timestamp(col("timestamp_utc_ts"), "Asia/Kolkata")) \
    .withColumn("date", expr("date_format(timestamp_ist_ts, 'yyyy-MM-dd')"))

# compute congestion_index if missing, and congestion_level
parsed3 = parsed2.withColumn("congestion_index",
                             expr("CASE WHEN congestion_index IS NULL AND speed_freeflow_kmph IS NOT NULL AND speed_freeflow_kmph > 0 AND speed_current_kmph IS NOT NULL THEN round(1.0 - speed_current_kmph / speed_freeflow_kmph, 3) ELSE congestion_index END")) \
                 .withColumn("congestion_index", expr("CASE WHEN congestion_index < 0 THEN 0 WHEN congestion_index > 1 THEN 1 ELSE congestion_index END")) \
                 .withColumn("congestion_level", expr("CASE WHEN speed_freeflow_kmph IS NULL OR speed_freeflow_kmph = 0 THEN NULL ELSE (speed_freeflow_kmph - speed_current_kmph) / speed_freeflow_kmph END"))

def foreach_batch_function(batch_df: DataFrame, batch_id: int):
    # print batch information for debugging
    print(f"--- foreachBatch id={batch_id} rows={batch_df.count()} ---")
    if batch_df.count() > 0:
        batch_df.select("segment_id", "timestamp_utc", "timestamp_ist_ts", "speed_current_kmph", "speed_freeflow_kmph", "congestion_index").show(10, truncate=False)
    # write to parquet append
    batch_df.write.mode("append").parquet(SILVER_PATH)

query = parsed3.writeStream \
    .foreachBatch(foreach_batch_function) \
    .option("checkpointLocation", CHECKPOINT_PATH) \
    .start()

print("Spark job started — writing silver parquet to", SILVER_PATH)
query.awaitTermination()
