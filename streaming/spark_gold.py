import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import window, avg, col, expr
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType

# Paths
SILVER_PATH = "data/silver"
GOLD_PATH = "data/gold"
GOLD_CHECKPOINT = os.path.join(GOLD_PATH, "_checkpoints")

# Spark Session
spark = SparkSession.builder.appName("GoldLayerTraffic").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Ensure directories exist
os.makedirs(GOLD_PATH, exist_ok=True)
os.makedirs(GOLD_CHECKPOINT, exist_ok=True)

# ✅ Define schema same as silver parquet
schema = StructType([
    StructField("segment_id", StringType(), True),
    StructField("timestamp_utc", StringType(), True),
    StructField("timestamp_ist_ts", TimestampType(), True),
    StructField("speed_current_kmph", DoubleType(), True),
    StructField("speed_freeflow_kmph", DoubleType(), True),
    StructField("congestion_index", DoubleType(), True)
])

# ✅ Read from Silver parquet stream
silver_stream = spark.readStream \
    .format("parquet") \
    .schema(schema) \
    .load(SILVER_PATH)

# ✅ Rename timestamp for convenience
silver_stream = silver_stream.withColumnRenamed("timestamp_ist_ts", "timestamp_event")

# ✅ Compute congestion level
silver_stream = silver_stream.withColumn(
    "congestion_level",
    expr("CASE WHEN speed_freeflow_kmph = 0 THEN NULL ELSE (speed_freeflow_kmph - speed_current_kmph) / speed_freeflow_kmph END")
)

# ✅ Aggregate over 30-second window (faster testing)
aggregated = silver_stream.withWatermark("timestamp_event", "1 minutes") \
    .groupBy(
        col("segment_id"),
        window(col("timestamp_event"), "30 seconds")
    ).agg(
        avg("speed_current_kmph").alias("avg_speed"),
        avg("congestion_index").alias("avg_congestion_index"),
        avg("congestion_level").alias("mean_congestion_level")
    ).select(
        col("segment_id").alias("road_id"),
        col("window.start").alias("window_start"),
        col("window.end").alias("window_end"),
        "avg_speed",
        "avg_congestion_index",
        "mean_congestion_level"
    )

# ✅ Write batches to gold parquet
def foreach_batch(batch_df, batch_id):
    count = batch_df.count()
    print(f">>> Gold foreachBatch id={batch_id}, rows={count}")
    if count > 0:
        batch_df.show(truncate=False)
        batch_df.write.mode("append").parquet(GOLD_PATH)

query = aggregated.writeStream \
    .foreachBatch(foreach_batch) \
    .option("checkpointLocation", GOLD_CHECKPOINT) \
    .start()

print(f"✅ Gold streaming job started — watching {SILVER_PATH}")
spark.streams.awaitAnyTermination()
