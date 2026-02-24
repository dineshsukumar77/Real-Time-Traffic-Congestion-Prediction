# src/ingestion/producer.py
import json
import time
import random
import os
from datetime import datetime, timezone, timedelta
from kafka import KafkaProducer, errors

BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPIC = os.getenv("RAW_TOPIC", "traffic.raw")

# Sample Bangalore points (you can add more)
POINTS = [
    {"segment_id": "blr_mgroad", "lat": 12.9755, "lon": 77.6041},
    {"segment_id": "blr_silkboard", "lat": 12.9177, "lon": 77.6233},
    {"segment_id": "blr_majestic", "lat": 12.9784, "lon": 77.5720},
    {"segment_id": "blr_hebball", "lat": 13.0358, "lon": 77.5970},
]

def make_message(p):
    now_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")
    # simulate freeflow & current speeds
    freeflow = random.uniform(40.0, 60.0)
    # sometimes simulate congestion
    if random.random() < 0.15:
        current = max(3.0, random.uniform(5.0, freeflow * 0.5))
    else:
        current = random.uniform(freeflow * 0.6, freeflow)
    congestion = None
    if freeflow and freeflow > 0:
        congestion = round(max(0.0, min(1.0, 1.0 - (current / freeflow))), 3)
    msg = {
        "segment_id": p["segment_id"],
        "timestamp_utc": now_utc,
        "lat": p["lat"],
        "lon": p["lon"],
        "speed_current_kmph": round(current, 2),
        "speed_freeflow_kmph": round(freeflow, 2),
        "travel_time_sec": int(random.uniform(60, 600)),
        "congestion_index": congestion,
        "source": "simulator"
    }
    return msg

def create_producer():
    return KafkaProducer(
        bootstrap_servers=[BROKER],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8"),
        acks="all",
        retries=5
    )

def main(interval_sec=5):
    print("Producer connecting to", BROKER)
    producer = None
    backoff = 1.0
    try:
        while True:
            if producer is None:
                try:
                    producer = create_producer()
                    print("Connected to Kafka broker.")
                    backoff = 1.0
                except Exception as e:
                    print("Kafka connect failed:", e)
                    time.sleep(backoff)
                    backoff = min(30, backoff * 2)
                    continue

            for p in POINTS:
                msg = make_message(p)
                key = f"{p['segment_id']}|{msg['timestamp_utc']}"
                try:
                    producer.send(TOPIC, key=key, value=msg)
                    producer.flush()
                    print(f"Produced -> {key} | {msg}")
                except errors.KafkaError as e:
                    print("Kafka send error:", e)
                    # close and force reconnect in next loop
                    try:
                        producer.close()
                    except Exception:
                        pass
                    producer = None
                    break  # break for loop to re-create producer
            time.sleep(interval_sec)
    except KeyboardInterrupt:
        print("Producer interrupted by user")
    finally:
        if producer:
            try:
                producer.close()
            except Exception:
                pass
        print("Producer stopped.")

if __name__ == "__main__":
    main(interval_sec=int(os.getenv("PRODUCER_INTERVAL", "30")))
