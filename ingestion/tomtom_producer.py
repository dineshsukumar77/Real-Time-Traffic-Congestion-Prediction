# src/ingestion/tomtom_producer.py
import os
import json
import time
import asyncio
import aiohttp
from datetime import datetime, timezone
from kafka import KafkaProducer

BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPIC = os.getenv("RAW_TOPIC", "traffic.raw")
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY", "")

POINTS = [
    {"segment_id": "blr_mgroad", "lat": 12.9755, "lon": 77.6041},
    {"segment_id": "blr_silkboard", "lat": 12.9177, "lon": 77.6233},
    {"segment_id": "blr_majestic", "lat": 12.9784, "lon": 77.5720},
    {"segment_id": "blr_hebball", "lat": 13.0358, "lon": 77.5970},
]

if not TOMTOM_API_KEY:
    raise RuntimeError("❌ Set TOMTOM_API_KEY environment variable to use TomTom producer")

def create_producer():
    return KafkaProducer(
        bootstrap_servers=[BROKER],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8"),
        acks="all",
        retries=3
    )

async def fetch_tomtom(session, lat, lon):
    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json?key={TOMTOM_API_KEY}&point={lat},{lon}"
    try:
        async with session.get(url, timeout=8) as r:
            if r.status == 200:
                return await r.json()
            else:
                print(f"⚠️ HTTP {r.status} for {lat},{lon}")
    except Exception as e:
        print(f"❌ Fetch error for {lat},{lon}: {e}")
    return None

async def fetch_all_points():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_tomtom(session, p["lat"], p["lon"]) for p in POINTS]
        return await asyncio.gather(*tasks)

async def main_async(interval_sec=30):
    producer = create_producer()
    try:
        while True:
            now_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            results = await fetch_all_points()

            for p, data in zip(POINTS, results):
                if not data or "flowSegmentData" not in data:
                    print(f"TomTom returned no flowSegmentData for {p['segment_id']}")
                    continue

                flow = data["flowSegmentData"]
                freeflow = flow.get("freeFlowSpeed")
                current = flow.get("currentSpeed")
                congestion = None
                if freeflow and freeflow > 0 and current is not None:
                    congestion = round(max(0.0, min(1.0, 1.0 - (current / freeflow))), 3)

                msg = {
                    "segment_id": p["segment_id"],
                    "timestamp_utc": now_utc,
                    "lat": p["lat"],
                    "lon": p["lon"],
                    "speed_current_kmph": current,
                    "speed_freeflow_kmph": freeflow,
                    "travel_time_sec": flow.get("currentTravelTime"),
                    "congestion_index": congestion,
                    "source": "tomtom"
                }

                key = f"{p['segment_id']}|{now_utc}"
                producer.send(TOPIC, key=key, value=msg)
                print(f"🚦 Sent -> {key} | {msg}")

            producer.flush()
            await asyncio.sleep(interval_sec)

    except KeyboardInterrupt:
        print("TomTom producer stopped by user")
    finally:
        producer.close()

if __name__ == "__main__":
    asyncio.run(main_async(interval_sec=int(os.getenv("TOMTOM_INTERVAL", "30"))))
