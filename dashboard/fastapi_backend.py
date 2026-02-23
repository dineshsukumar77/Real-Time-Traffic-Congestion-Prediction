import os
import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Union, Dict

import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------- Configuration ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("traffic-api")

app = FastAPI(title="Traffic Congestion Prediction API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths (adjust as needed)
DATA_SILVER_PATH = "../data/silver/"
DATA_GOLD_PATH = "../../data/gold"
MODEL_PATH = "../models"
LOOKBACK = 6  # number of records (e.g., 6 * 5min = 30 minutes)

# Global variables for models and scaler
models: Dict[str, object] = {}
scaler = None


# ---------- Pydantic models ----------
class PredictionRequest(BaseModel):
    road_id: str
    model_name: str = "LSTM"


class PredictionResponse(BaseModel):
    road_id: str
    model: str
    predicted_timestamp: str
    predicted_congestion_index: float
    current_congestion_index: float
    current_speed: float
    last_update: str
    confidence_score: float


class RealtimeData(BaseModel):
    road_id: str
    timestamp: str
    avg_speed: float
    avg_congestion_index: float
    mean_congestion_level: Optional[Union[float, str]]


# ---------- Helper classes ----------
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("WebSocket connected. Total connections: %d", len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        try:
            self.active_connections.remove(websocket)
            logger.info("WebSocket disconnected. Total connections: %d", len(self.active_connections))
        except ValueError:
            # connection already removed
            pass

    async def broadcast(self, message: dict):
        # iterate over a copy to allow safe removal
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning("Failed to send websocket message to a client: %s", e)
                # try to remove broken connection
                try:
                    self.active_connections.remove(connection)
                except ValueError:
                    pass


manager = ConnectionManager()


# ---------- Utility functions ----------
def get_congestion_level(index: float) -> str:
    """Convert congestion index to level (0-1 scale)"""
    try:
        if index < 0.3:
            return "Low"
        elif index < 0.6:
            return "Medium"
        elif index < 0.8:
            return "High"
        else:
            return "Severe"
    except Exception:
        return "Unknown"


def calculate_trend(df: pd.DataFrame) -> str:
    """Calculate congestion trend from last 3 records"""
    try:
        if len(df) < 3:
            return "stable"

        recent = df.sort_values("window_start").tail(3)["avg_congestion_index"].values
        # handle possible NaNs
        recent = np.nan_to_num(recent, nan=0.0)

        if recent[-1] > recent[0] * 1.1:
            return "increasing"
        elif recent[-1] < recent[0] * 0.9:
            return "decreasing"
        else:
            return "stable"
    except Exception as e:
        logger.warning("Error calculating trend: %s", e)
        return "stable"


# ---------- Model & scaler loading ----------
def load_models_and_scaler():
    """Load trained models and scaler into memory"""
    global models, scaler
    try:
        # Load scaler
        scaler_path = os.path.join(MODEL_PATH, "scaler.joblib")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info("✓ Scaler loaded from %s", scaler_path)
        else:
            logger.warning("Scaler file not found at %s", scaler_path)

        # Define model filenames (adjust names/extensions as you have them)
        model_files = {
            "LSTM": "lstm.keras",
            "GRU": "gru.keras",
            "LSTM_GRU": "lstm_gru.keras",
            "GRU_LSTM": "gru_lstm.keras",
            "Random_Forest": "random_forest.joblib",
        }

        for model_name, filename in model_files.items():
            model_path = os.path.join(MODEL_PATH, filename)
            if os.path.exists(model_path):
                if model_name == "Random_Forest":
                    models[model_name] = joblib.load(model_path)
                else:
                    models[model_name] = load_model(model_path)
                logger.info("✓ %s model loaded from %s", model_name, model_path)
            else:
                logger.debug("Model file not found (skipping): %s", model_path)

        if not models:
            logger.warning("No models were loaded. Ensure model files exist in %s", MODEL_PATH)
        else:
            logger.info("Available models: %s", list(models.keys()))

    except Exception as e:
        logger.exception("Error loading models or scaler: %s", e)


# ---------- Data preprocessing ----------
def preprocess_data(df: pd.DataFrame, road_id: Optional[str] = None) -> pd.DataFrame:
    """Preprocess data for prediction - returns filled dataframe with engineered features"""
    if "window_start" not in df.columns:
        raise ValueError("Expected 'window_start' column in dataframe")

    df = df.copy()
    df["window_start"] = pd.to_datetime(df["window_start"])

    if road_id:
        df = df[df["road_id"] == road_id].copy()

    df = df.sort_values("window_start").reset_index(drop=True)

    # Time features
    df["hour"] = df["window_start"].dt.hour
    df["minute"] = df["window_start"].dt.minute
    df["day_of_week"] = df["window_start"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"] = (
        (df["hour"].between(7, 10)) | (df["hour"].between(17, 20))
    ).astype(int)

    # Cyclical features
    df["time_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["time_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Change features
    df["speed_change"] = df["avg_speed"].diff().fillna(0.0)
    df["congestion_change"] = df["avg_congestion_index"].diff().fillna(0.0)

    # Rolling features
    for window in [3, 6]:
        df[f"speed_rolling_mean_{window}"] = (
            df["avg_speed"].rolling(window=window, min_periods=1).mean().fillna(method="bfill").fillna(method="ffill")
        )
        df[f"speed_rolling_std_{window}"] = (
            df["avg_speed"].rolling(window=window, min_periods=1).std().fillna(0.0)
        )
        df[f"congestion_rolling_mean_{window}"] = (
            df["avg_congestion_index"].rolling(window=window, min_periods=1).mean().fillna(method="bfill").fillna(method="ffill")
        )
        df[f"congestion_rolling_std_{window}"] = (
            df["avg_congestion_index"].rolling(window=window, min_periods=1).std().fillna(0.0)
        )

    # Fill any remaining NaNs
    df = df.bfill().ffill()

    return df


# ---------- Data reading ----------
def get_latest_data(road_id: Optional[str]) -> pd.DataFrame:
    """Get latest aggregated data from gold folder. Raises HTTPException on problems."""
    gold_path = Path(DATA_GOLD_PATH)
    if not gold_path.exists():
        raise HTTPException(status_code=404, detail=f"Gold folder not found: {DATA_GOLD_PATH}")

    gold_files = list(gold_path.glob("*.parquet"))
    if not gold_files:
        raise HTTPException(status_code=404, detail="No data files found in gold folder")

    latest_file = max(gold_files, key=lambda x: x.stat().st_mtime)
    try:
        df = pd.read_parquet(latest_file)
    except Exception as e:
        logger.exception("Error reading parquet %s: %s", latest_file, e)
        raise HTTPException(status_code=500, detail=f"Error reading parquet file: {e}")

    # Ensure expected columns and types
    if "window_start" in df.columns:
        df["window_start"] = pd.to_datetime(df["window_start"])
    else:
        # If different column existed (like 'window.start'), try to normalize (defensive)
        if "window.start" in df.columns:
            df = df.rename(columns={"window.start": "window_start"})
            df["window_start"] = pd.to_datetime(df["window_start"])
        else:
            logger.warning("Parquet file missing 'window_start' column")

    if road_id:
        df = df[df["road_id"] == road_id]

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for road_id: {road_id}")

    return df


# ---------- Startup hook ----------
@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Starting up - loading models and scaler...")
    load_models_and_scaler()


# ---------- Root & health ----------
@app.get("/")
async def root():
    return {
        "message": "Traffic Congestion Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "roads": "/api/roads",
            "current_data": "/api/current/{road_id}",
            "predict": "/api/predict",
            "historical": "/api/historical/{road_id}",
            "websocket": "/ws",
        },
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "scaler_loaded": scaler is not None,
        "available_models": list(models.keys()),
    }


# ---------- Endpoints ----------
@app.get("/api/roads")
async def get_available_roads():
    """Get list of available road IDs from all parquet files"""
    try:
        gold_files = list(Path(DATA_GOLD_PATH).glob("*.parquet"))
        logger.info("Gold files found: %s", [f.name for f in gold_files])

        if not gold_files:
            logger.warning("No parquet files found in %s", DATA_GOLD_PATH)
            return {"roads": []}

        # Read road_id column from all parquet files
        dfs = []
        for f in gold_files:
            try:
                df = pd.read_parquet(f, columns=["road_id"])
                dfs.append(df)
            except Exception as e:
                logger.warning("Skipping %s due to read error: %s", f, e)

        if not dfs:
            raise HTTPException(status_code=500, detail="No readable parquet files found")

        full_df = pd.concat(dfs, ignore_index=True)
        if "road_id" not in full_df.columns:
            raise HTTPException(status_code=500, detail="Parquet files missing 'road_id' column")

        roads = full_df["road_id"].dropna().unique().tolist()
        logger.info("Unique road_ids found: %s", roads)
        return {"roads": roads}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in get_available_roads: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/current/{road_id}")
async def get_current_data(road_id: str):
    """Get current traffic data for a road"""
    try:
        df = get_latest_data(road_id)
        # Get the most recent record
        latest = df.sort_values("window_start").iloc[-1]

        return {
            "road_id": road_id,
            "timestamp": latest["window_start"].isoformat() if pd.notna(latest["window_start"]) else str(latest.get("window_start", "")),
            "avg_speed": float(latest["avg_speed"]),
            "avg_congestion_index": float(latest["avg_congestion_index"]),
            "congestion_level": get_congestion_level(float(latest["avg_congestion_index"])),
            "trend": calculate_trend(df),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in get_current_data: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_congestion(request: PredictionRequest):
    """Predict next 5 minutes congestion"""
    try:
        # model existence check
        if request.model_name not in models:
            raise HTTPException(
                status_code=400,
                detail=f"Model {request.model_name} not available. Available: {list(models.keys())}",
            )

        # scaler loaded check
        if scaler is None:
            raise HTTPException(status_code=500, detail="Scaler not loaded. Cannot scale features for prediction.")

        # Get latest data
        df = get_latest_data(request.road_id)

        # Preprocess
        df = preprocess_data(df, request.road_id)

        # Get last LOOKBACK records
        latest_data = df.tail(LOOKBACK).copy()

        if len(latest_data) < LOOKBACK:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough data. Need at least {LOOKBACK} records (configured lookback).",
            )

        # Feature columns
        feature_cols = [
            "avg_speed",
            "avg_congestion_index",
            "hour",
            "minute",
            "day_of_week",
            "is_weekend",
            "is_rush_hour",
            "time_sin",
            "time_cos",
            "day_sin",
            "day_cos",
            "speed_change",
            "congestion_change",
        ]
        rolling_cols = [col for col in latest_data.columns if "rolling" in col]
        feature_cols.extend(rolling_cols)
        feature_cols = [col for col in feature_cols if col in latest_data.columns]

        if not feature_cols:
            raise HTTPException(status_code=500, detail="No feature columns found for prediction.")

        # Scale features
        try:
            latest_scaled = scaler.transform(latest_data[feature_cols])
        except Exception as e:
            logger.exception("Error scaling features: %s", e)
            raise HTTPException(status_code=500, detail=f"Error scaling features: {e}")

        # Prepare input shape
        try:
            X_pred = latest_scaled.reshape(1, LOOKBACK, len(feature_cols))
        except Exception as e:
            logger.exception("Error shaping input for model: %s", e)
            raise HTTPException(status_code=500, detail=f"Error shaping input for model: {e}")

        # Make prediction
        if request.model_name == "Random_Forest":
            X_pred_rf = X_pred.reshape(1, -1)
            prediction = float(models[request.model_name].predict(X_pred_rf)[0])
            confidence = 0.85  # placeholder
        else:
            model = models[request.model_name]
            try:
                pred_raw = model.predict(X_pred)
                # If model.predict returns array shape (1,1) or (1,), handle both
                if isinstance(pred_raw, (list, np.ndarray)):
                    prediction = float(np.array(pred_raw).reshape(-1)[0])
                else:
                    prediction = float(pred_raw)
            except Exception as e:
                logger.exception("Model prediction error: %s", e)
                raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

            # Calculate confidence robustly
            recent_variance = latest_data["avg_congestion_index"].std()
            try:
                recent_variance = float(recent_variance if not np.isnan(recent_variance) else 0.0)
            except Exception:
                recent_variance = 0.0
            confidence = max(0.5, min(0.95, 1 - recent_variance))

        # Timestamp handling
        last_timestamp = df["window_start"].iloc[-1]
        if pd.isna(last_timestamp):
            next_timestamp = datetime.utcnow() + timedelta(minutes=5)
            last_ts_out = datetime.utcnow()
        else:
            next_timestamp = pd.to_datetime(last_timestamp) + timedelta(minutes=5)
            last_ts_out = pd.to_datetime(last_timestamp)

        return PredictionResponse(
            road_id=request.road_id,
            model=request.model_name,
            predicted_timestamp=next_timestamp.isoformat(),
            predicted_congestion_index=float(max(0.0, prediction)),
            current_congestion_index=float(df["avg_congestion_index"].iloc[-1]),
            current_speed=float(df["avg_speed"].iloc[-1]),
            last_update=last_ts_out.isoformat(),
            confidence_score=float(confidence),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in predict_congestion: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/historical/{road_id}")
async def get_historical_data(road_id: str, hours: int = 2):
    """Get historical data for plotting"""
    try:
        df = get_latest_data(road_id)
        df = df.sort_values("window_start")

        cutoff_time = df["window_start"].max() - timedelta(hours=hours)
        df_filtered = df[df["window_start"] >= cutoff_time]

        data = []
        for _, row in df_filtered.iterrows():
            data.append(
                {
                    "timestamp": row["window_start"].isoformat() if pd.notna(row["window_start"]) else str(row.get("window_start", "")),
                    "avg_speed": float(row["avg_speed"]),
                    "avg_congestion_index": float(row["avg_congestion_index"]),
                }
            )

        return {"data": data}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in get_historical_data: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            await asyncio.sleep(5)  # broadcast frequency
            try:
                gold_files = list(Path(DATA_GOLD_PATH).glob("*.parquet"))
                if gold_files:
                    latest_file = max(gold_files, key=lambda x: x.stat().st_mtime)
                    try:
                        df = pd.read_parquet(latest_file)
                    except Exception as e:
                        logger.exception("Error reading parquet in websocket loop: %s", e)
                        continue

                    # ensure types
                    if "window_start" in df.columns:
                        df["window_start"] = pd.to_datetime(df["window_start"])

                    updates = []
                    for road_id in df["road_id"].unique():
                        try:
                            road_data = df[df["road_id"] == road_id].sort_values("window_start").iloc[-1]
                        except Exception as e:
                            logger.warning("No rows for road_id %s: %s", road_id, e)
                            continue

                        updates.append(
                            {
                                "road_id": road_id,
                                "timestamp": road_data["window_start"].isoformat() if pd.notna(road_data["window_start"]) else str(road_data.get("window_start", "")),
                                "avg_speed": float(road_data["avg_speed"]),
                                "avg_congestion_index": float(road_data["avg_congestion_index"]),
                                "congestion_level": get_congestion_level(float(road_data["avg_congestion_index"])),
                            }
                        )

                    if updates:
                        await manager.broadcast({"type": "update", "data": updates})

            except Exception as e:
                logger.exception("Unexpected error in websocket broadcast loop: %s", e)
                # continue the while loop (don't crash the websocket)
                continue

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.exception("WebSocket endpoint error: %s", e)
    finally:
        # ensure the websocket is removed if still present
        manager.disconnect(websocket)


# ---------- Run (if executed directly) ----------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
