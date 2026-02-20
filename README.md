# Real-Time Traffic Prediction

This project provides a real-time traffic congestion prediction system using machine learning models (LSTM, GRU, Random Forest) and a FastAPI backend. It ingests live traffic data, processes it, and serves predictions via a REST API and a web dashboard.

## Features

- **Data Ingestion**: Collects live traffic data from TomTom API and streams it using Kafka.
- **Data Processing**: Aggregates and preprocesses data for model training and inference.
- **Model Training**: Supports LSTM, GRU, and Random Forest models for traffic prediction.
- **API Backend**: FastAPI server provides endpoints for current, historical, and predicted traffic data.
- **Web Dashboard**: Frontend dashboard for live visualization (see `src/dashboard/frontend/`).

## Folder Structure

```
.
├── src/
│   ├── dashboard/
│   │   ├── fastapi_backend.py
│   │   └── frontend/
│   ├── ingestion/
│   │   └── tomtom_producer.py
│   └── models/
├── notebooks/
│   └── Realtime_Traffic_Predictor.ipynb
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Setup

### 1. Install Dependencies

```sh
pip install -r requirements.txt
```

### 2. Environment Variables

Copy `.env.example` to `.env` and set the required variables (e.g., `TOMTOM_API_KEY`, `KAFKA_BROKER`, etc.).

### 3. Start Kafka & Zookeeper

```sh
docker-compose up -d
```

### 4. Start Data Ingestion

```sh
python src/ingestion/tomtom_producer.py
```

### 5. Start FastAPI Backend

```sh
uvicorn src.dashboard.fastapi_backend:app --reload
```

### 6. Access Dashboard

Open `src/dashboard/frontend/index.html` in your browser.

## API Endpoints

- `GET /api/roads` — List available road IDs.
- `GET /api/current/{road_id}` — Get current traffic data for a road.
- `POST /api/predict` — Predict next 5 minutes congestion (provide `road_id` and `model_name`).
- `GET /api/historical/{road_id}` — Get historical data for a road.
- `GET /health` — Health check.

## Model Training

See [notebooks/Realtime_Traffic_Predictor.ipynb](notebooks/Realtime_Traffic_Predictor.ipynb) for model training and evaluation.

## Notes

- Ensure Kafka and Zookeeper are running before starting ingestion.
- Place trained models in `src/models/` (e.g., `.keras` and `.joblib` files).
- Data files are expected in `data/gold/` as parquet files.

---

**Author:**  
*Real-Time Traffic Prediction Project*