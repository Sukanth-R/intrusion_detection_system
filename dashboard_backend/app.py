# dashboard_backend/app.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
import os
import json
import urllib.request
import urllib.error
from collections import deque
from dotenv import load_dotenv
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "ids_db")
client = None
db = None
alerts_col = None
preproc_live_col = None
if MONGO_URI:
    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB]
        alerts_col = db["alerts"]
        preproc_live_col = db["preprocessing_live"]
    except Exception as e:
        print(f"WARNING: MongoDB connection failed; running without DB. Error: {e}")
        client = None
        db = None
        alerts_col = None
        preproc_live_col = None

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

INFERENCE_API = os.getenv("INFERENCE_API", "http://localhost:8000")

# In-memory fallback for live preprocessing, so the dashboard can still show
# real-time feature extraction even when MongoDB isn't configured.
_live_preproc = deque(maxlen=int(os.getenv("LIVE_PREPROC_MAX", "200")))


class LivePreprocIn(BaseModel):
    ts: float | None = None
    src: str | None = None
    window_start: float | None = None
    window_end: float | None = None
    window_secs: float | None = None
    records: int | None = None
    features: dict | None = None
    feature_vector_preview: list[float] | None = None
    feature_vector: list[float] | None = None

@app.get("/alerts")
def get_alerts(limit: int = 100):
    if alerts_col is None:
        return []
    docs = list(alerts_col.find().sort("ts", -1).limit(limit))
    for d in docs:
        d["_id"] = str(d["_id"])
    return docs

@app.get("/health")
def health():
    return {"status": "ok", "mongo": bool(alerts_col is not None)}


@app.get("/preprocessing")
def preprocessing():
    url = f"{INFERENCE_API.rstrip('/')}/preprocessing"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            data = resp.read().decode("utf-8")
            return json.loads(data)
    except Exception as e:
        # Fallback keeps the dashboard usable even if inference API isn't running.
        return {
            "error": "inference_api_unavailable",
            "detail": str(e),
            "steps": [
                "Extract per-window numeric features (78-dim vector).",
                "Stack last SEQ_LEN windows into a sequence.",
                "Apply saved scaler to normalize features.",
                "Predict label with CNN+BiLSTM+Attention model.",
            ],
        }


@app.get("/preprocessing/live")
def preprocessing_live(limit: int = 20):
    lim = max(1, min(int(limit or 20), 200))
    # Prefer MongoDB if available and has data; otherwise use in-memory buffer.
    if preproc_live_col is not None:
        docs = list(preproc_live_col.find().sort("ts", -1).limit(lim))
        if docs:
            for d in docs:
                d["_id"] = str(d["_id"])
            return docs
    return list(_live_preproc)[-lim:][::-1]


@app.post("/preprocessing/live")
def ingest_preprocessing_live(payload: LivePreprocIn):
    doc = payload.model_dump()
    # Add basic defaults.
    if doc.get("ts") is None:
        import time
        doc["ts"] = float(time.time())

    _live_preproc.append(doc)

    # If MongoDB is configured, upsert by src for fast reads.
    if preproc_live_col is not None and doc.get("src"):
        try:
            preproc_live_col.update_one({"src": doc["src"]}, {"$set": doc}, upsert=True)
        except Exception as e:
            # Keep endpoint best-effort; UI can still use memory.
            return {"status": "ok", "stored": "memory", "mongo_error": str(e)}

    return {"status": "ok", "stored": ("mongo" if preproc_live_col is not None else "memory")}
