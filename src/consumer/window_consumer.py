# src/consumer/window_consumer.py
import os, sys, time, json, joblib, threading, socket
import urllib.request
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
from collections import defaultdict, deque

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from bloom_filter import CountingBloomFilter
from consumer.feature_extractor import extract_window_features, N_FEATURES
from consumer.feature_extractor import FEATURE_NAMES
import numpy as np
from pymongo import MongoClient
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except Exception:
    # Optional dependency; env vars can still be provided by the shell.
    pass

KAFKA = os.getenv("KAFKA_BOOTSTRAP", "localhost:29092")
TOPIC = os.getenv("FLOWS_TOPIC", "flows")
WINDOW_SECS = int(os.getenv("WINDOW_SECS", "5"))
SEQ_LEN = int(os.getenv("SEQ_LEN", "6"))
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "ids_group")
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "torch").strip().lower()  # torch | synthetic_sklearn
MAX_WINDOW_RECORDS = int(os.getenv("MAX_WINDOW_RECORDS", "0"))  # 0 => disabled

DASHBOARD_BACKEND = os.getenv("DASHBOARD_BACKEND", "http://127.0.0.1:5000").rstrip("/")
PUSH_LIVE_PREPROC = os.getenv("PUSH_LIVE_PREPROC", "1").strip() not in ("0", "false", "no", "off")

KAFKA_MAX_WAIT_SECS = float(os.getenv("KAFKA_MAX_WAIT_SECS", "0"))  # 0 => no retry


def _can_connect(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _parse_bootstrap_first(bootstrap: str) -> tuple[str, int]:
    # Supports typical "host:port,host2:port2" strings.
    first = (bootstrap or "").split(",", 1)[0].strip()
    if ":" not in first:
        return first, 9092
    host, port_str = first.rsplit(":", 1)
    return host.strip(), int(port_str.strip())


def _create_kafka_consumer() -> KafkaConsumer:
    host, port = _parse_bootstrap_first(KAFKA)
    start = time.time()

    while True:
        if host and _can_connect(host, port, timeout=1.0):
            break
        if KAFKA_MAX_WAIT_SECS <= 0 or (time.time() - start) >= KAFKA_MAX_WAIT_SECS:
            raise RuntimeError(
                "Kafka broker is not reachable.\n"
                f"  bootstrap: {KAFKA}\n"
                f"  probed: {host}:{port}\n\n"
                "If you are using Docker, start Kafka with:\n"
                "  docker compose -f docker\\docker-compose.yml up -d\n\n"
                "Then re-run this consumer. You can also override the broker via:\n"
                "  set KAFKA_BOOTSTRAP=localhost:29092\n"
                "(or KAFKA_BOOTSTRAP=kafka:9092 if running inside the compose network)."
            )
        time.sleep(1.0)

    try:
        return KafkaConsumer(
            TOPIC,
            bootstrap_servers=KAFKA,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="latest",
            group_id=KAFKA_GROUP_ID,
        )
    except NoBrokersAvailable as e:
        raise RuntimeError(
            "Kafka is reachable by TCP, but kafka-python still reports NoBrokersAvailable.\n"
            f"bootstrap: {KAFKA}\n\n"
            "Common causes: wrong listener/advertised listener, broker still starting, or firewall/AV interception.\n"
            "Try waiting a few seconds and re-running, or set KAFKA_MAX_WAIT_SECS=30."
        ) from e

# MongoDB Atlas connection (optional)
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "ids_db")
REQUIRE_MONGO = os.getenv("REQUIRE_MONGO", "0") == "1"

alerts_col = None
preproc_live_col = None
if MONGO_URI:
    try:
        mongo = MongoClient(MONGO_URI)
        db = mongo[MONGO_DB]
        alerts_col = db["alerts"]
        preproc_live_col = db["preprocessing_live"]
    except Exception as e:
        print(f"WARNING: MongoDB connection failed; alerts will not be saved. Error: {e}")
        alerts_col = None
        preproc_live_col = None
elif REQUIRE_MONGO:
    raise RuntimeError("MONGO_URI not set in environment or .env")
else:
    print("WARNING: MONGO_URI not set; running without MongoDB persistence (alerts will only print).")

MODELS_DIR = os.path.join(BASE_DIR, "models")


def _load_torch_model():
    import torch
    from models.cnn_bilstm_attn import CNN_BiLSTM_Attn

    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNN_BiLSTM_Attn(n_features=N_FEATURES, seq_len=SEQ_LEN, num_classes=len(le.classes_)).to(device)
    return scaler, le, device, model, torch


def _load_synthetic_sklearn_pipeline():
    from models.synthetic_sklearn import load_or_train_synthetic_pipeline

    try:
        bundle = load_or_train_synthetic_pipeline(MODELS_DIR)
    except ModuleNotFoundError as e:
        # Common on Windows when user runs the system/python-store interpreter instead of the venv.
        if str(e).strip() in ("No module named 'sklearn'", 'No module named "sklearn"'):
            raise RuntimeError(
                "MODEL_BACKEND=synthetic_sklearn requires scikit-learn, but 'sklearn' is not importable.\n\n"
                f"You are currently running: {sys.executable}\n\n"
                "Fix (recommended): run the consumer with the project venv interpreter, e.g.:\n"
                "  C:/Users/hp/ids-project/.venv/Scripts/python.exe src/consumer/window_consumer.py\n\n"
                "Or install scikit-learn into THIS interpreter:\n"
                f"  {sys.executable} -m pip install scikit-learn\n"
            ) from e
        raise
    pipeline = bundle["pipeline"]
    classes = bundle.get("classes") or list(getattr(pipeline[-1], "classes_", []))
    return pipeline, classes


if MODEL_BACKEND == "torch":
    scaler, le, device, model, torch = _load_torch_model()
elif MODEL_BACKEND == "synthetic_sklearn":
    sk_pipeline, sk_classes = _load_synthetic_sklearn_pipeline()
else:
    raise RuntimeError(
        "Unsupported MODEL_BACKEND. Use MODEL_BACKEND=torch or MODEL_BACKEND=synthetic_sklearn. "
        f"Got: {MODEL_BACKEND!r}"
    )

def _load_checkpoint_into_model(model_obj, ckpt_path, map_location):
    # Only used for torch backend.
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if hasattr(ckpt, "state_dict") and callable(getattr(ckpt, "state_dict")):
        state = ckpt.state_dict()
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

    # Backward compatibility: some checkpoints may have saved submodule weights
    # without the top-level prefix (e.g., conv.* instead of cnn.conv.*).
    remapped = {}
    for k, v in state.items():
        if k.startswith("cnn."):
            remapped[k] = v
        elif k.startswith("conv.") or k.startswith("bn.") or k.startswith("relu."):
            remapped[f"cnn.{k}"] = v
        elif k.startswith("att.") or k.startswith("lstm.") or k.startswith("fc."):
            remapped[k] = v
        else:
            # Keep as-is; may match directly.
            remapped[k] = v

    incompatible = model_obj.load_state_dict(remapped, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print("WARNING: Checkpoint did not fully match model architecture.")
        if incompatible.missing_keys:
            print("  Missing keys:", incompatible.missing_keys)
        if incompatible.unexpected_keys:
            print("  Unexpected keys:", incompatible.unexpected_keys)

if MODEL_BACKEND == "torch":
    _load_checkpoint_into_model(model, os.path.join(MODELS_DIR, "cnn_bilstm_attn.pt"), device)
    model.eval()

# state
buckets = defaultdict(list)
# Start time of the current active window per src.
window_start_ts = defaultdict(float)
window_queues = defaultdict(lambda: deque(maxlen=SEQ_LEN))
bloom = CountingBloomFilter(capacity=200000, error_rate=0.01)


def _save_live_preprocessing(src: str, flows: list[dict], feat: np.ndarray, window_start: float | None, window_end: float | None):
    if preproc_live_col is None:
        # Still allow HTTP push even without Mongo.
        pass

    # Keep the payload small: just named features + a short preview vector.
    named = {}
    for i, name in enumerate(FEATURE_NAMES[: min(10, len(FEATURE_NAMES))]):
        try:
            named[name] = float(feat[i])
        except Exception:
            named[name] = None

    doc = {
        "ts": float(time.time()),
        "src": src,
        "window_start": float(window_start) if window_start else None,
        "window_end": float(window_end) if window_end else None,
        "window_secs": WINDOW_SECS,
        "records": int(len(flows) if flows is not None else 0),
        "features": named,
        "feature_vector_preview": [float(x) for x in feat[:10].tolist()],
        "feature_vector": [float(x) for x in feat.tolist()],
    }
    if preproc_live_col is not None:
        try:
            # Upsert so each src shows latest values quickly.
            preproc_live_col.update_one({"src": src}, {"$set": doc}, upsert=True)
        except Exception as e:
            print(f"WARNING: Failed to write preprocessing_live to MongoDB: {e}")

    if PUSH_LIVE_PREPROC:
        try:
            body = json.dumps(doc).encode("utf-8")
            req = urllib.request.Request(
                f"{DASHBOARD_BACKEND}/preprocessing/live",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=1.0) as _resp:
                pass
        except Exception:
            # Best-effort; don't crash consumer if backend isn't up.
            pass

def on_window_complete(src, feat):
    sig = f"{src}:{int(feat[0])}:{int(feat[2])}"
    if sig in bloom:
        return
    window_queues[src].append(feat)
    if len(window_queues[src]) == SEQ_LEN:
        seq = np.stack(list(window_queues[src]))    # (SEQ_LEN, N_FEATURES)

        if MODEL_BACKEND == "torch":
            seq_scaled = scaler.transform(seq)
            x = torch.from_numpy(seq_scaled.astype("float32")).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
                pred = logits.argmax(dim=1).cpu().numpy()[0]
                label = le.inverse_transform([pred])[0]
                score = float(torch.softmax(logits, dim=1)[0, pred].cpu().numpy())
        else:
            # Synthetic sklearn backend: classify based on window features.
            # If SEQ_LEN > 1, average the sequence into one feature vector.
            feat_vec = np.mean(seq, axis=0).reshape(1, -1)
            proba = sk_pipeline.predict_proba(feat_vec)[0]
            idx = int(np.argmax(proba))
            label = str(sk_pipeline.classes_[idx])
            score = float(proba[idx])

            if os.getenv("PRINT_PREDICTIONS", "0") == "1":
                print(
                    f"PRED src={src} label={label} score={score:.4f} "
                    f"window_bytes={int(feat[0])} pkts={int(feat[1])} syn={int(feat[3])}"
                )

            if label.lower() not in ("benign", "normal"):
                alert = {
                    "src": src,
                    "label": label,
                    "score": score,
                    "ts": time.time(),
                    "window_features": seq.tolist()
                }
                if alerts_col is not None:
                    try:
                        alerts_col.insert_one(alert)
                    except Exception as e:
                        print(f"WARNING: Failed to write alert to MongoDB: {e}")
                print("ALERT ->", alert)
                bloom.add(sig)
            else:
                bloom.add(sig)

def window_flush_worker():
    while True:
        now = time.time()
        for src in list(buckets.keys()):
            flows = buckets.get(src)
            if not flows:
                continue

            start = window_start_ts.get(src, 0.0) or 0.0
            window_elapsed = (now - start) if start else 0.0
            too_many = MAX_WINDOW_RECORDS > 0 and len(flows) >= MAX_WINDOW_RECORDS

            # Flush on fixed time windows (preferred) and optionally if the bucket grows too large.
            if (start and window_elapsed >= WINDOW_SECS) or too_many:
                flows = buckets.pop(src, [])
                window_start_ts.pop(src, None)
                feat = extract_window_features(flows)
                _save_live_preprocessing(src, flows, feat, start, now)
                on_window_complete(src, feat)
        time.sleep(0.5)


def _main():
    consumer = _create_kafka_consumer()

    t = threading.Thread(target=window_flush_worker, daemon=True)
    t.start()
    print("Window consumer started — listening to flows and writing alerts to Atlas...")
    for msg in consumer:
        rec = msg.value
        # Optional demo shortcut: force an alert for synthetic attack messages.
        # Default is OFF so detections come from the ML model.
        if os.getenv("FORCE_SYNTHETIC_ATTACK", "0") == "1" and rec.get("mode") == "attack":
            alert = {
                "src": rec.get("src", "synthetic"),
                "label": "SYNTHETIC_ATTACK",
                "score": 0.99,
                "ts": time.time(),
            }
            if alerts_col is not None:
                try:
                    alerts_col.insert_one(alert)
                except Exception as e:
                    print(f"WARNING: Failed to write alert to MongoDB: {e}")
            print("ALERT (forced):", alert)
            continue

        src = rec.get("src")
        ts = rec.get("ts", time.time())
        if src is None:
            continue
        buckets[src].append(rec)
        # Initialize a fixed window start time for this src if this is the first record.
        if not window_start_ts.get(src):
            window_start_ts[src] = float(ts)


if __name__ == "__main__":
    _main()

