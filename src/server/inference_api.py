# src/server/inference_api.py
from fastapi import FastAPI
import joblib, numpy as np, torch, os, sys
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.cnn_bilstm_attn import CNN_BiLSTM_Attn
from consumer.feature_extractor import FEATURE_NAMES
from dotenv import load_dotenv
load_dotenv()

SEQ_LEN = int(os.getenv("SEQ_LEN", "6"))
N_FEATURES = int(os.getenv("N_FEATURES", "78"))

ARTIFACT_DIR = os.getenv("ARTIFACT_DIR") or os.path.join(BASE_DIR, "models")

scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.joblib"))
le = joblib.load(os.path.join(ARTIFACT_DIR, "label_encoder.joblib"))

device = "cuda" if torch.cuda.is_available() else "cpu"

_weights_env = os.getenv("MODEL_WEIGHTS")
_weight_candidates = [
    _weights_env,
    os.path.join(ARTIFACT_DIR, "cnn_bilstm_attn_best.pt"),
    os.path.join(ARTIFACT_DIR, "cnn_bilstm_attn.pt"),
]
WEIGHTS_PATH = next((p for p in _weight_candidates if p and os.path.exists(p)), None)
if WEIGHTS_PATH is None:
    raise RuntimeError(
        "Model weights not found. Set MODEL_WEIGHTS or place weights at "
        f"{os.path.join(ARTIFACT_DIR, 'cnn_bilstm_attn.pt')}"
    )

_model = None
_model_load_error = None


def _get_model():
    global _model, _model_load_error
    if _model is not None:
        return _model
    if _model_load_error is not None:
        raise _model_load_error

    try:
        m = CNN_BiLSTM_Attn(
            n_features=N_FEATURES, seq_len=SEQ_LEN, num_classes=len(le.classes_)
        ).to(device)
        m.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        m.eval()
        _model = m
        return _model
    except Exception as e:
        _model_load_error = RuntimeError(f"Failed to load model weights: {e}")
        raise _model_load_error

app = FastAPI()

class WindowIn(BaseModel):
    window: list


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": device,
        "weights": os.path.basename(WEIGHTS_PATH),
        "model_loaded": _model is not None,
        "model_error": (str(_model_load_error) if _model_load_error is not None else None),
    }


@app.get("/preprocessing")
def preprocessing():
    scaler_type = type(scaler).__name__
    scaler_features_in = getattr(scaler, "n_features_in_", None)
    classes = getattr(le, "classes_", None)
    classes_preview = None
    if classes is not None:
        classes_preview = list(map(str, classes[: min(20, len(classes))]))

    return {
        "seq_len": SEQ_LEN,
        "n_features": N_FEATURES,
        "feature_names": FEATURE_NAMES,
        "steps": [
            "Extract per-window numeric features (78-dim vector).",
            "Stack last SEQ_LEN windows into a sequence for the model.",
            "Apply the saved scaler (scaler.joblib) to normalize features.",
            "Run CNN+BiLSTM+Attention model to predict class index.",
            "Decode class index to label using label_encoder.joblib.",
        ],
        "artifacts": {
            "scaler": {"type": scaler_type, "n_features_in": scaler_features_in},
            "label_encoder": {
                "n_classes": (len(classes) if classes is not None else None),
                "classes_preview": classes_preview,
                "classes_truncated": (len(classes) > 20 if classes is not None else None),
            },
            "model": {"device": device, "architecture": "CNN_BiLSTM_Attn"},
        },
    }

@app.post("/predict")
def predict(payload: WindowIn):
    window = np.array(payload.window, dtype=float)
    X = scaler.transform(window)
    x = torch.from_numpy(X.astype('float32')).unsqueeze(0).to(device)
    model = _get_model()
    with torch.no_grad():
        logits = model(x)
        p = logits.argmax(dim=1).cpu().numpy()[0]
        label = le.inverse_transform([p])[0]
    return {"label": label}
