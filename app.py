"""
app.py — บริการทำนายความน่าจะเป็นชนะของไม้ (FastAPI)

  GET  /health    ตรวจสถานะ + เวอร์ชันโมเดล (ไม่ต้องใช้ API key)
  POST /predict   รับค่าดิบจาก n8n -> คืน prob (ความน่าจะเป็นที่ netPnl > 0)

ENV:
  API_KEY            (จำเป็น) คีย์ที่ n8n ส่งมาใน header  x-api-key
  ALLOW_NO_AUTH      ตั้งเป็น "1" เพื่อปิดการตรวจคีย์ (ใช้เฉพาะตอนทดสอบในเครื่อง)
  MAX_MISSING_RATIO  สัดส่วน feature ที่ขาดได้สูงสุด (ค่าเริ่มต้น 0.2 = 20%)
  MODEL_PATH         path ของ model.json

n8n ควรตรวจ: ready == true, enforceOk == true, unknownKeys ว่าง, prob >= threshold
"""
import hmac
import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Body, FastAPI, Header, HTTPException

from features import MODEL_FEATURES, RAW_KEYS, Z_CLIP, engineer

log = logging.getLogger("btc-ml-api")
logging.basicConfig(level=logging.INFO)

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "model.json"))
API_KEY = os.environ.get("API_KEY", "")
ALLOW_NO_AUTH = os.environ.get("ALLOW_NO_AUTH", "") == "1"


def _read_ratio(name: str, default: float) -> float:
    try:
        v = float(os.environ.get(name, default))
        return v if 0.0 <= v <= 1.0 else default
    except ValueError:
        return default


MAX_MISSING_RATIO = _read_ratio("MAX_MISSING_RATIO", 0.2)

app = FastAPI(title="BTC ML API", version="1.2")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def is_valid_number(v: Any) -> bool:
    """ตัวเลขจริงที่จำกัด (ไม่ใช่ None / bool / str / NaN / Infinity)"""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    return math.isfinite(v)


def load_model(path: str = MODEL_PATH) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        log.error("cannot load model at %s", path)
        return None

    if m.get("features", []) != MODEL_FEATURES:
        # ฟีเจอร์ในโมเดลไม่ตรงกับโค้ด -> ไม่ใช้ เพื่อกันทำนายผิดเงียบ ๆ
        log.error("model features do not match features.py")
        return None

    coef = m.get("coefficients", m.get("coef", {}))
    try:
        if not is_valid_number(m.get("intercept")):
            raise ValueError("intercept")
        for k in m["features"]:
            if not is_valid_number(m["mean"][k]) or not is_valid_number(m["std"][k]):
                raise ValueError(f"mean/std of {k}")
            if not is_valid_number(coef.get(k, 0.0)):
                raise ValueError(f"coef of {k}")
        z = m.get("zClip", Z_CLIP)
        if not is_valid_number(z) or z <= 0:
            raise ValueError("zClip")
    except (KeyError, TypeError, ValueError) as e:
        log.error("model file is incomplete or invalid: %s", e)
        return None
    return m


MODEL = load_model()

if not API_KEY and not ALLOW_NO_AUTH:
    log.warning("API_KEY is not set -> /predict will return 503 (fail closed). "
                "Set API_KEY in Render Environment, or ALLOW_NO_AUTH=1 for local testing only.")
if ALLOW_NO_AUTH:
    log.warning("ALLOW_NO_AUTH=1 -> /predict is OPEN to anyone. Do not use in production.")


def check_key(x_api_key: Optional[str]) -> None:
    if not API_KEY:
        if ALLOW_NO_AUTH:
            return
        raise HTTPException(status_code=503, detail="server auth is not configured")
    if not x_api_key or not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="invalid api key")


def predict_prob(model: Dict[str, Any], raw: Dict[str, Any]) -> Tuple[Optional[float], List[str]]:
    feats = engineer(raw)                      # raise ValueError ถ้า side ผิด
    coef_dict = model.get("coefficients", model.get("coef", {}))
    z_clip = float(model.get("zClip", Z_CLIP))

    missing: List[str] = []
    logit = float(model["intercept"])

    for k in model["features"]:
        v = feats.get(k)
        if not is_valid_number(v):
            missing.append(k)                  # None / NaN / Inf / ชนิดข้อมูลผิด
            continue
        std = max(float(model["std"][k]), 1e-9)
        z = max(-z_clip, min(z_clip, (float(v) - float(model["mean"][k])) / std))
        logit += float(coef_dict.get(k, 0.0)) * z

    if not math.isfinite(logit):
        return None, missing

    prob = 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, logit))))
    return prob, missing


def response(model: Optional[Dict[str, Any]], *, prob: Optional[float] = None, ready: bool = False,
             missing: Optional[List[str]] = None, unknown: Optional[List[str]] = None,
             note: str = "") -> Dict[str, Any]:
    return {
        "prob": None if prob is None else round(prob, 4),
        "ready": ready,
        "enforceOk": bool(model.get("enforceOk", False)) if (model and ready) else False,
        "modelVersion": model.get("modelVersion") if model else None,
        "nTrain": model.get("nTrain", 0) if model else 0,
        "missing": missing or [],
        "unknownKeys": unknown or [],
        "note": note,
    }


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/")
@app.get("/health")
def health():
    return {
        "status": "ok",
        "ready": MODEL is not None,
        "modelVersion": MODEL.get("modelVersion") if MODEL else None,
        "nTrain": MODEL.get("nTrain") if MODEL else 0,
        "enforceOk": bool(MODEL.get("enforceOk", False)) if MODEL else False,
        "authConfigured": bool(API_KEY),
        "maxMissingRatio": MAX_MISSING_RATIO,
    }


@app.post("/predict")
def predict(payload: Dict[str, Any] = Body(...), x_api_key: Optional[str] = Header(default=None)):
    check_key(x_api_key)

    if MODEL is None:
        return response(None, note="no model loaded")

    # คีย์ที่ไม่อยู่ใน RAW_KEYS มักแปลว่า n8n สะกดผิดหรือเปลี่ยนชื่อ -> รายงานให้ n8n เห็น
    unknown = sorted(k for k in payload if k not in RAW_KEYS)

    try:
        prob, missing = predict_prob(MODEL, payload)
    except Exception as e:  # เช่น side ผิด หรือ payload ผิดรูปแบบ
        log.warning("predict rejected: %s", e)
        return response(MODEL, unknown=unknown, note=f"bad input: {e}")

    n_feat = len(MODEL["features"])
    if n_feat and len(missing) / n_feat > MAX_MISSING_RATIO:
        return response(
            MODEL, missing=missing, unknown=unknown,
            note=f"too many missing features: {len(missing)}/{n_feat} (max {MAX_MISSING_RATIO:.0%})",
        )

    if prob is None or not math.isfinite(prob):
        return response(MODEL, missing=missing, unknown=unknown, note="invalid prediction")

    return response(MODEL, prob=prob, ready=True, missing=missing, unknown=unknown)
