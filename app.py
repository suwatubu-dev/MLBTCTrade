"""
app.py — บริการทำนายความน่าจะเป็นชนะของไม้ (FastAPI)

  GET  /health    ตรวจสถานะ + เวอร์ชันโมเดล
  POST /predict   รับค่าดิบจาก n8n -> คืน prob (ความน่าจะเป็นที่ netPnl > 0)

ทำนายด้วยคณิตศาสตร์ล้วน (logistic regression จาก model.json) ไม่ต้องใช้ numpy/sklearn ตอนรัน
ถ้าไม่มี model.json จะตอบ ready=false และ prob=null (n8n จะข้ามไป ไม่กระทบการเทรด)
"""
import hmac
import json
import math
import os
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, Header, HTTPException

from features import MODEL_FEATURES, engineer

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "model.json"))
API_KEY = os.environ.get("API_KEY", "")
Z_CLIP = 5.0

app = FastAPI(title="BTC ML API", version="1.0")


def load_model(path: str = MODEL_PATH) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    feats = m.get("features", [])
    if feats != MODEL_FEATURES:
        # ฟีเจอร์ในโมเดลไม่ตรงกับโค้ด -> ไม่ใช้ เพื่อกันทำนายผิดเงียบ ๆ
        return None
    return m


MODEL = load_model()


def check_key(x_api_key: Optional[str]):
    if not API_KEY:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="invalid api key")


def predict_prob(model: Dict[str, Any], raw: Dict[str, Any]):
    feats = engineer(raw)
    missing = [k for k in model["features"] if feats.get(k) is None]
    logit = float(model["intercept"])
    for k in model["features"]:
        v = feats.get(k)
        if v is None:
            continue                          # ขาดค่า = ใช้ค่าเฉลี่ย (z = 0)
        std = max(float(model["std"][k]), 1e-9)
        z = max(-Z_CLIP, min(Z_CLIP, (v - float(model["mean"][k])) / std))
        logit += float(model["coef"][k]) * z
    prob = 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, logit))))
    return prob, missing


@app.get("/")
@app.get("/health")
def health():
    return {
        "status": "ok",
        "ready": MODEL is not None,
        "modelVersion": MODEL.get("version") if MODEL else None,
        "nTrain": MODEL.get("n_train") if MODEL else 0,
    }


@app.post("/predict")
def predict(payload: Dict[str, Any] = Body(...), x_api_key: Optional[str] = Header(default=None)):
    check_key(x_api_key)
    if MODEL is None:
        return {"prob": None, "ready": False, "enforceOk": False, "modelVersion": None,
                "nTrain": 0, "missing": [], "note": "no model loaded"}
    prob, missing = predict_prob(MODEL, payload)
    return {
        "prob": round(prob, 4),
        "ready": True,
        "enforceOk": bool(MODEL.get("enforce_ok", False)),
        "modelVersion": MODEL.get("version"),
        "nTrain": MODEL.get("n_train"),
        "missing": missing,
        "note": "",
    }
