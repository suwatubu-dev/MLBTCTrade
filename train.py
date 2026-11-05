#!/usr/bin/env python3
"""
train.py — ดึงข้อมูลจาก Google Sheets ผ่าน Service Account มาเทรนโมเดล ML แล้วสร้าง model.json
"""
import argparse
import datetime as dt
import json
import os
import sys

import gspread
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from features import MODEL_FEATURES, engineer, to_num

FEATURE_COLS = ["mlFeatures", "r_ratios"]
Z_CLIP = 5.0

# ── 0. ดึงข้อมูลจาก Google Sheets ──────────────────────────────────────────────

def fetch_data_from_google_sheets(sheet_name="Sheet1"):
    gcp_json_str = os.environ.get("GCP_CREDENTIALS_JSON")
    if not gcp_json_str:
        raise ValueError("❌ ไม่พบตัวแปร GCP_CREDENTIALS_JSON ใน Environment Variables / GitHub Secrets")
    
    creds_dict = json.loads(gcp_json_str)
    gc = gspread.service_account_from_dict(creds_dict)
    
    # ⚠️ ใส่ชื่อไฟล์ Google Sheet ของคุณตรงนี้ (หรือใช้ URL/Key แทนได้)
    # เช่น gc.open("BTC_Trading_Log").worksheet(sheet_name)
    sh = gc.open_by_key(creds_dict.get("sheet_id", "")) if "sheet_id" in creds_dict else gc.open_all()[0]
    worksheet = sh.worksheet(sheet_name) if sheet_name in [ws.title for ws in sh.worksheets()] else sh.get_worksheet(0)
    
    records = worksheet.get_all_records()
    return pd.DataFrame(records)


# ── 1. อ่านและแปลงข้อมูล ─────────────────────────────────────────────────────────

def parse_json_cell(v):
    s = str(v or "").strip()
    if not s.startswith("{"):
        return None
    try:
        d = json.loads(s)
        return d if isinstance(d, dict) else None
    except json.JSONDecodeError:
        return None


def legacy_raw(row):
    side = "SHORT" if (str(row.get("direction", "")).upper() == "BEARISH"
                       or str(row.get("side", "")).upper() == "SELL") else "LONG"
    entry, atr = to_num(row.get("entry")), to_num(row.get("atr"))
    return {
        "side": side,
        "rsi": row.get("rsi"), "macdLine": row.get("macdLine"),
        "macdHistogram": row.get("macdHistogram"), "atr": row.get("atr"),
        "atrPct": (atr / entry * 100) if (atr and entry) else None,
        "volRatio": row.get("volRatio"), "edge": row.get("scoreRatio"),
    }


def load_dataset_from_df(df, use_legacy, pnl_threshold):
    rows, skipped = [], {"not_closed": 0, "no_pnl": 0, "no_features": 0}

    for i, r in df.iterrows():
        row = r.to_dict()
        if "status" in row and str(row["status"]).strip().upper() != "CLOSED":
            skipped["not_closed"] += 1
            continue
        pnl = to_num(row.get("netPnl"))
        if pnl is None:
            skipped["no_pnl"] += 1
            continue
        raw = None
        for col in FEATURE_COLS:
            if col in row:
                raw = parse_json_cell(row[col])
                if raw:
                    break
        if raw is None and use_legacy:
            raw = legacy_raw(row)
        if raw is None:
            skipped["no_features"] += 1
            continue
        rows.append({
            "t": row.get("openTime", ""), "idx": i, "y": 1 if pnl > pnl_threshold else 0,
            "pnl": pnl, **engineer(raw),
        })

    data = pd.DataFrame(rows)
    if data.empty:
        return data, skipped
    data["t"] = pd.to_datetime(data["t"], errors="coerce", format="mixed")
    data = data.sort_values(["t", "idx"], kind="stable", na_position="last").reset_index(drop=True)
    return data, skipped


# ── 2. เทรนและประเมินโมเดล ───────────────────────────────────────────────────

def fit_model(X, y, C):
    mean = X.mean(skipna=True).fillna(0.0)
    std = X.std(ddof=0, skipna=True).fillna(1.0)
    std = std.where(std > 1e-9, 1.0)
    Z = ((X - mean) / std).clip(-Z_CLIP, Z_CLIP).fillna(0.0)
    lr = LogisticRegression(C=C, max_iter=2000)
    lr.fit(Z.values, y)
    return mean, std, lr


def predict_model(mean, std, lr, X):
    Z = ((X - mean) / std).clip(-Z_CLIP, Z_CLIP).fillna(0.0)
    return lr.predict_proba(Z.values)[:, 1]


def walk_forward(X, y, C):
    n = len(y)
    n_splits = max(2, min(5, n // 25))
    ys, ps, base = [], [], []
    for tr, te in TimeSeriesSplit(n_splits=n_splits).split(X):
        if len(set(y[tr])) < 2:
            continue
        mean, std, lr = fit_model(X.iloc[tr], y[tr], C)
        p = predict_model(mean, std, lr, X.iloc[te])
        ys.extend(y[te]); ps.extend(p); base.extend([y[tr].mean()] * len(te))
    if not ys:
        return None
    ys, ps, base = np.array(ys), np.array(ps), np.array(base)
    auc = float(roc_auc_score(ys, ps)) if len(set(ys)) == 2 else None
    brier = float(brier_score_loss(ys, ps))
    brier_base = float(brier_score_loss(ys, base))
    maj = max(ys.mean(), 1 - ys.mean())
    return {
        "n_oos": int(len(ys)), "auc": auc,
        "brier": brier, "brier_baseline": brier_base,
        "brier_improvement": brier_base - brier,
        "accuracy@0.5": float(((ps >= 0.5) == (ys == 1)).mean()),
        "accuracy_majority_baseline": float(maj),
    }


# ── 3. ฟังก์ชันหลัก ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="ถ้าใส่ path csv จะอ่านจากไฟล์แทน Google Sheets")
    ap.add_argument("--out", default="model.json")
    ap.add_argument("--min-train", type=int, default=100)
    ap.add_argument("--min-enforce", type=int, default=200)
    ap.add_argument("--min
