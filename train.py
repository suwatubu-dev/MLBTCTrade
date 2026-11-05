#!/usr/bin/env python3
"""
train.py — เทรนโมเดลจากบันทึกการเทรดใน Google Sheet แล้วสร้าง model.json

ขั้นตอน
  1) ใน Google Sheet: File -> Download -> CSV (เฉพาะแท็บ Sheet1)
  2) python train.py --csv trades.csv
  3) นำ model.json ไปไว้ในโฟลเดอร์นี้ แล้ว commit/push ให้ Render deploy ใหม่

ข้อมูลที่ใช้: แถวที่ปิดแล้ว (status = CLOSED) และมี netPnl
  - ฟีเจอร์: JSON ในคอลัมน์ mlFeatures (หรือ r_ratios ตามค่าเริ่มต้นของ workflow นี้)
  - ป้ายคำตอบ (label): netPnl > 0  => 1 (ชนะ), มิฉะนั้น 0

โมเดล: logistic regression (L2) บนฟีเจอร์ที่ทำให้มีทิศทางแล้ว (ดู features.py)
ประเมินด้วย walk-forward (เรียงตามเวลา ไม่สุ่ม) เพื่อไม่ให้อนาคตรั่วเข้าอดีต
"""
import argparse
import datetime as dt
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from features import MODEL_FEATURES, engineer, to_num

FEATURE_COLS = ["mlFeatures", "r_ratios"]
Z_CLIP = 5.0


# ── อ่านข้อมูล ────────────────────────────────────────────────────────────────

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
    """แถวเก่าที่ยังไม่มี mlFeatures: ประกอบเท่าที่มีจากคอลัมน์เดิม (ขาดหลายตัว -> ใช้ค่าเฉลี่ยตอนเทรน)"""
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


def load_dataset(csv_path, use_legacy, pnl_threshold):
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
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


# ── โมเดล ─────────────────────────────────────────────────────────────────────

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="model.json")
    ap.add_argument("--min-train", type=int, default=100, help="จำนวนไม้ขั้นต่ำที่ยอมสร้างโมเดล")
    ap.add_argument("--min-enforce", type=int, default=200, help="จำนวนไม้ขั้นต่ำก่อนอนุญาตให้ใช้บล็อกเทรด")
    ap.add_argument("--min-auc", type=float, default=0.58)
    ap.add_argument("--C", type=float, default=0.3, help="ค่าน้อย = กด regularization แรง (กัน overfit)")
    ap.add_argument("--pnl-threshold", type=float, default=0.0)
    ap.add_argument("--use-legacy", action="store_true", help="ใช้แถวเก่าที่ไม่มี mlFeatures ด้วย (ฟีเจอร์ไม่ครบ)")
    ap.add_argument("--force", action="store_true", help="สร้างโมเดลแม้ข้อมูลน้อย (เพื่อทดสอบท่อเท่านั้น)")
    a = ap.parse_args()

    data, skipped = load_dataset(a.csv, a.use_legacy, a.pnl_threshold)
    n = len(data)
    print(f"แถวที่ใช้เทรดได้: {n} | ข้าม: {skipped}")
    if n == 0:
        print("ไม่มีข้อมูลที่ใช้ได้ (ต้องมีสถานะ CLOSED, netPnl และ JSON ฟีเจอร์ในคอลัมน์ mlFeatures/r_ratios)")
        return 2

    y = data["y"].to_numpy()
    X = data[MODEL_FEATURES].astype(float)
    print(f"ชนะ {int(y.sum())} / แพ้ {int(n - y.sum())}  (อัตราชนะ {y.mean():.1%})")

    if n < a.min_train and not a.force:
        print(f"\nข้อมูลน้อยเกินไป ({n} < {a.min_train}) จึงไม่สร้าง model.json")
        print("โมเดลที่เทรนจากข้อมูลระดับนี้จะจำสัญญาณรบกวน ไม่ใช่รูปแบบจริง — เก็บข้อมูลต่อแล้วรันใหม่")
        return 2

    metrics = walk_forward(X, y, a.C) if n >= 8 else None
    if metrics:
        auc = metrics["auc"]
        print("\n== ผล walk-forward (ข้อมูลที่โมเดลไม่เคยเห็น) ==")
        print(f"  ตัวอย่างนอกกลุ่มเทรน : {metrics['n_oos']}")
        print(f"  AUC                  : {auc:.3f}" if auc is not None else "  AUC                  : n/a")
        print(f"  Brier (ต่ำ=ดี)       : {metrics['brier']:.4f}  | baseline {metrics['brier_baseline']:.4f}")
        print(f"  ความแม่น @0.5        : {metrics['accuracy@0.5']:.1%} | ทายตามเสียงข้างมาก {metrics['accuracy_majority_baseline']:.1%}")

    enforce_ok = bool(
        metrics and metrics["auc"] is not None
        and n >= a.min_enforce and metrics["auc"] >= a.min_auc
        and metrics["brier_improvement"] > 0
    )
    print(f"\nอนุญาตให้บล็อกเทรด (enforce_ok): {enforce_ok}")
    if not enforce_ok:
        print("  -> ควรรันโหมด SHADOW (บันทึกอย่างเดียว ไม่บล็อก) ต่อไป")

    mean, std, lr = fit_model(X, y, a.C)          # เทรนใหม่ด้วยข้อมูลทั้งหมดเพื่อนำไปใช้งาน
    coef = dict(zip(MODEL_FEATURES, lr.coef_[0].tolist()))
    print("\n== น้ำหนักฟีเจอร์ (ค่า z, บวก = เพิ่มโอกาสชนะ) ==")
    for k, v in sorted(coef.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {k:15s} {v:+.3f}")

    model = {
        "version": dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S"),
        "features": MODEL_FEATURES,
        "mean": {k: float(mean[k]) for k in MODEL_FEATURES},
        "std": {k: float(std[k]) for k in MODEL_FEATURES},
        "coef": coef,
        "intercept": float(lr.intercept_[0]),
        "n_train": int(n),
        "base_rate": float(y.mean()),
        "label": f"netPnl > {a.pnl_threshold}",
        "C": a.C,
        "metrics": metrics,
        "enforce_ok": enforce_ok,
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)
    print(f"\nบันทึก {a.out} (version {model['version']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
