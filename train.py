#!/usr/bin/env python3
"""
train.py — ดึงข้อมูลจาก Google Sheets (หรือ CSV) มาเทรนโมเดล ML แล้วสร้าง model.json

ENV (โหมด Google Sheets):
  GCP_CREDENTIALS_JSON  JSON ของ service account (อาจมีคีย์ "sheet_id" ปนอยู่ก็ได้ เหมือนเดิม)
  SHEET_ID              (แนะนำ) id ของ Google Sheet — ถ้าไม่ตั้งจะใช้ sheet_id ใน JSON; ไม่มีทั้งคู่ = error
  SHEET_NAME            ชื่อ worksheet (ค่าเริ่มต้น "Sheet1") — ไม่พบ = error (ไม่เดาชีทอื่นอีกต่อไป)

enforceOk = จำนวนข้อมูลพอ AND ผ่านเกณฑ์คุณภาพ walk-forward (AUC, Brier ดีกว่า baseline, จำนวน OOS)
"""
import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from features import MODEL_FEATURES, Z_CLIP, engineer, to_num

FEATURE_COLS = ["mlFeatures", "r_ratios"]
NET_PNL_COLS = ["netPnl", "net_pnl", "pnl"]      # ใช้เฉพาะ PnL สุทธิ (หลังหักค่าธรรมเนียม)
GROSS_PNL_COLS = ["grossPnl"]                   # ใช้เมื่อสั่ง --allow-gross เท่านั้น


# ── 0. ดึงข้อมูลจาก Google Sheets ──────────────────────────────────────────────

def fetch_data_from_google_sheets():
    import gspread  # import ตรงนี้ เพื่อให้โหมด --csv ไม่ต้องติดตั้ง gspread

    gcp_json_str = os.environ.get("GCP_CREDENTIALS_JSON")
    if not gcp_json_str:
        raise SystemExit("❌ ไม่พบตัวแปร GCP_CREDENTIALS_JSON ใน Environment Variables / GitHub Secrets")
    creds_dict = json.loads(gcp_json_str)

    sheet_id = os.environ.get("SHEET_ID") or creds_dict.get("sheet_id")
    if not sheet_id:
        raise SystemExit("❌ ไม่พบ SHEET_ID (ตั้ง env SHEET_ID หรือใส่ sheet_id ใน GCP_CREDENTIALS_JSON) "
                         "— ไม่เดาไฟล์แรกที่เข้าถึงได้อีกต่อไป")
    sheet_name = os.environ.get("SHEET_NAME", "Sheet1")

    gc = gspread.service_account_from_dict(creds_dict)
    sh = gc.open_by_key(sheet_id)
    names = [ws.title for ws in sh.worksheets()]
    print(f"📗 Spreadsheet: '{sh.title}' | worksheets: {names}")
    if sheet_name not in names:
        raise SystemExit(f"❌ ไม่พบ worksheet ชื่อ '{sheet_name}' (มี: {names}) — ตั้ง env SHEET_NAME ให้ถูกต้อง")

    df = pd.DataFrame(sh.worksheet(sheet_name).get_all_records())
    print(f"🔍 อ่านจาก '{sheet_name}' ได้ {len(df)} แถว, คอลัมน์: {list(df.columns)}")
    return df


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


def pick_pnl(row, allow_gross, counts):
    cols = NET_PNL_COLS + (GROSS_PNL_COLS if allow_gross else [])
    for c in cols:
        if c in row and row[c] is not None and str(row[c]).strip() != "":
            v = to_num(row[c])
            if v is not None:
                counts[c] = counts.get(c, 0) + 1
                return v
    return None


def load_dataset_from_df(df, use_legacy, pnl_threshold, allow_gross=False):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rows = []
    skipped = {"not_closed": 0, "no_pnl": 0, "no_features": 0, "bad_side": 0, "bad_time": 0}
    pnl_sources, n_legacy = {}, 0

    for i, r in df.iterrows():
        row = r.to_dict()

        status = str(row.get("status", "")).strip().upper()
        if status and status not in ["CLOSED", "NAN", "NONE", ""]:
            skipped["not_closed"] += 1
            continue

        pnl = pick_pnl(row, allow_gross, pnl_sources)
        if pnl is None:
            skipped["no_pnl"] += 1
            continue

        raw, is_legacy = None, False
        for col in FEATURE_COLS:
            if col in row:
                raw = parse_json_cell(row[col])
                if raw:
                    break
        if raw is None and use_legacy:
            raw, is_legacy = legacy_raw(row), True
        if raw is None:
            skipped["no_features"] += 1
            continue

        try:
            feats = engineer(raw)
        except ValueError:                      # side ว่าง/ไม่รู้จัก
            skipped["bad_side"] += 1
            continue

        n_legacy += int(is_legacy)
        rows.append({
            "t": row.get("openTime", ""), "idx": i, "y": 1 if pnl > pnl_threshold else 0,
            "pnl": pnl, "legacy": is_legacy, **feats,
        })

    data = pd.DataFrame(rows)
    info = {"pnl_sources": pnl_sources, "legacy_rows": n_legacy}
    if data.empty:
        return data, skipped, info

    data["t"] = pd.to_datetime(data["t"], errors="coerce", format="mixed")
    bad_t = int(data["t"].isna().sum())
    if bad_t:
        skipped["bad_time"] = bad_t             # เวลาอ่านไม่ได้ -> ทิ้ง (ไม่ให้ถูกวางเป็น "อนาคต")
        data = data[data["t"].notna()]
    data = data.sort_values(["t", "idx"], kind="stable").reset_index(drop=True)
    return data, skipped, info


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


def make_splitter(n, gap):
    """TimeSeriesSplit พร้อม gap (กันผลของไม้ที่ยังไม่ปิดรั่วข้ามชุด); ลด gap อัตโนมัติถ้าข้อมูลน้อย"""
    n_splits = max(2, min(5, n // 25))
    for g in (gap, 0):
        try:
            splitter = TimeSeriesSplit(n_splits=n_splits, gap=g)
            list(splitter.split(np.arange(n)))   # ทดสอบว่าใช้ได้
            return splitter, g
        except ValueError:
            continue
    return None, 0


def walk_forward(X, y, C, gap):
    n = len(y)
    splitter, used_gap = make_splitter(n, gap)
    if splitter is None:
        return None
    ys, ps, base = [], [], []
    for tr, te in splitter.split(X):
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
        "n_oos": int(len(ys)), "gap": int(used_gap), "auc": auc,
        "brier": brier, "brier_baseline": brier_base,
        "brier_improvement": brier_base - brier,
        "accuracy@0.5": float(((ps >= 0.5) == (ys == 1)).mean()),
        "accuracy_majority_baseline": float(maj),
    }


def evaluate_gate(wf, n_samples, args):
    """คืน (ผ่านไหม, เหตุผลที่ไม่ผ่าน)"""
    reasons = []
    if n_samples < args.min_enforce:
        reasons.append(f"ข้อมูล {n_samples} < min-enforce {args.min_enforce}")
    if not wf:
        reasons.append("walk-forward คำนวณไม่ได้")
    else:
        if wf["n_oos"] < args.min_oos:
            reasons.append(f"n_oos {wf['n_oos']} < min-oos {args.min_oos}")
        if wf["auc"] is None or wf["auc"] < args.min_auc:
            reasons.append(f"AUC {wf['auc']} < min-auc {args.min_auc}")
        if wf["brier_improvement"] <= 0:
            reasons.append(f"Brier ไม่ดีกว่า baseline (improvement {wf['brier_improvement']:.5f})")
    return (not reasons), reasons


# ── 3. ฟังก์ชันหลัก ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="ถ้าใส่ path csv จะอ่านจากไฟล์แทน Google Sheets")
    ap.add_argument("--out", default="model.json", help="Path สำหรับเซฟไฟล์โมเดล JSON")
    ap.add_argument("--min-train", type=int, default=10, help="จำนวนข้อมูลขั้นต่ำสำหรับเทรน")
    ap.add_argument("--min-enforce", type=int, default=150, help="จำนวนข้อมูลขั้นต่ำสำหรับ enforceOk")
    ap.add_argument("--min-oos", type=int, default=100, help="จำนวนตัวอย่าง out-of-sample ขั้นต่ำสำหรับ enforceOk")
    ap.add_argument("--min-auc", type=float, default=0.55, help="AUC walk-forward ขั้นต่ำสำหรับ enforceOk")
    ap.add_argument("--gap", type=int, default=5, help="จำนวนแถวที่เว้นระหว่างชุดเทรนกับชุดทดสอบ")
    ap.add_argument("--pnl-threshold", type=float, default=0.0, help="เกณฑ์กำไร PnL สำหรับตัดสินชนะ/แพ้")
    ap.add_argument("--C", type=float, default=1.0, help="ค่า Regularization C ของ LogisticRegression")
    ap.add_argument("--force", action="store_true", help="บังคับสร้างโมเดลแม้ข้อมูลน้อยกว่า min-train")
    ap.add_argument("--no-legacy", action="store_true", help="ปิดการดึงฟีเจอร์แบบเก่า")
    ap.add_argument("--allow-gross", action="store_true", help="อนุญาตใช้ grossPnl เมื่อไม่มี netPnl (ไม่แนะนำ)")
    args = ap.parse_args()

    print("🚀 เริ่มต้นกระบวนการเทรนโมเดล...")

    if args.csv:
        print(f"📖 กำลังโหลดข้อมูลจากไฟล์ CSV: {args.csv}")
        df = pd.read_csv(args.csv)
    else:
        print("🌐 กำลังดึงข้อมูลจาก Google Sheets...")
        df = fetch_data_from_google_sheets()

    data, skipped, info = load_dataset_from_df(
        df, use_legacy=not args.no_legacy, pnl_threshold=args.pnl_threshold, allow_gross=args.allow_gross)
    n_samples = len(data)

    print(f"📊 ทั้งหมด {len(df)} แถว, พร้อมเทรน {n_samples} แถว | Skipped: {skipped}")
    print(f"💰 แหล่ง PnL ที่ใช้: {info['pnl_sources']} | แถว legacy: {info['legacy_rows']}")
    if info["legacy_rows"]:
        print("⚠️ มีแถว legacy (ฟีเจอร์ไม่ครบและสเกล edge อาจต่างจากข้อมูลใหม่) — พิจารณา --no-legacy")

    if n_samples < args.min_train and not args.force:
        print(f"⚠️ ข้อมูลไม่เพียงพอสำหรับเทรน (ต้องการขั้นต่ำ {args.min_train} แถว แต่มี {n_samples} แถว)")
        sys.exit(0)
    if n_samples == 0:
        raise SystemExit("❌ ไม่มีข้อมูลเลย")

    features = [f for f in MODEL_FEATURES if f in data.columns]
    if features != MODEL_FEATURES:
        raise SystemExit(f"❌ features ไม่ครบ: {sorted(set(MODEL_FEATURES) - set(features))}")

    X = data[features].apply(pd.to_numeric, errors="coerce").astype(float)
    y = data["y"].to_numpy()

    if len(np.unique(y)) < 2:
        print("⚠️ label มีคลาสเดียว (ชนะหมดหรือแพ้หมด) เทรนไม่ได้ — ไม่สร้างโมเดลใหม่")
        sys.exit(0)

    coverage = X.notna().mean()
    dead = [f for f in features if coverage[f] == 0]
    sparse = {f: round(float(coverage[f]), 2) for f in features if 0 < coverage[f] < 0.8}
    if dead:
        print(f"⚠️ ฟีเจอร์ที่ไม่มีข้อมูลเลย (ไม่มีผลต่อโมเดล): {dead}")
    if sparse:
        print(f"⚠️ ฟีเจอร์ที่มีข้อมูล < 80%: {sparse}")

    mean, std, lr = fit_model(X, y, args.C)
    wf = walk_forward(X, y, args.C, args.gap)
    gate_ok, reasons = evaluate_gate(wf, n_samples, args)

    coefs = dict(zip(features, lr.coef_[0].tolist()))
    print("📐 coefficients (เรียงตามขนาด):")
    for k, v in sorted(coefs.items(), key=lambda kv: -abs(kv[1])):
        print(f"   {k:15s} {v:+.4f}")
    print(f"📈 walk-forward: {wf}")
    print(f"🚦 enforceOk = {gate_ok}" + ("" if gate_ok else f"  (เหตุผล: {'; '.join(reasons)})"))

    model_artifact = {
        "modelVersion": dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S"),
        "nTrain": n_samples,
        "features": features,
        "zClip": Z_CLIP,
        "mean": mean.to_dict(),
        "std": std.to_dict(),
        "coefficients": coefs,
        "intercept": float(lr.intercept_[0]),
        "walkForward": wf,
        "deadFeatures": dead,
        "gateReasons": reasons,
        "ready": True,
        "enforceOk": bool(gate_ok),
    }

    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(model_artifact, f, indent=2, allow_nan=False)
    os.replace(tmp, args.out)                   # เขียนแบบ atomic กันไฟล์ค้างครึ่งเดียว

    print(f"✅ บันทึกโมเดลเรียบร้อยแล้วไปที่: {args.out}")


if __name__ == "__main__":
    main()
