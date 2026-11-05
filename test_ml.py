import importlib
import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import features  # noqa: E402
import train     # noqa: E402


def make_csv(path, n, signal=True, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        side = rng.choice(["LONG", "SHORT"])
        raw = {
            "side": side, "rsi": float(rng.uniform(25, 75)), "rsiH1": float(rng.uniform(30, 70)),
            "rsiH4": float(rng.uniform(35, 65)), "macdLine": float(rng.normal(0, 150)),
            "macdSignal": float(rng.normal(0, 150)), "macdHistogram": float(rng.normal(0, 60)),
            "atr": float(rng.uniform(120, 400)), "atrPct": float(rng.uniform(0.15, 0.6)),
            "volRatio": float(rng.uniform(0.6, 2.5)), "volPercentile": float(rng.uniform(0, 1)),
            "stochK": float(rng.uniform(0, 100)), "stochD": float(rng.uniform(0, 100)),
            "takerAvg3": float(rng.uniform(0.8, 1.25)), "oiDeltaShortPct": float(rng.normal(0, 0.4)),
            "fundingRatePct": float(rng.normal(0, 0.01)), "premiumPct": float(rng.normal(0, 0.05)),
            "rawScore": float(rng.uniform(60, 90)), "edge": float(rng.uniform(40, 90)),
        }
        f = features.engineer(raw)
        if signal:   # ผลลัพธ์ขึ้นกับ taker_dir และ macdHist_dir จริง
            logit = 9.0 * f["taker_dir"] + 3.5 * f["macdHist_dir"]
            win = rng.random() < 1 / (1 + math.exp(-logit))
        else:
            win = rng.random() < 0.5
        rows.append({
            "orderId": str(1000 + i), "openTime": f"2026-10-{1 + i // 20:02d} {i % 20:02d}:00:00",
            "status": "CLOSED", "netPnl": "25.5" if win else "-31.2", "mlFeatures": json.dumps(raw),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def run_train(csv, out, *extra):
    return subprocess.run([sys.executable, os.path.join(ROOT, "train.py"), "--csv", str(csv), "--out", str(out), *extra],
                          capture_output=True, text=True, cwd=ROOT)


# ── features ──────────────────────────────────────────────────────────────────

def test_direction_flips_sign_but_not_magnitudes():
    base = {"rsi": 70, "macdHistogram": 30, "atr": 300, "takerAvg3": 1.2, "atrPct": 0.3, "volRatio": 1.5}
    L = features.engineer({**base, "side": "LONG"})
    S = features.engineer({**base, "side": "SHORT"})
    assert L["rsi_dir"] == pytest.approx(0.4) and S["rsi_dir"] == pytest.approx(-0.4)
    assert L["macdHist_dir"] == pytest.approx(-S["macdHist_dir"])
    assert L["taker_dir"] == pytest.approx(-S["taker_dir"])
    assert L["atrPct"] == S["atrPct"] and L["volRatio_log"] == S["volRatio_log"]
    assert L["isShort"] == 0.0 and S["isShort"] == 1.0
    assert features.engineer({"side": "SELL"})["isShort"] == 1.0


def test_bad_values_become_none():
    f = features.engineer({"side": "LONG", "rsi": "abc", "atr": 0, "macdHistogram": 5, "volRatio": float("nan")})
    assert f["rsi_dir"] is None and f["macdHist_dir"] is None and f["volRatio_log"] is None


# ── training ──────────────────────────────────────────────────────────────────

def test_demo_mode_trains_on_small_data_and_allows_enforce(tmp_path):
    csv, out = tmp_path / "t.csv", tmp_path / "model.json"
    make_csv(csv, 30)
    r = run_train(csv, out)
    assert r.returncode == 0, r.stdout + r.stderr
    m = json.load(open(out))
    assert m["n_train"] == 30 and m["demo"] is True and m["enforce_ok"] is True
    assert "คำเตือน" in r.stdout            # เตือนว่าข้อมูลน้อย แต่ไม่ปฏิเสธ


def test_production_flags_still_available(tmp_path):
    csv, out = tmp_path / "t.csv", tmp_path / "model.json"
    make_csv(csv, 30)
    r = run_train(csv, out, "--min-train", "100")
    assert r.returncode == 2 and not out.exists()
    r = run_train(csv, out, "--min-enforce", "200")
    assert r.returncode == 0 and json.load(open(out))["enforce_ok"] is False


def test_technical_minimum_and_single_class(tmp_path):
    csv, out = tmp_path / "t.csv", tmp_path / "model.json"
    make_csv(csv, 5)
    assert run_train(csv, out).returncode == 2 and not out.exists()
    make_csv(csv, 40)
    df = pd.read_csv(csv, dtype=str); df["netPnl"] = "10"; df.to_csv(csv, index=False)
    r = run_train(csv, out)
    assert r.returncode == 2 and not out.exists() and "แต่ไม้ชนะ" in r.stdout


def test_learns_real_signal_and_passes_production_gates(tmp_path):
    csv, out = tmp_path / "t.csv", tmp_path / "model.json"
    make_csv(csv, 400, signal=True)
    r = run_train(csv, out, "--min-train", "100", "--min-enforce", "200", "--require-quality")
    assert r.returncode == 0, r.stderr + r.stdout
    m = json.load(open(out))
    assert m["features"] == features.MODEL_FEATURES
    assert m["metrics"]["auc"] > 0.65
    assert m["enforce_ok"] is True and m["demo"] is False
    top = sorted(m["coef"].items(), key=lambda kv: -abs(kv[1]))[:2]
    assert {k for k, _ in top} == {"taker_dir", "macdHist_dir"}
    assert m["coef"]["taker_dir"] > 0 and m["coef"]["macdHist_dir"] > 0


def test_quality_gate_blocks_pure_noise_when_requested(tmp_path):
    csv, out = tmp_path / "t.csv", tmp_path / "model.json"
    make_csv(csv, 400, signal=False, seed=3)
    r = run_train(csv, out, "--require-quality")
    assert r.returncode == 0
    assert json.load(open(out))["enforce_ok"] is False
    r = run_train(csv, out)                      # โหมด demo: ไม่มีเกณฑ์คุณภาพ
    assert json.load(open(out))["enforce_ok"] is True


def test_falls_back_to_legacy_rows_when_no_feature_json(tmp_path):
    csv, out = tmp_path / "t.csv", tmp_path / "model.json"
    rng = np.random.default_rng(1)
    rows = []
    for i in range(40):
        win = i % 2 == 0
        rows.append({"orderId": str(i), "openTime": f"10/{1 + i // 10}/2026 {i % 10}:00:00", "status": "CLOSED",
                     "netPnl": "20" if win else "-30", "direction": "BEARISH", "side": "SELL", "entry": "77000",
                     "rsi": f"{rng.uniform(25, 70):.2f}", "macdLine": f"{rng.normal(0, 100):.2f}",
                     "macdHistogram": f"{rng.normal(0, 50):.2f}", "atr": "250", "volRatio": f"{rng.uniform(0.6, 2):.2f}",
                     "scoreRatio": "70", "mlFeatures": "", "r_ratios": "— / — / — / —"})
    pd.DataFrame(rows).to_csv(csv, index=False)
    r = run_train(csv, out)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ใช้แถวเก่าเสริมอัตโนมัติ" in r.stdout and json.load(open(out))["n_train"] == 40


def test_unclosed_rows_and_missing_pnl_are_skipped(tmp_path):
    csv, out = tmp_path / "t.csv", tmp_path / "model.json"
    make_csv(csv, 120)
    df = pd.read_csv(csv, dtype=str)
    df.loc[0, "status"] = "PENDING_FILL"; df.loc[1, "netPnl"] = ""
    df.to_csv(csv, index=False)
    data, skipped = train.load_dataset(csv, False, 0.0)
    assert len(data) == 118 and skipped["not_closed"] == 1 and skipped["no_pnl"] == 1


def test_features_can_live_in_legacy_r_ratios_column(tmp_path):
    csv = tmp_path / "t.csv"
    make_csv(csv, 60)
    df = pd.read_csv(csv, dtype=str).rename(columns={"mlFeatures": "r_ratios"})
    df.to_csv(csv, index=False)
    data, _ = train.load_dataset(csv, False, 0.0)
    assert len(data) == 60


# ── service ───────────────────────────────────────────────────────────────────

@pytest.fixture
def client_factory(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    def make(model_path=None, api_key=""):
        monkeypatch.setenv("MODEL_PATH", str(model_path or tmp_path / "missing.json"))
        monkeypatch.setenv("API_KEY", api_key)
        import app as app_module
        importlib.reload(app_module)
        return TestClient(app_module.app), app_module
    return make


RAW = {"side": "SHORT", "rsi": 28, "rsiH1": 37, "rsiH4": 52, "macdLine": -40, "macdSignal": -46,
       "macdHistogram": 6, "atr": 244, "atrPct": 0.3, "volRatio": 1.4, "volPercentile": 0.6,
       "stochK": 64, "stochD": 74, "takerAvg3": 0.83, "oiDeltaShortPct": -0.2,
       "fundingRatePct": 0.0005, "premiumPct": -0.04, "rawScore": 70, "edge": 60}


def test_no_model_means_no_opinion(client_factory):
    c, _ = client_factory()
    assert c.get("/health").json()["ready"] is False
    r = c.post("/predict", json=RAW).json()
    assert r["prob"] is None and r["ready"] is False and r["enforceOk"] is False


def test_service_matches_sklearn_exactly(client_factory, tmp_path):
    csv = tmp_path / "t.csv"; make_csv(csv, 300)
    data, _ = train.load_dataset(csv, False, 0.0)
    X, y = data[features.MODEL_FEATURES].astype(float), data["y"].to_numpy()
    mean, std, lr = train.fit_model(X, y, 0.3)
    mp = tmp_path / "model.json"
    json.dump({"version": "t1", "features": features.MODEL_FEATURES, "n_train": 300, "enforce_ok": True,
               "mean": mean.to_dict(), "std": std.to_dict(),
               "coef": dict(zip(features.MODEL_FEATURES, lr.coef_[0].tolist())),
               "intercept": float(lr.intercept_[0])}, open(mp, "w"))
    c, _ = client_factory(mp)
    got = c.post("/predict", json=RAW).json()
    expect = train.predict_model(mean, std, lr, pd.DataFrame([features.engineer(RAW)])[features.MODEL_FEATURES].astype(float))[0]
    assert got["ready"] and got["enforceOk"] and got["modelVersion"] == "t1"
    assert got["prob"] == pytest.approx(expect, abs=1e-4)
    assert got["missing"] == []
    # ขาดฟีเจอร์ -> ยังทำนายได้ และรายงานว่าขาดอะไร
    part = {k: v for k, v in RAW.items() if k not in ("rsi", "takerAvg3")}
    r2 = c.post("/predict", json=part).json()
    assert 0 <= r2["prob"] <= 1 and set(r2["missing"]) == {"rsi_dir", "taker_dir"}
    # ค่าสุดโต่งต้องไม่ทำให้ล้น
    r3 = c.post("/predict", json={**RAW, "macdHistogram": 1e12, "atr": 1e-6}).json()
    assert 0 <= r3["prob"] <= 1


def test_api_key_enforced(client_factory):
    c, _ = client_factory(api_key="secret")
    assert c.post("/predict", json=RAW).status_code == 401
    assert c.post("/predict", json=RAW, headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.post("/predict", json=RAW, headers={"X-API-Key": "secret"}).status_code == 200


def test_model_with_different_feature_list_is_rejected(client_factory, tmp_path):
    mp = tmp_path / "model.json"
    json.dump({"version": "old", "features": ["a", "b"]}, open(mp, "w"))
    c, _ = client_factory(mp)
    assert c.get("/health").json()["ready"] is False
