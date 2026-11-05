"""
features.py — แปลงค่าดิบจาก workflow เป็นฟีเจอร์ของโมเดล (ใช้ร่วมกันทั้งตอนเทรนและตอนทำนาย)

หลักคิด: ทำให้ฟีเจอร์ "มีทิศทาง" (directional) ด้วยการคูณ +1 (LONG) / -1 (SHORT)
เพื่อให้โมเดลตัวเดียวใช้ได้ทั้งสองฝั่ง เช่น RSI = 70 เป็นข้อดีของ LONG แต่เป็นข้อเสียของ SHORT
ค่าที่ได้จึงอ่านความหมายได้ตรงกันเสมอว่า "บวก = เอื้อต่อทิศทางที่เข้าไม้"
"""
import math

# คีย์ดิบที่ n8n ส่งมา (และบันทึกลงชีตเป็น JSON ตอนเข้าไม้)
RAW_KEYS = [
    "side",            # LONG | SHORT  (หรือ BUY/SELL/BULLISH/BEARISH)
    "rsi", "rsiH1", "rsiH4",
    "macdLine", "macdSignal", "macdHistogram",
    "atr", "atrPct",
    "volRatio", "volPercentile",
    "stochK", "stochD",
    "takerAvg3", "oiDeltaShortPct",
    "fundingRatePct", "premiumPct",
    "rawScore", "edge",
]

# ฟีเจอร์ที่โมเดลใช้จริง (เรียงลำดับคงที่ — ห้ามเปลี่ยนลำดับหลังเทรนแล้ว)
MODEL_FEATURES = [
    "rsi_dir", "rsiH1_dir", "rsiH4_dir", "stoch_dir",
    "macdHist_dir", "macdLine_dir",
    "taker_dir", "oi_dir", "funding_dir", "premium_dir",
    "atrPct", "volRatio_log", "volPercentile",
    "score_n", "edge_n", "isShort",
]


def to_num(v):
    """แปลงเป็นตัวเลข; ค่าว่าง/NaN/inf/ข้อความที่แปลงไม่ได้ -> None"""
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(str(v).replace(",", "").strip()) if not isinstance(v, (int, float)) else float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def direction(side):
    """+1 = LONG, -1 = SHORT"""
    s = str(side or "").strip().upper()
    return -1 if s in ("SHORT", "SELL", "BEARISH") else 1


def engineer(raw):
    """raw(dict) -> dict ของ MODEL_FEATURES (ค่าที่คำนวณไม่ได้ = None)"""
    d = direction(raw.get("side"))
    g = lambda k: to_num(raw.get(k))

    def dirs(key, center=0.0, scale=1.0):
        v = g(key)
        return None if v is None else d * (v - center) / scale

    atr = g("atr")
    hist, line = g("macdHistogram"), g("macdLine")
    vol = g("volRatio")
    score, edge = g("rawScore"), g("edge")

    return {
        "rsi_dir":       dirs("rsi", 50, 50),
        "rsiH1_dir":     dirs("rsiH1", 50, 50),
        "rsiH4_dir":     dirs("rsiH4", 50, 50),
        "stoch_dir":     dirs("stochK", 50, 50),
        "macdHist_dir":  (d * hist / atr) if (hist is not None and atr and atr > 0) else None,
        "macdLine_dir":  (d * line / atr) if (line is not None and atr and atr > 0) else None,
        "taker_dir":     dirs("takerAvg3", 1.0, 1.0),
        "oi_dir":        dirs("oiDeltaShortPct", 0.0, 1.0),
        "funding_dir":   dirs("fundingRatePct", 0.0, 1.0),
        "premium_dir":   dirs("premiumPct", 0.0, 1.0),
        "atrPct":        g("atrPct"),
        "volRatio_log":  math.log1p(vol) if (vol is not None and vol >= 0) else None,
        "volPercentile": g("volPercentile"),
        "score_n":       None if score is None else score / 100.0,
        "edge_n":        None if edge is None else edge / 100.0,
        "isShort":       1.0 if d < 0 else 0.0,
    }
