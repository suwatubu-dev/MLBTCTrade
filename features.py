"""
features.py — แปลงค่าดิบจาก workflow เป็นฟีเจอร์ของโมเดล (ใช้ร่วมกันทั้งตอนเทรนและตอนทำนาย)

หลักคิด: ทำให้ฟีเจอร์ "มีทิศทาง" (directional) ด้วยการคูณ +1 (LONG) / -1 (SHORT)
เพื่อให้โมเดลตัวเดียวใช้ได้ทั้งสองฝั่ง ค่าที่ได้คือ "บวก = ไปทางเดียวกับทิศทางที่เข้าไม้"
(ความหมายว่าเอื้อหรือไม่เอื้อ โมเดลเรียนรู้เองผ่าน coefficient ไม่ได้ถูกกำหนดตายตัว)

การเปลี่ยนแปลงจากเวอร์ชันก่อน:
  - side ว่าง/ไม่รู้จัก -> raise ValueError (เดิมถูกตีเป็น LONG เงียบ ๆ)
  - Z_CLIP อยู่ที่เดียว (ใช้ร่วมกับ train.py / app.py)
  - to_num ไม่แปลง "0,5" เป็น 5.0 อีกต่อไป (comma แบบทศนิยมถือว่ากำกวม -> None)
"""
import math
import re

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

# ค่า clip ของ z-score (ใช้ร่วมกันทั้งตอนเทรนและตอนทำนาย)
Z_CLIP = 5.0

LONG_SIDES = {"LONG", "BUY", "BULLISH"}
SHORT_SIDES = {"SHORT", "SELL", "BEARISH"}

_THOUSANDS = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")


def to_num(v):
    """แปลงเป็นตัวเลข; ค่าว่าง/NaN/inf/ข้อความที่แปลงไม่ได้/comma กำกวม -> None"""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        x = float(v)
    else:
        s = str(v).strip()
        if not s:
            return None
        if "," in s:
            if _THOUSANDS.match(s):
                s = s.replace(",", "")      # 1,234.5 -> 1234.5
            else:
                return None                 # "0,5" กำกวม (ทศนิยมแบบยุโรป?) -> ไม่เดา
        try:
            x = float(s)
        except ValueError:
            return None
    return x if math.isfinite(x) else None


def direction(side):
    """+1 = LONG, -1 = SHORT, None = ไม่รู้จัก (ห้ามเดา)"""
    s = str(side or "").strip().upper()
    if s in SHORT_SIDES:
        return -1
    if s in LONG_SIDES:
        return 1
    return None


def engineer(raw):
    """raw(dict) -> dict ของ MODEL_FEATURES (ค่าที่คำนวณไม่ได้ = None)
    raise ValueError ถ้า side ว่างหรือไม่รู้จัก"""
    if not isinstance(raw, dict):
        raise ValueError("raw must be an object")

    d = direction(raw.get("side"))
    if d is None:
        raise ValueError("invalid or missing side")

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
