# -*- coding: utf-8 -*-
"""타임라인 정렬된 큐 목록 → SRT 파일.

큐: dict {t_in, t_out, ja, ko}  (t_in/t_out 은 시퀀스 타임라인 기준 초)
mode:
  'both' → 한국어(위) + 일본어(아래)
  'ja'   → 일본어만
  'ko'   → 한국어만
"""


def tc(sec):
    if sec < 0:
        sec = 0
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def normalize(cues, min_dur=0.6):
    """시간순 정렬 + 최소 길이 보장 + 다음 큐와 겹침 방지."""
    cues = sorted(cues, key=lambda c: c["t_in"])
    out = []
    for k, c in enumerate(cues):
        ti, to = c["t_in"], c["t_out"]
        if to - ti < min_dur:
            to = ti + min_dur
        if k + 1 < len(cues) and to > cues[k + 1]["t_in"]:
            to = max(ti + 0.5, cues[k + 1]["t_in"] - 0.02)
        d = dict(c)
        d["t_in"], d["t_out"] = ti, to
        out.append(d)
    return out


def write_srt(path, cues, mode="both"):
    cues = normalize(cues)
    n = 0
    blocks = []
    for c in cues:
        ja = (c.get("ja") or "").strip()
        ko = (c.get("ko") or "").strip()
        if mode == "ja":
            body = ja
        elif mode == "ko":
            body = ko
        else:
            body = "\n".join([x for x in (ko, ja) if x])
        if not body:
            continue
        n += 1
        blocks.append(f"{n}\n{tc(c['t_in'])} --> {tc(c['t_out'])}\n{body}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(blocks))
    return n
