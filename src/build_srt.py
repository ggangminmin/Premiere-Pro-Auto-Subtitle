# -*- coding: utf-8 -*-
"""타임라인 정렬된 큐 목록 → SRT 파일.

큐: dict {t_in, t_out, ja, ko}  (t_in/t_out 은 시퀀스 타임라인 기준 초)
mode:
  'both' → 한국어(위) + 일본어(아래)
  'ja'   → 일본어만
  'ko'   → 한국어만

출력 직전에 자막 규칙(docs/RULES.md)을 한 번 더 강제 적용한다(어떤 경로로 들어온
텍스트든 — 트랜스크립션/번역 워크시트/수기 — 똑같이 보장):
  - 반복 추임새("Oh oh oh oh…")는 2회로 축약            (trim_filler)
  - 한 큐에 ' / ' 로 합쳐진 여러 가사 줄은 별도 큐로 분리  (split_merged_lines)
"""

try:                                  # run.py 는 src/ 를 sys.path 에 넣음
    from transcribe import trim_filler
except ImportError:                   # src.build_srt 형태로 import 될 때
    from src.transcribe import trim_filler

LINE_SEP = " / "   # 한 큐 안에서 가사 줄 경계를 표시하는 구분자


def tc(sec):
    if sec < 0:
        sec = 0
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _klen(s):
    """시간 분배용 가중치: 공백 제외 글자 수(최소 1)."""
    return max(1, len((s or "").replace(" ", "")))


def split_merged_lines(cues, sep=LINE_SEP, warn=True):
    """한 큐에 ' / ' 로 합쳐진 여러 가사 줄을 별도 큐로 분리(시간 비례 분배).

    ja/ko 양쪽을 같은 구분자로 나눠 짝을 맞춘다. 규칙:
      - 양쪽 모두 sep 으로 같은 개수로 나뉘면 → 짝지어 분리
      - 한쪽만 sep 이고 다른 쪽이 비어 있으면 → 그 쪽 개수대로 분리(반대쪽은 빈칸)
      - 그 외(개수 불일치, 또는 한쪽만 sep 인데 다른 쪽은 한 덩어리) → 짝을 확신할 수
        없으므로 자동 분리하지 않고 경고만 출력(사람이 확인). 이때는 양쪽 모두 같은
        위치에 ' / ' 를 넣어 주면 자동 분리된다.
    반환: 새 큐 리스트(입력은 변경하지 않음).
    """
    out = []
    for c in cues:
        ja, ko = (c.get("ja") or ""), (c.get("ko") or "")
        ja_parts = ja.split(sep) if sep in ja else None
        ko_parts = ko.split(sep) if sep in ko else None

        if ja_parts is None and ko_parts is None:
            out.append(dict(c))
            continue

        # 분리 개수 결정 / 정합성 검사
        if ja_parts and ko_parts:
            if len(ja_parts) != len(ko_parts):
                if warn:
                    print(f"[build_srt] ' / ' 분리 보류 (ja {len(ja_parts)}개 vs "
                          f"ko {len(ko_parts)}개 불일치) @ {tc(c['t_in'])}: {ko or ja}")
                out.append(dict(c))
                continue
            n = len(ko_parts)
        elif ja_parts:                       # ja 만 sep
            if ko.strip():                   # ko 가 한 덩어리 → 짝 불명확, 보류
                if warn:
                    print(f"[build_srt] ' / ' 분리 보류 (ja 만 분할됨, ko 짝 불명확) "
                          f"@ {tc(c['t_in'])}: {ja}")
                out.append(dict(c))
                continue
            n = len(ja_parts)
            ko_parts = [""] * n
        else:                                # ko 만 sep
            if ja.strip():                   # ja 가 한 덩어리 → 짝 불명확, 보류
                if warn:
                    print(f"[build_srt] ' / ' 분리 보류 (ko 만 분할됨, ja 짝 불명확) "
                          f"@ {tc(c['t_in'])}: {ko}")
                out.append(dict(c))
                continue
            n = len(ko_parts)
            ja_parts = [""] * n

        ja_parts = [p.strip() for p in ja_parts]
        ko_parts = [p.strip() for p in ko_parts]

        # 시간 비례 분배 (한국어 길이 기준, 없으면 일본어)
        weights = [_klen(ko_parts[i] or ja_parts[i]) for i in range(n)]
        total = sum(weights)
        ti, to = c["t_in"], c["t_out"]
        dur = to - ti
        cur = ti
        for i in range(n):
            seg_end = to if i == n - 1 else round(cur + dur * weights[i] / total, 3)
            d = dict(c)
            d["t_in"], d["t_out"] = round(cur, 3), seg_end
            d["ja"], d["ko"] = ja_parts[i], ko_parts[i]
            out.append(d)
            cur = seg_end
    return out


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


def write_srt(path, cues, mode="both", max_filler=2):
    # 규칙 강제: ' / ' 분리 → 추임새 축약 → 정렬/겹침 보정
    cues = split_merged_lines(cues)
    for c in cues:
        if c.get("ja"):
            c["ja"] = trim_filler(c["ja"], max_filler)
        if c.get("ko"):
            c["ko"] = trim_filler(c["ko"], max_filler)
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
