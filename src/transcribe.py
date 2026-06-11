# -*- coding: utf-8 -*-
"""faster-whisper 로 오디오를 받아 [시작, 끝, 텍스트] 세그먼트를 얻는다.

가창(노래) 인식에 맞춰 옵션을 튜닝하고, 자막 규칙(docs/RULES.md)을 적용한다:
  - 반복 추임새("oh oh oh oh…")는 2회로 축약
  - 각 세그먼트를 개별 자막 큐로 유지 (여러 줄을 '/' 로 합치지 않음)
faster_whisper 를 import 하기 전에 cuda_setup.enable_cuda_dlls() 가 호출되어야 한다.
"""
import re

# 반복 추임새로 간주할 토큰(소문자 비교). 필요시 추가.
FILLER_WORDS = {"oh", "ah", "la", "na", "woah", "whoa", "yeah", "uh", "mm", "hmm", "오", "아", "라"}
# 가짜 자막(영상 끝 크레딧/유튜브 멘트 등) 제거용
JUNK_RE = re.compile(r"(作詞|作曲|編曲|ご視聴|チャンネル登録|視聴ありがとう|ご覧いただき|提供|スポンサー|最後まで)")


def trim_filler(text, max_repeat=2):
    """반복 추임새를 max_repeat 회로 축약.

    두 경우를 처리:
      (a) 토큰 반복   "Oh, oh, oh, oh"   → "Oh oh"
      (b) 글자 반복   "おうおうおうおう" → "おう おう"  (띄어쓰기 없이 붙은 경우)
    """
    t = text.strip()
    # (a) 쉼표/공백으로 분리된 동일 추임새
    parts = [p for p in re.split(r"[,\s]+", t) if p]
    if parts:
        norm = [re.sub(r"[^\wぁ-んァ-ヶ가-힣]", "", p).lower() for p in parts]
        base = norm[0]
        if base in FILLER_WORDS and all(n == base or n == "" for n in norm):
            return " ".join([parts[0].rstrip(",")] * min(max_repeat, len(parts)))

    # (b) 짧은 단위가 4회 이상 연속 반복 (공백 제거 후 검사) → 가창 추임새로 간주
    s = re.sub(r"\s+", "", t)
    for unit_len in range(1, 5):
        unit = s[:unit_len]
        if not unit:
            break
        k = len(s) // unit_len
        if k >= 4 and unit * k == s:
            return " ".join([unit] * max_repeat)
    return text


def split_on_word_gaps(seg, gap=0.8):
    """word_timestamps 가 있을 때, 단어 사이 큰 공백(gap초 이상)에서 세그먼트를 분할.

    노래에서 한 세그먼트에 두 가사 줄이 붙어버린 경우를 분리한다.
    반환: list of (start, end, text)
    """
    words = getattr(seg, "words", None)
    if not words:
        return [(seg.start, seg.end, seg.text.strip())]
    chunks = []
    cur = [words[0]]
    for w in words[1:]:
        if w.start - cur[-1].end >= gap:
            chunks.append(cur)
            cur = [w]
        else:
            cur.append(w)
    chunks.append(cur)
    out = []
    for ch in chunks:
        txt = "".join(w.word for w in ch).strip()
        if txt:
            out.append((round(ch[0].start, 3), round(ch[-1].end, 3), txt))
    return out


def transcribe_file(model, audio_path, language="ja", split_gap=0.8,
                    drop_junk=True, max_filler=2):
    """오디오 한 곡 → 규칙 적용된 [(start, end, text), ...] (오디오 파일 기준 초)."""
    segments, info = model.transcribe(
        audio_path, language=language, vad_filter=False, beam_size=5,
        word_timestamps=True, condition_on_previous_text=False,
        no_speech_threshold=0.6,
    )
    cues = []
    for seg in segments:
        for st, en, txt in split_on_word_gaps(seg, split_gap):
            txt = txt.strip()
            if not txt:
                continue
            if drop_junk and JUNK_RE.search(txt):
                continue
            txt = trim_filler(txt, max_filler)
            cues.append((round(st, 3), round(en, 3), txt))
    return cues, round(info.duration, 2)


def load_model(model_name="medium", device="cuda", compute_type="float16"):
    """faster-whisper 모델 로드. cuda 실패 시 CPU(int8)로 자동 폴백."""
    from faster_whisper import WhisperModel
    try:
        return WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception as e:
        if device == "cuda":
            print(f"[transcribe] CUDA 로드 실패({e}). CPU(int8) 로 폴백.")
            return WhisperModel(model_name, device="cpu", compute_type="int8")
        raise
