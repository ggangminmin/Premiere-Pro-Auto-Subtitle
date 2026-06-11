# -*- coding: utf-8 -*-
"""Premiere Pro 자동 자막 생성 — 전체 파이프라인.

사용법:
    python run.py config.json

동작:
  1) .prproj 에서 지정 시퀀스의 오디오 클립(곡) 배치를 추출
  2) 각 곡을 faster-whisper 로 받아 시퀀스 타임라인에 맞춰 타이밍 추출
  3) 일본어(원문) SRT 자동 생성  +  번역용 워크시트(JSON) 생성
  4) 워크시트의 'ko' 칸이 채워져 있으면 한국어/통합 SRT 까지 생성

자세한 준비물·규칙은 README.md, docs/RULES.md 참고.
"""
import os, sys, json, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import cuda_setup
cuda_setup.enable_cuda_dlls(verbose=True)  # faster_whisper import 전에 필수

from extract_layout import extract_audio_clips, list_sequences, load_project
import transcribe as tr
from build_srt import write_srt


def find_lyric(lyrics_dir, number):
    """번호로 가사 파일 매칭 (파일명에 같은 숫자가 든 .txt)."""
    if not lyrics_dir or not os.path.isdir(lyrics_dir) or number is None:
        return None
    import re
    for fn in os.listdir(lyrics_dir):
        if not fn.lower().endswith(".txt"):
            continue
        nums = re.findall(r"(\d+)", os.path.splitext(fn)[0])
        if nums and int(nums[-1]) == number:
            return os.path.join(lyrics_dir, fn)
    return None


def run(cfg, out_default=None):
    prproj = cfg["prproj"]
    seq = cfg["sequence"]
    out_dir = cfg.get("output_dir") or out_default or os.path.dirname(os.path.abspath(prproj))
    audio_dir = cfg.get("audio_dir")            # path 없을 때 폴백
    lyrics_dir = cfg.get("lyrics_dir")
    language = cfg.get("language", "ja")
    model_name = cfg.get("model", "medium")
    device = cfg.get("device", "cuda")
    split_gap = float(cfg.get("split_gap", 0.8))
    base = cfg.get("output_basename", f"{seq}_자막")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[1/4] '{seq}' 시퀀스 클립 추출 …")
    clips = extract_audio_clips(prproj, seq)
    if not clips:
        print("오디오 클립을 찾지 못함. 시퀀스 목록:", list_sequences(load_project(prproj)[0]))
        return
    for c in clips:
        print(f"   {c['start']:8.2f}s ~ {c['end']:8.2f}s  #{c['number']}  {c['media']}")

    print(f"\n[2/4] faster-whisper 모델 로드 ({model_name}, {device}) …")
    model = tr.load_model(model_name, device, cfg.get("compute_type", "float16"))

    print("\n[3/4] 곡별 트랜스크립션 (타임라인 정렬) …")
    cues = []
    cache = {}  # 같은 파일은 한 번만 트랜스크립션
    for c in clips:
        path = c["path"]
        if not (path and os.path.exists(path)) and audio_dir and c["media"]:
            path = os.path.join(audio_dir, c["media"])
        if not (path and os.path.exists(path)):
            print(f"   ! 오디오 파일 없음: {c['media']} (건너뜀)")
            continue
        if path not in cache:
            segs, dur = tr.transcribe_file(model, path, language, split_gap)
            cache[path] = segs
            print(f"   ✓ {c['media']}: {len(segs)}개 세그먼트")
        clip_len = c["end"] - c["start"]
        for st, en, txt in cache[path]:
            if st >= clip_len + 0.5:
                continue
            en = min(en, clip_len)
            cues.append({"t_in": round(c["start"] + st, 3),
                         "t_out": round(c["start"] + en, 3),
                         "ja": txt, "ko": ""})

    # 번역 워크시트(있으면 기존 ko 유지)
    ws_path = os.path.join(out_dir, f"{base}_translation.json")
    prev = {}
    if os.path.exists(ws_path):
        try:
            for r in json.load(open(ws_path, encoding="utf-8")):
                prev[(round(r["t_in"], 2), r["ja"])] = r.get("ko", "")
        except Exception:
            pass
    for c in cues:
        c["ko"] = prev.get((round(c["t_in"], 2), c["ja"]), "")

    print("\n[4/4] SRT / 워크시트 출력 …")
    n_ja = write_srt(os.path.join(out_dir, f"{base}_일본어.srt"), cues, "ja")
    json.dump([{"t_in": c["t_in"], "t_out": c["t_out"], "ja": c["ja"], "ko": c["ko"]} for c in cues],
              open(ws_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"   일본어 SRT: {n_ja}줄  ->  {base}_일본어.srt")
    print(f"   번역 워크시트 -> {base}_translation.json  (각 줄 'ko' 채우면 통합 자막 생성)")

    if any(c["ko"].strip() for c in cues):
        n_both = write_srt(os.path.join(out_dir, f"{base}_한일통합.srt"), cues, "both")
        n_ko = write_srt(os.path.join(out_dir, f"{base}_한국어.srt"), cues, "ko")
        print(f"   한일통합 SRT: {n_both}줄, 한국어 SRT: {n_ko}줄")
    else:
        print("   (번역 전 — 'ko' 가 비어 있어 일본어 SRT만 생성됨)")
        print("\n  ▶ 한국어 자막을 만들려면, 이 터미널의 AI 에게 이렇게 말하세요:")
        print(f'      "{os.path.basename(ws_path)} 파일의 ja 를 한국어로 번역해서 ko 칸을 채워줘"')
        print("    그런 다음 같은 명령을 다시 실행하면 한일통합/한국어 SRT 가 생성됩니다.")

    print("\n완료. 출력 폴더:", out_dir)


def main():
    if len(sys.argv) < 2:
        print("usage:")
        print("  python run.py config.json                    # 설정 파일로 실행")
        print("  python run.py \"프로젝트.prproj 또는 폴더\"      # 자동 설정 후 실행")
        print("  python run.py \"프로젝트.prproj\" \"시퀀스 06\"   # 시퀀스 지정")
        sys.exit(1)

    arg = sys.argv[1]
    if arg.lower().endswith(".json"):
        cfg = json.load(open(arg, encoding="utf-8"))
        run(cfg, out_default=os.path.dirname(os.path.abspath(arg)))
    else:
        # 자동 설정 모드: .prproj / 폴더 경로만 받아 알아서 설정
        import autoconfig
        seq = sys.argv[2] if len(sys.argv) > 2 else None
        print("[자동 설정] 프로젝트 분석 중 …")
        cfg = autoconfig.build_config(arg, sequence=seq)
        # 생성된 설정을 프로젝트 폴더에 저장(다음에 재사용/수정 가능)
        proj_dir = os.path.dirname(os.path.abspath(cfg["prproj"]))
        cfg_out = os.path.join(proj_dir, f"{cfg['output_basename']}_config.json")
        json.dump(cfg, open(cfg_out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[자동 설정] 설정 저장 → {cfg_out}\n")
        run(cfg)


if __name__ == "__main__":
    main()
