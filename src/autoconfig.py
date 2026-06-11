# -*- coding: utf-8 -*-
"""프로젝트 경로만 주면 설정(config)을 자동으로 만들어 준다.

`.prproj`(또는 프로젝트 폴더)를 받아:
  - 자막 넣을 시퀀스 (오디오 클립이 가장 많은 것) 자동 선택
  - 오디오 폴더(audio_dir)  : 클립 미디어 경로에서 추론
  - 가사 폴더(lyrics_dir)   : 프로젝트 주변에서 .txt 가 든 폴더 탐색
  - 출력 폴더(output_dir)   : <프로젝트폴더>/자막출력
다른 영상에 적용할 때 경로를 일일이 안 적어도 되게 한다.
"""
import os, glob, re
from extract_layout import load_project, list_sequences, extract_audio_clips


def _resolve_prproj(target):
    """폴더면 내부 .prproj 중 가장 큰 것, 파일이면 그대로."""
    if os.path.isdir(target):
        cands = glob.glob(os.path.join(target, "**", "*.prproj"), recursive=True)
        if not cands:
            raise FileNotFoundError(f"'{target}' 안에서 .prproj 를 찾지 못함")
        return max(cands, key=lambda p: os.path.getsize(p))
    if not os.path.exists(target):
        raise FileNotFoundError(target)
    return target


def _seq_audio_counts(prproj):
    """각 시퀀스의 오디오 클립 수 {이름: 개수}."""
    root, _ = load_project(prproj)
    counts = {}
    for name in list_sequences(root):
        try:
            counts[name] = len(extract_audio_clips(prproj, name))
        except Exception:
            counts[name] = 0
    return counts


def _find_lyrics_dir(project_dir):
    """프로젝트 폴더와 그 부모에서 .txt 가사 파일이 든 폴더를 탐색."""
    roots = [project_dir, os.path.dirname(project_dir)]
    best, best_score = None, 0
    for r in roots:
        if not r or not os.path.isdir(r):
            continue
        for dirpath, _dirs, files in os.walk(r):
            # 너무 깊이 들어가지 않기
            if dirpath.count(os.sep) - r.count(os.sep) > 2:
                continue
            txts = [f for f in files if f.lower().endswith(".txt")]
            # 파일명에 숫자가 든 txt 가 여러 개면 가사 폴더일 확률↑
            numbered = [f for f in txts if re.search(r"\d", f)]
            score = len(numbered)
            name_bonus = 2 if re.search(r"가사|lyric", os.path.basename(dirpath), re.I) else 0
            score += name_bonus
            if score > best_score and len(txts) >= 1:
                best, best_score = dirpath, score
    return best


def build_config(target, sequence=None, verbose=True):
    """target(.prproj 또는 폴더) → config dict 생성."""
    prproj = _resolve_prproj(target)
    project_dir = os.path.dirname(os.path.abspath(prproj))

    counts = _seq_audio_counts(prproj)
    if verbose:
        print("[autoconfig] 시퀀스별 오디오 클립 수:")
        for n, c in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"   {c:3d}곡  {n}")
    if sequence is None:
        if not counts or max(counts.values(), default=0) == 0:
            raise ValueError("오디오 클립이 있는 시퀀스를 찾지 못함. sequence 를 직접 지정하세요.")
        sequence = max(counts, key=lambda k: counts[k])  # 곡이 가장 많은 시퀀스
        if verbose:
            print(f"[autoconfig] 자동 선택된 시퀀스 → '{sequence}' ({counts[sequence]}곡)")

    clips = extract_audio_clips(prproj, sequence)
    audio_dirs = {os.path.dirname(c["path"]) for c in clips if c.get("path")}
    audio_dir = next(iter(audio_dirs)) if len(audio_dirs) == 1 else (project_dir)

    lyrics_dir = _find_lyrics_dir(project_dir)
    if verbose and lyrics_dir:
        print(f"[autoconfig] 가사 폴더 추정 → {lyrics_dir}")

    safe_seq = re.sub(r"\s+", "", sequence)
    cfg = {
        "prproj": prproj.replace("\\", "/"),
        "sequence": sequence,
        "audio_dir": audio_dir.replace("\\", "/"),
        "lyrics_dir": (lyrics_dir or "").replace("\\", "/"),
        "output_dir": os.path.join(project_dir, "자막출력").replace("\\", "/"),
        "output_basename": f"{safe_seq}_자막",
        "language": "ja",
        "model": "medium",
        "device": "cuda",
        "compute_type": "float16",
        "split_gap": 0.8,
    }
    return cfg
