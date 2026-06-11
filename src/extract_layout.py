# -*- coding: utf-8 -*-
"""Premiere Pro 프로젝트(.prproj)에서 특정 시퀀스의 오디오 클립 배치를 추출한다.

.prproj 는 gzip 으로 압축된 XML 이다. 압축을 풀어 파싱한 뒤,
  Sequence(이름으로 검색) → TrackGroups → AudioTrackGroup → Tracks → Track
  → ClipItems → TrackItems → AudioClipTrackItem(Start/End ticks, SubClip→미디어)
경로를 따라가 각 곡의 [미디어 파일, 타임라인 시작/끝(초), 곡 번호]를 얻는다.
"""
import gzip, re, os
import xml.etree.ElementTree as ET

TICK = 254016000000  # Premiere 의 1초 = 254,016,000,000 ticks
AUDIO_EXT = (".wav", ".mp3", ".aif", ".aiff", ".m4a", ".flac", ".ogg")


def load_project(prproj_path):
    """gzip .prproj 를 풀어 (root, byid) 반환. byid: ObjectID/ObjectUID -> element."""
    with open(prproj_path, "rb") as f:
        head = f.read(2)
    if head == b"\x1f\x8b":  # gzip magic
        with gzip.open(prproj_path, "rb") as f:
            data = f.read()
    else:  # 이미 평문 XML 인 경우
        with open(prproj_path, "rb") as f:
            data = f.read()
    root = ET.fromstring(data)
    byid = {}
    for el in root.iter():
        for a in ("ObjectID", "ObjectUID"):
            v = el.get(a)
            if v:
                byid[v] = el
    return root, byid


def list_sequences(root):
    """프로젝트 안의 시퀀스 이름 목록."""
    names = []
    for el in root.iter("Sequence"):
        nm = el.find("Name")
        if nm is not None and nm.text:
            names.append(nm.text)
    return names


def _find_sequence(root, name):
    for el in root.iter("Sequence"):
        nm = el.find("Name")
        if nm is not None and nm.text == name:
            return el
    return None


def _resolve(byid, ref):
    return byid.get(str(ref)) if ref else None


def _audio_trackgroup(byid, seq):
    """시퀀스의 TrackGroups 중 AudioTrackGroup 객체를 반환."""
    tgs = seq.find("TrackGroups")
    if tgs is None:
        return None
    for tg in tgs.findall("TrackGroup"):
        second = tg.find("Second")
        if second is None:
            continue
        obj = _resolve(byid, second.get("ObjectRef"))
        if obj is not None and obj.tag == "AudioTrackGroup":
            return obj
    return None


def _media_for_item(byid, clip_item):
    """AudioClipTrackItem -> (미디어 이름, 미디어 파일 경로). 참조 그래프를 제한적으로 탐색."""
    cti = clip_item.find("ClipTrackItem")
    if cti is None:
        return None, None
    sc = cti.find("SubClip")
    start = sc.get("ObjectRef") if sc is not None else None
    name, path = None, None
    seen = set()
    stack = [start]
    while stack:
        r = stack.pop()
        if not r or r in seen:
            continue
        seen.add(r)
        e = byid.get(r)
        if e is None:
            continue
        for tag in ("FilePath", "ActualMediaFilePath"):
            ch = e.find(tag)
            if ch is not None and ch.text and ch.text.lower().endswith(AUDIO_EXT):
                path = ch.text
        for nm in e.iter("Name"):
            t = nm.text or ""
            if t.lower().endswith(AUDIO_EXT):
                name = t
        if path:
            break
        for d in e.iter():
            for a in ("ObjectRef", "ObjectURef"):
                v = d.get(a)
                if v and v not in seen:
                    stack.append(v)
    if name is None and path:
        name = os.path.basename(path)
    return name, path


def _ticks_to_sec(el, tag, default=0.0):
    ch = el.find(tag)
    if ch is not None and ch.text and ch.text.strip().lstrip("-").isdigit():
        return int(ch.text) / TICK
    return default


def extract_audio_clips(prproj_path, sequence_name):
    """지정 시퀀스의 오디오 클립 목록을 추출.

    반환: list of dict {track, index, media, path, number, start, end}
          (start/end 는 시퀀스 타임라인 기준 초)
    """
    root, byid = load_project(prproj_path)
    seq = _find_sequence(root, sequence_name)
    if seq is None:
        raise ValueError(
            f"시퀀스 '{sequence_name}' 를 찾을 수 없음. 사용 가능: {list_sequences(root)}"
        )
    atg = _audio_trackgroup(byid, seq)
    if atg is None:
        raise ValueError("이 시퀀스에서 오디오 트랙 그룹을 찾지 못함.")

    inner = atg.find("TrackGroup")
    tracks_el = inner.find("Tracks") if inner is not None else None
    if tracks_el is None:
        return []

    clips = []
    for tr in tracks_el.findall("Track"):
        track_idx = tr.get("Index")
        track_obj = _resolve(byid, tr.get("ObjectURef") or tr.get("ObjectRef"))
        if track_obj is None:
            continue
        # AudioClipTrack -> ClipTrack -> ClipItems -> TrackItems -> TrackItem[ObjectRef]
        for ti_ref in track_obj.iter("TrackItem"):
            ref = ti_ref.get("ObjectRef")
            if not ref:
                continue
            item = byid.get(ref)
            if item is None or item.tag != "AudioClipTrackItem":
                continue
            cti = item.find("ClipTrackItem")
            tiel = cti.find("TrackItem") if cti is not None else None
            if tiel is None:
                continue
            start = _ticks_to_sec(tiel, "Start", 0.0)
            end = _ticks_to_sec(tiel, "End", 0.0)
            if end <= 0:
                continue
            name, path = _media_for_item(byid, item)
            num = None
            if name:
                m = re.findall(r"(\d+)", name)
                if m:
                    num = int(m[-1])  # 파일명 끝 숫자 = 곡 순서
            clips.append({
                "track": track_idx, "media": name, "path": path,
                "number": num, "start": round(start, 3), "end": round(end, 3),
            })
    clips.sort(key=lambda c: c["start"])
    return clips


if __name__ == "__main__":
    import sys, json, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    if len(sys.argv) < 3:
        print("usage: python extract_layout.py <project.prproj> <시퀀스 이름>")
        if len(sys.argv) == 2:
            r, _ = load_project(sys.argv[1])
            print("시퀀스 목록:", list_sequences(r))
        sys.exit(1)
    out = extract_audio_clips(sys.argv[1], sys.argv[2])
    print(json.dumps(out, ensure_ascii=False, indent=2))
