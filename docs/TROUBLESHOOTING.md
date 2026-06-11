# 문제 해결

## `RuntimeError: Library cublas64_12.dll is not found or cannot be loaded`
GPU(CUDA)용 DLL을 못 찾는 경우. faster-whisper + CUDA 12 조합의 흔한 문제입니다.

**해결:**
```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```
- 이 프로젝트는 `src/cuda_setup.py` 가 실행 시 자동으로 위 휠의 DLL 경로를 등록합니다
  (`run.py` 가 faster_whisper import 전에 호출).
- 그래도 안 되면 NVIDIA 드라이버를 최신으로 업데이트하세요.
- GPU가 아예 없으면 `config.json` 에서 `"device": "cpu"`, `"compute_type": "int8"` 로 설정
  (느리지만 동작). 코드도 CUDA 실패 시 자동으로 CPU로 폴백합니다.

## 시퀀스를 찾을 수 없다고 나옴
- `config.json` 의 `sequence` 이름이 Premiere의 시퀀스 이름과 **정확히** 같아야 합니다(공백 포함).
- 목록 확인:
  ```bash
  python src/extract_layout.py "C:/.../프로젝트.prproj"
  ```

## 오디오 파일을 못 찾음
- 프로젝트에 저장된 미디어 경로가 현재 PC와 다를 수 있습니다.
- `config.json` 에 `"audio_dir"` 로 실제 오디오 폴더를 지정하세요. 파일명으로 다시 찾습니다.

## 타이밍이 안 맞음 / 줄이 너무 길거나 짧음
- 노래 인식 특성상 약간의 오차는 정상입니다. Premiere에서 미세 조정하세요.
- 한 자막에 두 줄이 붙으면 `split_gap` 을 낮추고, 너무 잘게 쪼개지면 높이세요(기본 0.8).

## 한글이 깨져 보임
- 모든 입출력은 UTF-8 입니다. 가사 `.txt` 파일도 UTF-8 로 저장하세요.

## .prproj 파싱 실패
- Premiere 버전에 따라 내부 구조가 다를 수 있습니다. 최신 Premiere Pro(2023+) 기준으로 작성되었습니다.
- 압축이 안 된 평문 XML `.prproj` 도 지원합니다.
