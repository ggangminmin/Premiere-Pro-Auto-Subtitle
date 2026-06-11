# -*- coding: utf-8 -*-
"""faster-whisper가 GPU(CUDA)를 쓰도록 nvidia cuBLAS/cuDNN DLL 경로를 등록한다.

Windows에서 `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` 로 설치한 DLL을
파이썬이 찾을 수 있게 add_dll_directory 로 등록한다. faster_whisper 를 import 하기
*전에* 반드시 enable_cuda_dlls() 를 먼저 호출해야 한다.
"""
import os, sys, site


def _candidate_site_dirs():
    dirs = set()
    try:
        for d in site.getsitepackages():
            dirs.add(d)
            dirs.add(os.path.join(d, "Lib", "site-packages"))
    except Exception:
        pass
    try:
        dirs.add(site.getusersitepackages())
    except Exception:
        pass
    dirs.add(os.path.join(sys.prefix, "Lib", "site-packages"))
    return [d for d in dirs if d and os.path.isdir(d)]


def enable_cuda_dlls(verbose=False):
    """nvidia cublas/cudnn bin 폴더를 DLL 검색 경로에 등록. 등록된 경로 리스트 반환."""
    registered = []
    for sp in _candidate_site_dirs():
        for sub in (("nvidia", "cublas", "bin"), ("nvidia", "cudnn", "bin")):
            p = os.path.join(sp, *sub)
            if os.path.isdir(p) and p not in registered:
                try:
                    os.add_dll_directory(p)
                    os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
                    registered.append(p)
                except Exception as e:
                    if verbose:
                        print(f"[cuda_setup] {p} 등록 실패: {e}")
    if verbose:
        if registered:
            print("[cuda_setup] CUDA DLL 경로 등록:", *registered, sep="\n  ")
        else:
            print("[cuda_setup] nvidia cublas/cudnn 휠을 찾지 못함 (CPU로 동작).")
    return registered
