"""
저장된 PSF 보호본 청크를 복원해 실제시간 mp4 생성 (코랩 GPU / 로컬 공용).

라즈베리파이가 저장한 청크 폴더(frame.jpg + face_N.npy + face_N_box.json
+ meta.json)를 받아, INN 역변환으로 원본 얼굴을 복원한다.
카메라·서버 없이 저장된 보호본만으로 동작하며, INN이 GPU에서 돌면 빠르다.

각 프레임을 촬영 시각(meta.json ts) 간격만큼 유지해 실제 시간 길이 영상 생성.

사용:
  python restore_chunk.py <청크폴더경로> [비밀번호] [출력.mp4]
예:
  python restore_chunk.py recordings/2026-06/29/오후/14시/14-00 forensic2026 out.mp4
"""
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

import config as c
from core.anonymizer import INNAnonymizer


def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _is_chunk(path):
    """path가 청크 폴더인지(숫자 하위폴더에 frame.jpg 존재) 확인."""
    try:
        subs = sorted(d for d in os.listdir(path) if d.isdigit())
    except Exception:
        return False
    return bool(subs) and os.path.exists(os.path.join(path, subs[0], "frame.jpg"))


def _find_chunk(path):
    """path 자체가 청크가 아니면 하위를 재귀 탐색해 청크 폴더를 찾는다."""
    if _is_chunk(path):
        return path
    for root, _dirs, _fs in os.walk(path):
        if _is_chunk(root):
            return root
    return None


def restore_chunk(chunk_path, password="forensic2026", out_path="restored.mp4",
                  out_fps=15):
    import time as _time
    _t_restore = _time.time()   # 💡 복원 시간 측정 시작
    # 경로 안에서 실제 청크 폴더 자동 탐색 (압축 해제 최상위를 줘도 됨)
    found = _find_chunk(chunk_path)
    if found is None:
        print(f"[Restore] 청크(frame.jpg 있는 숫자 폴더)를 찾지 못함: {chunk_path}")
        return None
    chunk_path = found
    print(f"[Restore] 청크 폴더: {chunk_path}")
    frame_dirs = sorted(d for d in os.listdir(chunk_path) if d.isdigit())
    if not frame_dirs:
        print("[Restore] 프레임이 없습니다.")
        return None

    print(f"[Restore] {len(frame_dirs)}프레임 복원 시작...")
    anon = INNAnonymizer(checkpoint_path=c.INN_CHECKPOINT)  # GPU 있으면 자동 cuda

    # 각 프레임 촬영 시각 수집 (실제 시간 길이 재현용)
    ts_list = []
    for fid in frame_dirs:
        m = _load_json(os.path.join(chunk_path, fid, "meta.json"))
        ts_list.append(m.get("ts", 0.0) if m else 0.0)

    raw_path = out_path.replace(".mp4", "_raw.mp4")
    writer = None
    n = len(frame_dirs)

    for idx, fid in enumerate(frame_dirs):
        snap = os.path.join(chunk_path, fid)
        frame = cv2.imread(os.path.join(snap, "frame.jpg"))
        if frame is None:
            continue

        # 저장된 타일(보호본)에서 INN 역변환 복원
        for i in range(20):
            npy = os.path.join(snap, f"face_{i}.npy")
            box = os.path.join(snap, f"face_{i}_box.json")
            if not os.path.exists(npy):
                break
            try:
                tile = np.load(npy)  # 깨진 파일이면 여기서 예외
                crop = _load_json(box)
                frame = anon.restore_roi(frame, tile, crop, password)
            except Exception as e:
                # 손상된 프레임 하나 때문에 전체가 죽지 않도록 건너뜀
                print(f"[Restore] {fid} face_{i} 건너뜀: {e}")
                continue

        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(
                raw_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h)
            )

        # 실제 시간 반영: 다음 프레임까지 간격만큼 이 화면 유지
        if idx < n - 1 and ts_list[idx] > 0 and ts_list[idx + 1] > 0:
            dur = ts_list[idx + 1] - ts_list[idx]
        else:
            dur = 1.0 / out_fps
        hold = max(1, min(round(dur * out_fps), out_fps * 30))
        for _ in range(hold):
            writer.write(frame)

        if (idx + 1) % 50 == 0:
            print(f"[Restore] {idx + 1}/{n} 프레임 처리")

    if writer is None:
        print("[Restore] 복원 실패 (유효 프레임 없음)")
        return None
    writer.release()

    _elapsed = _time.time() - _t_restore
    print(f"✅ [복원완료] {n}프레임 {_elapsed:.1f}초 "
          f"({_elapsed / max(1, n) * 1000:.0f}ms/frame)")

    # ffmpeg으로 H.264 변환 (브라우저 재생 호환)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            subprocess.run(
                [ffmpeg, "-y", "-i", raw_path,
                 "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", out_path],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            os.remove(raw_path)
            print(f"[Restore] 완료 (H.264): {out_path}")
            return out_path
        except Exception as e:
            print(f"[Restore] ffmpeg 변환 실패 → mp4v 반환: {e}")
            return raw_path
    print(f"[Restore] ffmpeg 없음 → mp4v 반환: {raw_path}")
    return raw_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용: python restore_chunk.py <청크폴더> [비밀번호] [출력.mp4]")
        sys.exit(1)
    chunk = sys.argv[1]
    pw = sys.argv[2] if len(sys.argv) > 2 else "forensic2026"
    out = sys.argv[3] if len(sys.argv) > 3 else "restored.mp4"
    restore_chunk(chunk, pw, out)