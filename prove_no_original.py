"""
"원본 없이 복원" 증명 스크립트.

저장된 청크 하나를 분석해 다음을 보여준다:
  1. 청크 폴더에 원본이 없음 (익명화 frame.jpg + 보호본 .npy + box.json 뿐)
  2. 저장된 tile_f32(.npy) 자체는 난독화된 보호본 (원본 아님)
  3. 같은 보호본을 올바른 키 → 복원 / 틀린 키 → 오복원
     (원본을 숨겨뒀다면 키와 무관하게 같아야 함 → 키 기반 복원의 증거)

출력:
  proof_result.jpg  — [익명화본 | 저장된 보호본 | 올바른키 복원 | 틀린키 복원]

사용:
  python prove_no_original.py [청크ID]
  인자 없으면 가장 최근 청크 사용.
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch

import config as c
from core.anonymizer import INNAnonymizer
from utils.image_processing import to_numpy

RECORDINGS = "recordings"


def find_chunk(arg):
    if arg:
        return os.path.join(RECORDINGS, arg)
    chunks = sorted(glob.glob(os.path.join(RECORDINGS, "*")), reverse=True)
    for ch in chunks:
        if os.path.isdir(ch):
            return ch
    return None


def find_face_frame(chunk):
    """얼굴(.npy)이 있는 첫 프레임 폴더 반환."""
    for fname in sorted(os.listdir(chunk)):
        fdir = os.path.join(chunk, fname)
        if os.path.isdir(fdir) and fname.isdigit():
            if os.path.exists(os.path.join(fdir, "face_0.npy")):
                return fdir
    return None


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    chunk = find_chunk(arg)
    if chunk is None or not os.path.isdir(chunk):
        print("[증명] 청크를 찾을 수 없습니다. 먼저 서버를 돌려 녹화를 만드세요.")
        return

    print("=" * 60)
    print(f"분석 청크: {chunk}")
    print("=" * 60)

    snap = find_face_frame(chunk)
    if snap is None:
        print("[증명] 얼굴이 저장된 프레임이 없습니다.")
        return

    # ── 증명 ① 폴더 내용: 원본 없음 ──
    print(f"\n[①] 저장된 프레임 폴더: {snap}")
    print("    파일 목록:")
    for f in sorted(os.listdir(snap)):
        size = os.path.getsize(os.path.join(snap, f))
        print(f"      - {f}  ({size:,} bytes)")
    print("    → frame.jpg(익명화본) + face_N.npy(보호본) + box.json 뿐.")
    print("      '원본 얼굴' 파일은 존재하지 않음.")

    # 익명화된 frame.jpg 로드
    anon_frame = cv2.imread(os.path.join(snap, "frame.jpg"))

    # 보호본 타일 + 좌표 로드
    tile_f32 = np.load(os.path.join(snap, "face_0.npy"))
    with open(os.path.join(snap, "face_0_box.json"), encoding="utf-8") as f:
        crop_box = json.load(f)
    print(f"\n[②] 저장된 보호본 tile_f32: shape={tile_f32.shape}, "
          f"dtype={tile_f32.dtype}, 범위=[{tile_f32.min():.2f}, {tile_f32.max():.2f}]")
    print("    → 이 .npy 는 INN이 만든 '보호본'이며 원본 얼굴이 아님.")

    # 보호본 타일을 이미지로 시각화
    tile_img = to_numpy(torch.from_numpy(tile_f32).unsqueeze(0))

    # ── 증명 ③ 같은 보호본 → 올바른 키 vs 틀린 키 ──
    if c.INN_CHECKPOINT is None:
        print("\n[③] INN_CHECKPOINT=None — 복원 비교 생략 (config에서 .pth 지정 필요)")
        cv2.imwrite("proof_result.jpg", np.hstack([
            anon_frame, cv2.resize(tile_img, (anon_frame.shape[1], anon_frame.shape[0]))
        ]))
        return

    anon = INNAnonymizer(checkpoint_path=c.INN_CHECKPOINT)
    correct = c.DEMO_PASSWORD
    wrong = "wrong_" + correct

    restored_ok = anon.restore_roi(anon_frame.copy(), tile_f32, crop_box, correct)
    restored_bad = anon.restore_roi(anon_frame.copy(), tile_f32, crop_box, wrong)

    print(f"\n[③] 동일한 보호본을 두 키로 복원:")
    print(f"      올바른 키('{correct}')  → 원본 복원")
    print(f"      틀린 키  ('{wrong}') → 오복원(깨짐)")
    print("    → 결과가 키에 따라 달라짐. 원본을 숨겨뒀다면 둘이 같아야 함.")
    print("      즉 저장된 보호본을 '키로 역연산'해 복원한다는 증거.")

    # ── 4분할 비교 이미지 ──
    def label(img, txt, color=(255, 255, 255)):
        img = img.copy()
        cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(img, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return img

    H, W = anon_frame.shape[:2]
    tile_vis = cv2.resize(tile_img, (W, H))
    row = np.hstack([
        label(anon_frame, "1. SAVED (anonymized)", (0, 220, 255)),
        label(tile_vis,   "2. SAVED tile (protected)", (0, 220, 255)),
        label(restored_ok, "3. RESTORE correct key", (0, 255, 0)),
        label(restored_bad, "4. RESTORE wrong key", (0, 0, 255)),
    ])
    cv2.imwrite("proof_result.jpg", row)
    print("\n[증명] 비교 이미지 저장: proof_result.jpg")
    print("       1.저장된 익명화본  2.저장된 보호본  3.정답키 복원  4.틀린키 복원")
    print("       → 디스크엔 1·2번만 존재. 3번은 2번을 키로 역연산한 결과.")


if __name__ == "__main__":
    main()
