"""
INN 복원 라운드트립 검증.

흐름:
  얼굴 이미지 → SCRFD 탐지 → protect_roi(익명화+tile) → restore_roi(복원)
  → original / anonymized / restored 3장 나란히 저장 + PSNR 출력

사용:
  python test_restore.py [이미지경로]
  인자 없으면 demo.mp4 첫 프레임 또는 registered_faces 첫 이미지 사용.
"""
import glob
import os
import sys

import cv2
import numpy as np
from insightface.app import FaceAnalysis

import config as c
from core.anonymizer import INNAnonymizer


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def find_best_face_frame(video_path: str, detector, samples: int = 40):
    """
    영상을 samples개 구간으로 나눠 스캔하며 가장 큰 얼굴이 있는 프레임 선택.
    얼굴 bbox 면적이 클수록(=가깝고 선명할수록) 좋은 프레임으로 간주.

    Returns: (best_frame, best_bbox, best_frame_idx) 또는 (None, None, -1)
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    step = max(1, total // samples)

    best_area = 0
    best_frame = None
    best_bbox = None
    best_idx = -1

    idx = 0
    while idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        bboxes, _ = detector.detect(frame, max_num=0, metric="default")
        if bboxes is not None and len(bboxes) > 0:
            for b in bboxes:
                x1, y1, x2, y2 = b[:4]
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best_frame = frame.copy()
                    best_bbox = [int(x1), int(y1), int(x2), int(y2)]
                    best_idx = idx
        idx += step

    cap.release()
    if best_frame is not None:
        w = best_bbox[2] - best_bbox[0]
        h = best_bbox[3] - best_bbox[1]
        print(f"[Test] 최적 프레임 #{best_idx} 선택 — 얼굴 크기 {w}x{h}px")
    return best_frame, best_bbox, best_idx


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    print(f"[Test] INN_CHECKPOINT = {c.INN_CHECKPOINT}")

    # SCRFD 탐지기 준비
    fa = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
    fa.prepare(ctx_id=-1, det_thresh=0.5)
    detector = fa.models["detection"]

    # 입력 선택: 인자 이미지 > demo.mp4 최적 프레임 > registered_faces
    frame = None
    bbox = None
    if arg and os.path.exists(arg):
        frame = cv2.imread(arg)
    elif os.path.exists("demo.mp4"):
        print("[Test] demo.mp4 스캔 — 가장 큰 얼굴 프레임 탐색 중...")
        frame, bbox, _ = find_best_face_frame("demo.mp4", detector)
    if frame is None:
        imgs = (glob.glob("registered_faces/*.jpg")
                + glob.glob("registered_faces/*.png"))
        if imgs:
            print(f"[Test] {imgs[0]} 사용")
            frame = cv2.imread(imgs[0])

    if frame is None:
        print("[Test] 테스트 이미지를 찾을 수 없습니다. "
              "python test_restore.py <얼굴사진.jpg>")
        return

    print(f"[Test] 입력 프레임 shape={frame.shape}")

    # bbox가 아직 없으면(이미지 입력) 가장 큰 얼굴 탐지
    if bbox is None:
        bboxes, _ = detector.detect(frame, max_num=0, metric="default")
        if bboxes is None or len(bboxes) == 0:
            print("[Test] 얼굴 탐지 실패 — 다른 이미지로 시도하세요.")
            return
        # 가장 큰 얼굴 선택
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
        best = int(np.argmax(areas))
        bbox = bboxes[best, :4].astype(int).tolist()
    x1, y1, x2, y2 = bbox
    print(f"[Test] 얼굴 영역: [{x1},{y1},{x2},{y2}]  크기={x2-x1}x{y2-y1}px")

    # INN 준비
    anon = INNAnonymizer(checkpoint_path=c.INN_CHECKPOINT)
    pw = c.DEMO_PASSWORD

    # protect
    modified, tile_f32, crop_box = anon.protect_roi(frame, [x1, y1, x2, y2], pw)
    print(f"[Test] protect 완료 — tile shape={tile_f32.shape}, "
          f"범위=[{tile_f32.min():.2f}, {tile_f32.max():.2f}]")

    # restore (올바른 비밀번호)
    restored = anon.restore_roi(modified.copy(), tile_f32, crop_box, pw)

    # restore (틀린 비밀번호) — 오복원 확인
    wrong = anon.restore_roi(modified.copy(), tile_f32, crop_box, "wrongpassword")

    # 얼굴 영역만 잘라서 PSNR 비교
    ox1, oy1, ox2, oy2 = crop_box
    orig_face = frame[oy1:oy2, ox1:ox2]
    rest_face = restored[oy1:oy2, ox1:ox2]
    p = psnr(orig_face, rest_face)
    print(f"\n[결과] 원본 vs 복원 PSNR = {p:.2f} dB")
    if p > 25:
        print("       ✅ 복원 양호 (25dB↑)")
    elif p > 18:
        print("       ⚠️  복원 다소 흐림 (18~25dB)")
    else:
        print("       ❌ 복원 실패 — 가중치 미로딩 또는 키 불일치 의심")

    # 4분할 비교 이미지 저장
    def label(img, txt):
        img = img.copy()
        cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(img, txt, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        return img

    h = frame.shape[0]
    row = np.hstack([
        label(frame, "1. ORIGINAL"),
        label(modified, "2. ANONYMIZED"),
        label(restored, "3. RESTORED (correct pw)"),
        label(wrong, "4. WRONG pw"),
    ])
    out_path = "test_restore_result.jpg"
    cv2.imwrite(out_path, row)
    print(f"\n[Test] 비교 이미지 저장: {out_path}")
    print("       1.원본  2.익명화  3.복원(정답)  4.복원(틀린키)")


if __name__ == "__main__":
    main()
