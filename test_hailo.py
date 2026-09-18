"""
Hailo SCRFD 탐지 + ArcFace 임베딩 단독 검증.
camera_stream 통합 전에 Hailo 추론이 제대로 되는지 확인.

사용:
  python test_hailo.py [이미지경로]
  인자 없으면 카메라(CAMERA_DEVICE)에서 한 프레임 캡처.
"""
import sys
import time

from core.camera_manager import CameraManager
import cv2
import numpy as np

import config as c
from hailo_infer import HAILO_AVAILABLE, HailoSCRFD, HailoArcFace

SCRFD_HEF = "/usr/share/hailo-models/scrfd_2.5g_h8l.hef"
ARCFACE_HEF = "/usr/share/hailo-models/arcface_mobilefacenet.hef"


def get_image(arg):
    if arg:
        return cv2.imread(arg)
    cap = CameraManager().open()
    for _ in range(10):
        ret, f = cap.read()
        if ret and f is not None and f.mean() > 3:
            cap.release()
            return f
    cap.release()
    return None


def main():
    if not HAILO_AVAILABLE:
        print("[Test] hailo_platform import 실패 — HailoRT Python 패키지 확인")
        return

    arg = sys.argv[1] if len(sys.argv) > 1 else None
    img = get_image(arg)
    if img is None:
        print("[Test] 이미지를 얻지 못했습니다.")
        return
    print(f"[Test] 입력 {img.shape}")

    # SCRFD
    print("[Test] SCRFD 로드...")
    det_model = HailoSCRFD(SCRFD_HEF, conf_thresh=0.5)
    t0 = time.time()
    det, kpss = det_model.detect(img, max_num=0)
    dt = time.time() - t0
    if det is None:
        print(f"[Test] 얼굴 미탐지 ({dt*1000:.0f}ms). conf_thresh를 낮춰보세요.")
        return
    print(f"[Test] 탐지 {len(det)}명 ({dt*1000:.0f}ms)")
    for i, b in enumerate(det):
        print(f"   #{i}: bbox={b[:4].astype(int).tolist()} score={b[4]:.2f}")

    # ArcFace
    print("[Test] ArcFace 로드...")
    from insightface.utils import face_align
    rec_model = HailoArcFace(ARCFACE_HEF)
    aligned = face_align.norm_crop(img, landmark=kpss[0], image_size=112)
    t0 = time.time()
    emb = rec_model.get_feat(aligned)
    dt = time.time() - t0
    print(f"[Test] 임베딩 shape={emb.shape} norm={np.linalg.norm(emb):.3f} "
          f"({dt*1000:.0f}ms)")
    print(f"   앞 5개 값: {emb[:5]}")

    # 시각화 저장
    vis = img.copy()
    for b in det:
        x1, y1, x2, y2 = b[:4].astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for kp in kpss:
        for p in kp:
            cv2.circle(vis, tuple(p.astype(int)), 2, (0, 0, 255), -1)
    cv2.imwrite("test_hailo_result.jpg", vis)
    print("[Test] 시각화 저장: test_hailo_result.jpg")
    print("[Test] ✅ Hailo 탐지/인식 동작 확인 완료")


if __name__ == "__main__":
    main()
