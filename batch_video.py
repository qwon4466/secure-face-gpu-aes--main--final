"""
영상 배치 익명화 + 복원 검증.

입력 영상의 매 프레임(또는 N프레임마다)에 대해:
  1. SCRFD로 얼굴 탐지
  2. INN protect_roi 익명화 → 익명화 영상 저장
  3. 같은 프레임을 INN restore_roi 복원 → 복원 영상 저장

출력:
  out_anonymized.mp4   — 익명화 영상
  out_restored.mp4     — 복원 영상 (정답 비밀번호)
  out_compare.mp4      — 원본|익명화|복원 나란히

사용:
  python batch_video.py                  # demo.mp4 처리
  python batch_video.py input.mp4        # 특정 영상
  python batch_video.py input.mp4 2      # 2프레임마다 처리 (빠름)
"""
import sys
import time

import cv2
import numpy as np
from insightface.app import FaceAnalysis

import config as c
from core.anonymizer import INNAnonymizer


def pick_providers():
    """GPU 가능하면 CUDA, 아니면 CPU. (로컬·코랩 공용)"""
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" in ort.get_available_providers():
            print("[Batch] GPU 감지 → CUDAExecutionProvider 사용")
            return ["CUDAExecutionProvider", "CPUExecutionProvider"], 0
    except Exception:
        pass
    print("[Batch] CPU 모드 (CPUExecutionProvider)")
    return ["CPUExecutionProvider"], -1


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else "demo.mp4"
    frame_skip = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    # 처리할 최대 프레임 수 (0이면 전체). 검증용으로 일부만 처리할 때.
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Batch] 영상 열기 실패: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = fps / frame_skip
    print(f"[Batch] {video_path}  {W}x{H}  {fps:.0f}fps  총 {total}프레임")
    print(f"[Batch] {frame_skip}프레임마다 처리 → 출력 {out_fps:.1f}fps")
    print(f"[Batch] INN_CHECKPOINT = {c.INN_CHECKPOINT}")

    # 모델 준비 (한 번만)
    providers, ctx_id = pick_providers()
    fa = FaceAnalysis(name="buffalo_s", providers=providers)
    fa.prepare(ctx_id=ctx_id, det_thresh=0.5)
    detector = fa.models["detection"]
    anon = INNAnonymizer(checkpoint_path=c.INN_CHECKPOINT)
    pw = c.DEMO_PASSWORD

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w_anon = cv2.VideoWriter("out_anonymized.mp4", fourcc, out_fps, (W, H))
    w_rest = cv2.VideoWriter("out_restored.mp4", fourcc, out_fps, (W, H))
    w_comp = cv2.VideoWriter("out_compare.mp4", fourcc, out_fps, (W * 3, H))

    idx = 0
    processed = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        idx += 1
        if (idx - 1) % frame_skip != 0:
            continue

        original = frame.copy()

        # 얼굴 탐지 (모든 얼굴)
        bboxes, _ = detector.detect(frame, max_num=0, metric="default")
        tiles = []  # (tile_f32, crop_box)
        anon_frame = frame.copy()

        if bboxes is not None and len(bboxes) > 0:
            for b in bboxes:
                x1, y1, x2, y2 = b[:4].astype(int).tolist()
                try:
                    anon_frame, tile_f32, crop_box = anon.protect_roi(
                        anon_frame, [x1, y1, x2, y2], pw
                    )
                    tiles.append((tile_f32, crop_box))
                except Exception as e:
                    print(f"[Batch] protect 실패 frame#{idx}: {e}")

        # 복원: 익명화 프레임 + 저장 타일 → 원본 얼굴 복원
        rest_frame = anon_frame.copy()
        for tile_f32, crop_box in tiles:
            try:
                rest_frame = anon.restore_roi(rest_frame, tile_f32, crop_box, pw)
            except Exception as e:
                print(f"[Batch] restore 실패 frame#{idx}: {e}")

        # 라벨
        def lab(img, txt, color):
            img = img.copy()
            cv2.rectangle(img, (0, 0), (260, 28), (0, 0, 0), -1)
            cv2.putText(img, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            return img

        o = lab(original, "ORIGINAL", (255, 255, 255))
        a = lab(anon_frame, "ANONYMIZED", (0, 200, 255))
        r = lab(rest_frame, "RESTORED", (0, 255, 0))

        w_anon.write(a)
        w_rest.write(r)
        w_comp.write(np.hstack([o, a, r]))

        processed += 1
        if max_frames and processed >= max_frames:
            print(f"[Batch] max_frames={max_frames} 도달 → 중단")
            break
        if processed % 10 == 0:
            elapsed = time.time() - t_start
            spd = processed / elapsed
            eta = (total / frame_skip - processed) / spd if spd > 0 else 0
            print(f"[Batch] {processed}프레임 처리 "
                  f"({spd:.1f}fps, 남은시간 ~{eta:.0f}s, "
                  f"얼굴 {len(tiles)}명)")

    cap.release()
    w_anon.release()
    w_rest.release()
    w_comp.release()
    elapsed = time.time() - t_start
    print(f"\n[Batch] 완료! {processed}프레임, {elapsed:.0f}초")
    print("  out_anonymized.mp4  — 익명화 영상")
    print("  out_restored.mp4    — 복원 영상")
    print("  out_compare.mp4     — 원본|익명화|복원 비교")


if __name__ == "__main__":
    main()
