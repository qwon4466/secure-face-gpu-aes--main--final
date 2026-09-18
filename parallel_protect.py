"""
병렬 INN 보호본 생성 워커 (multiprocessing 워커 프로세스 측 코드).

라즈베리파이 CPU에서 INN protect_roi(약 2.3초/프레임)가 저장의 병목이다.
탐지/인식(Hailo)은 VDevice가 1개뿐이라 프로세스 간 공유가 안 되므로 메인에
남겨두고, 순수 PyTorch(CPU)로 도는 INN protect_roi만 이 워커들에서 병렬 처리한다.

메인(recorder)이 프레임마다 탐지·인식을 끝낸 뒤 "보호할 얼굴 목록(faces)"과
원본 프레임을 payload로 넘기면, 워커는 그 얼굴들에 INN protect_roi를 적용해
보호본 프레임과 복원용 타일(tile_f32, crop_box)을 만들어 되돌려준다.

프로세스당 INN 모델 사본 1개를 init_worker에서 로드한다(torch 스레드=1로 고정해
워커 간 코어 과다경쟁 방지). 워커 수는 config.PARALLEL_WORKERS로 조절.
"""
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config as c
from core.anonymizer import INNAnonymizer

# ── 워커 전역 상태 (init_worker에서 1회 채움) ────────────────────────────────
_ANON = None        # 이 워커의 INN 모델 사본
_PW = None          # 복원 비밀번호

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Ubuntu/라즈베리파이
    "C:/Windows/Fonts/NanumGothic.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/gulim.ttc",
]


def _load_font(size: int) -> "ImageFont.ImageFont":
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _put_text(frame, text, pos, size, color_bgr):
    b, g, r = color_bgr
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    ImageDraw.Draw(pil).text(pos, text, font=_load_font(size), fill=(r, g, b))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _mosaic(frame, x1, y1, x2, y2):
    out = frame.copy()
    h, w = out.shape[:2]
    bx1, by1 = max(0, int(x1)), max(0, int(y1))
    bx2, by2 = min(w, int(x2)), min(h, int(y2))
    if bx2 > bx1 and by2 > by1:
        roi = out[by1:by2, bx1:bx2]
        if roi.size > 0:
            out[by1:by2, bx1:bx2] = cv2.GaussianBlur(roi, (99, 99), 30)
    return out


def _draw(frame, x1, y1, x2, y2, name, group, sim):
    if name != "Unknown":
        color = (0, 200, 0)
        label = f"허가자 ({sim:.2f})"
    else:
        color = (0, 0, 220)
        label = f"비허가자 ({sim:.2f})"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text_y = max(y1 - 28, 5)
    return _put_text(frame, label, (x1, text_y), 18, color)


def init_worker(checkpoint, password, threads=1):
    """
    각 워커 프로세스가 시작될 때 1회 호출. INN 모델을 로드한다.
    threads: 이 워커가 INN 추론에 쓸 CPU 스레드 수(torch intra-op).
             워커 여러 개가 코어를 나눠 쓰도록 config에서 조절한다.
    """
    global _ANON, _PW
    try:
        t = max(1, int(threads))
    except Exception:
        t = 1
    try:
        import torch
        torch.set_num_threads(t)
    except Exception:
        pass
    try:
        # OpenCV 연산도 워커 간 코어 경쟁을 줄이도록 같은 값으로 제한
        cv2.setNumThreads(t)
    except Exception:
        pass
    _ANON = INNAnonymizer(checkpoint_path=checkpoint) if checkpoint else None
    _PW = password



def load_anonymizer(checkpoint, password="forensic2026"):
    """
    서버(PC GPU)에서 INN 모델을 1회 로드하기 위한 헬퍼 (init_worker의 서버판).
    INNAnonymizer는 CUDA가 있으면 자동으로 GPU에 올라간다.
    """
    global _ANON, _PW
    if _ANON is None:
        _ANON = INNAnonymizer(checkpoint_path=checkpoint) if checkpoint else None
    _PW = password
    return _ANON


def protect_one(frame, faces, password):
    """
    탐지·인식이 끝난 얼굴 목록(faces)에 INN protect_roi를 적용해 보호본
    프레임과 복원용 타일을 만든다. 로드된 _ANON(INN)을 사용한다.

    faces: [{"box":[x1,y1,x2,y2], "protect":bool, "name","group","sim"}, ...]
    반환: (anon_frame, tiles)
      tiles: [{"tile_f32": np.float32, "crop_box": [...]}, ...]  # 복원용
    """
    out = frame
    tiles = []
    for f in faces:
        x1, y1, x2, y2 = f["box"]
        if f.get("protect"):
            if _ANON is not None:
                try:
                    out, tile_f32, crop_box = _ANON.protect_roi(
                        out, [x1, y1, x2, y2], password
                    )
                    tiles.append({"tile_f32": tile_f32, "crop_box": crop_box})
                except Exception as e:
                    print(f"[INN] protect 실패 → 모자이크: {e}")
                    out = _mosaic(out, x1, y1, x2, y2)
            else:
                out = _mosaic(out, x1, y1, x2, y2)
        out = _draw(out, x1, y1, x2, y2, f["name"], f["group"], f["sim"])
    return out, tiles


def protect_job(payload):
    """
    한 프레임의 보호본을 생성한다 (워커 프로세스에서 실행).

    payload: (frame_id, frame(np.uint8 HxWx3), faces, ts)
    반환: (frame_id, anon_frame, tiles, ts)
    """
    frame_id, frame, faces, ts = payload
    out, tiles = protect_one(frame, faces, _PW)
    return frame_id, out, tiles, ts
