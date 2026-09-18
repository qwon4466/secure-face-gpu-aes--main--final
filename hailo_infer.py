"""
HailoRT 추론 래퍼 — SCRFD 얼굴 탐지 + ArcFace 인식 (Hailo-8L).

insightface의 detector/recognizer를 대체한다.
  HailoSCRFD.detect(img, max_num) → (det[N,5], kpss[N,5,2])   ← insightface와 동일 시그니처
  HailoArcFace.get_feat(aligned112) → (512,) float32 L2정규화

전처리/정규화는 HEF 내부에 컴파일돼 있다고 가정(UINT8 0~255 입력).
출력은 FLOAT32로 받아 자동 dequantize.

주의: 첫 통합이므로 라즈베리파이에서 test_hailo.py로 검증하며
score 임계값/색공간 등을 조정해야 할 수 있다.
"""
import threading

import cv2
import numpy as np

# Hailo는 한 번에 하나의 network group만 activate 가능.
# 여러 스레드(카메라/녹화)의 추론을 직렬화.
_HAILO_LOCK = threading.Lock()

try:
    from hailo_platform import (
        HEF, VDevice, ConfigureParams, HailoStreamInterface,
        InferVStreams, InputVStreamParams, OutputVStreamParams, FormatType,
    )
    HAILO_AVAILABLE = True
except Exception as e:  # noqa
    HAILO_AVAILABLE = False
    _IMPORT_ERR = e


# ── SCRFD 후처리 헬퍼 (insightface 로직 포팅) ──────────────────────────────

def _distance2bbox(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points, distance):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def _nms(dets, scores, thresh):
    x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


# ── HailoRT 단일 네트워크 래퍼 ─────────────────────────────────────────────

# Hailo-8L은 물리 디바이스 1개 → 모든 모델이 하나의 VDevice를 공유해야 함.
_SHARED_VDEVICE = None


def _get_vdevice():
    global _SHARED_VDEVICE
    if _SHARED_VDEVICE is None:
        _SHARED_VDEVICE = VDevice()
    return _SHARED_VDEVICE


class _HailoNet:
    def __init__(self, hef_path: str):
        if not HAILO_AVAILABLE:
            raise RuntimeError(f"hailo_platform import 실패: {_IMPORT_ERR}")
        self.hef = HEF(hef_path)
        self.target = _get_vdevice()  # 공유 VDevice
        cfg = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_group = self.target.configure(self.hef, cfg)[0]
        self.ng_params = self.network_group.create_params()
        self.input_info = self.hef.get_input_vstream_infos()[0]
        self.input_vparams = InputVStreamParams.make(
            self.network_group, format_type=FormatType.UINT8
        )
        self.output_vparams = OutputVStreamParams.make(
            self.network_group, format_type=FormatType.FLOAT32
        )

    def infer(self, img_hwc: np.ndarray) -> dict:
        """img_hwc: (H,W,3) uint8 RGB → {출력이름: (1,...) float32}"""
        data = {self.input_info.name: np.expand_dims(img_hwc, 0).astype(np.uint8)}
        # 전역 락으로 activate 충돌 방지 (한 번에 하나의 추론만)
        with _HAILO_LOCK:
            with InferVStreams(
                self.network_group, self.input_vparams, self.output_vparams
            ) as pipeline:
                with self.network_group.activate(self.ng_params):
                    return pipeline.infer(data)


# ── SCRFD 탐지 ─────────────────────────────────────────────────────────────

class HailoSCRFD:
    def __init__(self, hef_path: str, conf_thresh: float = 0.5,
                 nms_thresh: float = 0.4):
        self.net = _HailoNet(hef_path)
        self.size = 640
        self.conf = conf_thresh
        self.nms = nms_thresh
        self.num_anchors = 2
        self._anchor_cache = {}

    def _anchors(self, H, W, stride):
        key = (H, W, stride)
        if key not in self._anchor_cache:
            ac = np.stack(np.mgrid[:H, :W][::-1], axis=-1).astype(np.float32)
            ac = (ac * stride).reshape(-1, 2)
            ac = np.stack([ac] * self.num_anchors, axis=1).reshape(-1, 2)
            self._anchor_cache[key] = ac
        return self._anchor_cache[key]

    def detect(self, img: np.ndarray, max_num: int = 0, metric: str = "default"):
        h0, w0 = img.shape[:2]
        scale = min(self.size / h0, self.size / w0)
        nh, nw = int(round(h0 * scale)), int(round(w0 * scale))
        resized = cv2.resize(img, (nw, nh))
        padded = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        padded[:nh, :nw] = resized
        inp = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

        out = self.net.infer(inp)

        # 출력을 채널 수로 분류: 2=score, 8=bbox, 20=kps
        by_stride = {}
        for name, arr in out.items():
            a = arr[0]  # (H,W,C)
            H, W, C = a.shape
            stride = self.size // H
            d = by_stride.setdefault(stride, {})
            if C == 2:
                d["score"] = a.reshape(-1)
            elif C == 8:
                d["bbox"] = a.reshape(-1, 4) * stride
            elif C == 20:
                d["kps"] = a.reshape(-1, 10) * stride
            d["HW"] = (H, W)

        scores_l, bboxes_l, kpss_l = [], [], []
        for stride, d in by_stride.items():
            if "score" not in d or "bbox" not in d or "kps" not in d:
                continue
            H, W = d["HW"]
            ac = self._anchors(H, W, stride)
            score = d["score"]
            pos = np.where(score >= self.conf)[0]
            if len(pos) == 0:
                continue
            scores_l.append(score[pos])
            bboxes_l.append(_distance2bbox(ac, d["bbox"])[pos])
            kpss_l.append(_distance2kps(ac, d["kps"])[pos])

        if not bboxes_l:
            return None, None

        scores = np.concatenate(scores_l)
        bboxes = np.concatenate(bboxes_l) / scale
        kpss = np.concatenate(kpss_l) / scale

        keep = _nms(bboxes, scores, self.nms)
        if max_num > 0 and len(keep) > max_num:
            keep = keep[:max_num]
        det = np.concatenate([bboxes[keep], scores[keep, None]], axis=1).astype(np.float32)
        kpss = kpss[keep].reshape(-1, 5, 2).astype(np.float32)
        return det, kpss


# ── ArcFace 인식 ───────────────────────────────────────────────────────────

class HailoArcFace:
    def __init__(self, hef_path: str):
        self.net = _HailoNet(hef_path)
        self.out_name = self.net.hef.get_output_vstream_infos()[0].name

    def get_feat(self, aligned112: np.ndarray) -> np.ndarray:
        """aligned112: (112,112,3) BGR uint8 → (512,) float32 L2정규화"""
        inp = cv2.cvtColor(aligned112, cv2.COLOR_BGR2RGB)
        out = self.net.infer(inp)
        emb = np.asarray(out[self.out_name]).reshape(-1).astype(np.float32)
        n = np.linalg.norm(emb)
        return emb / n if n > 0 else emb
