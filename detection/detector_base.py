"""
검출기 플러그인 인터페이스 (FR-1.2)
MTCNN ↔ YOLO 교체 가능하도록 추상화.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class DetectionResult:
    bbox: list[int]             # [x1, y1, x2, y2] (원본 프레임 좌표)
    conf: float
    landmarks: np.ndarray | None = None  # (5, 2) 또는 None


class FaceDetector(ABC):
    """모든 검출기가 구현해야 하는 인터페이스."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[DetectionResult]:
        """
        image: HWC BGR uint8
        반환 : DetectionResult 리스트 (없으면 [])
        """

    def set_threshold(self, conf: float | None = None, iou: float | None = None):
        """신뢰도·NMS 임계값 변경 (FR-1.3)."""
