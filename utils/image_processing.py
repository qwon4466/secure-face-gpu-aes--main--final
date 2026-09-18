"""
Obfuscator — 원본 ProFace S utils/image_processing.py 의 난독화 클래스 이식.

원본 지원 타입:
  'blur'       : Blur(kernel, sigma_min, sigma_max) — F.gaussian_blur
  'pixelate'   : Pixelate(block_size)
  'median'     : MedianBlur(kernel) — kornia
  'mask'       : Mask (CartoonSet 스티커)
  'hybridAll'  : 위 전체 무작위 (학습 시 사용)

입력/출력: torch.Tensor (B, 3, H, W), [0, 1] 범위
targ_img : FaceShifter/SimSwap용 (기본 None)

강도 하한 검증 (FR-3.3): blur sigma < BLUR_SIGMA_MIN 이면 경고
"""

import random
import warnings

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import config as c


# ── 개별 난독화 클래스 ─────────────────────────────────────────────

class Blur:
    """
    원본: Blur(kernel, sigma_min, sigma_max)
    - 학습 시: sigma = random.uniform(sigma_min, sigma_max)
    - 추론 시: sigma = (sigma_min + sigma_max) / 2 권장
    """
    def __init__(
        self,
        kernel: int   = c.BLUR_KERNEL_SIZE,
        sigma_min: float = c.BLUR_SIGMA_MIN,
        sigma_max: float = c.BLUR_SIGMA,
        train_mode: bool = False,
    ):
        if kernel % 2 == 0:
            kernel += 1
        self.kernel    = kernel
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.train_mode = train_mode

    def __call__(self, x: torch.Tensor, _targ=None) -> torch.Tensor:
        sigma = (
            random.uniform(self.sigma_min, self.sigma_max)
            if self.train_mode
            else (self.sigma_min + self.sigma_max) / 2.0
        )
        return TF.gaussian_blur(x, self.kernel, sigma)


class Pixelate:
    """원본: Pixelate(block_size) — 블록 평균 다운샘플 후 업샘플"""
    def __init__(self, block_size: int = c.PIXELATE_BLOCK):
        self.block = block_size

    def __call__(self, x: torch.Tensor, _targ=None) -> torch.Tensor:
        B, C, H, W = x.shape
        small = F.avg_pool2d(x, self.block, stride=self.block)
        return F.interpolate(small, size=(H, W), mode='nearest')


class MedianBlur:
    """
    원본: MedianBlur(kernel) — kornia.filters.median_blur
    kornia 없으면 average pool로 근사
    """
    def __init__(self, kernel: int = c.MEDIAN_KERNEL):
        if kernel % 2 == 0:
            kernel += 1
        self.kernel = kernel
        self._kornia = None
        try:
            import kornia.filters as kf
            self._kornia = kf.median_blur
        except ImportError:
            warnings.warn("[MedianBlur] kornia 없음 - avg_pool로 근사합니다.", stacklevel=1)

    def __call__(self, x: torch.Tensor, _targ=None) -> torch.Tensor:
        if self._kornia is not None:
            return self._kornia(x, (self.kernel, self.kernel))
        # fallback: reflect pad + avg_pool
        p = self.kernel // 2
        xp = F.pad(x, [p]*4, mode='reflect')
        return F.avg_pool2d(xp, self.kernel, stride=1)


class Mask:
    """
    원본: Mask — 검정 마스크 (CartoonSet 스티커 대체 단순 구현)
    운영 시 CartoonSet 스티커를 로드하도록 확장 가능.
    """
    def __init__(self, fill: float = 0.0):
        self.fill = fill

    def __call__(self, x: torch.Tensor, _targ=None) -> torch.Tensor:
        return torch.full_like(x, self.fill)


# ── 통합 Obfuscator 클래스 ────────────────────────────────────────

class Obfuscator:
    """
    원본 Obfuscator 클래스에 대응.
    obf_type 문자열로 난독화 방식을 동적 선택.

    지원 타입: 'blur', 'pixelate', 'median', 'mask', 'hybridAll'
    """
    TYPES = ('blur', 'pixelate', 'median', 'mask', 'hybridAll')

    def __init__(
        self,
        obf_type: str  = c.DEFAULT_OBFUSCATOR,
        train_mode: bool = False,
        **kwargs,
    ):
        if obf_type not in self.TYPES:
            raise ValueError(f"obf_type 은 {self.TYPES} 중 하나여야 합니다.")
        self.obf_type   = obf_type
        self.train_mode = train_mode

        self._blur      = Blur(train_mode=train_mode)
        self._pixelate  = Pixelate()
        self._median    = MedianBlur()
        self._mask      = Mask()
        self._pool = {
            'blur':     self._blur,
            'pixelate': self._pixelate,
            'median':   self._median,
            'mask':     self._mask,
        }

    def _check_strength(self):
        """FR-3.3: 블러 sigma 하한 검증."""
        sigma = (self._blur.sigma_min + self._blur.sigma_max) / 2.0
        if sigma < c.BLUR_SIGMA_MIN:
            warnings.warn(
                f"[Obfuscator] sigma={sigma:.1f} < 하한 {c.BLUR_SIGMA_MIN}. "
                "사전난독이 약하면 보호 강도가 저하됩니다 (FR-3.3).",
                stacklevel=3,
            )

    def __call__(self, x: torch.Tensor, targ_img=None) -> torch.Tensor:
        if self.obf_type == 'hybridAll':
            fn = random.choice(list(self._pool.values()))
        else:
            self._check_strength()
            fn = self._pool[self.obf_type]
        return fn(x, targ_img)


# ── 이미지 변환 유틸 ──────────────────────────────────────────────

def to_tensor(img_np, device: torch.device | None = None) -> torch.Tensor:
    """numpy HWC BGR uint8 → tensor BCHW float32 [-1,1]

    원본 input_trans:
        transforms.ToTensor()               # [0,1]
        transforms.Normalize(mean=0.5, std=0.5)  # [-1,1]
    """
    import numpy as np
    import cv2
    # BGR → RGB
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
    t = t.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    t = t * 2.0 - 1.0                    # [0,1] → [-1,1]
    if device is not None:
        t = t.to(device)
    return t


def to_numpy(tensor: torch.Tensor) -> "np.ndarray":
    """tensor BCHW [-1,1] → numpy HWC BGR uint8

    원본 normalize(x): (x - (-1)) / (1 - (-1)) = (x + 1) / 2  → [0,1]
    """
    import numpy as np
    import cv2
    t = tensor.squeeze(0).permute(1, 2, 0).clamp(-1.0, 1.0)
    arr = ((t.detach().cpu().float().numpy() + 1.0) / 2.0 * 255).astype(np.uint8)
    # RGB → BGR
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
