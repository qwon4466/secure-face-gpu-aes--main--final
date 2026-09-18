"""
DWT / IWT — 원본 ProFace S 의 modules/Unet_common.py 를 그대로 이식
dwt_init() / IWT 클래스의 수식을 변경 없이 사용.

DWT: (B, C, H, W) → (B, 4C, H/2, W/2)
     서브밴드 순서: LL, HL, LH, HH  (원본과 동일)

IWT: (B, 4C, H/2, W/2) → (B, C, H, W)
     device 파라미터: 출력 텐서 생성 위치 (원본 IWT 시그니처 유지)
"""

import torch
import torch.nn as nn


# ── DWT ────────────────────────────────────────────────────────────
def dwt_init(x: torch.Tensor) -> torch.Tensor:
    """
    원본 ProFace S Unet_common.py 의 dwt_init() 함수 그대로.
    /2 스케일링 + LL/HL/LH/HH 서브밴드 생성
    """
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL =  x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH =  x1 - x2 - x3 + x4
    return torch.cat((x_LL, x_HL, x_LH, x_HH), dim=1)


class DWT(nn.Module):
    """원본 DWT 클래스 (requires_grad=False, dwt_init 사용)"""
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return dwt_init(x)


class IWT(nn.Module):
    """
    원본 IWT 클래스.
    forward 시그니처: (x, device) — 원본 코드와 호환 유지.
    device 기본값은 x.device 로 자동 설정.
    """
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
        if device is None:
            device = x.device
        r = 2
        in_batch, in_channel, in_height, in_width = x.size()
        out_channel = in_channel // (r ** 2)
        out_height  = in_height  * r
        out_width   = in_width   * r

        x1 = x[:, 0            : out_channel    , :, :] / 2
        x2 = x[:, out_channel  : out_channel * 2, :, :] / 2
        x3 = x[:, out_channel*2: out_channel * 3, :, :] / 2
        x4 = x[:, out_channel*3: out_channel * 4, :, :] / 2

        h = torch.zeros(
            [in_batch, out_channel, out_height, out_width],
            dtype=x.dtype
        ).float().to(device)

        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
        return h
