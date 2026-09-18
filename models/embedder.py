"""
ModelDWT — 원본 ProFace S embedder.py 의 ModelDWT 클래스 이식.

핵심 수정 사항 (원본 코드 확인 후):
  ① 채널 concat 순서: cat(cover_dwt, secret_dwt)  ← cover 먼저!
  ② 반환 순서:        (output_z, output_steg_img)   ← z 먼저
  ③ 복원 concat:     cat(steg_dwt, output_z)        ← steg 먼저
  ④ IWT: device 파라미터 전달

보호 호출 (test_tcsvt.py 기준):
    xa_out_z, xa_proc = embedder(xa, xa_obfs, skey_dwt)
    # xa      = input1 = secret_img (원본)
    # xa_obfs = input2 = cover_img  (사전난독)
    → xa_out_z : 부산물 z  (12ch DWT 공간, 즉시 폐기)
    → xa_proc  : 보호본 ŷ  (3ch 이미지 공간)

복원 호출:
    key_rec = skey_dwt.repeat(1, 3, 1, 1)   # (B,12,H/2,W/2)
    xa_rev, _ = embedder(key_rec, xa_proc, skey_dwt, rev=True)
    # key_rec = input1 = output_z 대체
    # xa_proc = input2 = steg_img
    → xa_rev : 복원 이미지 x̌
"""

import torch
import torch.nn as nn
from models.modules import DWT, IWT
from models.hinet import Hinet
import config as c


# 모듈 레벨 DWT/IWT (원본 embedder.py 와 동일하게 모듈 레벨 선언)
dwt = DWT()
iwt = IWT()


class ModelDWT(nn.Module):
    def __init__(self, n_blocks: int = c.INV_BLOCKS):
        super(ModelDWT, self).__init__()
        self.model  = Hinet(n_blocks)
        self.device = torch.device('cpu')

    def to(self, device):
        super(ModelDWT, self).to(device)
        self.device = device
        return self

    def forward(
        self,
        input1: torch.Tensor,
        input2: torch.Tensor,
        password: torch.Tensor,
        rev: bool = False,
    ):
        if not rev:
            # ── 보호 경로 ─────────────────────────────────────────
            # input1 = secret_img (원본 x),  input2 = cover_img (사전난독 y)
            secret_img, cover_img = input1, input2

            cover_dwt  = dwt(cover_img)               # (B, 12, H/2, W/2)
            secret_dwt = dwt(secret_img)              # (B, 12, H/2, W/2)

            # 원본: cat(cover, secret) — cover 먼저
            input_dwt  = torch.cat((cover_dwt, secret_dwt), 1)  # (B, 24, H/2, W/2)

            output = self.model(input_dwt, password)  # (B, 24, H/2, W/2)

            # 원본: narrow(1, 0, 4*channels_in) = 앞 12ch = steg
            output_steg_dwt = output.narrow(1, 0,                    4 * c.channels_in)
            # 원본: narrow(1, 4*channels_in, ...) = 뒤 12ch = z
            output_z        = output.narrow(1, 4 * c.channels_in,
                                            output.shape[1] - 4 * c.channels_in)

            output_steg_img = iwt(output_steg_dwt, self.device)  # (B, 3, H, W)

            # 원본 반환 순서: (z, steg_img)
            return output_z, output_steg_img

        else:
            # ── 복원 경로 ─────────────────────────────────────────
            # input1 = output_z 대체 (key_rec, 12ch DWT 공간)
            # input2 = output_steg_img (보호본 ŷ, 3ch 이미지 공간)
            output_z, output_steg_img = input1, input2

            output_steg_dwt = dwt(output_steg_img)   # (B, 12, H/2, W/2)

            # 원본: cat(steg_dwt, output_z) — steg 먼저
            output_rev = torch.cat((output_steg_dwt, output_z), 1)  # (B, 24, H/2, W/2)

            output_dwt = self.model(output_rev, password, rev=True)

            # 복원된 secret (원본 얼굴): 뒤 12ch
            secret_rev_dwt = output_dwt.narrow(
                1, 4 * c.channels_in,
                output_dwt.shape[1] - 4 * c.channels_in
            )
            secret_rev_img = iwt(secret_rev_dwt, self.device)   # (B, 3, H, W)

            # 복원된 cover: 앞 12ch  (보통 사용 안 함)
            cover_rev_dwt  = output_dwt.narrow(1, 0, 4 * c.channels_in)
            cover_rev_img  = iwt(cover_rev_dwt, self.device)

            return secret_rev_img, cover_rev_img


def init_model(mod: nn.Module, device: torch.device):
    """원본 embedder.py 의 init_model 함수 (conv5 출력만 0 초기화)."""
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = c.init_scale * torch.randn(param.data.shape).to(device)
            if split[-2] == 'conv5':
                param.data.fill_(0.)
