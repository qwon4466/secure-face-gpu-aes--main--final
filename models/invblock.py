"""
INV_block_affine — SACB (Secure Affine Coupling Block)
원본 ProFace S invblock.py 의 INV_block_affine 클래스 이식.

채널 구성 (config.channels_in=3 기준):
  split_len1 = channels_in * 4 = 12  (cover/steg 쪽)
  split_len2 = channels_in * 4 = 12  (secret/z 쪽)
  password_ch = 4                     (DWT된 1ch 키 → 4ch)

서브넷 입력/출력:
  r, y : (split_len1 + password_ch) → split_len2  = 16 → 12
  f, p : (split_len2 + password_ch) → split_len1  = 16 → 12

e(s) = exp( clamp * 2 * (sigmoid(s) - 0.5) )
     → 범위: [exp(-clamp), exp(clamp)] = [exp(-2), exp(2)]

c(x, password) = torch.cat((x, password), 1)  — 채널 방향 concat

Forward (보호):
  x1, x2 = split(x)          x1=cover쪽, x2=secret쪽
  y2 = x2 * e(y(c(x1,K))) + r(c(x1,K))
  y1 = x1 * e(p(c(y2,K))) + f(c(y2,K))
  return [y1 | y2]            y1=steg쪽, y2=z쪽

Inverse (복원):
  y1, y2 = split(y)           y1=steg쪽, y2=z(≈key_rec)쪽
  x1 = (y1 - f(c(y2,K))) / e(p(c(y2,K)))
  x2 = (y2 - r(c(x1,K))) / e(y(c(x1,K)))
  return [x1 | x2]            x1=cover복원, x2=original복원
"""

import torch
import torch.nn as nn
import config as c
from models.rrdb_denselayer import ResidualDenseBlock_out


class INV_block_affine(nn.Module):
    def __init__(
        self,
        in_1: int = c.channels_in,
        in_2: int = c.channels_in,
        clamp: float = c.clamp,
        harr: bool = True,
        in_1_size: int = 2,
        in_2_size: int = 2,
    ):
        super(INV_block_affine, self).__init__()

        self.clamp = clamp

        # DWT 후 채널 수: channels_in * 4
        self.split_len1 = in_1 * 4  # 12
        self.split_len2 = in_2 * 4  # 12

        # password(key) 채널 수: DWT된 1채널 → 4채널
        self.password_channel = 4

        # imp: ADJ_UTILITY=False 이면 0
        self.imp = 1 if c.ADJ_UTILITY else 0

        # 서브넷 입력 = split + imp + password_ch
        in_r = self.split_len1 + self.imp + self.password_channel  # 16
        in_f = self.split_len2 + self.password_channel             # 16

        self.r = ResidualDenseBlock_out(in_r, self.split_len2)  # 16 → 12
        self.y = ResidualDenseBlock_out(in_r, self.split_len2)  # 16 → 12
        self.f = ResidualDenseBlock_out(in_f, self.split_len1 + self.imp)  # 16 → 12
        self.p = ResidualDenseBlock_out(in_f, self.split_len1 + self.imp)  # 16 → 12

    # ── e(s): 클램프된 지수 스케일 ────────────────────────────────
    def e(self, s: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    # ── c(x, password): 채널 방향 concat ─────────────────────────
    def c(self, x: torch.Tensor, password: torch.Tensor) -> torch.Tensor:
        return torch.cat((x, password), 1)

    def forward(
        self, x: torch.Tensor, password: torch.Tensor, rev: bool = False
    ) -> torch.Tensor:
        # 원본 invblock.py 의 narrow 사용 (ADJ_UTILITY=False 이면 imp=0)
        x1 = x.narrow(1, 0,                          self.split_len1 + self.imp)
        x2 = x.narrow(1, self.split_len1 + self.imp, self.split_len2)

        if not rev:
            # ── Forward (원본 INV_block_affine 과 동일) ───────────
            # Step 1: x2 → x1 방향 변환  (φ, ψ 서브넷)
            t2 = self.f(self.c(x2, password))        # φ: split_len2+pwd → split_len1
            s2 = self.p(self.c(x2, password))        # ψ: split_len2+pwd → split_len1
            y1 = self.e(s2) * x1 + t2               # affine on x1

            # Step 2: y1 → x2 방향 변환  (ρ, η 서브넷)
            s1 = self.r(self.c(y1, password))        # ρ: split_len1+pwd → split_len2
            t1 = self.y(self.c(y1, password))        # η: split_len1+pwd → split_len2
            y2 = self.e(s1) * x2 + t1               # affine on x2

            return torch.cat((y1, y2), 1)            # [y1(steg) | y2(z)]

        else:
            # ── Inverse (원본과 동일, 변수명 swap 주의) ───────────
            # 입력: x1 = y1(steg), x2 = y2(z≈key_rec)
            # Step 1: x1(=y1) → x2 복원  (ρ, η 서브넷)
            s1 = self.r(self.c(x1, password))
            t1 = self.y(self.c(x1, password))
            y2 = (x2 - t1) / self.e(s1)             # 원본 x2 복원

            # Step 2: y2(복원된 x2) → x1 복원  (φ, ψ 서브넷)
            t2 = self.f(self.c(y2, password))
            s2 = self.p(self.c(y2, password))
            y1 = (x1 - t2) / self.e(s2)             # 원본 x1 복원

            return torch.cat((y1, y2), 1)            # [y1(cover복원) | y2(original복원)]
