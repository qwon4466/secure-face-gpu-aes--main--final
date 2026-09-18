"""
Hinet — 원본 ProFace S hinet.py 이식
INV_block_affine 을 n_blocks 회 스택.

원본 기본값: n_blocks=6
실제 사용 시: Hinet(n_blocks=c.INV_BLOCKS) = Hinet(3)
"""

import torch
import torch.nn as nn
from models.invblock import INV_block_affine
import config as c


class Hinet(nn.Module):
    def __init__(self, n_blocks: int = 6):
        super(Hinet, self).__init__()
        self.inv_blocks = nn.ModuleList(
            [INV_block_affine() for _ in range(n_blocks)]
        )

    def forward(
        self, x: torch.Tensor, password: torch.Tensor, rev: bool = False
    ) -> torch.Tensor:
        if not rev:
            for inv_block in self.inv_blocks:
                x = inv_block(x, password)
        else:
            for inv_block in reversed(self.inv_blocks):
                x = inv_block(x, password, rev=True)
        return x
