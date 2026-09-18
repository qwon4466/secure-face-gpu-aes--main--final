"""
ResidualDenseBlock_out — 원본 ProFace S 의 rrdb_denselayer.py 이식
SACB(INV_block_affine) 서브넷 ρ·η·φ·ψ 의 공통 구현체.

초기화 규칙 (원본 그대로):
  - conv1~conv4: kaiming_normal_(fan_in) 기본
  - conv5      : kaiming_normal_ 후 weight *= 0.  (출력 0 초기화)
  → INN 학습 초기 항등 변환에 가깝게 시작 (안정적 학습)
"""

import torch
import torch.nn as nn
import torch.nn.init as init


def initialize_weights(net_l, scale: float = 1.0):
    """원본 module_util.py의 initialize_weights 함수."""
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class ResidualDenseBlock_out(nn.Module):
    """
    Dense-connection 서브넷.
    input  : in_ch 채널
    output : out_ch 채널

    구조: 5단 Dense conv
      x → conv1 → x1
      [x, x1] → conv2 → x2
      [x, x1, x2] → conv3 → x3
      [x, x1, x2, x3] → conv4 → x4
      [x, x1, x2, x3, x4] → conv5 → output
    """
    INNER = 32   # 각 dense 레이어 내부 채널

    def __init__(self, in_ch: int, out_ch: int, bias: bool = True):
        super(ResidualDenseBlock_out, self).__init__()
        ic = self.INNER
        self.conv1 = nn.Conv2d(in_ch,          ic,     3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(in_ch + ic,     ic,     3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(in_ch + ic*2,   ic,     3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(in_ch + ic*3,   ic,     3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(in_ch + ic*4,   out_ch, 3, 1, 1, bias=bias)
        # 원본: LeakyReLU(inplace=True), negative_slope 기본값(0.01)
        self.lrelu = nn.LeakyReLU(inplace=True)
        # 원본: conv5 만 scale=0 초기화
        initialize_weights([self.conv5], 0.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5
