"""
KeyGen — 원본 test_tcsvt.py 의 generate_key() 그대로 이식 (SRS §3.2)

원본 코드:
    def generate_key(password, bs, w, h):
        salt = 1
        key = PBKDF2(password, salt, int(w * h / 8), count=10, hmac_hash_module=SHA512)
        list_int = list(key)
        array_uint8 = np.array(list_int, dtype=np.uint8)
        array_bits = np.unpackbits(array_uint8).astype(int) * 2 - 1
        array_bits_2d = array_bits.reshape((w, h))
        skey_tensor = torch.tensor(array_bits_2d).repeat(bs, 1, 1, 1)
        return skey_tensor   # (bs, 1, w, h) — 2D tensor에 repeat(bs,1,1,1) → 4D

주의: salt=1, count=10 은 demonstration 값 (NFR-SEC-2)
"""

import numpy as np
import torch
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Hash import SHA512
import config as c


def generate_key(
    password,
    bs: int,
    w: int = c.NORM_RESOLUTION,
    h: int = c.NORM_RESOLUTION,
    salt=None,
    count: int = None,
) -> torch.Tensor:
    """
    원본 generate_key() 함수.

    password : str 또는 int (원본은 int 0 도 사용)
    반환      : (bs, 1, w, h) float32 tensor, 값: {-1, +1}

    이후 DWT 적용:
        skey_dwt = dwt(generate_key(pw, bs, w, h).float())
        → (bs, 4, w/2, h/2)
    """
    if salt is None:
        salt = c.KEY_SALT
    if count is None:
        count = c.KEY_COUNT

    # 원본: PBKDF2(password, salt, int(w*h/8), count=10, hmac_hash_module=SHA512)
    if isinstance(password, int):
        password = str(password).encode()
    elif isinstance(password, str):
        password = password.encode()
    # bytes 는 그대로

    n_bytes = int(w * h / 8)
    key = PBKDF2(password, salt, n_bytes, count=count, hmac_hash_module=SHA512)

    array_uint8  = np.array(list(key), dtype=np.uint8)
    array_bits   = np.unpackbits(array_uint8).astype(int) * 2 - 1   # {0,1} → {-1,+1}
    array_bits_2d = array_bits.reshape((w, h))

    # 원본: torch.tensor(array_bits_2d).repeat(bs, 1, 1, 1)
    # 2D tensor에 4개 dim repeat → PyTorch가 앞에 1을 붙여 (1,1,w,h) → (bs,1,w,h)
    skey_tensor = torch.tensor(array_bits_2d, dtype=torch.float32).repeat(bs, 1, 1, 1)
    return skey_tensor   # (bs, 1, w, h)


def make_key_rec(skey_dwt: torch.Tensor) -> torch.Tensor:
    """
    복원 보조입력 생성 (SRS FR-11.1 SECRET_KEY_AS_NOISE=True)

    skey_dwt : (B, 4, H/2, W/2)  — DWT된 키
    반환      : (B, 12, H/2, W/2) — 채널 3회 반복 (원본: skey_dwt.repeat(1,3,1,1))
    """
    return skey_dwt.repeat(1, 3, 1, 1)
