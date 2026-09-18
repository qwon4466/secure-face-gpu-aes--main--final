"""
AES-256 File Encryption Module

Algorithm : AES-256 CBC
Key : SHA256(password)
"""

import os
from datetime import datetime
import config as c
import hashlib

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad


PASSWORD = "0000"

BUFFER_SIZE = 64 * 1024


def make_key(password: str):
    return hashlib.sha256(password.encode("utf-8")).digest()


def encrypt_file(src_path: str,
                 dst_path: str,
                 password: str = PASSWORD):

    key = make_key(password)

    iv = get_random_bytes(16)

    cipher = AES.new(key,
                     AES.MODE_CBC,
                     iv)

    filesize = os.path.getsize(src_path)

    with open(src_path, "rb") as fin, \
            open(dst_path, "wb") as fout:

        # 원본 파일 크기 저장
        fout.write(filesize.to_bytes(8, "big"))

        # IV 저장
        fout.write(iv)

        while True:

            chunk = fin.read(BUFFER_SIZE)

            if len(chunk) == 0:
                break

            if len(chunk) % AES.block_size != 0:
                chunk = pad(chunk, AES.block_size)
                fout.write(cipher.encrypt(chunk))
                break

            fout.write(cipher.encrypt(chunk))


def decrypt_file(src_path: str,
                 dst_path: str,
                 password: str = PASSWORD):

    key = make_key(password)

    with open(src_path, "rb") as fin:

        # 1. 헤더에서 원본 파일 크기와 IV(초기화 벡터) 읽기
        filesize = int.from_bytes(fin.read(8), "big")
        iv = fin.read(16)

        cipher = AES.new(key, AES.MODE_CBC, iv)

        with open(dst_path, "wb") as fout:

            while True:
                chunk = fin.read(BUFFER_SIZE)

                if len(chunk) == 0:
                    break

                # 2. 골치 아픈 unpad 로직을 걷어내고 곧바로 복호화 및 쓰기
                chunk = cipher.decrypt(chunk)
                fout.write(chunk)

            # 3. 만약 암호화 때 패딩이 들어갔더라도, 여기서 원본 크기(filesize) 
            # 밖의 더미 데이터는 완벽하게 잘려나갑니다. (초강력 안전장치)
            fout.truncate(filesize)


def encrypt_and_remove(video_path):

    encrypt_path = video_path + ".enc"

    encrypt_file(video_path,
                 encrypt_path)

    os.remove(video_path)

    return encrypt_path


def encrypt_and_remove_to(src_path: str, dst_path: str,
                          password: str = PASSWORD):
    """기존 AES 포맷으로 암호화하되 출력 파일명을 지정한다."""
    encrypt_file(src_path, dst_path, password)
    os.remove(src_path)
    return dst_path


def iter_decrypted_frames(src_path: str, frame_size: int,
                          password: str = PASSWORD):
    """기존 filesize+IV+CBC 포맷을 메모리 bounded 방식으로 복호화한다."""
    if frame_size <= 0:
        raise ValueError("frame_size must be positive")
    key = make_key(password)
    with open(src_path, "rb") as fin:
        header_size = 24
        filesize_raw = fin.read(8)
        iv = fin.read(16)
        if len(filesize_raw) != 8 or len(iv) != 16:
            raise ValueError("AES header가 손상되었습니다.")
        filesize = int.from_bytes(filesize_raw, "big")
        encrypted_size = os.path.getsize(src_path) - header_size
        if encrypted_size < 0 or encrypted_size % AES.block_size:
            raise ValueError("AES 암호문 길이가 올바르지 않습니다.")
        cipher = AES.new(key, AES.MODE_CBC, iv)
        pending = bytearray()
        remaining = encrypted_size
        plaintext_remaining = filesize
        while remaining:
            block = fin.read(min(BUFFER_SIZE, remaining))
            if not block or len(block) % AES.block_size:
                raise ValueError("AES 암호문을 끝까지 읽지 못했습니다.")
            remaining -= len(block)
            plain = cipher.decrypt(block)
            if plaintext_remaining < len(plain):
                plain = plain[:plaintext_remaining]
            plaintext_remaining -= len(plain)
            pending.extend(plain)
            while len(pending) >= frame_size:
                yield bytes(pending[:frame_size])
                del pending[:frame_size]
        if plaintext_remaining != 0 or pending:
            raise ValueError("복호화 데이터가 완전한 프레임 단위가 아닙니다.")


def decrypt_to_temp(enc_path):

    if enc_path.endswith(".enc"):
        out_path = enc_path[:-4]
    else:
        out_path = enc_path + ".tmp"

    decrypt_file(enc_path,
                 out_path)

    return out_path

def make_chunk_directory(chunk_time=None):
    """
    encrypted_originals/
        MMDD/
            HH/
                HH-MM-SS/
    """

    if chunk_time is None:
        chunk_time = datetime.now()

    date_dir = chunk_time.strftime("%m%d")
    hour_dir = chunk_time.strftime("%H")
    chunk_dir = chunk_time.strftime("%H-%M-%S")

    path = os.path.join(
        c.RAW_ENCRYPT_DIR,
        date_dir,
        hour_dir,
        chunk_dir
    )

    os.makedirs(path, exist_ok=True)

    return path

def save_encrypted_frame(frame_bytes, frame_index, password, chunk_time=None):
    """
    JPEG bytes -> AES 암호화 -> 저장
    """

    folder = make_chunk_directory(chunk_time)

    encrypted = encrypt_bytes(frame_bytes, password)

    filename = f"frame_{frame_index:04d}.enc"

    filepath = os.path.join(folder, filename)

    with open(filepath, "wb") as f:
        f.write(encrypted)

    return filepath
