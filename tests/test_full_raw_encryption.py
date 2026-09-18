import json
import secrets
import uuid

import numpy as np
import pytest

from core.selective_crypto import (GROUPS, RAW_META_NAME, RAW_NAME, RestoreError,
    decrypt_original, generate_keys, inspect_chunk, keyset, save_chunk)
from core.selective_index import completed


def test_full_original_aes_gcm_roundtrip_and_authentication(work):
    key_dir=work/'keys';generate_keys(key_dir);keys=keyset(key_dir)
    frames=[np.random.default_rng(i).integers(0,256,(24,32,3),dtype=np.uint8) for i in range(2)]
    chunk=save_chunk(work/'raw_data',frames,[[],[]],
        [{'capture_id':i,'timestamp_ms':i*100} for i in range(2)],keys,uuid.uuid4().hex,fps=10)
    data=json.loads((chunk/RAW_META_NAME).read_text())
    assert set(data)=={'version','algorithm','nonce','aad','encrypted_file','frame_count','fps',
        'width','height','channels','started_at','ended_at','tag','ciphertext_sha256'}
    assert data['algorithm']=='AES-256-GCM' and (chunk/RAW_NAME).stat().st_size>0
    restored=decrypt_original(chunk,data,keys['master'])
    assert all(np.array_equal(a,b) for a,b in zip(frames,restored))
    with pytest.raises(RestoreError):decrypt_original(chunk,data,secrets.token_bytes(32))
    ciphertext=bytearray((chunk/RAW_NAME).read_bytes());ciphertext[len(ciphertext)//2]^=1
    (chunk/RAW_NAME).write_bytes(ciphertext)
    with pytest.raises(RestoreError):decrypt_original(chunk,data,keys['master'])
    with pytest.raises(RestoreError):inspect_chunk(chunk)
    (chunk/RAW_NAME).unlink()
    assert completed(work/'raw_data')==[]
