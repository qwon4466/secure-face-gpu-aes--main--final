import json
import secrets
import uuid
import numpy as np
import pytest
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.keywrap import aes_key_unwrap
from core.selective_crypto import *
from core.selective_recognition import SyntheticAdapter,synthetic_frame,Classifier
from core.selective_recording import Recorder,process
from test_security import fixture

@pytest.mark.parametrize('part',['nonce','tag','aad'])
def test_gcm_itself_rejects(work,part):
    _,keys,chunk,_=fixture(work)
    doc,_=inspect_chunk(chunk);m=doc['manifest'];g='external';e=m['envelopes'][g]
    dek=aes_key_unwrap(keys[g],unb64(e['wraps'][0]));nonce=unb64(e['nonce'])
    ct=(chunk/(g+'.gcm')).read_bytes();aad=canonical({'meta':m['meta'],'group':g})
    if part=='nonce':nonce=(2).to_bytes(12,'big')
    elif part=='tag':ct=ct[:-1]+bytes([ct[-1]^1])
    else:aad+=b'changed'
    with pytest.raises(InvalidTag):AESGCM(dek).decrypt(nonce,ct,aad)


def test_save_io_failure_and_process_error(work,monkeypatch):
    import core.selective_crypto as core
    keys={g:secrets.token_bytes(32) for g in GROUPS};frame=synthetic_frame(1)
    actual=core.write_new
    def failure(path,data):
        if str(path).endswith('internal.gcm'):raise OSError('disk full')
        actual(path,data)
    monkeypatch.setattr(core,'write_new',failure)
    with pytest.raises(OSError):save_chunk(work/'chunks',[frame],[[]],[{'capture_id':0,'timestamp_ms':0}],keys,uuid.uuid4().hex)
    assert not [p for p in (work/'chunks').iterdir() if ID.fullmatch(p.name)]
    monkeypatch.setattr(core,'write_new',actual)
    class Failing(SyntheticAdapter):
        def observe(self,frame,capture_id):
            if capture_id==4:raise RuntimeError('detector failure')
            return super().observe(frame,capture_id)
    with pytest.raises(RuntimeError):process('synthetic',Failing(),work/'process',keys,max_frames=8,chunk_frames=3)
    complete=[p for p in (work/'process').iterdir() if ID.fullmatch(p.name)]
    assert len(complete)==1
    frames,_,_=restore_frames(complete[0],keys['master']);assert len(frames)==3


def test_ambiguous_tracking_resets():
    c=Classifier(); adapter=SyntheticAdapter()
    for i in range(6):c.classify(adapter.observe(synthetic_frame(i),i),i)
    obs=adapter.observe(synthetic_frame(6),6)
    obs.append(dict(obs[0]))
    out=c.classify(obs,6)
    assert out[0]['group']=='master' and out[-1]['group']=='master'
    assert out[0]['track_id']!=out[-1]['track_id']


def test_bounds_validation(work):
    keys={g:secrets.token_bytes(32) for g in GROUPS};f=synthetic_frame(0)
    for box in ([0,0,321,100],[-1,0,5,5],[0,0,0,5]):
        with pytest.raises(RestoreError):protect(f,[{'bbox':box,'group':'external'}])
    with pytest.raises(RestoreError):save_chunk(work/'bad',[f]*17,[[]]*17,[{}]*17,keys,uuid.uuid4().hex)
    with pytest.raises(RestoreError):process('synthetic',SyntheticAdapter(),work/'bad',keys,max_frames=0)
    r=Recorder(work/'buffer',keys,chunk_frames=16)
    large=np.zeros((1080,1920,3),np.uint8)
    for i in range(12):r.add(large,[],i,i*100)
    result=r.finish()
    assert result['peak_buffer_frames']<=MAX_RAW//large.nbytes
    assert len(result['chunks'])==2


def test_concurrent_video_replacement_is_rejected(work,monkeypatch):
    import core.selective_crypto as core
    _,keys,chunk,_=fixture(work)
    inspect=core.inspect_chunk
    def changed(p):
        result=inspect(p)
        f=p/'protected.avi';data=bytearray(f.read_bytes());data[-20]^=1;f.write_bytes(data)
        return result
    monkeypatch.setattr(core,'inspect_chunk',changed)
    with pytest.raises(RestoreError):restore_to(chunk,work/'out',keys['external'])
    assert not (work/'out').exists()
