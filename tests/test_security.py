import copy
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import uuid
import cv2
import numpy as np
import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.keywrap import aes_key_unwrap, InvalidUnwrap
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding,PublicFormat
from core.selective_crypto import *
from core.selective_recognition import Classifier, SyntheticAdapter, synthetic_frame, legacy_components, CPUAdapter
from core.selective_recording import Recorder, process


def fixture(work, count=3):
    # Evaluation originals never written to disk and never passed to restore worker.
    originals=[synthetic_frame(i+30) for i in range(count)]
    faces=[{'bbox':[20,30,60,90],'group':'external','track_id':1},
           {'bbox':[110,30,150,90],'group':'internal','track_id':2},
           {'bbox':[230,30,270,90],'group':'master','track_id':3},
           {'bbox':[80,150,130,200],'group':'external','track_id':4},
           {'bbox':[120,150,165,200],'group':'internal','track_id':5}]
    keys={g:secrets.token_bytes(32) for g in GROUPS}
    chunk=save_chunk(work/'chunks',originals,[faces]*count,
                     [{'capture_id':i*2,'timestamp_ms':i*200} for i in range(count)],keys,uuid.uuid4().hex)
    return originals,keys,chunk,faces


def worker(chunk,output,key):
    # Only bytes for the single supplied key go over stdin; no role or other keys or original references.
    return subprocess.run([sys.executable,'-B','-m','core.selective_worker',str(chunk),str(output)],
                          input=key or b'',cwd=ROOT,capture_output=True,timeout=30)


@pytest.mark.parametrize('group',[None,'external','internal','master','wrong'])
def test_policy_separate_process(work,group):
    originals,keys,chunk,faces=fixture(work)
    output=work/'out'
    supplied=secrets.token_bytes(32) if group=='wrong' else keys.get(group)
    result=worker(chunk,output,supplied)
    if group=='wrong':
        assert result.returncode!=0 and not output.exists()
        assert not list(work.glob('.restore-*'))
        return
    assert result.returncode==0,result.stdout+result.stderr
    actual=read_video(output/'restored.avi',3,[240,320,3])
    for i,(raw,restored) in enumerate(zip(originals,actual)):
        protected,masks,_=protect(raw,faces)
        allowed=np.zeros(raw.shape[:2],bool)
        for g in GROUPS if group=='master' else ([group] if group else []):
            allowed|=masks[g]
        assert np.array_equal(restored[allowed],raw[allowed])
        assert np.array_equal(restored[~allowed],protected[~allowed])
        assert np.array_equal(cv2.imread(str(output/f'{i:06d}.png')),restored)
    if group=='master':
        assert all(np.array_equal(a,b) for a,b in zip(originals,actual))


@pytest.mark.parametrize('supplied,forbidden',[('external','internal'),('external','master'),('internal','external'),('internal','master')])
def test_direct_unwrap_denied(work,supplied,forbidden):
    _,keys,chunk,_=fixture(work)
    doc,_=inspect_chunk(chunk)
    for wrapped in doc['manifest']['envelopes'][forbidden]['wraps']:
        with pytest.raises(InvalidUnwrap):
            aes_key_unwrap(keys[supplied],unb64(wrapped))


def test_sparse_payload_has_no_other_group_pixels(work):
    originals,keys,chunk,faces=fixture(work)
    doc,masks=inspect_chunk(chunk); manifest=doc['manifest']
    for g in GROUPS:
        e=manifest['envelopes'][g]
        dek=aes_key_unwrap(keys[g],unb64(e['wraps'][0]))
        raw=AESGCM(dek).decrypt(unb64(e['nonce']),(chunk/(g+'.gcm')).read_bytes(),canonical({'meta':manifest['meta'],'group':g}))
        assert raw==b''.join(f[m[g]].tobytes() for f,m in zip(originals,masks))
        for m in masks:
            for other in GROUPS:
                if other!=g:
                    assert not np.any(m[g]&m[other])
    # Blur and encryption union use exactly the detector boxes; no padding reaches background.
    exact=np.zeros((240,320),bool)
    for face in faces:
        x1,y1,x2,y2=face['bbox'];exact[y1:y2,x1:x2]=True
    assert np.array_equal(np.logical_or.reduce(list(masks[0].values())),exact)
    _,_,records=protect(originals[0],faces)
    assert all(record['write_roi']==record['bbox'] for record in records)
    # Cross-group ROI intersection and 2-pixel guard are master only.
    assert masks[0]['master'][170,125]
    assert not masks[0]['external'][170,125]
    assert not masks[0]['internal'][170,125]


def resign(doc):
    signer=Ed25519PrivateKey.generate()
    doc['manifest']['meta']['signing_public']=b64(signer.public_key().public_bytes(Encoding.Raw,PublicFormat.Raw))
    doc['signature']=b64(signer.sign(canonical({'manifest':doc['manifest'],'seals':doc['seals']})))


@pytest.mark.parametrize('attack',['ciphertext','tag','nonce','metadata','video','video_rehash','wrap','seal','swap','missing','complete','mask','oversize','duplicate','role','resigned'])
def test_tamper_no_partial_output(work,attack):
    _,keys,chunk,_=fixture(work)
    doc=json.loads((chunk/'manifest.json').read_bytes())
    m=doc['manifest']; e=m['envelopes']['external']
    if attack in ('ciphertext','tag'):
        p=chunk/'external.gcm'; data=bytearray(p.read_bytes()); data[0 if attack=='ciphertext' else -1]^=1; p.write_bytes(data)
        # Attacker updates plain hashes too; authentication must still reject.
        e['sha256']=digest(data)
    elif attack=='nonce':
        e['nonce']=b64(secrets.token_bytes(12))
    elif attack=='metadata':
        m['meta']['frames'][0]['timestamp_ms']+=1
    elif attack in ('video','video_rehash'):
        p=chunk/'protected.avi'; data=bytearray(p.read_bytes()); data[-20]^=1; p.write_bytes(data)
        if attack=='video_rehash': m['meta']['protected_sha256']=digest(data)
    elif attack=='wrap': e['wraps'][0]=b64(secrets.token_bytes(40))
    elif attack=='seal': doc['seals']['internal']=b64(secrets.token_bytes(16))
    elif attack=='swap':
        _,_,other,_=fixture(work/'second')
        (chunk/'external.gcm').write_bytes((other/'external.gcm').read_bytes())
        e['sha256']=digest((chunk/'external.gcm').read_bytes())
    elif attack=='missing': (chunk/'internal.gcm').unlink()
    elif attack=='complete': m['meta']['complete']=False
    elif attack=='mask': m['meta']['frames'][0]['masks']['external']=[[0,240*320+1]]
    elif attack=='oversize': m['meta']['shape']=[100000,100000,3]
    elif attack=='role': e['role']='master'
    elif attack=='resigned':
        m['meta']['frames'][0]['timestamp_ms']+=1
        resign(doc)  # Public signer replacement cannot bypass authenticated public key / AAD.
    if attack=='duplicate':
        (chunk/'manifest.json').write_bytes(b'{"manifest":{},"manifest":{}}')
    else:
        (chunk/'manifest.json').write_bytes(canonical(doc))
    for g in ('external','master'):
        out=work/('out-'+g)
        run=worker(chunk,out,keys[g])
        assert run.returncode!=0,(attack,run.stdout)
        assert not out.exists() and not list(work.glob('.restore-*'))


def test_key_filename_cannot_raise_privilege(work):
    _,keys,chunk,_=fixture(work)
    p=work/'master.key'; write_new(p,keys['external'])
    result=restore_to(chunk,work/'out',load_key(p))
    assert result['groups']==['external']


def test_temporal_classification():
    c=Classifier(); a=SyntheticAdapter(); frames=[]
    for i in range(6): frames.append(c.classify(a.observe(synthetic_frame(i),i),i))
    assert frames[0][0]['group']=='master'
    assert frames[3][0]['group']=='master'
    assert frames[4][0]['group']=='external'
    assert frames[1][1]['group']=='master'
    assert frames[2][1]['group']=='internal'
    assert all(f[2]['group']=='master' for f in frames)
    # A gap or changed track geometry resets certainty.
    assert c.classify(a.observe(synthetic_frame(0),20),20)[0]['group']=='master'
    observations=a.observe(synthetic_frame(0),21)
    observations[0]['bbox']=[280,200,315,230]
    assert c.classify(observations,21)[0]['group']=='master'
    c=Classifier()
    for i in range(3): r=c.classify(a.observe(synthetic_frame(i),i),i)
    observations=a.observe(synthetic_frame(3),3); observations[1]['identity']='different'
    assert c.classify(observations,3)[1]['group']=='master'
    # Empty/missing gallery never establishes outsider status.
    c=Classifier()
    for i in range(8):
        observations=a.observe(synthetic_frame(i),i)
        for o in observations:o['gallery_valid']=False
        assert all(o['group']=='master' for o in c.classify(observations,i))


def test_empty_boundary_drop_chunk_retry(work):
    frame=synthetic_frame(0); keys={g:secrets.token_bytes(32) for g in GROUPS}
    r=Recorder(work/'chunks',keys,chunk_frames=2)
    for i in (0,2,3,9,10):r.add(frame,[],i,i*100)
    report=r.finish()
    assert len(report['chunks'])==3 and report['peak_buffer_frames']==2 and report['queue_capacity']==0
    for cid in report['chunks']:
        out,_,_=restore_frames(work/'chunks'/cid,keys['master']); assert all(np.array_equal(f,frame) for f in out)
    with pytest.raises(RestoreError):Recorder(work/'x',keys,chunk_frames=MAX_FRAMES+1)
    for fail in ('video','manifest'):
        with pytest.raises(OSError):
            save_chunk(work/'interrupted',[frame],[[]],[{'capture_id':0,'timestamp_ms':0}],keys,uuid.uuid4().hex,fail_at=fail)
    assert len(list((work/'interrupted').glob('.partial-*')))==2
    for p in (work/'interrupted').iterdir():
        with pytest.raises(RestoreError):restore_frames(p,keys['master'])
    a=save_chunk(work/'retry',[frame],[[]],[{'capture_id':0,'timestamp_ms':0}],keys,uuid.uuid4().hex)
    b=save_chunk(work/'retry',[frame],[[]],[{'capture_id':0,'timestamp_ms':0}],keys,uuid.uuid4().hex)
    da,_=inspect_chunk(a); db,_=inspect_chunk(b)
    for g in GROUPS:
        wa=da['manifest']['envelopes'][g]['wraps'][0]; wb=db['manifest']['envelopes'][g]['wraps'][0]
        assert aes_key_unwrap(keys[g],unb64(wa))!=aes_key_unwrap(keys[g],unb64(wb))
    # Writes are clipped and full restoration includes blur boundaries, even at image edge.
    p,m,_=protect(frame,[{'bbox':[0,0,9,10],'group':'internal'}])
    assert not m['internal'][40,40] and np.array_equal(p[40:,40:],frame[40:,40:])


def test_legacy_reuse_no_side_effects():
    Result,cosine=legacy_components()
    assert Result([1,2,3,4],.9).conf==.9
    assert cosine(np.ones(5),np.ones(5))==pytest.approx(1)
    assert 'config' not in sys.modules and 'camera_stream' not in sys.modules and 'raw_restore' not in sys.modules
    with pytest.raises(RestoreError):CPUAdapter('/missing.onnx','/missing2.onnx')


def test_no_overwrite_and_traversal(work):
    _,keys,chunk,_=fixture(work)
    with pytest.raises(RestoreError):restore_to(chunk,chunk,keys['master'])
    with pytest.raises(RestoreError):private_dir(ROOT/'outside-new')
    with pytest.raises(RestoreError):generate_keys(chunk)
    link=work/'link'; link.symlink_to(chunk,target_is_directory=True)
    with pytest.raises(RestoreError):restore_to(chunk,link/'out',keys['master'])


def test_real_file_cli_path(work):
    source=work/'synthetic-input.avi'
    frames=[synthetic_frame(i) for i in range(12)]
    write_video(source,frames,20)
    keys={g:secrets.token_bytes(32) for g in GROUPS}
    report=process(str(source),SyntheticAdapter(),work/'chunks',keys,max_frames=12,fps=10,chunk_frames=3)
    assert report['frames']==6 and report['drops']==6 and len(report['chunks'])==2
    originals=[frames[i] for i in range(0,12,2)]
    restored=[]
    for cid in report['chunks']:
        f,meta,_=restore_frames(work/'chunks'/cid,keys['master']);restored+=f
        assert all(x['capture_id']%2==0 for x in meta['frames'])
    assert all(np.array_equal(a,b) for a,b in zip(originals,restored))
