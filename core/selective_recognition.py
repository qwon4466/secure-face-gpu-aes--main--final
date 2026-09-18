"""Read-only legacy adapter. No import of config/main/camera_stream modules."""
from __future__ import annotations
import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import sys
import numpy as np
from .selective_crypto import ROOT, RestoreError

PROJECT=ROOT

def legacy_components():
    """Use the original interface and unchanged cosine function, after hash checks.

    Compile only the audited pure function, without importing camera hardware modules.
    Its AST is pinned separately so camera initialization can evolve independently.
    """
    baseline=json.loads((ROOT/'docs/selective-original-baseline.json').read_text())['files']
    for name in ('detection/detector_base.py',):
        if hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()!=baseline[name]:
            raise RestoreError('기존 어댑터 소스가 변경되었습니다; 재검토가 필요합니다')
    name='srx._legacy_detector_base'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,PROJECT/'detection/detector_base.py')
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module
        spec.loader.exec_module(module)
    tree=ast.parse((PROJECT/'camera_stream.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_cosine_sim')
    if hashlib.sha256(ast.dump(node).encode()).hexdigest() != '58e9e0c8248575b638006a60fceaf085b3b8d9d9355ae9fea314920ee3193c40':
        raise RestoreError('cosine 함수 변경: 재검토 필요')
    scope={'np':np}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(PROJECT/'camera_stream.py'),'exec'),scope)
    return sys.modules[name].DetectionResult,scope['_cosine_sim']

@dataclass(frozen=True)
class Policy:
    internal_score: float=0.55
    external_score: float=0.25
    quality: float=0.8
    internal_streak: int=3
    external_streak: int=5
    track_iou: float=0.35

def iou(a,b):
    x=max(0,min(a[2],b[2])-max(a[0],b[0])); y=max(0,min(a[3],b[3])-max(a[1],b[1]))
    return x*y/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-x*y)

class Classifier:
    def __init__(self, policy=Policy()):
        self.policy=policy; self.previous=[]; self.next_id=0; self.last_capture=None

    def classify(self, observations, capture_id):
        p=self.policy
        previous=self.previous if self.last_capture is not None and capture_id==self.last_capture+1 else []
        matches=[[j for j,old in enumerate(previous) if iou(o['bbox'],old['bbox'])>=p.track_iou] for o in observations]
        usage={j:sum(j in row for row in matches) for j in range(len(previous))}
        current=[]
        for o,candidates in zip(observations,matches):
            old=previous[candidates[0]] if len(candidates)==1 and usage[candidates[0]]==1 else None
            score=o.get('similarity'); identity=o.get('identity')
            reliable=o.get('quality',0)>=p.quality and o.get('gallery_valid',False) and score is not None and np.isfinite(score)
            evidence='master'
            if reliable and score>=p.internal_score and identity:
                evidence='internal'
            elif reliable and score<=p.external_score:
                evidence='external'
            token=identity if evidence=='internal' else evidence
            consistent=old is not None and old['evidence']==evidence and old['token']==token
            streak=old['streak']+1 if consistent else 1
            if old is None:
                self.next_id+=1
            track=old['track_id'] if old else self.next_id
            needed=p.internal_streak if evidence=='internal' else p.external_streak
            group=evidence if evidence!='master' and streak>=needed else 'master'
            current.append({**o,'group':group,'track_id':track,'streak':streak,'evidence':evidence,'token':token,
                            'reason':'consistent' if group!='master' else 'uncertain_or_track_reset'})
        self.previous=current; self.last_capture=capture_id
        return [{k:v for k,v in o.items() if k not in ('identity','token')} for o in current]

class CPUAdapter:
    """Same SCRFD/ArcFace detector.detect + norm_crop + get_feat as CameraProcessor._process.
    Explicit local ONNX models only; no FaceAnalysis model downloader or legacy recorder.
    """
    def __init__(self, detector_path, recognizer_path, gallery=None):
        self.Result,self.cosine=legacy_components()
        paths=[Path(detector_path),Path(recognizer_path)]
        if any(not p.is_file() or p.suffix!='.onnx' for p in paths):
            raise RestoreError('실제 SCRFD / ArcFace ONNX 모델 파일을 지정하세요')
        from insightface.model_zoo import get_model
        from insightface.utils import face_align
        self.align=face_align.norm_crop
        self.detector=get_model(str(paths[0]),providers=['CPUExecutionProvider'])
        self.recognizer=get_model(str(paths[1]),providers=['CPUExecutionProvider'])
        if not hasattr(self.detector,'detect') or not hasattr(self.recognizer,'get_feat'):
            raise RestoreError('SCRFD 탐지 모델 / ArcFace 인식 모델 종류를 확인하세요')
        self.detector.prepare(ctx_id=-1,input_size=(640,640),det_thresh=0.6)
        self.recognizer.prepare(ctx_id=-1)
        self.reload_gallery(gallery)

    def reload_gallery(self, gallery):
        users=[]
        if gallery:
            gp=Path(gallery).resolve(strict=True)
            conn=sqlite3.connect(gp.as_uri()+'?mode=ro&immutable=1',uri=True)
            try:
                for name,group,blob in conn.execute('SELECT name, auth_group, vector FROM users LIMIT 10001'):
                    if len(users)>=10000 or not isinstance(blob,bytes) or len(blob)>16384:
                        raise RestoreError('등록 DB 크기 / 벡터 형식 오류')
                    vec=np.load(io.BytesIO(blob),allow_pickle=False)
                    if vec.dtype.kind!='f' or vec.size not in (128,256,512) or not np.isfinite(vec).all():
                        raise RestoreError('등록 벡터 오류')
                    users.append((hashlib.sha256(str(name).encode()).hexdigest()[:16],vec))
            finally:
                conn.close()

        self.users=users

    def observe(self,frame,capture_id):
        boxes,landmarks=self.detector.detect(frame,max_num=0,metric='default')
        if boxes is None:
            return []
        out=[]; h,w=frame.shape[:2]
        for i,b in enumerate(boxes):
            coords=[max(0,int(b[0])),max(0,int(b[1])),min(w,int(b[2])),min(h,int(b[3]))]
            if coords[0]>=coords[2] or coords[1]>=coords[3]:
                raise RestoreError('탐지 bbox 오류')
            lm=landmarks[i] if landmarks is not None else None
            det=self.Result(coords,float(b[4]),lm)
            score=None; identity=None
            if lm is not None and np.isfinite(lm).all() and self.users:
                try:
                    emb=self.recognizer.get_feat(self.align(frame,landmark=lm,image_size=112))
                    if emb is not None and np.isfinite(emb).all():
                        ranked=[(self.cosine(emb,v),n) for n,v in self.users]
                        score,identity=max(ranked)
                except Exception as exc:
                    print('[ArcFace] 임베딩 생성 실패: 비허가자 표시 / 전체키 전용:',type(exc).__name__,flush=True)
            out.append({'bbox':det.bbox,'quality':det.conf,'similarity':score,'identity':identity,
                        'gallery_valid':bool(self.users)})
        return out

class SyntheticAdapter:
    """ONLY deterministic test scenes, never an actual face detector."""
    def observe(self,frame,capture_id):
        if frame.shape!=(240,320,3):
            raise RestoreError('합성 입력은 320×240이어야 합니다')
        return [
            {'bbox':[20,30,60,90],'quality':0.99,'similarity':0.1,'identity':None,'gallery_valid':True},
            {'bbox':[110,30,150,90],'quality':0.99,'similarity':0.9,'identity':'test-employee','gallery_valid':True},
            {'bbox':[230,30,270,90],'quality':0.5,'similarity':None,'identity':None,'gallery_valid':True},
            {'bbox':[80,150,130,200],'quality':0.99,'similarity':0.1,'identity':None,'gallery_valid':True},
            {'bbox':[120,150,165,200],'quality':0.99,'similarity':0.9,'identity':'test-employee-2','gallery_valid':True}]

def synthetic_frame(i):
    rng=np.random.default_rng(1729+i)
    return rng.integers(0,256,size=(240,320,3),dtype=np.uint8)
