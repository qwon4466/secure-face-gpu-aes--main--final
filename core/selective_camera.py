"""Camera interface for main.py: one raw read, selective recorder, protected monitor."""
import io
import os
from pathlib import Path
import sqlite3
import shutil
import threading
import time
import cv2
import numpy as np
from .selective_crypto import ROOT, RestoreError, generate_keys, keyset, private_dir, protect
from .selective_recognition import CPUAdapter, Classifier, Policy, SyntheticAdapter, synthetic_frame, iou
from .selective_recording import Recorder
from .camera_manager import CameraManager, SELECTED
from .face_display import korean_font

DB_PATH='security_system.db'
def _adapt_array(arr):
    buf=io.BytesIO();np.save(buf,arr,allow_pickle=False);return sqlite3.Binary(buf.getvalue())
def _convert_array(data):
    return np.load(io.BytesIO(data),allow_pickle=False)

class MainCamera:
    def __init__(self, config):
        self.config=config
        self.gallery_path=DB_PATH
        self.test=os.environ.get('SRX_TEST_MODE')=='synthetic'
        self.operational=not (self.test or os.environ.get('SRX_DATA_ROOT'))
        if self.operational:
            duration=getattr(config,'CHUNK_SECONDS',60)
            if isinstance(duration,bool) or not isinstance(duration,(int,float)) or not 0<duration<=86400:
                raise RestoreError('CHUNK_SECONDS는 0보다 큰 유효한 초 값이어야 합니다')
        default_root=ROOT/'tests/runtime/main-server-test' if self.test else getattr(config,'RAW_DATA_DIR',ROOT/'raw_data')
        self.root=private_dir(Path(os.environ.get('SRX_DATA_ROOT',str(default_root))).absolute())
        self.keydir=self.root/('.keys' if self.operational else 'keys')
        if not self.keydir.exists():
            old=ROOT/'data/selective/main-server/keys'
            if self.operational and old.is_dir():
                private_dir(self.keydir)
                for name in ('internal.key','external.key','master.key'):
                    shutil.copyfile(old/name,self.keydir/name);os.chmod(self.keydir/name,0o600)
            else:generate_keys(self.keydir)
        self.keys=keyset(self.keydir)
        self.chunks=self.root if self.operational else private_dir(self.root/'chunks')
        self.lock=threading.RLock();self.thread=None;self.stop_event=threading.Event()
        self.preview=None;self.raw=None;self.detector=None;self.recognizer=None;self.adapter=None
        self.error=None;self.faces=[];self.metrics={};self.running=False;self.recorder=None

    def reload_db(self):
        if self.adapter is not None and not self.test:
            self.adapter.reload_gallery(self.gallery_path)

    def _adapter(self):
        if self.test:return SyntheticAdapter()
        modeldir=Path(getattr(self.config,'SELECTIVE_MODEL_DIR',Path.home()/'.insightface/models/buffalo_s'))
        detector=getattr(self.config,'SELECTIVE_DETECTOR_PATH','')
        recognizer=getattr(self.config,'SELECTIVE_RECOGNIZER_PATH','')
        if not detector or not recognizer:
            candidates=list(modeldir.glob('*.onnx'))
            detector=next((str(p) for p in candidates if 'scrfd' in p.name.lower() or p.name.lower().startswith('det_')),'')
            recognizer=next((str(p) for p in candidates if 'w600k' in p.name.lower() or 'rec' in p.name.lower()),'')
        adapter=CPUAdapter(detector,recognizer,self.gallery_path if Path(self.gallery_path).is_file() else None)
        self.detector=adapter.detector;self.recognizer=adapter.recognizer
        return adapter

    def start(self):
        with self.lock:
            if self.thread and self.thread.is_alive():return
            self.stop_event.clear();self.running=True;self.error=None
            self.thread=threading.Thread(target=self._loop,name='selective-camera',daemon=True)
            self.thread.start()

    def _update_display_authorization(self, faces, observations, states, now):
        """Apply match confirmation + delayed revoke to prevent one-frame auth flicker."""
        confirm=max(1,int(getattr(self.config,'FACE_AUTH_CONFIRM_MATCHES',2)))
        window=max(confirm,int(getattr(self.config,'FACE_AUTH_CONFIRM_WINDOW',3)))
        revoke=max(1,int(getattr(self.config,'FACE_AUTH_REVOKE_MISMATCHES',5)))
        hold=max(0.0,float(getattr(self.config,'FACE_AUTH_HOLD_SECONDS',0.8)))
        threshold=float(getattr(self.config,'MATCH_THRESHOLD',0.45))
        retain=float(getattr(self.config,'FACE_AUTH_RETAIN_THRESHOLD',max(0.0,threshold-0.05)))
        next_states={}

        for face,observation in zip(faces,observations):
            track=face['track_id']
            score=face.get('similarity')
            identity=observation.get('identity')
            matched=(face.get('gallery_valid',False) and identity is not None and
                     score is not None and np.isfinite(score) and score>=threshold)

            previous=states.get(track,{
                'identity':None,'matches':0,'history':[],'misses':0,
                'authorized':False,'hold_until':0.0
            })
            state=dict(previous)

            valid=(face.get('gallery_valid',False) and identity is not None and
                   score is not None and np.isfinite(score))
            same_identity=valid and state.get('identity')==identity
            if matched:
                if not same_identity:
                    state['identity']=identity
                    state['history']=[]
                    state['matches']=0
                history=list(state.get('history',[]))[-window+1:]
                history.append(1)
                state['history']=history
                state['matches']=sum(history)
                state['misses']=0
                if state['matches']>=confirm:
                    state['authorized']=True
                    state['hold_until']=now+hold
            elif valid and same_identity and float(score)>=retain and state.get('authorized',False):
                # A small score dip from pose/landmark changes is retained.
                state['misses']=0
                state['hold_until']=now+hold
            elif valid:
                # Only a valid, sustained mismatch counts toward revocation.
                state['history']=[]
                state['matches']=0
                if state.get('authorized',False):
                    state['misses']=state.get('misses',0)+1
                    if state['misses']>=revoke:
                        state['authorized']=False
                        state['identity']=None
                        state['misses']=0
                else:
                    state['identity']=None
                    state['misses']=0
            elif not state.get('authorized',False):
                state['identity']=None
                state['history']=[]
                state['matches']=0
            # Missing/invalid embeddings are deliberately held for the short
            # continuity window and never counted as a mismatch.

            face['authorized']=bool(state.get('authorized',False))
            # Storage policy follows the final display decision.  There is no
            # operational third bucket that could make selective restore skip
            # an otherwise visible face.
            face['group']='internal' if face['authorized'] else 'external'
            next_states[track]=state

        return next_states

    def _hold_detector_dropouts(self, observations, previous_observations,
                                last_detected_at, now):
        """Reuse a recent bbox during a brief detector miss only."""
        if not previous_observations or now-last_detected_at > float(getattr(self.config,'FACE_AUTH_HOLD_SECONDS',0.8)):
            return observations
        threshold=float(getattr(self.config,'FACE_AUTH_TRACK_IOU',0.35))
        current=list(observations)
        if not current:
            current=[dict(item,quality=0.0,detector_hold=True) for item in previous_observations]
            return current
        for old in previous_observations:
            if any(iou(old['bbox'],item['bbox'])>=threshold for item in current):
                continue
            held=dict(old,quality=0.0,detector_hold=True)
            current.append(held)
        return current

    def _loop(self):
        cap=None
        try:
            korean_font()
            self.adapter=self._adapter()
            classifier=Classifier(Policy(track_iou=float(getattr(self.config,'FACE_AUTH_TRACK_IOU',0.35))))
            display_states={}
            previous_observations=[];last_detected_at=-1e9
            fps=getattr(self.config,'SELECTIVE_FPS',10)
            duration=getattr(self.config,'CHUNK_SECONDS',None) if self.operational else None
            self.recorder=Recorder(self.chunks,self.keys,fps=fps,chunk_frames=getattr(self.config,'SELECTIVE_CHUNK_FRAMES',8),duration=duration)
            source=getattr(self.config,'SELECTIVE_SOURCE','')
            if not source and getattr(self.config,'FORCE_VIDEO',False):source=getattr(self.config,'VIDEO_FALLBACK',None)
            is_file=bool(source)
            if not self.test:
                if not source:
                    cap=CameraManager(config=self.config,stop_event=self.stop_event)
                    while not self.stop_event.is_set():
                        try:
                            cap.open();self.error=None;break
                        except RuntimeError as exc:
                            self.error=str(exc);print('[Camera]',exc,flush=True)
                            self.stop_event.wait(3)
                else:
                    if not Path(source).is_file():raise RestoreError('설정한 입력 파일이 없습니다')
                    cap=cv2.VideoCapture(str(source))
                    if not cap.isOpened():raise RestoreError('파일 입력을 열 수 없습니다')
            capture=0;start=time.monotonic()
            limit=int(os.environ.get('SRX_TEST_FRAMES','24')) if self.test else None
            while not self.stop_event.is_set():
                tick=time.monotonic()
                if self.test:
                    if capture>=limit:break
                    frame=synthetic_frame(capture)
                else:
                    ok,frame=cap.read()
                    if not ok:
                        if is_file:break
                        if self.recorder.duration:self.recorder.abort()
                        else:self.recorder.flush()
                        classifier=Classifier(Policy(track_iou=float(getattr(self.config,'FACE_AUTH_TRACK_IOU',0.35))))
                        display_states={}
                        with self.lock:
                            self.error='카메라 연결 끊김: 재연결 중';self.preview=None;self.raw=None;self.faces=[]
                        print('[Camera] 프레임 읽기 실패: 동일 장치 재연결 시도',flush=True)
                        while not self.stop_event.wait(2):
                            try:
                                cap.reconnect();self.error=None;break
                            except RuntimeError as exc:
                                self.error=str(exc);print('[Camera]',exc,flush=True)
                        continue
                width=getattr(self.config,'PROCESS_WIDTH',640)
                if width and frame.shape[1]>width:
                    height=int(frame.shape[0]*width/frame.shape[1])//2*2
                    frame=cv2.resize(frame,(width//2*2,height))
                detected=self.adapter.observe(frame,capture)
                if detected:
                    last_detected_at=tick
                observations=self._hold_detector_dropouts(
                    detected,previous_observations,last_detected_at,tick)
                faces=classifier.classify(observations,capture)
                timestamp=cap.get(cv2.CAP_PROP_POS_MSEC) if cap is not None and is_file else (tick-start)*1000
                display_states=self._update_display_authorization(faces,observations,display_states,tick)
                previous_observations=list(observations)
                self.recorder.add(frame,faces,capture,timestamp)
                protected=protect(frame,faces)[0]
                ok,jpg=cv2.imencode('.jpg',protected)
                if not ok:raise RestoreError('보호 프레임 인코딩 실패')
                with self.lock:
                    self.preview=jpg.tobytes();self.raw=frame.copy();self.faces=faces;self.metrics=self.recorder.stats()
                capture+=1
                self.stop_event.wait(max(0,1/fps-(time.monotonic()-tick)))
            report=self.recorder.finish()
            with self.lock:self.metrics=report
        except Exception as exc:
            if self.recorder:self.recorder.abort()
            with self.lock:self.error=str(exc)
        finally:
            if cap:cap.release()
            with self.lock:self.running=False

    def stop(self):
        self.stop_event.set()
        if self.thread:self.thread.join(timeout=15)

    def get_jpeg(self):
        with self.lock:return self.preview
    def get_raw_jpeg(self):
        with self.lock:
            if self.raw is None:return None
            return cv2.imencode('.jpg',self.raw)[1].tobytes()
    def capture_raw_frame(self):
        with self.lock:return None if self.raw is None else self.raw.copy()
    def get_stats(self):
        with self.lock:
            return {'employee_count':sum(f.get('authorized',False) for f in self.faces),
                    'unknown_count':sum(not f.get('authorized',False) for f in self.faces),
                    'recording':self.running and self.error is None,'error':self.error,
                    'selective_restore':True,'camera':dict(SELECTED),**self.metrics}
    def get_save_progress(self):
        with self.lock:return self.recorder.progress() if self.recorder else {'in_progress':False}
    def get_perf(self):
        with self.lock:return self.metrics.copy()
    def get_debug_info(self):return self.get_stats()
    def web_state(self):
        with self.lock:return {'recording':self.running,'error':self.error,'faces':self.faces.copy(),'stats':self.metrics.copy()}

    def list_assets(self):
        from .selective_index import completed, summary
        result=[]
        for path,meta in completed(self.chunks):
            item=summary(path,meta,self.chunks)
            result.append({'chunk_id':item['id'],'display_name':item['name'],'date':item['date'],'start_time':item['started_at'][11:19],
                'duration_seconds':item['duration_seconds'],'frame_count':item['frames'],
                'total_faces':sum(item['groups'].values()),'complete':True,'format':'selective',
                'fps':item['fps'],'resolution':item['shape'][1::-1],'path':item['path']})
        return result
