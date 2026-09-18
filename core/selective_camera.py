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
from .selective_recognition import CPUAdapter, Classifier, SyntheticAdapter, synthetic_frame, iou
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
        # The old selective path hard-coded 0.60, which dropped many profile faces.
        # Reuse the project detection threshold so side faces are still detected and blurred.
        det_thresh=float(getattr(self.config,'DETECTOR_CONF_THRESHOLD',0.35))
        adapter=CPUAdapter(
            detector,
            recognizer,
            self.gallery_path if Path(self.gallery_path).is_file() else None,
            det_thresh=det_thresh,
        )
        self.detector=adapter.detector;self.recognizer=adapter.recognizer
        return adapter

    def start(self):
        with self.lock:
            if self.thread and self.thread.is_alive():return
            self.stop_event.clear();self.running=True;self.error=None
            self.thread=threading.Thread(target=self._loop,name='selective-camera',daemon=True)
            self.thread.start()

    def _update_display_authorization(self, faces, observations, states, now):
        """Stable live authorization with grant/retain hysteresis and restore-group sync.

        A face is granted only at MATCH_THRESHOLD. Once granted, a lower retain threshold,
        identity continuity, or short spatial continuity keeps it authorized through pose
        changes. Most importantly, every detected face is stored as internal/external using
        the same decision used by the monitor, so selective restore never silently falls into
        the legacy master-only bucket while the screen says '허가자'.
        """
        confirm=max(1,int(getattr(self.config,'FACE_AUTH_CONFIRM_MATCHES',2)))
        revoke=max(1,int(getattr(self.config,'FACE_AUTH_REVOKE_MISMATCHES',5)))
        hold=max(0.0,float(getattr(self.config,'FACE_AUTH_HOLD_SECONDS',0.8)))
        threshold=float(getattr(self.config,'MATCH_THRESHOLD',0.45))
        retain_threshold=float(getattr(self.config,'FACE_AUTH_RETAIN_THRESHOLD',max(0.25,threshold-0.15)))
        association_iou=float(getattr(self.config,'FACE_AUTH_TRACK_IOU',0.12))
        state_ttl=max(1.2,hold+0.4)
        next_states={}
        consumed=set()

        for face,observation in zip(faces,observations):
            track=face['track_id']
            bbox=face['bbox']
            score=face.get('similarity')
            identity=observation.get('identity')
            score_ok=score is not None and np.isfinite(score)
            gallery=face.get('gallery_valid',False)
            strong_match=bool(gallery and identity is not None and score_ok and score>=threshold)
            retain_match=bool(gallery and identity is not None and score_ok and score>=retain_threshold)

            preferred=f'id:{identity}' if identity is not None else f'track:{track}'
            state_key=preferred
            previous=states.get(preferred)

            # If ArcFace briefly loses identity during a head turn, recover the recent state
            # by spatial overlap instead of treating the same person as a brand-new stranger.
            if previous is None and not strong_match:
                best_key=None;best_state=None;best_iou=0.0
                for key,candidate in states.items():
                    if key in consumed or now-candidate.get('last_seen',now)>state_ttl:
                        continue
                    cbbox=candidate.get('bbox')
                    if cbbox is None:
                        continue
                    overlap=iou(bbox,cbbox)
                    if overlap>best_iou:
                        best_iou=overlap;best_key=key;best_state=candidate
                if best_state is not None and best_iou>=association_iou:
                    state_key=best_key;previous=best_state

            if previous is None:
                previous={
                    'identity':None,'matches':0,'misses':0,'authorized':False,
                    'hold_until':0.0,'bbox':bbox,'last_seen':now,'track_id':track,
                }
            state=dict(previous)
            previous_identity=state.get('identity')
            spatial_same=iou(bbox,state.get('bbox',bbox))>=association_iou

            if strong_match:
                if previous_identity in (None,identity):
                    state['matches']=state.get('matches',0)+1
                else:
                    state['matches']=1
                    state['authorized']=False
                state['identity']=identity
                state['misses']=0
                if state['matches']>=confirm:
                    state['authorized']=True
                if state.get('authorized',False):
                    state['hold_until']=now+hold
            else:
                same_identity=identity is not None and previous_identity==identity
                may_retain=bool(
                    state.get('authorized',False) and (
                        (same_identity and retain_match) or
                        spatial_same or
                        now<state.get('hold_until',0.0)
                    )
                )
                if may_retain:
                    # A decent lower-threshold ArcFace match does not count as a miss.
                    if same_identity and retain_match:
                        state['misses']=0
                        state['hold_until']=max(state.get('hold_until',0.0),now+hold)
                    else:
                        state['misses']=state.get('misses',0)+1
                    if state['misses']>=revoke and now>=state.get('hold_until',0.0):
                        state['authorized']=False
                        state['identity']=None
                        state['matches']=0
                        state['misses']=0
                else:
                    state['authorized']=False
                    state['matches']=0
                    state['misses']=0
                    if identity is None or not retain_match:
                        state['identity']=None

            authorized=bool(state.get('authorized',False))
            face['authorized']=authorized

            # Selective restore must follow the same classification shown on screen.
            # Strongly matched registered faces are internal immediately; an authorized held
            # face stays internal. Every other detected face is external. This intentionally
            # avoids the old 'master' bucket that made 0001/0002 appear to restore nothing.
            face['group']='internal' if (strong_match or authorized) else 'external'
            face['reason']='authorized_or_strong_match' if face['group']=='internal' else 'unrecognized_external'

            state['bbox']=bbox
            state['last_seen']=now
            state['track_id']=track
            if strong_match:
                state['identity']=identity
            final_key=f'id:{state["identity"]}' if state.get('identity') is not None else f'track:{track}'
            next_states[final_key]=state
            consumed.add(state_key)
            consumed.add(final_key)

        # Keep recently authorized identity state for a short detector/landmark dropout so a
        # reappearing side face does not have to build authorization from zero again.
        for key,state in states.items():
            if key in consumed or key in next_states:
                continue
            if state.get('authorized',False) and now-state.get('last_seen',now)<=state_ttl:
                next_states[key]=dict(state)

        return next_states

    def _loop(self):
        cap=None
        try:
            korean_font()
            self.adapter=self._adapter();classifier=Classifier()
            display_states={}
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
                        classifier=Classifier();display_states={}
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
                observations=self.adapter.observe(frame,capture)
                faces=classifier.classify(observations,capture)
                timestamp=cap.get(cv2.CAP_PROP_POS_MSEC) if cap is not None and is_file else (tick-start)*1000
                display_states=self._update_display_authorization(faces,observations,display_states,tick)
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
