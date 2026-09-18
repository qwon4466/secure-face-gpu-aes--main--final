from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import resource
import subprocess
import time
import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import cv2
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from .selective_recognition import Classifier, synthetic_frame
from .selective_crypto import MAX_FRAMES, MAX_RAW, MAX_CHUNK_RAW, MAX_PIXELS, RestoreError, canonical, private_dir, save_chunk, write_new

class Recorder:
    """Synchronous backpressure, no unbounded queue. At most one chunk in memory."""
    def __init__(self, output, keys, fps=10, chunk_frames=8, on_frame=None, on_chunk=None, duration=None):
        if not 1<=chunk_frames<=MAX_FRAMES or not 0<fps<=120:
            raise RestoreError('청크 프레임 / FPS 범위 오류')
        if duration is not None and (not isinstance(duration,(int,float)) or not 0<duration<=86400):
            raise RestoreError('CHUNK_SECONDS는 0보다 큰 유효한 초 값이어야 합니다')
        self.duration=duration;self.chunk_start=None;self.chunk_wall=None
        self.output=private_dir(output); self.keys=keys; self.fps=fps; self.limit=chunk_frames
        self.session=uuid.uuid4().hex; self.frames=[]; self.faces=[]; self.timing=[]
        self.on_frame=on_frame; self.on_chunk=on_chunk
        self.chunks=[]; self.latencies=[]; self.frames_total=0; self.drops=0; self.peak_buffer=0
        self.buffer_cipher=AESGCM(AESGCM.generate_key(bit_length=256));self.buffer_bytes=0
        self.last_capture=-1; self.shape=None; self.started=time.perf_counter()

    def add(self, frame, faces, capture_id, timestamp_ms):
        if capture_id<=self.last_capture or frame.nbytes>MAX_RAW or frame.shape[0]*frame.shape[1]>MAX_PIXELS:
            raise RestoreError('촬영 ID / 프레임 상한 오류')
        if self.duration and self.frames and time.monotonic()-self.chunk_start>=self.duration:
            self.flush(completed=True)
        if self.frames and (frame.shape!=self.shape or self.buffer_bytes+frame.nbytes>(MAX_CHUNK_RAW if self.duration else MAX_RAW)):
            if self.duration:raise RestoreError('60초 청크 메모리 상한 또는 해상도 변경; 미완성 청크 게시 안 함')
            self.flush()
        if not self.frames and self.duration:
            self.chunk_start=time.monotonic();self.chunk_wall=datetime.now(ZoneInfo('Asia/Seoul'))
            target=self.chunk_wall.strftime('%Y-%m-%d/%H/%H-%M-%S')
            print(f'[CHUNK] 녹화 시작 | 시작={self.chunk_wall:%Y-%m-%d %H:%M:%S} KST | 목표={self.duration:g}초 | 임시경로=raw_data/{target.rsplit("/",1)[0]}/.{target.rsplit("/",1)[1]}.partial',flush=True)
        self.shape=frame.shape; self.last_capture=capture_id
        nonce=capture_id.to_bytes(12,"big")
        self.frames.append((nonce,self.buffer_cipher.encrypt(nonce,frame.tobytes(),None)))
        self.buffer_bytes+=frame.nbytes
        self.faces.append(faces)
        self.timing.append({'capture_id':capture_id,'timestamp_ms':float(timestamp_ms),'source_pts_ms':float(timestamp_ms)})
        self.frames_total+=1; self.peak_buffer=max(self.peak_buffer,len(self.frames))
        if self.on_frame:
            from .selective_crypto import protect
            self.on_frame(protect(frame,faces)[0],faces,self.stats())
        if not self.duration and len(self.frames)>=self.limit:
            self.flush()

    def flush(self,completed=False):
        if not self.frames:
            return
        if self.duration and not completed:return
        t=time.perf_counter()
        elapsed=self.duration if self.duration else len(self.frames)/self.fps
        actual_fps=len(self.frames)/elapsed
        dropped=max(0,round(self.fps*elapsed)-len(self.frames)) if self.duration else 0
        final_path=self.output/self.chunk_wall.strftime('%Y-%m-%d/%H/%H-%M-%S') if self.duration else None
        originals=[np.frombuffer(self.buffer_cipher.decrypt(nonce,data,None),dtype=np.uint8).reshape(self.shape) for nonce,data in self.frames]
        try:
            chunk=save_chunk(self.output,originals,self.faces,self.timing,self.keys,self.session,actual_fps,final_path=final_path,duration=elapsed,drops=dropped,started_at=self.chunk_wall if self.duration else None,ended_at=(self.chunk_wall+timedelta(seconds=elapsed)) if self.duration else None)
        except Exception as exc:
            print(f'[CHUNK][ERROR] 저장 실패 | 임시경로={final_path} | 원인={exc}',flush=True)
            raise
        if self.duration:
            total=sum(f.stat().st_size for f in chunk.iterdir() if f.is_file())/1048576
            end=datetime.now(ZoneInfo('Asia/Seoul'))
            capture_end=self.chunk_wall+timedelta(seconds=elapsed)
            print(f'[CHUNK] 저장 완료 | 시작={self.chunk_wall:%Y-%m-%d %H:%M:%S} KST | 촬영종료={capture_end:%Y-%m-%d %H:%M:%S} KST | 저장완료={end:%Y-%m-%d %H:%M:%S} KST | 경로={chunk.relative_to(self.output)} | 목표={self.duration:g}초 | 길이={elapsed:.2f}초 | 프레임={len(originals)} | FPS={actual_fps:.2f} | 드롭={dropped} | 크기={total:.1f}MB',flush=True)
        self.latencies.append(time.perf_counter()-t); self.chunks.append(chunk.name)
        self.frames.clear(); self.faces.clear(); self.timing.clear();self.buffer_bytes=0;self.chunk_start=None
        if self.on_chunk:
            self.on_chunk(chunk)

    def progress(self):
        if not self.duration or self.chunk_start is None:
            return {'in_progress':False}
        elapsed=max(0,time.monotonic()-self.chunk_start)
        return {'in_progress':True,'saved':len(self.frames),'target':round(self.fps*self.duration),
                'fps':self.fps,'elapsed_seconds':round(elapsed,2),
                'eta_seconds':max(0,round(self.duration-elapsed)),
                'drops':max(0,round(self.fps*elapsed)-len(self.frames))}

    def stats(self):
        elapsed=time.perf_counter()-self.started
        return {'session':self.session,'resolution':list(self.shape[:2][::-1]) if self.shape else None,
                'frames':self.frames_total,'processing_fps':self.frames_total/max(elapsed,1e-9),
                'elapsed_seconds':elapsed,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'chunk_write_seconds':self.latencies.copy(),'drops':self.drops,
                'device_drops':'unknown','queue_capacity':0,'buffer_limit_frames':self.limit,
                'peak_buffer_frames':self.peak_buffer,'chunks':self.chunks.copy()}

    def finish(self):
        if not self.duration:self.flush()
        else:self.abort()  # a short final segment is never published as a completed chunk
        report=self.stats()
        if not self.duration:write_new(self.output/(self.session+'-stats.json'),canonical(report))
        return report

    def abort(self):
        # Discard uncommitted originals without ever persisting them.
        self.frames.clear(); self.faces.clear(); self.timing.clear();self.buffer_bytes=0


def process(source, adapter, output, keys, max_frames=80, fps=10, chunk_frames=8,
            camera=False, on_frame=None, on_chunk=None):
    if not 1<=max_frames<=100000:
        raise RestoreError('입력 프레임 범위는 1..100000입니다')
    recorder=Recorder(output,keys,fps,chunk_frames,on_frame,on_chunk)
    classifier=Classifier(); cap=None
    try:
        if source=='synthetic' and not camera:
            for i in range(max_frames):
                f=synthetic_frame(i)
                recorder.add(f,classifier.classify(adapter.observe(f,i),i),i,i*1000/fps)
        else:
            if camera:
                from .camera_manager import CameraManager
                cap=CameraManager(device=source).open()
            else:
                if not Path(source).is_file():raise RestoreError('로컬 입력 영상 파일이 없습니다')
                cap=cv2.VideoCapture(str(source))
            if not cap.isOpened():
                raise RestoreError('입력 영상을 열 수 없습니다')
            # Native decoder allocation is bounded by validating declared dimensions first.
            width,height=cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if not 0<width*height<=MAX_PIXELS:
                raise RestoreError('입력 해상도 상한 초과')
            source_fps=cap.get(cv2.CAP_PROP_FPS)
            if not 0<source_fps<=1000:
                source_fps=fps
            next_ms=0.0; i=0; started=time.monotonic()
            while i<max_frames:
                ok,f=cap.read()
                if not ok:
                    break
                timestamp=(time.monotonic()-started)*1000 if camera else i*1000/source_fps
                capture_id=i; i+=1
                if timestamp+1e-6<next_ms:
                    recorder.drops+=1
                    continue
                next_ms=timestamp+1000/fps
                faces=classifier.classify(adapter.observe(f,capture_id),capture_id)
                recorder.add(f,faces,capture_id,timestamp)
            if not recorder.frames_total:
                raise RestoreError('저장할 프레임이 없습니다')
        report=recorder.finish()
        report['policy']=asdict(classifier.policy)
        return report
    except BaseException:
        recorder.abort()
        raise
    finally:
        if cap is not None:
            cap.release()
