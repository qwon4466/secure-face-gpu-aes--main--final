"""Read actual cached SCRFD/ArcFace weights; run real main.py on synthetic file frames."""
import os,sys,time,json,subprocess,socket
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import cv2,numpy as np,httpx
from core.selective_crypto import write_video
from core.selective_camera import MainCamera
import config
out=ROOT/'tests/runtime/cpu-file';out.mkdir(parents=True,exist_ok=True)
source=out/'input.avi'
if not source.exists():write_video(source,[np.zeros((240,320,3),np.uint8) for _ in range(4)],10)
env=dict(os.environ,SRX_TEST_MODE='file',SRX_VIDEO_SOURCE=str(source),SRX_DATA_ROOT=str(out),PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(out/'mpl'),XDG_CACHE_HOME=str(out/'cache'))
with socket.socket() as s:
 s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('127.0.0.1',8000))
log=(out/'main.log').open('w');p=subprocess.Popen([str(ROOT/'hk_env/bin/python'),'main.py'],cwd=ROOT,env=env,stdout=log,stderr=log)
try:
 with httpx.Client(base_url='http://127.0.0.1:8000',timeout=20,trust_env=False) as c:
  for _ in range(100):
   try:c.get('/').raise_for_status();break
   except httpx.TransportError:time.sleep(.1)
  import config
  c.post('/api/admin/login',json={'id':config.ADMIN_ID,'password':config.ADMIN_PASSWORD}).raise_for_status()
  c.get('/selective/')
  for _ in range(200):
   state=c.get('/selective/api/status').json()
   if not state['recording']:break
   time.sleep(.2)
  assert state['error'] is None,state
  assert state['stats']['frames']==4,state
  assert len(c.get('/selective/api/chunks').json())>=1
  result={'actual_cpu_detector_file_pipeline':True,'frames':4,'state':state,'meaning':'blank synthetic frames, not real face accuracy'}
finally:
 p.terminate();p.wait(timeout=30);log.close()
# Execute the actual ArcFace get_feat API on a synthetic aligned crop as a compatibility check.
from core.selective_recognition import CPUAdapter
model=Path.home()/'.insightface/models/buffalo_s'
a=CPUAdapter(model/'det_500m.onnx',model/'w600k_mbf.onnx')
embedding=a.recognizer.get_feat(np.zeros((112,112,3),np.uint8))
assert embedding.size==512 and np.isfinite(embedding).all()
result['actual_arcface_embedding_shape']=list(embedding.shape)
(out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False))
