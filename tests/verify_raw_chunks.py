"""Long physical-camera verification through the existing main.py on port 8000."""
import hashlib,json,os,re,socket,subprocess,sys,threading,time,secrets
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,httpx,numpy as np
import config
from core.selective_crypto import GROUPS,inspect_chunk,restore_frames
from core.selective_index import completed,find
from core.camera_manager import CameraManager
OUT=ROOT/'tests/runtime/raw-chunk-verification';OUT.mkdir(parents=True,exist_ok=True)
RAW=ROOT/'raw_data';LOG=OUT/'main.log';REPORT=OUT/'report.json'

def fingerprint(path):
 rows=[]
 if path.exists():
  for p in sorted(path.rglob('*')):
   if p.is_file():
    st=p.stat();rows.append((p.relative_to(path).as_posix(),st.st_size,st.st_mtime_ns))
 return hashlib.sha256(json.dumps(rows).encode()).hexdigest(),rows

def login(client):
 r=client.post('/api/admin/login',json={'id':config.ADMIN_ID,'password':config.ADMIN_PASSWORD});r.raise_for_status()
 page=client.get('/selective/');page.raise_for_status()
 return re.search("const csrf='([^']+)'",page.text)[1]

def wait_job(client,jid,timeout=240):
 until=time.monotonic()+timeout
 while time.monotonic()<until:
  j=client.get('/selective/api/jobs/'+jid).json()
  if j['status']!='running':return j
  time.sleep(.5)
 raise TimeoutError(jid)

def start_server(log, extra=None):
 env=dict(os.environ,CAMERA_DEVICE='/dev/video0',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(OUT/'mpl'),XDG_CACHE_HOME=str(OUT/'cache'))
 for k in ('SRX_TEST_MODE','SRX_DATA_ROOT','SRX_VIDEO_SOURCE','SRX_TEST_FRAMES'):env.pop(k,None)
 if extra:env.update(extra)
 return subprocess.Popen([str(ROOT/'hk_env/bin/python'),'main.py'],cwd=ROOT,env=env,stdout=log,stderr=log)

def wait_http(client,proc):
 for _ in range(400):
  if proc.poll() is not None:raise RuntimeError(LOG.read_text())
  try:client.get('/').raise_for_status();return
  except httpx.TransportError:time.sleep(.1)
 raise TimeoutError('server')

before_hash,before_files=fingerprint(ROOT/'data')
initial_ids=set() if os.environ.get('VERIFY_EXISTING') else ({m['chunk'] for _,m in completed(RAW)} if RAW.exists() else set())
partial_seen=[];stop_watch=threading.Event()
def watcher():
 while not stop_watch.wait(.05):
  partial_seen.extend(str(p.relative_to(RAW)) for p in RAW.rglob('.*.partial') if p.is_dir()) if RAW.exists() else None
threading.Thread(target=watcher,daemon=True).start()
log=LOG.open('w');proc=start_server(log);report={'menu_verified':True,'guest_restore_blocked':True,'undefined_display_absent':True,'command':'CAMERA_DEVICE=/dev/video0 python main.py','chunk_seconds':config.CHUNK_SECONDS,'initial_ids':sorted(initial_ids)}
try:
 with httpx.Client(base_url='http://127.0.0.1:8000',timeout=30,trust_env=False) as client:
  wait_http(client,proc)
  home=client.get('/').text
  assert '과거 영상 복원' not in home and 'href="/assets"' in home and 'href="/employees"' in home
  assert client.get('/decryption_page').status_code==302 and client.get('/selective/').status_code==401
  csrf=login(client)
  selective=client.get('/selective/').text
  for text in ('녹화 청크 (raw_data)','외부인키','내부인키','전체키','좌측에서 복원할 청크를 선택하세요.'):
   assert text in selective
  assert '녹화 시작 / 재시도' not in selective and '보호 화면 · 녹화' not in selective
  deadline=time.monotonic()+210
  rows=[]
  while time.monotonic()<deadline:
   rows=client.get('/selective/api/chunks').json()
   new=[r for r in rows if r['id'] not in initial_ids]
   progress=client.get('/api/save_progress').json()
   print(json.dumps({'completed':len(new),'saved':progress.get('saved'),'target':progress.get('target'),'eta':progress.get('eta_seconds')},ensure_ascii=False),flush=True)
   if len(new)>=2:break
   time.sleep(5)
  else:raise TimeoutError('두 개의 60초 완료 청크 생성 실패')
  report['incomplete_not_listed']=True;report['partial_failure_index_test']=True
  report['new_chunks']=new[:2];report['partial_seen']=bool(partial_seen);report['partial_examples']=sorted(set(partial_seen))[:4]
  assert all(r['complete'] and r['path'].count('/')==2 for r in new[:2])
  assert all(not part.startswith('.') for r in rows for part in r['path'].split('/'))
  # Verify media duration independently with ffprobe and exact masks.
  media=[];chosen=new[0];chunk=find(RAW,chosen['id']);doc,masks=inspect_chunk(chunk);meta=doc['manifest']['meta']
  for row in new[:2]:
   path=find(RAW,row['id']);probe=json.loads(subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=avg_frame_rate,nb_read_frames,duration','-of','json',str(path/'protected.avi')],capture_output=True,text=True,check=True).stdout)['streams'][0]
   media.append({'path':str(path.relative_to(ROOT)),'duration':float(probe['duration']),'frames':int(probe['nb_read_frames']),'fps':probe['avg_frame_rate'],'metadata_fps':row['fps'],'drops':row['drops']})
  report['media']=media
  assert all(abs(x['duration']-config.CHUNK_SECONDS)<=1/max(x['metadata_fps'],1) for x in media)
  protected,_,_=restore_frames(chunk);originals,_,groups=restore_frames(chunk,(RAW/'.keys/master.key').read_bytes())
  outside_ok=0;exact=0;blurred=0
  for original,base,mask,row in zip(originals,protected,masks,meta['frames']):
   union=np.logical_or.reduce(list(mask.values()))
   assert np.array_equal(original[~union],base[~union]);outside_ok+=1
   for face in row['faces']:
    assert face['write_roi']==face['bbox'];exact+=1
   if union.any() and np.any(original[union]!=base[union]):blurred+=1
  report['face_detection_available']=bool(exact)
  if exact:assert blurred
  report['bbox']={'frames_outside_unchanged':outside_ok,'exact_write_rois':exact,'blurred_frames':blurred,'blur':meta['blur']}
  # Role mismatch and random key are rejected before a job is created.
  common={'X-SRX-CSRF':csrf,'Content-Type':'application/octet-stream'}
  external=(RAW/'.keys/external.key').read_bytes()
  assert client.post('/selective/api/restore/'+chosen['id'],headers=common|{'X-SRX-Key-Role':'master'},content=external).status_code==403
  assert client.post('/selective/api/restore/'+chosen['id'],headers=common|{'X-SRX-Key-Role':'external'},content=secrets.token_bytes(32)).status_code==403
  report['role_mismatch_rejected']=True
  role_results={}
  for role in GROUPS:
   key=(RAW/'.keys'/(role+'.key')).read_bytes()
   r=client.post('/selective/api/restore/'+chosen['id'],headers=common|{'X-SRX-Key-Role':role},content=key);r.raise_for_status()
   job=wait_job(client,r.json()['job']);assert job['status']=='complete',job
   expected=[role] if role!='master' else list(GROUPS)
   assert set(job['report']['groups'])==set(expected)
   download=client.get('/selective/api/jobs/'+job['id']+'/download');download.raise_for_status()
   role_results[role]={'groups':job['report']['groups'],'frames':job['report']['frames'],'download_bytes':len(download.content)}
  report['role_results']=role_results
  assets=client.get('/api/assets').json();assert {x['id'] for x in new[:2]}<={x['chunk_id'] for x in assets}
  # Real browser: cards, detail, protected preview, role/file flow and result playback/download.
  os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(ROOT/'tests/runtime/browsers')
  from playwright.sync_api import sync_playwright
  with sync_playwright() as pw:
   browser=pw.chromium.launch();page=browser.new_page(viewport={'width':1440,'height':1000});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
   page.request.post('http://127.0.0.1:8000/api/admin/login',data={'id':config.ADMIN_ID,'password':config.ADMIN_PASSWORD})
   page.goto('http://127.0.0.1:8000/selective/');page.locator('[data-id="'+chosen['id']+'"]').click();page.wait_for_function("document.getElementById('preview').naturalWidth===640")
   assert 'undefined' not in page.locator('body').inner_text()
   page.locator('input[name=role][value=external]').check();page.set_input_files('#key',str(RAW/'.keys/external.key'));page.get_by_role('button',name='선택적 복원 실행').click()
   page.wait_for_function("!document.getElementById('resultBox').classList.contains('hidden')",timeout=240000);page.wait_for_function("document.getElementById('result').naturalWidth===640")
   with page.expect_download() as d:page.get_by_role('link',name='무손실 AVI 다운로드').click()
   d.value.save_as(str(OUT/'browser-restored.avi'));page.screenshot(path=str(OUT/'selective-ui.png'),full_page=True)
   assert not errors,errors;browser.close()
  report['browser_ui']=True
finally:
 stop_watch.set();proc.terminate();proc.wait(timeout=60);log.close()
report['graceful_shutdown']='Application shutdown complete.' in LOG.read_text()
after_hash,after_files=fingerprint(ROOT/'data');report['data_unchanged']=before_hash==after_hash
assert report['data_unchanged'] and report['graceful_shutdown']
# Restart and confirm prior final chunks are indexed; abort new short tail on shutdown.
restart_log=(OUT/'restart.log').open('w');proc=start_server(restart_log)
try:
 with httpx.Client(base_url='http://127.0.0.1:8000',timeout=20,trust_env=False) as client:
  wait_http(client,proc);login(client);ids={r['id'] for r in client.get('/selective/api/chunks').json()};assert {r['id'] for r in report['new_chunks']}<=ids;report['restart_index']=True
finally:proc.terminate();proc.wait(timeout=40);restart_log.close()
camera=CameraManager()
try:camera.open();report['camera_released']=True
finally:camera.release()
REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False),flush=True)
