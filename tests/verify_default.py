"""Check unmodified launch command, with no synthetic mode or alternate input."""
from pathlib import Path
import os,sys,subprocess,time,socket,json,httpx
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root));out=root/'tests/runtime/default-start';out.mkdir(parents=True,exist_ok=True)
with socket.socket() as s:
 s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('127.0.0.1',8000))
env=dict(os.environ)
for name in ('SRX_TEST_MODE','SRX_DATA_ROOT','SRX_TEST_FRAMES','SRX_PORT','SRX_VIDEO_SOURCE'):env.pop(name,None)
env.update(PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(out/'mpl'),XDG_CACHE_HOME=str(out/'cache'))
log=(out/'server.log').open('w');p=subprocess.Popen([str(root/'hk_env/bin/python'),'main.py'],cwd=root,env=env,stdout=log,stderr=log)
try:
 with httpx.Client(base_url='http://127.0.0.1:8000',trust_env=False) as c:
  for _ in range(100):
   try:c.get('/').raise_for_status();break
   except httpx.TransportError:time.sleep(.1)
  import config
  c.post('/api/admin/login',json={'id':config.ADMIN_ID,'password':config.ADMIN_PASSWORD}).raise_for_status()
  c.get('/selective/')
  for _ in range(200):
   status=c.get('/selective/api/status').json()
   if status['error'] or status['stats'].get('frames'):break
   time.sleep(.1)
  result={'normal_main_start':True,'address':'http://127.0.0.1:8000','test_mode':False,'camera_status':status,'key_permissions':{f.name:oct(f.stat().st_mode&0o777) for f in (root/'raw_data/.keys').glob('*.key')}}
  (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False))
finally:
 p.terminate();p.wait(timeout=30);log.close()
