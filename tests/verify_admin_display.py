"""Exercise the existing main.py with a physical camera and an isolated registration DB."""
import os,sys,time,subprocess,json,socket
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import httpx
import config
from playwright.sync_api import sync_playwright
out=ROOT/'tests/runtime/ui-review';out.mkdir(parents=True,exist_ok=True)
run=out/('after-'+str(time.time_ns()));run.mkdir(mode=0o700)
env=dict(os.environ,CAMERA_DEVICE='/dev/video0',SRX_DATA_ROOT=str(run),SRX_TEST_MODE='camera')
with socket.socket() as sock:
 sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);sock.bind(('127.0.0.1',8000))
log=(run/'server.log').open('w');p=subprocess.Popen([str(ROOT/'hk_env/bin/python'),'main.py'],env=env,stdout=log,stderr=log)
report={'run':str(run),'camera':'/dev/video0','synthetic':False,'production_gallery_modified':False}
try:
 with httpx.Client(base_url='http://127.0.0.1:8000',trust_env=False,timeout=30) as c:
  for _ in range(250):
   try:
    stats=c.get('/api/stats').json()
    if stats.get('frames',0)>=8:break
   except httpx.TransportError:pass
   time.sleep(.2)
  assert not stats.get('error'),stats
  for path in ('/selective/','/selective/api/status','/selective/api/chunks','/selective/api/jobs/fake/download','/api/assets','/api/debug'):
   assert c.get(path).status_code==401,path
  assert c.post('/selective/api/restore/'+'0'*32,content=b'').status_code==401
  assert 'href="/selective/"' not in c.get('/').text
  assert 'href="/selective/"' not in c.get('/admin/login').text
  assert not stats.get('chunks')
  report['guest_access_blocked']=True
  c.post('/api/admin/login',json={'id':config.ADMIN_ID,'password':config.ADMIN_PASSWORD}).raise_for_status()
  c.get('/selective/').raise_for_status()
  for path in ('/employees','/register','/audit','/decryption_page'):
   r=c.get(path);r.raise_for_status();assert 'href="/selective/"' in r.text,path
  report['admin_pages']=True
  os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(ROOT/'tests/runtime/browsers')
  with sync_playwright() as pw:
   browser=pw.chromium.launch();page=browser.new_page(viewport={'width':1440,'height':1000})
   page.goto('http://127.0.0.1:8000');page.wait_for_function('document.querySelector("#streamImg").naturalWidth===640')
   # No employee gallery: all actual detected faces must be denied on-screen.
   state=c.get('/selective/api/status').json()
   assert state['faces'] and all(not f['authorized'] and f['group']=='master' for f in state['faces'])
   page.screenshot(path=str(out/'after-unregistered.png'));report['unregistered_faces']=state['faces']
   registered=False
   for _ in range(12):
    r=c.post('/api/register_capture',json={'name':'ui_camera_verification','group':'staff'}).json()
    if r.get('status')=='success':registered=True;break
    time.sleep(.4)
   assert registered,r
   report['registration']=r
   for _ in range(200):
    state=c.get('/selective/api/status').json()
    if any(f['authorized'] for f in state['faces']):break
    time.sleep(.2)
   else:raise AssertionError('실제 등록 얼굴 허가자 판정 시간 초과')
   report['registered_faces']=state['faces'];report['stats']=c.get('/api/stats').json()
   assert report['stats']['employee_count']>0
   assert report['stats']['unknown_alerts']==report['stats']['unknown_count']
   page.wait_for_timeout(700);page.screenshot(path=str(out/'after-registered.png'))
   page.request.post('http://127.0.0.1:8000/api/admin/login',data={'id':config.ADMIN_ID,'password':config.ADMIN_PASSWORD})
   page.goto('http://127.0.0.1:8000/employees');page.screenshot(path=str(out/'admin-menu.png'))
   page.get_by_role('link',name='선택적 복원',exact=True).click();page.wait_for_selector('#chunks');report['admin_menu_opens_real_page']=True
   browser.close()
  c.post('/api/admin/logout').raise_for_status()
  assert c.get('/selective/api/chunks').status_code==401
  report['logout_revokes_access']=True
finally:
 p.terminate();p.wait(timeout=40);log.close()
report['graceful_shutdown']='Application shutdown complete.' in (run/'server.log').read_text()
from core.camera_manager import CameraManager
camera=CameraManager()
try:camera.open();report['camera_released']=True
finally:camera.release()
(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False))
