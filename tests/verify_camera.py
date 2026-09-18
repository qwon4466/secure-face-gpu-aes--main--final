"""Physical camera verification through the existing main.py, never a separate server."""
import os,sys,json,time,subprocess,socket
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import httpx,numpy as np,cv2
from core.camera_manager import CameraManager
from core.selective_crypto import restore_frames,protect
out=ROOT/'tests/runtime/camera-verification';out.mkdir(parents=True,exist_ok=True)
reports=[]
for label,device in [('auto',None),('manual','/dev/video0'),('invalid','/dev/secureface-nonexistent'),('metadata','/dev/video1')]:
    if os.environ.get('VERIFY_CAMERA_CASE') and os.environ['VERIFY_CAMERA_CASE']!=label:continue
    run=out/(label+'-'+str(time.time_ns()));run.mkdir(mode=0o700)
    env=dict(os.environ,SRX_DATA_ROOT=str(run),PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(out/'mpl'))
    for key in ('CAMERA_DEVICE','SRX_TEST_MODE','SRX_VIDEO_SOURCE'):env.pop(key,None)
    if device:env['CAMERA_DEVICE']=device
    with socket.socket() as sock:sock.bind(('127.0.0.1',8000))
    log=(run/'server.log').open('w')
    proc=subprocess.Popen([str(ROOT/'hk_env/bin/python'),'main.py'],cwd=ROOT,env=env,stdout=log,stderr=log)
    report={'case':label,'requested':device,'run':str(run)}
    try:
        with httpx.Client(base_url='http://127.0.0.1:8000',timeout=15,trust_env=False) as client:
            for _ in range(300):
                assert proc.poll() is None
                try:
                    state=client.get('/api/stats').json()
                    if state.get('frames',0)>=8:break
                except httpx.TransportError:pass
                time.sleep(.2)
            else:raise AssertionError('Camera startup timeout')
            assert not state['error'],state
            report['stats']=state
            frame=cv2.imdecode(np.frombuffer(client.get('/snapshot/cam_0').content,np.uint8),cv2.IMREAD_COLOR)
            assert frame.shape==(480,640,3)
            report['snapshot_shape']=list(frame.shape)
            if label in ('auto','manual'):
                os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(ROOT/'tests/runtime/browsers')
                from playwright.sync_api import sync_playwright
                with sync_playwright() as pw:
                    browser=pw.chromium.launch(headless=True)
                    page=browser.new_page();page.goto('http://127.0.0.1:8000/')
                    page.wait_for_function('document.querySelector("#streamImg").naturalWidth === 640')
                    report['browser_live_image']=page.locator('#streamImg').evaluate('(e)=>({width:e.naturalWidth,height:e.naturalHeight})')
                    browser.close()
    finally:
        proc.terminate();proc.wait(timeout=40);log.close()
    report['server_exit']=proc.returncode
    report['graceful_shutdown']='Application shutdown complete.' in (run/'server.log').read_text()
    assert report['graceful_shutdown']
    # Confirm the real captured chunk has the expected blur; decrypted originals stay in RAM.
    face_count=0;blur_checked=0
    for chunk in (run/'chunks').iterdir():
        if not chunk.is_dir():continue
        frames,meta,_=restore_frames(chunk,(run/'keys/master.key').read_bytes())
        protected,_,_=restore_frames(chunk)
        for frame,base,entry in zip(frames,protected,meta['frames']):
            faces=entry['faces'];face_count+=len(faces)
            assert np.array_equal(protect(frame,faces)[0],base)
            if faces and np.any(frame!=base):blur_checked+=1
    report.update(detected_faces=face_count,blur_verified_frames=blur_checked)
    camera=CameraManager()
    try:
        camera.open();report['released_and_reopened']=camera.isOpened()
    finally:camera.release()
    if label in ('invalid','metadata'):assert '자동 탐색으로 전환' in (run/'server.log').read_text()
    reports.append(report)
    (out/('report-'+os.environ['VERIFY_CAMERA_CASE']+'.json' if os.environ.get('VERIFY_CAMERA_CASE') else 'report.json')).write_text(json.dumps(reports,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False),flush=True)
