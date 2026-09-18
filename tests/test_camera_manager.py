from types import SimpleNamespace
import numpy as np
import pytest
from core import camera_manager as cm

@pytest.fixture
def fake(monkeypatch):
    monkeypatch.delenv('CAMERA_DEVICE', raising=False)
    monkeypatch.setattr(cm.Path, 'exists', lambda self: True)
    monkeypatch.setattr(cm, 'product', lambda path: 'Color Camera')
    monkeypatch.setattr(cm, 'candidates', lambda: ['/dev/video8','/dev/video9'])
    frame=np.random.default_rng(1).integers(0,255,(24,32,3),dtype=np.uint8)
    opened=[]; modes={}; active=[]
    class Capture:
        def __init__(self,path,backend):
            assert backend == cm.cv2.CAP_V4L2
            assert not active, 'simultaneous capture'
            active.append(self);opened.append(path);self.path=path;self.count=0
        def isOpened(self):return modes.get(self.path)!='closed'
        def set(self,*args):return True
        def get(self,prop):return 0 if modes.get(self.path)=='zero' else 30
        def read(self):
            self.count+=1
            mode=modes.get(self.path)
            if mode=='empty' or mode=='intermittent' and self.count==5:return False,None
            if mode=='gray':return True,np.repeat(frame[:,:,:1],3,axis=2)
            if mode=='solid':return True,np.full_like(frame,[10,40,80])
            if mode=='depth':return True,frame[:,:,0].astype(np.uint16)
            return True,frame
        def release(self):
            if self in active:active.remove(self)
    monkeypatch.setattr(cm.cv2,'VideoCapture',Capture)
    return modes,opened,active

@pytest.mark.parametrize('mode',['closed','zero','empty','intermittent','gray','solid','depth'])
def test_reject_and_release(fake, mode):
    modes,opened,active=fake;modes['/dev/video8']=mode
    camera=cm.CameraManager()
    try:
        camera.open();assert camera.selected['device']=='/dev/video9'
        assert len(camera.failures)==1
    finally:camera.release()
    assert not active

def test_manual_and_single_owner(fake,monkeypatch):
    monkeypatch.setenv('CAMERA_DEVICE','/dev/video4');config=SimpleNamespace()
    camera=cm.CameraManager(config=config)
    try:
        camera.open();assert fake[1]==['/dev/video4']
        assert config.CAMERA_SELECTED_DEVICE=='/dev/video4'
        with pytest.raises(RuntimeError,match='이미'):cm.CameraManager().open()
    finally:camera.release()

def test_invalid_manual_falls_back(fake,monkeypatch,capsys):
    monkeypatch.setenv('CAMERA_DEVICE','/missing')
    monkeypatch.setattr(cm.Path,'exists',lambda self:str(self)!='/missing')
    camera=cm.CameraManager()
    try:
        camera.open();assert camera.selected['device']=='/dev/video8'
        assert '자동 탐색으로 전환' in capsys.readouterr().out
    finally:camera.release()

def test_reconnect_same_then_scan(fake):
    camera=cm.CameraManager()
    try:
        camera.open();camera.reconnect();assert fake[1]==['/dev/video8']*2
        fake[0]['/dev/video8']='empty';camera.reconnect()
        assert camera.selected['device']=='/dev/video9'
    finally:camera.release()

def test_discovery_priority_and_numeric(monkeypatch):
    monkeypatch.setattr(cm.glob,'glob',lambda pattern: ['/dev/video12','/dev/video2'] if pattern=='/dev/video*' else [])
    monkeypatch.setattr(cm,'product',lambda p:'Logitech')
    assert cm.candidates()==['/dev/video2','/dev/video12']
    monkeypatch.setattr(cm,'product',lambda p:'RealSense RGB' if p.endswith('12') else 'Logitech')
    assert cm.candidates()==['/dev/video12','/dev/video2']

def test_requested_mode_failure_reopens_native(fake,monkeypatch,capsys):
    original=cm.cv2.VideoCapture
    monkeypatch.setattr(cm,'product',lambda path:'Logitech 046d:081b')
    def capture(path,backend):
        cap=original(path,backend)
        cap.set=lambda prop,value: prop!=cm.cv2.CAP_PROP_FRAME_WIDTH
        return cap
    monkeypatch.setattr(cm.cv2,'VideoCapture',capture)
    camera=cm.CameraManager()
    try:
        camera.open()
        assert fake[1]==['/dev/video8','/dev/video8']
        assert camera.selected['profile']=='device-native'
        assert '장치 기본 형식/해상도로 재시도' in capsys.readouterr().out
    finally:camera.release()

def test_no_devices_reports_reason(fake,monkeypatch):
    monkeypatch.setattr(cm,'candidates',lambda:[])
    with pytest.raises(RuntimeError,match='검색된 장치 없음'):cm.CameraManager().open()
    assert not cm._OWNER.locked()
