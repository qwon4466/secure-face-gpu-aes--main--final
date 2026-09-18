import io
import sqlite3
from types import SimpleNamespace
import time
import numpy as np
import pytest
from core.selective_recognition import CPUAdapter
from core.selective_camera import MainCamera
from core.selective_crypto import inspect_chunk,restore_frames,RestoreError

def test_main_camera_records_and_restarts(work,monkeypatch):
    monkeypatch.setenv('SRX_TEST_MODE','synthetic');monkeypatch.setenv('SRX_TEST_FRAMES','9');monkeypatch.setenv('SRX_DATA_ROOT',str(work))
    c=MainCamera(SimpleNamespace(SELECTIVE_FPS=120,SELECTIVE_CHUNK_FRAMES=8,PROCESS_WIDTH=640))
    for run in range(2):
        c.start();c.thread.join(timeout=10)
        assert not c.thread.is_alive() and c.error is None
        assert c.get_jpeg() is not None and c.capture_raw_frame().shape==(240,320,3)
        assert not hasattr(c,'_raw_recorder')
    folders=[p for p in c.chunks.iterdir() if p.is_dir()]
    assert len(folders)==4
    for p in folders:inspect_chunk(p)
    c.stop()

def test_main_model_failure_visible(work,monkeypatch):
    monkeypatch.delenv('SRX_TEST_MODE',raising=False);monkeypatch.setenv('SRX_DATA_ROOT',str(work))
    c=MainCamera(SimpleNamespace(SELECTIVE_MODEL_DIR=str(work/'no-models')))
    c.start();c.thread.join(timeout=5)
    assert not c.running and c.error and not list(c.chunks.iterdir())
    assert c.web_state()['error'] and not c.get_stats()['recording']

def test_gallery_reload(work):
    db=work/'users.db';conn=sqlite3.connect(db)
    conn.execute('CREATE TABLE users (name TEXT, auth_group TEXT, vector BLOB)')
    data=io.BytesIO();np.save(data,np.ones(512,dtype=np.float32),allow_pickle=False)
    conn.execute('INSERT INTO users VALUES (?,?,?)',('test','staff',data.getvalue()));conn.commit()
    a=CPUAdapter.__new__(CPUAdapter);a.reload_gallery(db);assert len(a.users)==1
    conn.execute('DELETE FROM users');conn.commit();a.reload_gallery(db);assert a.users==[]
    conn.close()

def test_invalid_operational_chunk_seconds_fails_before_start(monkeypatch):
    monkeypatch.delenv('SRX_TEST_MODE',raising=False);monkeypatch.delenv('SRX_DATA_ROOT',raising=False);monkeypatch.delenv('SRX_VIDEO_SOURCE',raising=False)
    with pytest.raises(RestoreError,match='CHUNK_SECONDS'):
        MainCamera(SimpleNamespace(CHUNK_SECONDS=0))
