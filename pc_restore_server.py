"""
PC 로컬 GPU 복원 서버 (Modal 대체 · 무료).

라즈베리파이가 저장한 PSF 보호본 청크(tar.gz)를 POST로 받아, 이 PC의 GPU로
INN 역변환 복원해 mp4를 돌려준다. Modal 엔드포인트와 요청/응답 계약이 같아
라즈베리파이의 "GPU 복원" 버튼이 그대로 이 서버를 호출하게 만들 수 있다.

  요청: POST multipart  file=<청크 tar.gz>, password=<복원 비밀번호>
  응답: video/mp4 (복원된 영상)

── PC(예: GTX 1660 Ti)에서 실행 ───────────────────────────────────────────
  1) CUDA용 PyTorch 설치 (예: CUDA 12.1)
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  2) 나머지 의존성
     pip install "fastapi[standard]" uvicorn python-multipart opencv-python numpy pillow pycryptodome
  3) 체크포인트(.pth)를 checkpoints/ 폴더에 두고 실행
     python pc_restore_server.py           # 0.0.0.0:8500 대기
  4) 라즈베리파이 config.py 에 이 PC 주소를 지정
     REMOTE_RESTORE_URL = "http://<이_PC_IP>:8500/restore"

  * ffmpeg 이 PATH에 있으면 브라우저 호환 H.264로, 없으면 mp4v로 반환.
  * 방화벽에서 8500 포트 인바운드 허용 필요 (아래 GET /health 로 확인).
"""
import asyncio
import io
import os
import sys
import tarfile
import tempfile
import threading

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

# 이 파일이 있는 폴더(=저장소 루트)를 import 경로에 추가 → restore_chunk 등 로드
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

app = FastAPI(title="SecureFace-RX PC GPU Restore")

# GPU를 저장(protect)/복원(restore)이 나눠 쓰므로 직렬화 + 복원 우선 처리.
# _restore_active가 켜져 있으면 저장 요청은 503(busy)로 즉시 반려 → 라즈베리파이가
# 저장을 잠시 멈추고 재시도. 복원이 GPU를 독점해 먼저 끝난다.
_gpu_lock = threading.Lock()
_restore_active = threading.Event()


@app.get("/health")
def health():
    """GPU 인식 여부 확인용. 브라우저로 http://<PC_IP>:8500/health 접속."""
    try:
        import torch
        cuda = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if cuda else "cpu"
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "cuda": cuda, "device": dev}


@app.post("/restore")
async def restore(file: UploadFile = File(...),
                  password: str = Form("forensic2026")):
    """청크 tar.gz를 받아 GPU로 복원해 mp4 반환 (Modal restore와 동일 계약)."""
    import torch

    data = await file.read()
    workdir = tempfile.mkdtemp(prefix="pcrestore_")
    try:
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            t.extractall(workdir)
    except Exception as e:
        return JSONResponse({"error": f"압축 해제 실패: {e}"}, status_code=400)

    from restore_chunk import restore_chunk
    out = os.path.join(workdir, "restored.mp4")

    def _work():
        with _gpu_lock:               # 저장과 GPU 동시 사용 방지
            with torch.no_grad():
                return restore_chunk(workdir, password=password, out_path=out)

    # 복원 우선: 플래그를 켜서 저장 요청이 GPU를 양보하게 함 (저장은 503으로 반려)
    _restore_active.set()
    print("[protect] ⏸ 복원 우선 처리 시작 — 저장 잠시 대기")
    try:
        result = await asyncio.to_thread(_work)
    except Exception as e:
        return JSONResponse({"error": f"복원 실패: {e}"}, status_code=500)
    finally:
        _restore_active.clear()
        print("[protect] ▶ 복원 완료 — 저장 재개")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if result is None or not os.path.exists(result):
        return JSONResponse({"error": "복원할 청크를 찾지 못했거나 실패"},
                            status_code=500)
    with open(result, "rb") as f:
        mp4 = f.read()
    return Response(content=mp4, media_type="video/mp4")


def _tar_add(tar, name, data):
    """메모리 bytes를 tar에 파일로 추가."""
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


@app.post("/protect")
async def protect(frame: UploadFile = File(...),
                  faces: str = Form(...),
                  password: str = Form("forensic2026"),
                  ts: str = Form("0")):
    """
    저장(INN 보호본 생성)을 이 PC의 GPU로 수행 (시험용).

    라즈베리파이가 탐지·인식(Hailo)을 끝낸 뒤 원본 프레임(JPEG)과 얼굴 목록을
    보내면, GPU로 INN protect_roi를 적용해 보호본 프레임 + 복원용 타일을
    tar.gz(frame.jpg, face_i.npy, face_i_box.json)로 되돌려준다.

    ⚠️ 익명화 前 '원본' 프레임을 받는 경로다. 시험/데모 용도로만 사용.
    """
    import json

    import cv2
    import numpy as np
    import torch

    import config as cfg
    import parallel_protect as pp

    # 복원이 우선 처리 중이면 저장은 즉시 반려(503) → 라즈베리파이가 잠시 멈추고
    # 재시도. 복원이 GPU를 독점해 먼저 끝나게 한다.
    if _restore_active.is_set():
        return JSONResponse({"busy": "restore"}, status_code=503)

    # INN 모델 1회 로드 (CUDA 있으면 자동 GPU)
    pp.load_anonymizer(getattr(cfg, "INN_CHECKPOINT", None), password)

    data = await frame.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "프레임 디코드 실패"}, status_code=400)
    try:
        face_list = json.loads(faces)
    except Exception as e:
        return JSONResponse({"error": f"faces 파싱 실패: {e}"}, status_code=400)

    import time as _time
    _t0 = _time.time()

    def _work():
        with _gpu_lock:               # 복원과 GPU 동시 사용 방지
            with torch.no_grad():
                return pp.protect_one(img, face_list, password)

    try:
        anon, tiles = await asyncio.to_thread(_work)
    except Exception as e:
        return JSONResponse({"error": f"보호 실패: {e}"}, status_code=500)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _protect_ms = (_time.time() - _t0) * 1000

    # 보호본 프레임 + 타일을 tar.gz로 포장 (청크 스냅샷과 같은 파일 구성)
    _t1 = _time.time()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        _ok, enc = cv2.imencode(".jpg", anon)
        _tar_add(t, "frame.jpg", enc.tobytes())
        for i, td in enumerate(tiles):
            nb = io.BytesIO()
            np.save(nb, td["tile_f32"])
            _tar_add(t, f"face_{i}.npy", nb.getvalue())
            _tar_add(t, f"face_{i}_box.json",
                     json.dumps(td["crop_box"]).encode("utf-8"))
    buf.seek(0)
    _payload = buf.getvalue()
    _pack_ms = (_time.time() - _t1) * 1000
    print(f"[protect] GPU/보호 {_protect_ms:.0f}ms  포장(gzip) {_pack_ms:.0f}ms  "
          f"타일 {len(tiles)}개  응답 {len(_payload) // 1024}KB")
    return Response(content=_payload, media_type="application/gzip")


@app.post("/protect_batch")
async def protect_batch(file: UploadFile = File(...),
                        password: str = Form("forensic2026")):
    """
    여러 프레임을 한 번에 보호 (배치). 요청 tar에 프레임별 폴더(숫자)로
    frame.jpg + faces.json + meta.json 을 담아 보내면, 각 프레임을 GPU로
    보호해 같은 구조의 tar.gz로 되돌려준다. gzip 포장을 배치당 1회만 한다.
    """
    if _restore_active.is_set():
        return JSONResponse({"busy": "restore"}, status_code=503)

    import json

    import cv2
    import numpy as np
    import torch

    import config as cfg
    import parallel_protect as pp

    pp.load_anonymizer(getattr(cfg, "INN_CHECKPOINT", None), password)

    data = await file.read()
    workdir = tempfile.mkdtemp(prefix="pbatch_")
    try:
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            t.extractall(workdir)
    except Exception as e:
        return JSONResponse({"error": f"압축 해제 실패: {e}"}, status_code=400)

    dirs = sorted(d for d in os.listdir(workdir)
                  if d.isdigit() and os.path.isdir(os.path.join(workdir, d)))

    def _work():
        results = []
        with _gpu_lock:
            with torch.no_grad():
                for d in dirs:
                    fp = os.path.join(workdir, d)
                    img = cv2.imread(os.path.join(fp, "frame.jpg"))
                    if img is None:
                        continue
                    with open(os.path.join(fp, "faces.json"), encoding="utf-8") as f:
                        faces = json.load(f)
                    anon, tiles = pp.protect_one(img, faces, password)
                    results.append((d, anon, tiles))
        return results

    try:
        results = await asyncio.to_thread(_work)
    except Exception as e:
        return JSONResponse({"error": f"보호 실패: {e}"}, status_code=500)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for d, anon, tiles in results:
            _ok, enc = cv2.imencode(".jpg", anon)
            _tar_add(t, f"{d}/frame.jpg", enc.tobytes())
            for i, td in enumerate(tiles):
                nb = io.BytesIO()
                np.save(nb, td["tile_f32"])
                _tar_add(t, f"{d}/face_{i}.npy", nb.getvalue())
                _tar_add(t, f"{d}/face_{i}_box.json",
                         json.dumps(td["crop_box"]).encode("utf-8"))
    buf.seek(0)
    print(f"[protect_batch] {len(results)}프레임 처리 → {len(buf.getvalue())//1024}KB")
    return Response(content=buf.getvalue(), media_type="application/gzip")


if __name__ == "__main__":
    port = int(os.environ.get("PC_RESTORE_PORT", "8500"))
    print(f"[PC복원서버] http://0.0.0.0:{port}  (GET /health 로 GPU 확인)")
    uvicorn.run(app, host="0.0.0.0", port=port)
