"""
Modal 서버리스 GPU 복원 앱.

라즈베리파이가 저장한 PSF 보호본 청크(tar.gz)를 POST로 받아,
GPU에서 INN 역변환으로 복원해 mp4를 반환한다.

배포:
  pip install modal
  modal setup                 # (최초 1회) 토큰 인증
  modal deploy modal_app.py   # 배포 → 웹 URL 출력

개발 테스트:
  modal serve modal_app.py    # 임시 URL로 실행

배포하면 출력되는 URL(예: https://<계정>--securefacerx-restore-restore.modal.run)을
라즈베리파이 config 의 MODAL_RESTORE_URL 에 넣는다.
"""
import modal
from fastapi import File, UploadFile

app = modal.App("securefacerx-restore")

# 코드 전체 + 체크포인트(.pth)를 이미지에 포함. 대용량/불필요 폴더는 제외.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch", "torchvision", "opencv-python-headless", "numpy",
        "pycryptodome", "fastapi[standard]", "python-multipart",
    )
    .add_local_dir(
        ".", remote_path="/app",
        ignore=[
            "recordings/**", "*.mp4", ".git/**", "**/__pycache__/**",
            "*.tar.gz", "registered_faces/**", "*.db", "debug_*",
        ],
    )
)


@app.function(gpu="a10g", image=image, timeout=1800)
@modal.fastapi_endpoint(method="POST")
async def restore(file: UploadFile = File(...)):
    """
    file: 청크 폴더를 압축한 tar.gz (multipart 업로드)
    반환: 복원된 mp4 (video/mp4)
    """
    import io
    import os
    import sys
    import tarfile
    import tempfile

    from fastapi.responses import JSONResponse, Response

    sys.path.insert(0, "/app")
    os.chdir("/app")

    data = await file.read()
    workdir = tempfile.mkdtemp()
    try:
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            t.extractall(workdir)
    except Exception as e:
        return JSONResponse({"error": f"압축 해제 실패: {e}"}, status_code=400)

    from restore_chunk import restore_chunk
    out = os.path.join(workdir, "restored.mp4")
    result = restore_chunk(workdir, password="forensic2026", out_path=out)
    if result is None or not os.path.exists(result):
        return JSONResponse({"error": "복원할 청크를 찾지 못했거나 실패"}, status_code=500)

    with open(result, "rb") as f:
        mp4 = f.read()
    return Response(content=mp4, media_type="video/mp4")
