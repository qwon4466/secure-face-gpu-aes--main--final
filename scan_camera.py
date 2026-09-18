"""기존 진단 명령: 서버 종료 상태에서 공통 V4L2 탐색을 확인합니다."""
from core.camera_manager import CameraManager
if __name__ == '__main__':
    camera = CameraManager()
    try:
        camera.open()
        print(camera.selected)
    finally:
        camera.release()
