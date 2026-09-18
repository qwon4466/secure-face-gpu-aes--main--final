# 카메라 호환성 및 실제 검증

## 실행

프로젝트 디렉터리에서 기존 환경을 활성화한 후 기존 서버 하나만 실행합니다.

```bash
source hk_env/bin/activate
python main.py
```

브라우저: http://127.0.0.1:8000/

수동 지정 예시:

```bash
# Ubuntu Logitech
CAMERA_DEVICE=/dev/video0 python main.py

# Raspberry Pi RealSense D457 — 실제 RGB 노드가 이 경로일 때
CAMERA_DEVICE=/dev/video4 python main.py
```

번호는 예시입니다. 기본 장치 번호는 없습니다. 안정적인 `/dev/v4l/by-id/...` 경로도 환경변수로 지정할 수 있습니다. 진단 스크립트는 서버와 동시에 실행하지 않습니다.

## 선택과 재연결

- `CAMERA_DEVICE`를 먼저 확인합니다. 실패하면 경로·실패 원인과 **자동 탐색으로 전환** 메시지를 출력하고 다른 후보를 검색합니다.
- 자동 탐색은 `/dev/v4l/by-id/` 별칭을 우선 사용하고, 같은 실제 노드는 중복 검사하지 않습니다. 나머지 `/dev/video*`는 숫자순으로 확인합니다.
- RealSense RGB/Color 이름을 우선하고, RGB 표시가 없는 RealSense도 실제 프레임을 확인한 뒤 다른 제조사보다 먼저 선택합니다. 제품명에 `Depth Camera`가 들어간다는 이유만으로 컬러 노드를 제외하지 않습니다.
- 후보는 OpenCV `CAP_V4L2`로 열고 8회 연속 프레임을 읽습니다. 열기 실패·0 해상도·빈 프레임·비컬러/16비트 깊이·단색 영상·연속 읽기 실패는 제외하고 이유를 출력합니다. 어두운 화면이나 회색 장면은 컬러 확인에 실패할 수 있으므로 조명과 장면을 확인하세요.
- Logitech에는 MJPG·640×480·30 FPS를 요청합니다. 설정 또는 수신 검증이 실패하면 먼저 해제한 뒤 장치 기본 형식·해상도로 다시 엽니다. 다른 제조사에는 이 해상도를 강제하지 않습니다. 드라이버가 지원하는 값으로 조정하면 실제 수신 해상도와 FPS를 기록합니다.
- 선택 결과는 `config.CAMERA_SELECTED_DEVICE`, `core.camera_manager.SELECTED`, `/api/stats`의 `camera`에 표시됩니다. 로그에는 경로·제품명·해상도·FPS·FOURCC가 나옵니다. 장치 FPS와 AI 처리 FPS는 다릅니다.
- 읽기 실패 시 기존 캡처를 해제하고 같은 경로를 먼저 다시 엽니다. 실패하면 전체 재검색합니다. 전체 검색 실패는 3초 간격으로 재시도합니다. 재연결 사이에는 이전 미리보기와 원본 버퍼를 비우고 분류 추적을 초기화합니다.
- 프로세스 내부 잠금과 같은 사용자 프로세스 사이의 파일 잠금으로 공통 관리자의 동시 점유를 막습니다. 후보를 바꿀 때 기존 `VideoCapture`를 먼저 해제합니다. 다른 외부 카메라 앱의 점유는 V4L2 열기/읽기 오류로 드러납니다.
- 한 번 읽은 프레임에서 SCRFD·ArcFace·얼굴 블러·관제 JPEG·선택 복원 녹화를 처리합니다. 파일 재생/복원용 `VideoCapture`는 카메라를 열지 않습니다.

## Ubuntu 실제 결과 (2026-09-16)

연결 장치: Logitech `UVC Camera (046d:081b)`. RealSense는 연결되지 않았습니다.

| 검증 | 결과 |
|---|---|
| 자동 선택 | `/dev/v4l/by-id/usb-046d_081b_00315070-video-index0` 선택 |
| 수동 `/dev/video0` | 지정 경로 선택, MJPG·640×480·30 FPS |
| 메타데이터 `/dev/video1` | V4L2 열기 실패 원인 출력 후 컬러 노드 자동 선택 (기존 main.py에서 실제 검증) |
| 존재하지 않는 경로 | 명확한 오류와 자동 전환 로그 후 Logitech 선택 |
| 기존 main.py | 포트 8000 정상 시작 |
| 기존 관제 화면 | Chromium에서 실제 `#streamImg` 640×480 디코딩 확인; 자동/수동 모두 성공 |
| 실제 얼굴 탐지·블러 | 자동 13프레임, 수동 18프레임의 블러 픽셀 일치 확인 |
| 녹화·복원 | 실제 촬영 청크를 전체키로 메모리 복원하고, 다시 블러한 픽셀이 저장된 보호 영상과 동일함을 확인 |
| 종료 자원 해제 | `Application shutdown complete` 확인 후 같은 카메라 재개방 성공 |
| 기본 모드 폴백·재연결 | 모의 장치 단위 테스트; 물리적 USB 뽑기 시험은 미실행 |
| Raspberry Pi D457 | 환경변수 지정과 자동 탐색 구조 구현. **실장비 검증 미실행** |

실제 처리 속도는 이 짧은 VMware/CPU 시험에서 약 3 FPS였으며, 장치가 보고한 30 FPS가 전체 AI 처리 속도를 뜻하지 않습니다. MJPG 입력에서 `Corrupt JPEG data: ... extraneous bytes ...` 경고가 일부 발생했습니다. 프레임 수신·브라우저 표시·암호화 녹화·복원 검사는 성공했으나 경고 없는 입력이라고 주장하지 않습니다.

갤러리가 없는 얼굴은 전체키 전용으로 보존됩니다. 이번 시험은 개인 신원 인식 정확도나 직원 등록 성공률을 평가하지 않습니다.

## 변경 파일과 재현

- `core/camera_manager.py`: 공통 탐색·모드 요청/폴백·컬러 검증·점유 제어·재연결.
- `main.py`, `core/selective_camera.py`, `config.py`: 기존 서버 연결, 상태 공개, 재연결 시 분류 초기화.
- `camera_stream.py`, `core/selective_recording.py`: 기존 카메라 입력도 공통 관리자 사용.
- `core/selective_recognition.py`: 순수 cosine 함수의 검증 범위를 함수 AST로 분리하여 카메라 초기화 변경과 독립적으로 유지.
- `diag_camera.py`, `scan_camera.py`, `detect_realsys.py`, `test_hailo.py`, `test_camera_basic.py`: 정수 카메라 열기를 제거하고 공통 관리자 사용.
- `tests/test_camera_manager.py`, `tests/verify_camera.py`: 모의 장치 테스트 및 기존 main.py 실장비 검증 자동화. 별도 서버를 구현하지 않습니다.
- `README.md`, `docs/selective-restore.md`, `docs/selective-verification.md`, 이 문서: 최신 실행법과 검증 한계 기록.

```bash
hk_env/bin/python -m pytest tests -q
hk_env/bin/python tests/verify_camera.py
```

실장비 검증 로그와 JSON 결과: `tests/runtime/camera-verification/report.json` 및 각 실행 폴더의 `server.log`. 복원 원본은 검증 중 메모리에서만 비교합니다.

최종 카메라 단위 테스트 13개와 기존 회귀 테스트 42개가 통과했습니다(총 55개). 소스 전체 검색 결과 운영 코드에 `CAMERA_INDEX`, 고정 `/dev/video0`·`/dev/video4`, 정수 인자의 `VideoCapture`는 남아 있지 않습니다. 해당 경로는 문서 예시와 검증 입력에서만 사용합니다. 남은 직접 `VideoCapture`는 공통 V4L2 관리자 또는 로컬 영상 파일 디코더입니다. 메타데이터 노드 실험 결과는 `tests/runtime/camera-verification/report-metadata.json`입니다.
