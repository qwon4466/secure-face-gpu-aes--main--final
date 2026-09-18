# 선택적 복원 사용 방법

## 환경과 실행

프로젝트 루트의 기존 `hk_env`만 사용합니다. 현재 환경에는 필요한 서버·CPU 인식·암호화 패키지를 설치했습니다. 다른 환경에서 재설치할 때는 활성화한 hk_env에서 `python -m pip install -r requirements.txt`를 실행합니다. 이번 환경에서 설치된 정확한 버전은 `requirements-lock.txt`에 기록합니다.

```bash
source hk_env/bin/activate
python main.py
```

접속 주소는 **http://127.0.0.1:8000**입니다. 선택 복원용 다른 서버를 띄우지 않습니다.

## 기존 화면에서 사용하는 순서

- **실시간 모니터링**: 보호 영상과 등록 판정 표시. 허가자는 녹색, 나머지는 비허가자 적색으로 표시합니다. 보호 영상에도 같은 블러와 라벨을 저장합니다.
- **인원 등록**: 기존 관리자 로그인과 등록 페이지를 유지합니다. 기존 `security_system.db`에 등록하며 등록 후 갤러리를 갱신합니다.
- **보호 자산**: 과거 저장 형식과 새 선택 복원 형식을 함께 나열합니다. 새 청크를 선택하면 선택 복원 화면으로 연결합니다.
- **과거 영상 복원**: 기존 AES `.enc` 영상을 기존 비밀번호로 복원합니다. 관리자 로그인이 필요합니다. 과거 데이터에 집단별 픽셀이 저장되어 있지 않으면 선택 복원 대상으로 표시하지 않습니다.
- **선택적 복원 (관리자 로그인 후 관리자 메뉴)**: 완료된 청크 선택 → 키 파일 선택 → 선택 복원 → 진행/결과 확인 → 프레임 재생·AVI 다운로드.

결과에는 대상 청크, 시작/완료 시각, 복원된 집단, 프레임 수, FPS가 표시됩니다. 잘못된 키/변조 오류 시 성공 결과를 제공하지 않습니다. 출력 폴더는 작업 ID마다 달라 덮어쓰지 않습니다. 화면 재생은 저장된 PNG 프레임을 사용하며 AVI는 FFV1 무손실 파일입니다.

## 모델과 카메라

카메라 선택은 공통 `core/camera_manager.py`가 담당하며 `PROCESS_WIDTH`는 처리 크기를 조정합니다. 실행 방법과 탐색 규칙은 [카메라 호환성](camera-compatibility.md)을 참고하세요. 기본 CPU 모델 위치는 `~/.insightface/models/buffalo_s`입니다. 모델이 다른 경로에 있으면 다음 항목에 **실제 SCRFD·ArcFace ONNX 파일**을 지정합니다:

```python
SELECTIVE_DETECTOR_PATH = "/실제/경로/scrfd.onnx"
SELECTIVE_RECOGNIZER_PATH = "/실제/경로/arcface.onnx"
SELECTIVE_SOURCE = ""  # 카메라 사용; 파일 입력이면 로컬 영상 경로
SELECTIVE_FPS = 10
CHUNK_SECONDS = 60  # 운영 카메라 청크 길이
SELECTIVE_FPS = 10   # 목표 저장 FPS; 처리 부족분은 드롭으로 기록
```

기존 INN `.pth` 체크포인트는 이 두 모델을 대체하지 않습니다. 모델을 몰래 다운로드하거나 GPU/Hailo 경로를 추가하지 않습니다. 모델과 갤러리의 임베딩 공간이 일치해야 합니다. 갤러리가 없거나 인식 품질이 부족하면 전체키 전용으로 처리합니다.

기존 홈 디렉터리 캐시에서 det_500m.onnx와 w600k_mbf.onnx를 확인했습니다. 실제 SCRFD CPU 파일 입력 처리와 ArcFace 512차원 임베딩 생성을 검증했습니다. Ubuntu Logitech 실물 영상의 얼굴 탐지·블러·관제 화면까지 검증했습니다. 실물 등록과 신원 인식 정확도 평가는 별도이며, Raspberry Pi 실장비는 검증하지 않았습니다.

장비 없이 화면을 확인할 때만 `SRX_TEST_MODE=synthetic python main.py`를 사용합니다. 명시적 테스트 모드에서 24개 합성 프레임을 만들며, 운영 데이터와 다른 `main-server-test` 폴더를 사용합니다. 이 옵션 없이 합성 입력으로 자동 대체하지 않습니다.

## 데이터와 키

운영 데이터 기본 위치:

```text
raw_data/
  .keys/                         # external.key / internal.key / master.key (0600)
  YYYY-MM-DD/HH/HH-MM-SS/        # protected.avi, 세 집단 gcm, manifest.json
  .results/<작업ID>/             # restored.avi, 프레임 PNG, result.json
```

운영 청크는 `CHUNK_SECONDS`의 monotonic 경과 시간으로 닫습니다. 저장 중에는 `.HH-MM-SS.partial`이고 검증·fsync 후 최종 폴더로 rename합니다. 서버 종료 시 짧은 마지막 버퍼는 완료 청크로 게시하지 않습니다. 기존 `data`는 보존하지만 신규 녹화·목록·복원 입력에 쓰지 않습니다.

폴더는 0700, 키 파일은 0600입니다. 처음에만 암호학적 난수 키를 생성하며 기존 키를 덮어쓰지 않습니다. 키를 영상 청크에 넣거나 로그·URL·Git에 기록하지 않습니다. 키 폴더는 영상과 별도로 안전하게 백업하세요. 집단키 손실 시 전체키가 있으면 해당 집단을 복원할 수 있지만 전체키 전용 데이터는 전체키를 잃으면 복원할 수 없습니다.

새 녹화 경로는 같은 입력 프레임에서 전원 보호 영상과 집단별 원본 픽셀 암호문을 만들고, 전체 프레임 AES 사본을 추가로 저장하지 않습니다. 과거 복원 API는 기존 raw_data 범위만 받으며 새 포맷을 거부합니다. 과거 복원 결과도 작업별 새 파일명으로 저장합니다.

## 권한·판정

키 없음: 보호본만 저장. 외부인키/내부인키: 해당 집단의 허용 마스크만 복원. 전체키: 모든 저장 마스크 복원. 이름을 master.key로 바꿔도 실제 AES Key Wrap을 풀 수 있는 집단만 허용합니다.

내부인은 신뢰도 0.8 이상, 유사도 0.55 이상, 동일 후보 3연속. 외부인은 유효한 갤러리와 유사도 0.25 이하가 5연속일 때 확정합니다. IoU 0.35 이상의 유일한 추적 연결만 이어갑니다. 인식 흔들림·추적 변경·중첩은 보수적으로 전체키 전용 처리합니다. 탐지 누락이나 지속 오분류는 여전히 노출/권한 오분배 위험이 있으므로 실물 검증이 필요합니다.

## 제한·종료

운영 카메라는 최대 1024프레임/암호화 원본 1GiB, 파일·합성 회귀 경로는 16프레임/64MiB 상한입니다. 대기 큐는 0개입니다. 불완전 `.partial` 청크는 복원 목록에서 제외하며 자동 삭제/이어쓰기를 하지 않습니다. 재시도는 새 ID·DEK를 사용합니다. 웹 동시 복원 1개, 서버 수명 동안 최대 작업 24개/세션 64개입니다. 세션은 1시간이며 세션별 결과 접근을 검사합니다. OS 사용자 자체의 파일 접근까지 격리하는 외부 공개형 다중 사용자 서버는 아닙니다.

출력은 고정 FPS입니다. 실제 가변 FPS의 재생 시간 보존과 카메라 드라이버 드롭 측정은 보장하지 않습니다. 전체 복원의 픽셀 기준은 처리 해상도로 들어온 BGR 프레임이며 센서 원시 데이터가 아닙니다.

Ctrl+C로 서버를 종료합니다. 완료된 청크/결과/키는 남습니다. 다시 같은 `python main.py`로 시작하면 됩니다. 사용자 촬영 데이터와 모델은 이번 정리에서 삭제하지 않았습니다.

## 테스트

```bash
source hk_env/bin/activate
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests -q -o cache_dir=tests/runtime/pytest-cache
PLAYWRIGHT_BROWSERS_PATH=tests/runtime/browsers python -m playwright install chromium --only-shell
PYTHONDONTWRITEBYTECODE=1 python tests/verify_main.py
```

마지막 스크립트는 기존 8000번이 비어 있는지 확인한 뒤 실제 main.py를 실행하고 테스트 서버만 종료합니다. 테스트 원본·키·보호본·결과는 `tests/runtime/`에만 생성합니다. 최신 결과 경로는 `tests/runtime/latest-main.txt`입니다.

선택적 복원 페이지와 모든 API·결과 다운로드는 기존 관리자 세션이 필요합니다. 비로그인 요청은 401입니다. 관제 화면 상단에는 선택적 복원 링크가 없습니다. 현재 프레임의 허가자/비허가자 수를 표시하며 동일 얼굴을 프레임마다 누적하지 않습니다. [화면·블러 변경 및 실물 검증](admin-display-verification.md)을 참고하세요.
