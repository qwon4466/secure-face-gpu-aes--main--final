# main.py 직접 통합 최종 검증 보고서

## 최종 구조

사용자 요청에 따라 기존 main.py와 기존 웹페이지에 직접 통합했다. 진입점은 프로젝트 루트의 `source hk_env/bin/activate` → `python main.py`, 주소는 **http://127.0.0.1:8000**이다. `core/selective_routes.py`는 APIRouter이며 별도 FastAPI 앱/서버가 아니다. 복원 worker는 제공된 키만 받는 로컬 자식 프로세스이며 서버/포트를 열지 않는다.

이전 extensions/selective_restore는 최초 파일 목록에 없고 이 대화의 이전 작업에서 생성한 경로임을 확인했다. 그 안의 소스·독립 .venv·runtime 합성 산출물·run.py·독립 index.html·별도 requirements·문서를 이관/검증 후 모두 삭제했다. 삭제 전 목록은 selective-migration-inventory.json에 기록했다. hk_env와 사용자의 checkpoints는 보존했다. 기존 원래 파일 삭제는 **0건**이다.

## 수정한 기존 파일

- config.py
- requirements.txt
- main.py
- .gitignore
- README.md
- templates/monitor.html
- templates/index.html
- templates/decryption.html
- templates/assets.html

main.py는 기존 관제·등록·과거 AES API를 유지하며 선택 라우터를 포함한다. config.py는 선택 녹화 설정을 제공한다. requirements는 hk_env에 설치한 CPU/웹/암호화 의존성을 반영한다. templates는 기존 탐색 메뉴를 유지하고 선택적 복원을 추가한다. .gitignore는 새 운영 데이터/키/테스트 산출물을 제외한다.

Git 디렉터리는 비어 있어 `git status --porcelain=v1`은 시작·종료 모두 저장소 아님 오류였다. 원래 90개 파일의 해시와 비교했고, 사용자 요청에 따라 위 파일을 수정했다. 자동 커밋·푸시·Git reset/clean은 하지 않았다.

## 새 파일과 연결

| 파일 | 역할 |
|---|---|
| core/selective_crypto.py | AES-256-GCM, AES Key Wrap, 배타 마스크, 보호/복원 영상·원자적 청크 저장 |
| core/selective_recognition.py | 기존 DetectionResult/코사인 비교 재사용, 실제 SCRFD/ArcFace CPU 어댑터, 연속 판정 |
| core/selective_recording.py | 최대 16프레임/64MiB 버퍼·청크 회전·시각/드롭 통계 |
| core/selective_camera.py | 기존 main.py 카메라 인터페이스, 원본 1회 취득→보호 화면+집단 암호문, 등록 갤러리 갱신, O/X/? 표시 |
| core/selective_routes.py | main.app에 등록되는 일반 라우터, 키 제출·작업 상태·재생·다운로드·세션 소유권 |
| core/selective_worker.py | stdin의 단일 키로만 검증·복원; 다른 키/녹화 DEK 공유 없음 |
| templates/selective_restore.html | 기존 웹 메뉴에서 여는 한국어 복원 화면 |
| tests/test_*.py | 정책/변조/직접 DEK/경계/저장 중단/재시도/갤러리/카메라 인터페이스 검사 |
| tests/verify_main.py | 실제 main.py, 8000번, HTTP 및 Chromium 클릭·재생·다운로드 검증 |
| tests/verify_cpu_file.py | 캐시된 실제 CPU 모델과 파일 입력 경로 검증 |
| tests/verify_default.py | 테스트 모드 없는 기존 실행 방법의 시작·상태 확인 |
| requirements-dev.txt / requirements-lock.txt | 테스트 의존성과 hk_env 실제 설치 버전 |
| docs/selective-*.md/json | 사용법·암호화 포맷·검증·이관/파일 보존 기록 |

호출 연결: main.lifespan → MainCamera.start/_loop → CPUAdapter.observe → Classifier.classify → Recorder.add/save_chunk. 같은 처리 입력 프레임의 원본 픽셀과 보호본을 사용하며 RawRecorder를 함께 생성하지 않는다. 등록 페이지는 기존 DB/API를 사용하고 등록 후 reload_gallery를 호출한다.

## API와 웹 화면

- 기존 `/`, `/register`, `/employees`, `/assets`, `/snapshot/cam_0`, `/stream/cam_0` 유지.
- 메뉴 **선택적 복원** → `/selective/`.
- `/selective/api/chunks`, `/selective/api/status`, `/selective/api/preview`.
- `POST /selective/api/restore/{청크ID}`: 실제 32바이트 키만 제출. 역할 문자열 없음.
- `/selective/api/jobs/{작업ID}`, `/frame/{번호}`, `/download`.
- 대상 청크, 시작/완료 시각, 성공/실패, 허용 집단, 프레임 수/FPS 표시.
- `/decryption_page`, `/api/admin/chunks`, `/api/admin/verify_chunk`, `/api/admin/restore/...`: 과거 AES 데이터용으로 유지. 관리자 인증과 기존 데이터 루트 제한, 새 manifest 형식 거부.

## 수행 결과

자동 테스트: **42 passed in 13.20s**. 최종 모델 선택 수정 후 관련 3개 카메라/갤러리 회귀 테스트도 다시 통과했다.

| 검사 | 결과 |
|---|---|
| 키 없음 | 모든 픽셀 보호본 유지 |
| 외부인키 | 외부인 마스크만 원본, 다른 영역 보호본 일치 |
| 내부인키 | 내부인 마스크만 원본, 다른 영역 보호본 일치 |
| 전체키 | 저장 입력 BGR 프레임과 모든 픽셀 일치 |
| 잘못된 키 | 실패·성공 출력 없음 |
| 직접 키 포장 해제 | 외부인↔내부인 및 전체키 전용 DEK 접근 실패 |
| 파일명 master.key 변경 | 외부인키 권한 확대 없음 |
| 암호문/tag/nonce/AAD/영상/청크 교체 | 거부, 부분 성공 결과 노출 없음 |
| 직접 복호화 payload 검사 | 정확한 비허가 bbox 밖 원본 픽셀 제외 |
| 중첩/미확정/추적 변경/빈 프레임/누락 | 정책 및 원본/보호본 픽셀 비교 통과 |
| 중단/파일 누락/I/O 실패/재시도/버퍼 상한 | 불완전 청크 제외, 새 DEK 재시도, 상한 준수 |
| 실제 main.py/8000 | 정상 시작·관제/등록/직원/자산/과거 복원 페이지 표시 |
| 기존 정적 파일 | /recordings 테스트 파일 내용 확인 |
| 기존 과거 AES 복원 | 합성 과거 형식 .enc 입력→MP4 생성→4프레임/320×240 재생 디코딩 성공 |
| 과거 API에 새 청크 전달 | 거부 |
| 브라우저 실제 폼 | 기존 메뉴 클릭→키 파일 선택→복원 완료→PNG 프레임 재생→AVI 다운로드 성공 |
| 보호 미리보기 | 브라우저에서 실제 이미지 320픽셀 폭 디코딩 확인 |
| 세션/입력 검증 | 다른 세션 결과 거부, CSRF/키 길이/프레임 범위/청크 ID 검사 |
| 서버 종료 | 자신이 시작한 테스트 서버만 정상 종료 |

브라우저 테스트 최초 시도에서는 HTML option의 화면 표시를 기다리는 테스트 코드 때문에 타임아웃이 발생했다. option은 DOM 존재 상태로 기다리도록 수정했고 재실행을 통과했다. 실제 모델 검색에서 2d106det(랜드마크)를 선택할 수 있는 조건을 수정하여 det_500m(SCRFD)을 선택하도록 했고 모델 종류 API도 검사한다.

## 실제 CPU 모델 / 기본 실행

hk_env에 insightface 0.7.3, onnxruntime 1.30.0, OpenCV 5.0.0, cryptography 50.0.1 등이 설치되어 있다. `pip check`는 No broken requirements found였다.

프로젝트 밖 기존 `~/.insightface/models/buffalo_s`의 det_500m.onnx / w600k_mbf.onnx를 읽기만 했다. 다운로드·교체하지 않았다. 실제 SCRFD CPU로 빈 합성 파일 4프레임을 처리·저장했고, 실제 ArcFace get_feat에서 **[1, 512]**의 유한한 임베딩을 얻었다. 이는 모델 연결/호환성 검사이며 실물 얼굴 인식 정확도 증명이 아니다.

초기 검증 당시에는 카메라가 연결되지 않았다. 이후 Logitech 연결 상태에서 기존 main.py의 실제 영상·얼굴 탐지·블러·자원 해제를 검증했다. 최신 결과는 [카메라 호환성 검증](camera-compatibility.md)을 참조한다. 실물 등록 성공·실물 O 판정·인식 정확도 평가는 이 카메라 호환성 검증 범위에 포함하지 않는다.

## 측정

브라우저/HTTP 통합 실험: 320×240 합성 24프레임, 출력 10 FPS, 8프레임×3청크. 처리 9.55 FPS, peak RSS 129.15 MiB, 청크 저장 0.138s, 0.127s, 0.127s, 프로그램 드롭 0. 하드웨어 드롭 unknown. 실시간/Raspberry Pi 성능 보장이 아니다.

## 생성 파일

이 절은 최초 통합 당시 결과다. 최신 운영 저장 위치는 `raw_data/.keys`, 날짜/시간 청크, `.results`이며 실제 60초 검증은 [최신 raw_data 검증](raw-data-selective-ui.md)에 기록했다. 기존 `data`는 삭제·이동하지 않았다.

최종 브라우저·정책 검증 결과: **tests/runtime/main-8b6ed2f60a60455eba3a8743de46ebcd**

- `chunks/<ID>/protected.avi`, 집단별 `*.gcm`, `manifest.json`
- `results/<작업ID>/restored.avi`, PNG 및 result.json
- `browser.png`: 실제 기존 메뉴에서 진입해 복원한 화면
- `browser-restored.avi`: 실제 브라우저 다운로드
- `historical-restored.mp4`: 과거 AES 경로 회귀 결과
- `report.json`, `main.log`: 실제 명령·API 결과·성능·서버 종료 기록

추가 실제 CPU 파일 결과: `tests/runtime/cpu-file/report.json`. 기본 실행 결과: `tests/runtime/default-start/report.json`. 자동 테스트 결과: `tests/runtime/pytest-final.txt`.

## 실제 실행 명령

```bash
hk_env/bin/python -m pip install --no-cache-dir numpy==2.5.3 opencv-python-headless==5.0.0.93 cryptography==50.0.1 fastapi==0.141.1 uvicorn==0.53.0 jinja2==3.1.6 python-multipart==0.0.32 httpx==0.28.1 pytest==9.1.1 pillow==12.3.0 pycryptodome
hk_env/bin/python -m pip install --no-cache-dir insightface==0.7.3 'onnxruntime>=1.18,<2'
hk_env/bin/python -m pip install --no-cache-dir playwright
PLAYWRIGHT_BROWSERS_PATH=tests/runtime/browsers hk_env/bin/python -m playwright install chromium --only-shell
PYTHONDONTWRITEBYTECODE=1 hk_env/bin/python -m pytest tests -q -o cache_dir=tests/runtime/pytest-cache
PYTHONDONTWRITEBYTECODE=1 hk_env/bin/python tests/verify_main.py
MPLCONFIGDIR=tests/runtime/mpl PYTHONDONTWRITEBYTECODE=1 hk_env/bin/python tests/verify_cpu_file.py
PYTHONDONTWRITEBYTECODE=1 hk_env/bin/python tests/verify_default.py
hk_env/bin/python -m pip check
```

운영 실행은 **source hk_env/bin/activate → python main.py** 한 경로다. 종료는 Ctrl+C. 모델/장치 설정과 한계는 selective-restore.md에 정리했다. 가변 FPS 실제 PTS 보존, 카메라 드롭 측정, 실물 얼굴 정확도는 추가 확인이 필요하다. 발견한 남은 실행 오류는 카메라 장치 부재이며, 구현 테스트 실패는 남아 있지 않다.
