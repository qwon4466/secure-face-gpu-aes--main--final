# 관리자 전용 선택적 복원·강화 블러·한글 라벨

## 실행 및 메뉴

```bash
source hk_env/bin/activate
CAMERA_DEVICE=/dev/video0 python main.py
```

접속: http://127.0.0.1:8000/

공용 관제/보호 자산 상단의 선택적 복원 링크를 제거했다. 관리자 로그인 → 사원 관리·등록·감사 로그·원본 복원 화면의 **관리자 메뉴 → 선택적 복원**으로 이동한다. 원본 복원은 유지했다. 선택적 복원 페이지에도 같은 관리자 이동 메뉴와 로그아웃이 있다.

기존 `admin_session`과 `require_admin`을 라우터 전체의 의존성으로 재사용한다. `/selective/`와 모든 하위 API, 프레임, 다운로드 요청은 미로그인 시 401이다. 기존 CSRF와 작업별 소유 세션 검사도 유지한다. `/api/assets`, 자산 상세, `/api/debug` 역시 인증하고 공용 통계에서 청크 ID·세션 ID를 제외했다. 로그아웃 이후 남아 있는 복원용 쿠키만으로는 접근하지 못한다.

## 블러와 처리 순서

변경 전 실제 선택 복원 파이프라인은 고정 **31×31, sigma 12, bbox 확장 15px**였다. 기존 config에 더 강한 61×61 설정은 있었지만 이 경로에서는 사용하지 않았다. Git 기록은 사용 가능한 저장소가 아니어서 소스 설정을 비교했다.

현재 기본값:

- 커널: `max(61, 얼굴 긴 변 × 0.9)`를 홀수 보정.
- 최신 변경: bbox 확장 0. 탐지 bbox, 블러 ROI, 표시 테두리, 암호화 마스크가 일치합니다.
- 계산: ROI 긴 변을 최대 32px로 축소 → 비례 축소한 홀수 Gaussian 커널(최소 3), sigma=원래 커널×축소율/3 → 확대.
- 조정 위치: `config.py`의 `SELECTIVE_BLUR_*` 네 항목. 큰 얼굴에 거대한 고정 커널을 직접 적용하지 않는다.
- 모든 집단에 같은 블러를 적용한다. 라벨은 블러 이후에 그리며, 저장되는 보호본과 관제본이 같은 함수를 사용한다.

원본 탐지/인식 → 원본을 임시 AES-GCM 메모리 버퍼에 암호화 → 강한 블러 → 한글 라벨·테두리 → 관제/보호본 저장 순서다. 청크 확정 시 임시 버퍼를 메모리에서 복호화하고 기존 집단별 원본 픽셀 암호문으로 저장한다. 기존 v1 형식과 영상 해시 AAD를 유지하므로 최종 디스크 암호문 작성은 보호본 해시 확정 후 이루어진다. 원본 payload에 블러나 라벨 픽셀을 넣지 않는다. 라벨 쓰기도 bbox 복원 마스크 안으로 제한해 전체키 복원 시 원래 픽셀로 되돌린다.

## 판정·글꼴·통계

- 허가자: 기존 등록 DB ArcFace 비교에서 **MATCH_THRESHOLD=0.45 이상**, 같은 추적 ID·같은 등록 후보가 3회 연속 일치. 녹색 BGR `(0,255,0)` 테두리·배경, 검정 글자.
- 비허가자: 갤러리 없음, 불일치, 임베딩 실패, 인식 대기·후보 변경 등. 적색 BGR `(0,0,255)` 테두리·배경, 흰 글자.
- 저장 분류는 기존 0.55/0.25, 탐지 품질 0.8, 연속 확인 정책을 유지한다. 화면의 비허가자를 외부인키 집단으로 바꾸지 않는다. 화면의 허가자도 저장 기준이 부족하면 전체키 전용이다.
- 실제 사용 글꼴: `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`. 환경변수 `CAMERA_LABEL_FONT`, Noto CJK, Nanum 후보를 순서대로 탐색한다. 모두 없으면 터미널 오류와 카메라 상태 오류를 표시하며 물음표로 대체하지 않는다.
- 통계 두 카드는 현재 프레임의 동일한 `authorized` 판정으로 계산한다. 누적 감사 이벤트 수를 현재 외부인 수로 표시하던 불일치를 제거했다. 감사 로그의 과거 기록은 유지한다.

## 실제 검증 (Ubuntu Logitech, 2026-09-16)

`tests/verify_admin_display.py`가 기존 `hk_env/bin/python main.py`, 포트 8000, 실제 `/dev/video0`으로 실행했다. `SRX_TEST_MODE=camera`는 시험 등록 DB/이미지/녹화 경로만 분리한다. 합성 입력으로 바꾸지 않는다. 운영 `registered_faces`와 등록 DB는 변경하지 않았다.

| 항목 | 결과 |
|---|---|
| 미로그인 공용 메뉴 | 선택 복원 링크 없음 |
| 미로그인 페이지/API/청크/다운로드/POST | 401, 복원 작업 실행 안 됨 |
| 로그인 후 관리자 메뉴 | 관리·등록·감사·원본복원 모두 선택 복원 링크 있음 |
| 관리자 메뉴 클릭 | 실제 선택 복원 페이지와 청크 선택 UI 열림 |
| 로그아웃 후 요청 | 401 |
| 미등록 실제 얼굴 | 등록 전 빨간색 `비허가자`, 저장은 master |
| 실제 등록 | 기존 `/api/register_capture`로 좌측면 등록 성공 |
| 등록된 실제 얼굴 | 유사도 0.4711, 연속 일치 후 녹색 `허가자`, 사원 1/외부인 0 |
| 보수적 저장 분류 | 위 얼굴은 탐지 품질 0.6603으로 저장은 master 유지 |
| 강한 블러·한글 | 두 상태 스크린샷을 직접 확인; 눈·코·입의 디테일이 보이지 않음 |
| 실제 저장본 | 32프레임을 전체키로 메모리 복원한 후 블러·라벨 재적용, 저장 보호본과 픽셀 일치 |
| 키별 회귀 | 외부인키·내부인키·전체키 허용 영역 복원, 잘못된 키 실패, 변조 거부 |
| 기존 기능 | 관리자 페이지, 원본 AES 복원, 브라우저 선택 복원·재생·다운로드 통과 |
| 종료 | 정상 shutdown 로그, 카메라 재개방 성공 |

첫 실제 등록 시험에서는 표시 판정에 저장용 탐지 품질 0.8까지 적용해 허가 표시가 되지 않았다. 이를 분리했고, 최종 표시 판정은 기존 ArcFace 임계값 0.45와 동일 후보 연속성만 사용한다. 저장용 기준은 완화하지 않았다.

처리 FPS: 변경 전 짧은 관제 시험 **4.39 FPS**, 변경 후 실제 등록 시험 시점 **4.60 FPS**. 다른 순간·얼굴 자세의 측정이므로 성능 개선율로 해석하지 않는다. 장치 입력은 MJPG 640×480, 30 FPS이며 AI 처리 속도와 다르다. Raspberry Pi 성능은 미측정이다.

## 결과 파일

- [변경 전 관제](../tests/runtime/ui-review/before.png)
- [미등록·비허가자](../tests/runtime/ui-review/after-unregistered.png)
- [등록·허가자](../tests/runtime/ui-review/after-registered.png)
- [관리자 메뉴](../tests/runtime/ui-review/admin-menu.png)
- [실장비 검증 JSON](../tests/runtime/ui-review/report.json)
- 선택 복원 회귀 결과: `tests/runtime/main-1729623a872146e4a01d4dd4606ce748/report.json`

실행한 검증 명령:

```bash
hk_env/bin/python -m pytest tests -q
hk_env/bin/python tests/verify_main.py
hk_env/bin/python tests/verify_admin_display.py
```

단위/회귀 테스트 57개 통과. 브라우저 통합은 실제 main.py로 별도 수행했다. 새로운 서버·run.py·포트·가상환경은 만들지 않았다.

## 수정 파일

`main.py`, `config.py`, `core/selective_routes.py`, `core/selective_camera.py`, `core/selective_crypto.py`, `core/selective_recording.py`, `core/selective_recognition.py`, 새 `core/face_display.py`.

템플릿: `monitor.html`, `assets.html`, `employees.html`, `index.html`, `audit.html`, `decryption.html`, `selective_restore.html`.

검증: `tests/test_face_display.py`, `tests/verify_admin_display.py`, `tests/verify_main.py`, `tests/verify_default.py`, `tests/verify_cpu_file.py`. 문서: README, selective-restore, selective-format, 본 문서.

## 검증 한계

RealSense D457·Raspberry Pi 실장비와 USB 물리적 분리는 미검증이다. 미등록 시험은 동일 실제 얼굴을 등록하기 전 상태로 진행했으며 다른 사람을 대상으로 한 오인식률 시험은 하지 않았다. 실물 시험은 짧아 장시간 안정성을 보장하지 않는다. Logitech MJPG의 `Corrupt JPEG data ... extraneous bytes` 경고는 여전히 일부 발생하지만 영상·블러·저장·복원 검사는 통과했다. 탐지하지 못한 얼굴까지 보호하는 것은 보장하지 않는다.
