# SecureFace-RX

기존 관제·인원 등록·녹화·과거 영상 복원과 **집단키 선택적 복원**을 하나의 서버에서 제공합니다.

```bash
cd /home/hkl9292/mbappe/secure-face-gpu-aes--main-main
source hk_env/bin/activate
python main.py
```

**http://127.0.0.1:8000** → **관리자 로그인 → 관리자 메뉴 → 선택적 복원**.

1. 카메라에서 새 형식의 보호 청크를 자동 녹화합니다.
2. 완료 청크를 선택하고 키 파일을 제출합니다.
3. 선택 복원 후 화면에서 재생하거나 무손실 AVI를 다운로드합니다.

키는 첫 실행에 생성됩니다. `raw_data/.keys/`의 `external.key`는 외부인, `internal.key`는 내부인, `master.key`는 전체 복원용입니다. 미확정·집단 간 겹침은 전체키 전용입니다. 키 파일명으로 권한을 판단하지 않습니다.

카메라·얼굴 모델이 없으면 오류가 표시되며 실제 녹화는 시작되지 않습니다. 장비 없는 테스트는 **같은 서버**에서 명시적으로 실행합니다:

```bash
SRX_TEST_MODE=synthetic python main.py
```

이때 테스트키와 영상은 `tests/runtime/main-server-test/`에 분리됩니다. 종료는 Ctrl+C입니다.

- [설치·사용·모델 설정 및 저장 형식](docs/selective-restore.md)
- [실제 통합 검증 결과](docs/selective-verification.md)
- [암호화·마스크 상세](docs/selective-format.md)

별도 실행기·별도 FastAPI 앱·별도 가상환경은 없습니다. 이전 `extensions/selective_restore` 폴더는 통합 후 제거했습니다.

## 카메라 자동 선택

`python main.py`는 연결된 컬러 카메라를 자동 선택합니다. 수동 지정은 `CAMERA_DEVICE=/dev/video0 python main.py`처럼 실행합니다. 번호는 예시이며 기본값이 아닙니다. [설정·탐색 규칙·실제 검증 결과](docs/camera-compatibility.md)를 참고하세요.

[관리자 전용 복원·강화 블러·한글 라벨 검증](docs/admin-display-verification.md)

- [raw_data 청크·관리자 선택 복원 UI 검증](docs/raw-data-selective-ui.md)
