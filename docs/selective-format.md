# SRX v1 형식 · 암호화 경계

## 청크

운영 청크는 `raw_data/YYYY-MM-DD/HH/HH-MM-SS/` 아래에
`encrypted_raw.bin.enc`, `encrypted_raw.bin.json`, `protected.avi`,
`external.gcm`, `internal.gcm`, `master.gcm`, `manifest.json`을 저장한다.
보호 영상은 OpenCV FFV1/AVI, uint8 HWC **BGR24**. 인코딩 후 전 프레임을 다시 읽어 입력 보호 프레임과 정확히 일치하는지 확인한다. 짝수 가로·세로만 받는다.

1. 메모리의 입력 프레임에서 모든 탐지 영역을 보호한다.
2. 보호 영상만 디스크에 쓰고 닫아 SHA-256을 확정한다.
3. 해당 해시와 프레임/마스크 메타데이터를 AAD에 넣어 희소 원본 픽셀을 암호화한다.
4. 모든 파일 fsync → staging 디렉터리 fsync → 동일 부모 아래 원자적 rename → 부모 fsync.

`encrypted_raw.bin.enc`는 블러 전 BGR24 프레임 전체를 순서대로 이어 붙인
평문을 기존 32바이트 `master.key`로 AES-256-GCM 암호화한 별도 payload다.
매 청크에 12바이트 난수 nonce를 새로 사용하고 16바이트 tag는
`encrypted_raw.bin.json`에 저장한다. JSON에는 키나 원본 픽셀이 없으며,
nonce, tag, 공개 영상 속성, ciphertext SHA-256만 있다. JSON에서 tag와 해시를
제외한 정규화 객체 전체가 GCM AAD다. JSON 전체의 SHA-256은 서명되고 기존
master envelope로 인증되는 manifest에 포함된다. 전체키 복원은 manifest 인증과
원본 GCM 인증이 모두 끝난 뒤에만 원본 프레임을 반환한다.

`.partial-*`는 완료 청크가 아니며 목록/복원에서 제외한다. 재시작 시 이 디렉터리를 이어 쓰지 않는다. 재시도는 새 ID/DEK로 시작한다. 자동 삭제하지 않는다. 운영자가 저장 장애를 확인한 뒤 신규 경로 내 불완전한 파일을 정리할 수 있다. 이미 완료된 청크는 후속 입력 오류와 무관하게 유효하다. 저장이 끝난 복원 결과 역시 숨김 staging에서 원자적으로 별도 결과 디렉터리로 이동한다. 복원 저장 실패의 staging에는 사용자가 요청한 복원 데이터가 남을 수 있으므로 같은 비공개 권한으로 보관하며 API에 제공하지 않는다.

## 키와 인증

집단 KEK는 독립 난수 32바이트. 파일명은 CLI 녹화 키 선택에만 쓰이며 복원 권한에는 사용하지 않는다. 복원기는 제공된 단일 키로 AES Key Wrap을 실제 풀 수 있는 항목만 처리한다.

| DEK | 포장하는 KEK |
|---|---|
| external | external, master |
| internal | internal, master |
| master | master |

각 청크/집단에 독립 AES-256 DEK를 새로 만든다. 각 DEK에서 12바이트 nonce `000...000`으로 **한 번만** 픽셀 payload를 AES-GCM 암호화한다. ciphertext에는 cryptography가 생성한 16바이트 GCM tag가 포함된다. DEK는 저장하지 않고 AES Key Wrap 결과만 저장한다. 비어 있는 집단도 빈 payload를 암호화하여 키 확인과 인증을 유지한다.

AAD = canonical JSON `{"meta": 전체 메타데이터, "group": 집단}`.
직렬화는 UTF-8과 동일한 ASCII JSON (`ensure_ascii=True`, `sort_keys=True`, `separators=(',', ':')`, `allow_nan=False`). 중복 JSON 키와 비유한 수는 거부한다.

`meta`는 버전, 랜덤 세션 ID/청크 ID, 완료 상태, 해상도·색상·코덱·FPS, 프레임 정보, 정확한 복원 마스크, 블러 설정, 보호 영상 SHA-256, 청크별 서명 공개키를 포함한다.

`manifest`는 meta + 모든 집단 ciphertext SHA-256/nonce/감싼 DEK 목록이다. 각 집단 DEK의 **다른 nonce** `000...001`로 빈 평문을 암호화하여 전체 manifest를 인증한다(`seals`). 따라서 외부인키로도 내부인 ciphertext와 포장 정보의 변경을 검출할 수 있지만 내부인 원본은 풀 수 없다. 오류 후에는 DEK를 재사용하지 않는다. 랜덤 DEK 충돌 확률은 무시할 정도이나 수학적 0이라고 주장하지 않는다.

추가 Ed25519 서명은 청크마다 일회성 서명키를 생성해 manifest + seals 전체를 서명한다. 서명 개인키는 저장하지 않는다. 공개키가 DEK 인증 AAD 안에 있으므로 공격자가 새 서명키를 만들어도 제공된 집단키의 GCM 검증을 통과할 수 없다. 이 서명은 접근할 수 없는 집단의 seal 손상도 검출한다. 별도의 조직 신원/카메라 출처 인증 체계는 아니다.

**무키 복원**은 원본 payload를 전혀 풀지 않고 구조·파일 해시·청크 자체 서명만 검사해 보호본을 복사한다. 무키 사용자는 독립 신뢰 기준이 없으므로 통째로 위조된 새 묶음의 출처를 인증할 수 없다. 제공된 올바른 키가 있을 때 보호 영상 해시는 GCM AAD에 묶여 인증된다. 복원 시 파일 바이트를 다시 읽어 인증된 해시와 비교하고 메모리 스냅샷(memfd)에서 디코딩하므로 검증 후 경로 교체로 다른 영상을 읽지 않는다. 폴더 ID가 다르면 교체 청크는 거부한다. 정상 청크 디렉터리 전체를 그 원래 ID 그대로 제공하는 재생 공격은 외부 신뢰 카탈로그 없이는 판별할 수 없다.

[공식 AESGCM 문서](https://cryptography.io/en/latest/hazmat/primitives/aead/#cryptography.hazmat.primitives.ciphers.aead.AESGCM), [공식 AES Key Wrap 문서](https://cryptography.io/en/stable/hazmat/primitives/keywrap/)의 API를 사용한다. 직접 구현한 암호 알고리즘은 없다.

## 마스크와 payload

bbox는 탐지 좌표 `[x1,y1,x2,y2)`이며 영상 경계로 제한한다. **write ROI, 블러 ROI, 표시 bbox, 원본 암호화 마스크는 같은 좌표**를 사용하고 padding은 0이다. 커널은 max(61, 얼굴 긴 변×0.9)를 홀수로 보정한다. bbox ROI의 긴 변을 최대 32픽셀로 축소해 대응 커널과 sigma=대응 커널/3의 Gaussian blur를 적용한 뒤 정확히 같은 bbox 크기로 확대한다. bbox 밖 픽셀은 블러 입력·출력에 포함하지 않는다. 한글 라벨과 테두리는 블러 후 bbox 마스크 안에 쓰므로 전체키 복원 시 원본으로 돌아간다.

각 집단 ROI 합집합을 구하고, 다른 집단/미확정 ROI의 2픽셀 dilation 영역을 external/internal 마스크에서 제외한다. 제외한 겹침·보호 경계 및 미확정 영역은 master 마스크에 넣는다. 세 마스크는 서로 겹치지 않고 합집합은 실제 보호 ROI 합집합이다.

마스크는 row-major 평탄화한 `[시작 인덱스, 연속 픽셀 수]` 목록이다. 정렬·양수 길이·중복 없음·프레임 경계를 검사한다. payload는 프레임 순서대로 `original[mask]`의 BGR 3바이트만 이어 붙인다. bbox 전체 crop이나 비허가 영역의 원본은 포함하지 않는다. 복원은 마스크 인덱스에 직접 대입하며 보간·feathering을 하지 않는다.

## 프레임 대응 · 제한

`capture_id`는 입력에서 읽은 프레임 번호, `stored_index`는 청크 내 실제 저장 인덱스다. `timestamp_ms`/`source_pts_ms`는 파일에서 읽은 프레임 번호와 입력 FPS로 계산한 시각(카메라는 monotonic 경과 시간)이다. `pts`는 저장 표시 순서이고 `time_base`, `pts_scale`로 출력 고정 FPS에 대응한다. AVI/FFV1에는 B-frame 재정렬이 없다. 디코딩 순서와 픽셀 왕복 검사를 한다.

**가변 FPS 입력의 실제 demuxer PTS를 보존하지 않는다.** 파일 입력은 일정 FPS 영상만 지원 대상으로 삼는다. FPS 낮춤으로 건너뛴 입력 ID를 기록하며 분류 연속성이 끊기면 전체키 전용부터 재확인한다. 출력은 고정 FPS 재생이므로 실제 녹화 시간 공백을 그대로 재현하지 않는다. 카메라/드라이버 자체 드롭은 OpenCV에서 정확히 관측할 수 없어 `unknown`이다.

운영 시간 청크는 최대 1024프레임 / 암호화 원본 1GiB, 파일·합성 청크는 16프레임 / 64MiB, 프레임은 2,073,600픽셀 상한. 큐는 0개이며 저장 중 입력 읽기를 멈추는 동기 backpressure. 모델 처리 시간이 길면 카메라 드라이버 내부에서 드롭될 수 있다. RSS는 원본 버퍼 상한보다 크며 디코더·암호화·마스크·출력 버퍼를 포함한다.

실시간 저장 버퍼는 얼굴 탐지/인식 이후 원본 프레임을 세션별 임시 AES-GCM 키로 먼저 암호화합니다. 그 다음 보호본 블러와 라벨을 생성합니다. 청크 확정 시 메모리에서 원본 버퍼를 풀어 기존 집단별 마스크의 원본 픽셀을 암호화합니다. 디스크 형식 버전 1과 보호 영상 해시를 포함하는 AAD를 유지하므로 최종 디스크 암호문은 보호 영상 해시 확정 후 작성합니다. 블러된 픽셀이나 라벨을 원본 payload로 사용하지 않습니다. 전체키 복원은 라벨까지 원래 픽셀로 되돌립니다.
