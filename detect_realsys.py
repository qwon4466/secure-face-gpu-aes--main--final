from core.camera_manager import CameraManager
import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import sqlite3
import io
import os
from insightface.app import FaceAnalysis
from insightface.utils import face_align

import config as c
from core.anonymizer import INNAnonymizer

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",   # Ubuntu
    "C:/Windows/Fonts/NanumGothic.ttf",                   # Windows (나눔고딕 설치 시)
    "C:/Windows/Fonts/malgun.ttf",                         # Windows 기본 맑은 고딕
    "C:/Windows/Fonts/gulim.ttc",                          # Windows 기본 굴림
]

# ==========================================
# 1. DB 연동 및 Numpy 어댑터
# ==========================================
DB_PATH = "security_system.db"

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

sqlite3.register_converter("array", convert_array)

def load_registered_users():
    """DB에 등록된 모든 인원의 정보와 벡터를 메모리에 불러옵니다."""
    try:
        conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        cursor = conn.cursor()
        cursor.execute("SELECT name, auth_group, vector FROM users")
        data = cursor.fetchall()
        conn.close()
        return data
    except Exception as e:
        print(f"❌ DB 로드 실패 (먼저 main.py로 인원을 등록하세요): {e}")
        return []

def cosine_similarity(vec1, vec2):
    """두 벡터 간의 유사도를 계산합니다 (-1 ~ 1)"""
    v1=vec1.flatten()
    v2=vec2.flatten()
    
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# ---------------------------------------------------------
# ✨ 한글 출력 전용 함수 (여기를 복사해서 코드 윗부분에 붙여넣으세요)
# ---------------------------------------------------------
def _load_font(size):
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()

def put_korean_text(img, text, position, font_size, color):
    b, g, r = color
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    font = _load_font(font_size)
    draw.text(position, text, font=font, fill=(b, g, r))
    return np.array(img_pil)

# ==========================================
# 2. UI 표식 그리기 도우미
# ==========================================
def draw_green_circle(img, center, size=6):
    cv2.circle(img, center, size, (0, 255, 0), -1)


def draw_red_x(img, center, size=6, thickness=2):
    cx, cy = center
    cv2.line(img, (cx - size, cy - size), (cx + size, cy + size), (0, 0, 255), thickness)
    cv2.line(img, (cx + size, cy - size), (cx - size, cy + size), (0, 0, 255), thickness)

# ==========================================
# 3. 실시간 감시 파이프라인
# ==========================================
def run_security_camera():
    print("🧠 모델 로드 중...")
    app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_thresh=0.6)
    detector = app.models['detection']
    recognizer = app.models['recognition']

    db_users = load_registered_users()
    print(f"✅ DB에서 {len(db_users)}개의 등록된 얼굴 데이터를 불러왔습니다.")

    # INN 익명화 모듈 초기화
    anonymizer = INNAnonymizer(checkpoint_path=c.INN_CHECKPOINT)
    PASSWORD = c.DEMO_PASSWORD

    cap = CameraManager().open()

    MATCH_THRESHOLD = c.MATCH_THRESHOLD

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1)

        bboxes, kpss = detector.detect(frame, max_num=0, metric='default')

        if bboxes is not None and len(bboxes) > 0:
            for i in range(bboxes.shape[0]):
                x1, y1, x2, y2 = bboxes[i, 0:4].astype(int)
                conf = bboxes[i, 4]
                landmarks = kpss[i]

                # 1. 벡터 추출
                face_aligned = face_align.norm_crop(frame, landmark=landmarks, image_size=112)
                embedding = recognizer.get_feat(face_aligned)

                # 2. DB와 매칭 (가장 닮은 사람 찾기)
                best_name = "Unknown"
                best_group = "비허가"
                max_sim = -1

                if embedding is not None and len(db_users) > 0:
                    for db_name, db_group, db_vector in db_users:
                        sim = cosine_similarity(embedding, db_vector)
                        if sim > max_sim:
                            max_sim = sim
                            if sim > MATCH_THRESHOLD:
                                best_name = db_name
                                best_group = db_group

                # 3. Unknown 외부인 → INN 익명화 적용 (박스 그리기 전에)
                if best_name == "Unknown":
                    frame, _, _ = anonymizer.protect_roi(frame, [x1, y1, x2, y2], PASSWORD)

                # 4. 그룹별 색상 및 표식 매핑
                if best_group == "허가":
                    color = (0, 255, 0)
                    marker_func = draw_green_circle
                else:
                    color = (0, 0, 255)
                    marker_func = draw_red_x

                # 5. 시각화
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                ui_y = max(y1 - 10, 20)
                marker_center = (x1 + 10, ui_y - 4)
                text_start_x = x1 + 25

                marker_func(frame, marker_center)

                if best_name == "Unknown":
                    display_text = f"외부인 ({max_sim:.2f})"
                else:
                    display_text = f"{best_name} ({max_sim:.2f})"

                frame = put_korean_text(frame, display_text, (text_start_x, ui_y - 15), 20, color)

        # (for문 종료)
        cv2.imshow("Security Robot Vision", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_security_camera()
