from core.camera_manager import CameraManager
"""
 CameraProcessor — 카메라 캡처 + SCRFD 탐지 + ArcFace 인식 + AES 원본 저장
백그라운드 스레드에서 처리 후 MJPEG용 JPEG 버퍼를 유지한다.
"""
import io
import os
import threading
import sqlite3
import time
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from insightface.app import FaceAnalysis
from insightface.utils import face_align

import config as c
import event_log

DB_PATH = "security_system.db"

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Ubuntu
    "C:/Windows/Fonts/NanumGothic.ttf",                  # Windows (나눔고딕 설치)
    "C:/Windows/Fonts/malgun.ttf",                        # Windows 맑은 고딕
    "C:/Windows/Fonts/gulim.ttc",                         # Windows 굴림
]


def _load_font(size: int) -> ImageFont.ImageFont:
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _put_text(frame: np.ndarray, text: str, pos: tuple, size: int, color_bgr: tuple) -> np.ndarray:
    b, g, r = color_bgr
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    ImageDraw.Draw(pil).text(pos, text, font=_load_font(size), fill=(r, g, b))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# SQLite ↔ numpy 어댑터
def _adapt_array(arr: np.ndarray) -> sqlite3.Binary:
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    return sqlite3.Binary(buf.read())


def _convert_array(data: bytes) -> np.ndarray:
    buf = io.BytesIO(data)
    buf.seek(0)
    return np.load(buf)


sqlite3.register_adapter(np.ndarray, _adapt_array)
sqlite3.register_converter("array", _convert_array)


def _load_db() -> list:
    try:
        conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        rows = conn.execute("SELECT name, auth_group, vector FROM users").fetchall()
        conn.close()
        return rows
    except Exception as e:
        print(f"[DB] 로드 실패: {e}")
        return []


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten(), b.flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else -1.0


class CameraProcessor:
    """
    단일 카메라를 백그라운드 스레드로 처리.
    - SCRFD 탐지 → ArcFace 인식 → 사원/외부인 분기
    - 외부인: 모자이크 익명화 + 빨간 박스
    - 사원: 권한 컬러 박스 + 이름
    - get_jpeg(): 최신 처리 프레임을 JPEG bytes로 반환
    """

    def _load_models(self):
        """Hailo-8L NPU 가속기 및 CPU 폴백 모델 로드"""
        use_hailo = getattr(c, "USE_HAILO", False)
        if use_hailo:
            try:
                from hailo_infer import HAILO_AVAILABLE, HailoSCRFD
                if not HAILO_AVAILABLE:
                    raise RuntimeError("hailo_platform 미설치")
                det = HailoSCRFD(
                    c.SCRFD_HEF_PATH,
                    conf_thresh=getattr(c, "HAILO_DET_THRESH", 0.5),
                )
                # SCRFD는 Hailo에 유지하고 ArcFace는 CPU에서 실행한다.
                # 이렇게 하면 저장 경로의 ArcFace가 Hailo 전역 락을 점유하지 않는다.
                fa = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
                fa.prepare(ctx_id=-1, det_thresh=0.6)
                rec = fa.models["recognition"]
                print("[CameraProcessor] ⚡ Hailo-8L SCRFD + CPU ArcFace 사용")
                return det, rec
            except Exception as e:
                print(f"[CameraProcessor] Hailo 사용 불가({e}) → insightface 폴백")

        fa = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
        fa.prepare(ctx_id=-1, det_thresh=0.6)
        print("[CameraProcessor] insightface(CPU) 사용")
        return fa.models["detection"], fa.models["recognition"]

    def __init__(self):
        from raw_restore import RawRecorder
        print("[CameraProcessor] 모델 로드 중...")
        self.detector, self.recognizer = self._load_models()

        self._db_lock = threading.Lock()
        self._db_users: list = []
        self._last_detect_log = 0.0   # 비허가자 감지 로그 쿨다운용
        self.reload_db()
        self._raw_recorder = RawRecorder(
            save_root="encrypted_raw",
            fps=10,
            width=640,
            height=480,
        )

        #self._raw_recorder.write()
        self._frame_lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._latest_raw_jpeg: bytes | None = None  # 익명화 전 원본 (등록용)

        # 원본 저장은 캡처 스레드와 분리한다. 저장이 밀리면 bounded queue가
        # 오래된 raw 프레임부터 폐기해 실시간 입력을 지연시키지 않는다.
        self._raw_lock = threading.Lock()
        self._raw_io_lock = threading.Lock()
        self._raw_queue = deque(maxlen=2)
        self._raw_running = True
        self._raw_thread = threading.Thread(
            target=self._raw_writer_loop, daemon=True
        )
        self._raw_thread.start()

        self._stats_lock = threading.Lock()
        self._stats = {"employee_count": 0, "unknown_count": 0, "recording": True}
        self._perf_lock = threading.Lock()
        self._perf = {
            "processed_fps": 0.0,
            "process_ms": 0.0,
            "pending_queue": 0,
        }

        self._running = False
        print("[CameraProcessor] 준비 완료")

    # ── 공개 API ──────────────────────────────────────────────────────────
     
    def reload_db(self):
        with self._db_lock:
            self._db_users = _load_db()
        print(f"[DB] {len(self._db_users)}명 로드됨")

    def start(self, cam_id=None):
        self._running = True
        t = threading.Thread(target=self._loop, args=(cam_id,), daemon=True)
        t.start()
        print(f"[CameraProcessor] 카메라 {cam_id} 시작")

    def stop(self):
        self._running = False
        self._raw_running = False
        if getattr(self, "_raw_thread", None):
            self._raw_thread.join(timeout=1.0)
        try:
            with self._raw_io_lock:
                self._raw_recorder.close()
        except:
            pass

    def _raw_writer_loop(self):
        """캡처와 분리된 원본 raw 파일 writer."""
        while self._raw_running or self._raw_queue:
            with self._raw_lock:
                frame = self._raw_queue.popleft() if self._raw_queue else None
            if frame is None:
                time.sleep(0.005)
                continue
            try:
                with self._raw_io_lock:
                    self._raw_recorder.write(frame)
            except Exception:
                pass

    def get_jpeg(self) -> bytes | None:
        with self._frame_lock:
            return self._latest_jpeg

    def get_raw_jpeg(self) -> bytes | None:
        """익명화 전 원본 프레임 (사원 등록용)."""
        with self._frame_lock:
            return self._latest_raw_jpeg

    def capture_raw_frame(self) -> "np.ndarray | None":
        """현재 원본 프레임을 디코딩해 ndarray로 반환 (등록 처리용)."""
        jpeg = self.get_raw_jpeg()
        if jpeg is None:
            return None
        arr = np.frombuffer(jpeg, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def get_stats(self) -> dict:
        with self._stats_lock:
            return dict(self._stats)

    def get_debug_info(self) -> dict:
        with self._frame_lock:
            size = len(self._latest_jpeg) if self._latest_jpeg else 0
        return {
            "running": self._running,
            "jpeg_size": size,
            "has_frame": size > 0,
            "db_users": len(self._db_users),
            "stats": self.get_stats(),
            "perf": self.get_perf(),
        }

    def get_perf(self) -> dict:
        with self._perf_lock:
            return dict(self._perf)

    # ── 캡처 루프 ─────────────────────────────────────────────────────────

    def _loop(self, cam_id: int):
        if getattr(c, "FORCE_VIDEO", False):
            fallback = getattr(c, "VIDEO_FALLBACK", None)
            if fallback and os.path.exists(fallback):
                print(f"[Camera] FORCE_VIDEO=True → 영상 재생: {fallback}")
                self._video_loop(fallback)
                return

        cap = None
        manager = CameraManager(config=c)
        def _open_capture():
            try:
                return manager.reconnect() if manager.selected else manager.open()
            except RuntimeError as exc:
                print('[Camera]', exc, flush=True)
                return None

        # 최초 연결도 재귀 호출하지 않고 backoff를 둔 반복으로 처리한다.
        while self._running:
            cap = _open_capture()
            if cap is not None:
                break
            print(f"[Camera] 카메라 {cam_id} 열기 실패 → 5초 후 재시도")
            self._show_reconnecting()
            time.sleep(5)
        if cap is None:
            return

        print(f"[Camera] 카메라 {cam_id} 열림")

        state = {"frame": None, "run": True, "first": True}
        rlock = threading.Lock()

        def _reader():
            nonlocal cap
            last_raw = 0.0
            raw_dt = 1.0 / max(1.0, float(getattr(self._raw_recorder, "fps", 10)))
            read_failures = 0
            last_failure_log = 0.0

            def _reconnect():
                nonlocal cap
                print(f"[Camera] 카메라 reconnect 시작 (id={cam_id})")
                try:
                    cap.release()
                except Exception:
                    pass

                delay = 1.0
                while state["run"] and self._running:
                    time.sleep(delay)
                    candidate = _open_capture()
                    if candidate is None:
                        print(f"[Camera] 카메라 reconnect 실패 → {delay:.0f}초 후 재시도")
                        delay = min(10.0, delay * 2.0)
                        continue
                    try:
                        ok, first_frame = candidate.read()
                    except Exception as exc:
                        print(f"[Camera] reconnect 확인 중 예외: {exc}")
                        ok, first_frame = False, None
                    if ok and first_frame is not None:
                        cap = candidate
                        print(f"[Camera] 카메라 reconnect 성공 (id={cam_id})")
                        return first_frame
                    candidate.release()
                    print(f"[Camera] reconnect 실패: 정상 프레임 미수신 → {delay:.0f}초 후 재시도")
                    delay = min(10.0, delay * 2.0)
                return None

            while state["run"] and self._running:
                try:
                    ret, f = cap.read()
                except Exception as exc:
                    print(f"[Camera] _reader() 예외: {exc}")
                    ret, f = False, None

                if not ret or f is None:
                    read_failures += 1
                    now = time.monotonic()
                    if read_failures == 1 or now - last_failure_log >= 5.0:
                        print(f"[Camera] camera read 실패 (연속 {read_failures}회)")
                        last_failure_log = now
                    # 일시적인 단일 실패는 잠시 기다리고, 연속 실패부터 재연결한다.
                    if read_failures < 3:
                        time.sleep(0.05)
                        continue
                    f = _reconnect()
                    if f is None:
                        return
                    read_failures = 0
                    last_failure_log = 0.0

                try:
                    f = cv2.flip(f, 1)
                except Exception as exc:
                    print(f"[Camera] _reader() 프레임 처리 예외: {exc}")
                    time.sleep(0.05)
                    continue

                read_failures = 0

                try:
                    now = time.time()
                    raw_due = now - last_raw >= raw_dt
                    # 등록/원본 스냅샷용 JPEG는 raw recorder 기준 FPS로만 갱신한다.
                    # 매 입력 프레임마다 인코딩하던 중복 CPU 부하를 제거한다.
                    if raw_due:
                        _okr, _bufr = cv2.imencode(
                            ".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 90]
                        )
                        if _okr:
                            with self._frame_lock:
                                self._latest_raw_jpeg = _bufr.tobytes()
                        last_raw = now

                    with rlock:
                        state["frame"] = f

                    # raw 저장은 별도 writer에 전달한다. queue가 가득 차면
                    # 가장 오래된 raw 프레임이 버려지고 캡처는 계속된다.
                    if raw_due:
                        with self._raw_lock:
                            self._raw_queue.append(f.copy())
                except Exception as exc:
                    print(f"[Camera] _reader() 예외: {exc}")
                    time.sleep(0.05)

        threading.Thread(target=_reader, daemon=True).start()

        max_fps = getattr(c, "PROCESS_MAX_FPS", 15)
        min_dt = (1.0 / max_fps) if max_fps and max_fps > 0 else 0.0
        last_proc = 0.0
        perf_started = time.time()
        perf_count = 0
        perf_ms_total = 0.0
        while self._running:
            with rlock:
                frame = state["frame"]
                state["frame"] = None
                
            if frame is None:
                time.sleep(0.005)
                continue

            if state["first"]:
                state["first"] = False
                print(f"[Camera] 첫 프레임 수신 {frame.shape}")

            now = time.time()
            if min_dt > 0 and (now - last_proc) < min_dt:
                time.sleep(0.002)
                continue
            last_proc = now

            frame = self._maybe_downscale(frame)
            proc_started = time.time()
            try:
                frame, emp, unk = self._process(frame)
            except Exception as e:
                print(f"[Camera] _process 오류 (건너뜀): {e}")
                emp, unk = 0, 0
            proc_ms = (time.time() - proc_started) * 1000.0
            perf_count += 1
            perf_ms_total += proc_ms

            with self._stats_lock:
                self._stats["employee_count"] = emp
                self._stats["unknown_count"] = unk

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                with self._frame_lock:
                    self._latest_jpeg = buf.tobytes()

            perf_elapsed = time.time() - perf_started
            if perf_elapsed >= 5.0:
                with self._raw_lock:
                    pending_size = len(self._raw_queue)
                with self._perf_lock:
                    self._perf.update({
                        "processed_fps": perf_count / perf_elapsed,
                        "process_ms": perf_ms_total / max(1, perf_count),
                        "pending_queue": pending_size,
                    })
                print(
                    f"[Camera] 화면 처리 {perf_count / perf_elapsed:.1f}fps, "
                    f"평균 {perf_ms_total / max(1, perf_count):.0f}ms, "
                    f"저장 대기큐 {pending_size}장"
                )
                perf_started = time.time()
                perf_count = 0
                perf_ms_total = 0.0

        state["run"] = False
        cap.release()
        print(f"[Camera] 카메라 {cam_id} 종료")

    def _realsense_loop(self):
        # Compatibility entry point: RGB acquisition uses the same V4L2 manager.
        self._loop(None)

    def _video_loop(self, video_path: str):
        fps = getattr(c, "VIDEO_FALLBACK_FPS", 25)
        delay = 1.0 / max(1, fps)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Video] 영상 열기 실패: {video_path} → 더미 모드")
            self._dummy_loop()
            return

        print(f"[Video] 폴백 영상 재생 시작 (목표 fps={fps})")
        frame_count = 0
        while self._running:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_count += 1
            try:
                frame, emp, unk = self._process(frame)
            except Exception as e:
                print(f"[Video] _process 오류 (건너뜀): {e}")
                emp, unk = 0, 0

            cv2.putText(frame, f"DEMO (video) F{frame_count}",
                        (10, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)

            with self._stats_lock:
                self._stats["employee_count"] = emp
                self._stats["unknown_count"] = unk

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                with self._frame_lock:
                    self._latest_jpeg = buf.tobytes()

            elapsed = time.time() - t0
            if elapsed < delay:
                time.sleep(delay - elapsed)

        cap.release()
        print("[Video] 폴백 영상 종료")

    def _show_reconnecting(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (30, 30, 40)
        cv2.putText(frame, "Reconnecting...", (160, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 140, 200), 2)
        cv2.putText(frame, "Camera disconnected. Retrying in 5s", (60, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 100), 1)
        _, buf = cv2.imencode(".jpg", frame)
        with self._frame_lock:
            self._latest_jpeg = buf.tobytes()

    def _dummy_loop(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (30, 30, 40)
        cv2.putText(frame, "No Camera", (200, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 120), 2)
        cv2.putText(frame, "Connect webcam & restart server", (70, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 100), 1)
        _, buf = cv2.imencode(".jpg", frame)
        jpeg = buf.tobytes()
        with self._frame_lock:
            self._latest_jpeg = jpeg
        while self._running:
            time.sleep(1)

    # ── 모자이크 익명화 (Gaussian Blur) ──────────────────────────────

    def _mosaic(self, frame: np.ndarray, x1, y1, x2, y2) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        bx1, by1 = max(0, int(x1)), max(0, int(y1))
        bx2, by2 = min(w, int(x2)), min(h, int(y2))
        
        if bx2 > bx1 and by2 > by1:
            roi = out[by1:by2, bx1:bx2]
            if roi.size > 0:
                blurred = cv2.GaussianBlur(roi, (99, 99), 30)
                out[by1:by2, bx1:bx2] = blurred
        return out

    # ── 프레임 처리 ───────────────────────────────────────────────────────

    def _maybe_downscale(self, frame: np.ndarray) -> np.ndarray:
        pw = getattr(c, "PROCESS_WIDTH", 0)
        if pw and frame.shape[1] > pw:
            scale = pw / frame.shape[1]
            frame = cv2.resize(frame, (pw, int(frame.shape[0] * scale)))
        return frame

    def _process(self, frame: np.ndarray) -> tuple[np.ndarray, int, int]:
        bboxes, kpss = self.detector.detect(frame, max_num=0, metric="default")
        
        # 사람이 없을 때 불필요한 연산 방지
        if bboxes is None or len(bboxes) == 0:
            return frame, 0, 0

        anonymize_all = getattr(c, "ANONYMIZE_ALL", False)
        emp, unk = 0, 0
        for i in range(bboxes.shape[0]):
            x1, y1, x2, y2 = bboxes[i, :4].astype(int)
            lm = kpss[i]
            aligned = face_align.norm_crop(frame, landmark=lm, image_size=112)
            emb = self.recognizer.get_feat(aligned)
            name, group, sim = self._match(emb)

            if name == "Unknown":
                unk += 1
                # 비허가자 감지 이벤트 기록 (쿨다운으로 로그 폭주 방지)
                _now = time.time()
                if _now - self._last_detect_log >= getattr(c, "DETECT_LOG_COOLDOWN", 10):
                    self._last_detect_log = _now
                    event_log.log_detection(name, group, sim)
            else:
                emp += 1

            if name == "Unknown" or anonymize_all:
                frame = self._mosaic(frame, x1, y1, x2, y2)
            frame = self._draw(frame, x1, y1, x2, y2, name, group, sim)
        return frame, emp, unk

    def _match(self, emb) -> tuple[str, str, float]:
        best_name, best_group, best_sim = "Unknown", "비허가", -1.0
        with self._db_lock:
            if emb is not None and self._db_users:
                for db_name, db_group, db_vec in self._db_users:
                    s = _cosine_sim(emb, db_vec)
                    if s > best_sim:
                        best_sim = s
                        if s > c.MATCH_THRESHOLD:
                            best_name = db_name
                            best_group = db_group
        return best_name, best_group, best_sim

    def _draw(self, frame: np.ndarray, x1, y1, x2, y2, name, group, sim) -> np.ndarray:
        # 상태에 따른 텍스트와 색상 설정 (Unknown 기준 판별)
        if name != "Unknown":
            color = (0, 200, 0) # 초록색
            # 개인 이름 대신 '허가자'로 통일하고 유사도 표시
            label = f"허가자 ({sim:.2f})" 
        else:
            color = (0, 0, 220) # 빨간색
            # '외부인' 대신 '비허가자'로 변경하고 유사도 표시
            label = f"비허가자 ({sim:.2f})"

        # 네모 테두리 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # 텍스트 위치 계산 및 기존 _put_text 함수로 한글 출력
        text_y = max(y1 - 28, 5)
        frame = _put_text(frame, label, (x1, text_y), 18, color)
        
        return frame
