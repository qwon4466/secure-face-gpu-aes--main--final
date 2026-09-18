"""
PSFRecorder — 1시간 단위 PSF 청크 녹화 (Phase 3)

청크 구조:
  recordings/
    YYYY-MM-DD_HH/        ← 1시간 단위 청크
      manifest.json
      000001/             ← 스냅샷 (N초 간격)
        frame.jpg         ← 익명화된 프레임
        face_0.npy        ← float32 (3,256,256) 복원 타일
        face_0_box.json   ← crop_box [x1,y1,x2,y2]
      000002/
        ...
"""

import hashlib
import json
import os
import threading
import time
import shutil
from collections import deque
from datetime import datetime
import cv2
import numpy as np
import config as c

RECORDINGS_DIR = getattr(c, "RECORD_RAM_DIR", "recordings")


# ── JSON 유틸 ──────────────────────────────────────────────────────────────

def _load_json(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# ── 그림자 백업 (Shadow Backup) ──────────────────────────────────────────
def backup_chunk_to_sd(ram_chunk_path: str):
    """메인 시스템에 부하를 주지 않고 뒤에서 조용히 SD카드로 복사하는 thread"""
    ram_base = getattr(c, "RECORD_RAM_DIR", "recordings")
    sd_base = getattr(c, "RECORD_SD_DIR", "recordings")
    
    # 윈도우/맥처럼 RAM과 SD 경로가 동일하면 복사할 필요 없음
    if ram_base == sd_base:
        return

    def _copy_task():
        t_start = time.time()

        try:
            # RAM 경로에서 상대 경로(예: 2026-07/14/오후/14시/14-00-00)만 추출
            rel_path = os.path.relpath(ram_chunk_path, ram_base)
            sd_path = os.path.join(sd_base, rel_path)
            
            # SD카드 쪽에 폴더 만들고 통째로 덮어쓰기 복사
            os.makedirs(os.path.dirname(sd_path), exist_ok=True)
            shutil.copytree(ram_chunk_path, sd_path, dirs_exist_ok=True)
            # 💡 [추가] 복사 소요 시간 계산
            elapsed = time.time() - t_start
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            
            # 💡 [추가] 요구하신 "26-07-15 13시 00분" 포맷 생성
            now_str = datetime.now().strftime("%y-%m-%d %H시 %M분")
            print(f"[Backup] 💾 쉐도우 백업 완료 (SD카드 저장됨): {sd_path}")
        except Exception as e:
            print(f"[Backup Error] ❌ 쉐도우 백업 실패: {e}")

    # 데몬 스레드로 실행 (메인 서버가 꺼지면 같이 종료됨)
    threading.Thread(target=_copy_task, daemon=True).start()

def _json_default(o):
    """numpy 정수/실수/배열을 JSON 직렬화 가능 타입으로 변환."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def _save_json(path: str, data: dict):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)
    os.replace(tmp_path, path)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _first_frame_jpg(chunk_path: str) -> str | None:
    for name in sorted(os.listdir(chunk_path)):
        fdir = os.path.join(chunk_path, name)
        fpath = os.path.join(fdir, "frame.jpg")
        if os.path.isdir(fdir) and os.path.exists(fpath):
            return fpath
    return None


# 계층 폴더(월/일/오전오후/시/10분청크)와 안전한 chunk_id 간 변환.
# 실제 경로:  recordings/2026-06/29/오후/14시/14-00
# chunk_id : "2026-06__29__오후__14시__14-00"  (슬래시 → __, URL 안전)
def _path_to_id(chunk_path: str) -> str:
    """
    RAM이나 SD 어디서든 기준 디렉토리를 떼어내어 순수한 상대 경로로 ID 생성.
    """
    # RAM 디스크 경로에 속해 있다면 RECORD_RAM_DIR 기준으로 상대경로 추출
    if getattr(c, "RECORD_RAM_DIR", "recordings") in chunk_path:
        rel = os.path.relpath(chunk_path, getattr(c, "RECORD_RAM_DIR", "recordings"))
    # SD 카드 경로에 속해 있다면 RECORD_SD_DIR 기준으로 추출
    else:
        rel = os.path.relpath(chunk_path, getattr(c, "RECORD_SD_DIR", "recordings"))
    return rel.replace(os.sep, "__").replace("/", "__")


def _id_to_path(chunk_id: str) -> str:
    """
    1순위로 RAM 디스크(오늘 데이터)에 파일이 있는지 확인하고,
    없으면 2순위로 SD 카드(과거 데이터)에서 파일을 찾아 반환.
    """
    rel = chunk_id.replace("__", os.sep)
    ram_path = os.path.join(getattr(c, "RECORD_RAM_DIR", "recordings"), rel)
    sd_path = os.path.join(getattr(c, "RECORD_SD_DIR", "recordings"), rel)
    return ram_path if os.path.exists(ram_path) else sd_path


# ── PSFRecorder ────────────────────────────────────────────────────────────

class PSFRecorder:
    """
    CameraProcessor에서 N초 간격으로 스냅샷을 받아 PSF 청크로 저장.
    1시간마다 새 청크 폴더 자동 생성.
    """

    def __init__(self, camera, interval_sec: int = 5):
        self._camera = camera
        self._interval = interval_sec
        self._running = False
        # 저장 진행률 추적 (UI ETA 표시용)
        self._save_fps = 0.0
        self._cur_frame_id = 0
        self._cur_chunk = None
        self._save_times = deque(maxlen=20)
        os.makedirs(RECORDINGS_DIR, exist_ok=True)

    def start(self):
        # 현재 시스템은 AES RawRecorder만 사용한다. 이 Recorder는 과거
        # INN/PSF 저장 호환 코드이므로 실수로 시작되어도 아무 작업도 하지 않는다.
        print("[Recorder] PSF/INN Recorder 비활성화 (AES raw 저장 사용)")
        return

        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        threading.Thread(target=self._cleanup_loop, daemon=True).start()
        print(f"[Recorder] 녹화 시작 (간격={self._interval}s)")

    def _cleanup_loop(self):
        """보관 기간(RETENTION_DAYS)이 지난 청크를 주기적으로 삭제."""
        import config as c
        from datetime import timedelta
        days = int(getattr(c, "RETENTION_DAYS", 0) or 0)
        if days <= 0:
            return
        bases = {getattr(c, "RECORD_RAM_DIR", "recordings"),
                 getattr(c, "RECORD_SD_DIR", "recordings")}
        while self._running:
            try:
                cutoff = (datetime.now() - timedelta(days=days)).date()
                for base in bases:
                    self._purge_old(base, cutoff)
            except Exception as e:
                print(f"[Recorder] 저장소 정리 오류: {e}")
            # 약 1시간마다 (10초 단위로 나눠서 종료에 빠르게 반응)
            for _ in range(360):
                if not self._running:
                    return
                time.sleep(10)

    def _purge_old(self, base: str, cutoff):
        """base 아래 YYYY-MM/DD 폴더 중 cutoff(날짜)보다 오래된 것을 삭제."""
        if not os.path.isdir(base):
            return
        for month in os.listdir(base):
            mpath = os.path.join(base, month)
            if not os.path.isdir(mpath):
                continue
            try:
                y, mo = (int(x) for x in month.split("-"))
            except Exception:
                continue
            for day in os.listdir(mpath):
                dpath = os.path.join(mpath, day)
                if not os.path.isdir(dpath) or not day.isdigit():
                    continue
                try:
                    ddate = datetime(y, mo, int(day)).date()
                except Exception:
                    continue
                if ddate < cutoff:
                    shutil.rmtree(dpath, ignore_errors=True)
                    print(f"[Recorder] 🗑 보관기간 초과 삭제: {dpath}")
            try:
                if not os.listdir(mpath):
                    os.rmdir(mpath)
            except Exception:
                pass

    def stop(self):
        self._running = False

    # ── 내부 루프 ──────────────────────────────────────────────────────────

    def _chunk_dir(self, ts: float | None = None) -> str:
        """
        새 청크 폴더 경로를 만들고 반환 (청크 하나가 시작될 때 1회 호출).
        ts(첫 프레임 촬영 시각)를 그대로 HH-MM-SS 폴더명으로 사용한다.
        벽시계 버킷으로 자르지 않으므로, 프레임 수 기준으로 청크가 닫힐 때까지
        같은 폴더에 계속 쌓인다. 같은 초에 다른 청크가 이미 있으면 접미사(_N)로 구분.
        """
        now = datetime.fromtimestamp(ts) if ts else datetime.now()
        month = now.strftime("%Y-%m")            # 2026-06
        day = now.strftime("%d")                 # 29
        ampm = "오전" if now.hour < 12 else "오후"
        hour = f"{now.hour:02d}시"               # 14시
        base = f"{now.hour:02d}-{now.minute:02d}-{now.second:02d}"  # 정확한 시각
        parent = os.path.join(RECORDINGS_DIR, month, day, ampm, hour)
        os.makedirs(parent, exist_ok=True)
        # 같은 초에 이미 프레임이 든 청크가 있으면 새 접미사로 충돌 회피
        chunk = base
        n = 1
        while True:
            cand = os.path.join(parent, chunk)
            has_frames = os.path.isdir(cand) and any(
                d.isdigit() for d in os.listdir(cand)
            )
            if not has_frames:
                break
            chunk = f"{base}_{n}"
            n += 1
        path = os.path.join(parent, chunk)
        os.makedirs(path, exist_ok=True)
        return path

    def _loop(self):
        """
        저장 루프 진입점.
        1순위 REMOTE_PROTECT_URL(원격 PC GPU) → 2순위 PARALLEL_WORKERS(병렬)
        → 3순위 단일 프로세스.
        """
        import config as c
        if getattr(c, "REMOTE_PROTECT_URL", None):
            try:
                bs = int(getattr(c, "PROTECT_BATCH_SIZE", 1) or 1)
                if bs > 1:
                    self._loop_remote_gpu_batch(bs)
                else:
                    self._loop_remote_gpu()
                return
            except Exception as e:
                print(f"[Recorder] 원격 GPU 저장 초기화 실패 → 로컬 처리로 폴백: {e}")
        workers = int(getattr(c, "PARALLEL_WORKERS", 1) or 1)
        if workers > 1:
            try:
                self._loop_parallel(workers)
                return
            except Exception as e:
                print(f"[Recorder] 병렬 저장 초기화 실패 → 단일 프로세스로 폴백: {e}")
        self._loop_single()

    def _frames_per_chunk(self) -> int:
        import config as c
        return getattr(c, "FRAMES_PER_CHUNK", 0) or int(
            getattr(c, "CHUNK_SECONDS", 20) * getattr(c, "SAVE_FPS", 15)
        )

    def _roll_chunk(self, last_chunk, frame_id, frames_per_chunk,
                    chunk_start_time, ts):
        """
        프레임 수가 한도에 도달(또는 첫 프레임)하면 이전 청크를 완료 처리하고
        새 청크를 연다. 롤오버가 없으면 인자를 그대로 돌려준다.
        반환: (chunk, frame_id, chunk_start_time)
        """
        if last_chunk is not None and frame_id < frames_per_chunk:
            return last_chunk, frame_id, chunk_start_time
        if last_chunk is not None:
            elapsed_ram = time.time() - chunk_start_time
            r_mins = int(elapsed_ram // 60)
            r_secs = int(elapsed_ram % 60)
            now_str = datetime.now().strftime("%y-%m-%d %H시 %M분")
            self._mark_complete(last_chunk)
            print(f"[{now_str}] 🐏 RAM 청크 저장 완료 "
                  f"({frame_id}프레임), {r_mins}분 {r_secs}초 소요")
            backup_chunk_to_sd(last_chunk)  # 섀도우 백업 시작
        # 새 청크: 첫 프레임 촬영 시각으로 폴더 생성
        return self._chunk_dir(ts), 0, time.time()

    def _save_snapshot(self, chunk, frame_id, anon_frame, tiles, ts):
        """보호본 프레임 1장을 청크 폴더에 원자적으로 저장하고 manifest 갱신."""
        snap_dir = os.path.join(chunk, f"{frame_id:06d}")
        tmp_dir = snap_dir + ".tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        cv2.imwrite(os.path.join(tmp_dir, "frame.jpg"), anon_frame)
        _save_json(os.path.join(tmp_dir, "meta.json"), {"ts": ts})
        for i, td in enumerate(tiles):
            np.save(os.path.join(tmp_dir, f"face_{i}.npy"), td["tile_f32"])
            _save_json(os.path.join(tmp_dir, f"face_{i}_box.json"), td["crop_box"])
        os.replace(tmp_dir, snap_dir)

        mpath = os.path.join(chunk, "manifest.json")
        m = _load_json(mpath) or {
            "chunk_id": _path_to_id(chunk),
            "start_time": datetime.now().isoformat(),
            "frame_count": 0,
            "total_faces": 0,
        }
        m["frame_count"] += 1
        m["total_faces"] += len(tiles)
        m["last_update"] = datetime.now().isoformat()
        _save_json(mpath, m)

        # 진행률 추적 (UI ETA 표시용): 현재 청크·프레임, 최근 저장 fps
        self._cur_chunk = chunk
        self._cur_frame_id = frame_id
        now = time.time()
        self._save_times.append(now)
        if len(self._save_times) >= 2:
            span = self._save_times[-1] - self._save_times[0]
            if span > 0:
                self._save_fps = (len(self._save_times) - 1) / span

    def save_progress(self) -> dict:
        """진행 중인 청크의 저장 진행률·예상 완료 시간(ETA)을 반환 (UI용)."""
        target = self._frames_per_chunk()
        saved = self._cur_frame_id
        fps = self._save_fps
        remaining = max(0, target - saved)
        eta = (remaining / fps) if fps > 0 else None
        return {
            "in_progress": bool(self._running and self._cur_chunk and saved < target),
            "saved": saved,
            "target": target,
            "fps": round(fps, 2),
            "eta_seconds": round(eta) if eta is not None else None,
        }

    def _loop_single(self):
        """단일 프로세스 저장 루프 (PARALLEL_WORKERS<=1 또는 병렬 폴백)."""
        frames_per_chunk = self._frames_per_chunk()
        frame_id = 0
        last_chunk = None
        save_count = 0
        t_report = time.time()
        proc_ms = 0.0
        chunk_start_time = time.time()

        while self._running:
            if self._interval > 0:
                time.sleep(self._interval)
            popped = self._camera.pop_pending()
            if popped is None:
                time.sleep(0.05)
                continue
            raw, ts = popped
            _t0 = time.time()
            anon_frame, tiles = self._camera.make_protected(raw)
            proc_ms = (time.time() - _t0) * 1000
            save_count += 1

            if time.time() - t_report >= 5.0:
                fps = save_count / (time.time() - t_report)
                print(f"[Recorder] 저장 {fps:.1f}fps (INN {proc_ms:.0f}ms/frame) "
                      f"대기 큐 {self._camera.pending_size()}장")
                save_count = 0
                t_report = time.time()

            last_chunk, frame_id, chunk_start_time = self._roll_chunk(
                last_chunk, frame_id, frames_per_chunk, chunk_start_time, ts)
            frame_id += 1
            self._save_snapshot(last_chunk, frame_id, anon_frame, tiles, ts)

    def _loop_parallel(self, workers: int):
        """
        병렬 저장 루프: 탐지·인식(Hailo)은 메인에서, 무거운 INN은 워커 N개로 분산.

        imap이 제출 순서(=프레임 순서)대로 결과를 스트리밍하므로 프레임 순서가
        보존된다. 워커는 Hailo를 건드리지 않고 순수 INN protect_roi만 수행한다.
        """
        import multiprocessing as mp
        import config as c
        from parallel_protect import init_worker, protect_job

        frames_per_chunk = self._frames_per_chunk()

        # 리눅스(라즈베리파이): forkserver 우선 — 깨끗한 서버에서 워커를 생성해
        # __main__(서버) 재실행과 부모의 torch/Hailo 스레드 상태 상속을 피한다.
        # 없으면 spawn(윈도우 등) 폴백.
        try:
            ctx = mp.get_context("forkserver")
        except ValueError:
            ctx = mp.get_context("spawn")

        checkpoint = getattr(c, "INN_CHECKPOINT", None)
        password = getattr(c, "DEMO_PASSWORD", "forensic2026")
        # 워커당 스레드 수: 0(자동)이면 코어를 워커 수로 나눠 배분
        threads = int(getattr(c, "INN_THREADS_PER_WORKER", 1) or 0)
        if threads <= 0:
            cores = os.cpu_count() or workers
            threads = max(1, cores // workers)
        pool = ctx.Pool(processes=workers, initializer=init_worker,
                        initargs=(checkpoint, password, threads))
        print(f"[Recorder] 병렬 INN 저장 시작 (워커 {workers}개 × "
              f"{threads}스레드, 청크당 {frames_per_chunk}프레임)")

        def _payloads():
            idx = 0
            while self._running:
                popped = self._camera.pop_pending()
                if popped is None:
                    time.sleep(0.05)
                    continue
                raw, ts = popped
                # 탐지·인식은 메인 프로세스에서만 (Hailo VDevice)
                faces = self._camera.analyze_frame(raw)
                yield (idx, raw, faces, ts)
                idx += 1

        frame_id = 0
        last_chunk = None
        save_count = 0
        t_report = time.time()
        chunk_start_time = time.time()
        try:
            for _fid, anon_frame, tiles, ts in pool.imap(
                    protect_job, _payloads(), chunksize=1):
                save_count += 1
                if time.time() - t_report >= 5.0:
                    fps = save_count / (time.time() - t_report)
                    print(f"[Recorder] 저장 {fps:.1f}fps (병렬 {workers}워커) "
                          f"대기 큐 {self._camera.pending_size()}장")
                    save_count = 0
                    t_report = time.time()

                last_chunk, frame_id, chunk_start_time = self._roll_chunk(
                    last_chunk, frame_id, frames_per_chunk, chunk_start_time, ts)
                frame_id += 1
                self._save_snapshot(last_chunk, frame_id, anon_frame, tiles, ts)
        finally:
            pool.terminate()
            pool.join()

    def _unpack_protected(self, content: bytes):
        """PC /protect 응답(tar.gz)을 (anon_frame, tiles)로 되돌린다."""
        import io
        import json
        import tarfile

        tiles = []
        with tarfile.open(fileobj=io.BytesIO(content)) as t:
            names = set(t.getnames())
            anon_frame = cv2.imdecode(
                np.frombuffer(t.extractfile("frame.jpg").read(), np.uint8),
                cv2.IMREAD_COLOR,
            )
            i = 0
            while f"face_{i}.npy" in names:
                tf = np.load(io.BytesIO(t.extractfile(f"face_{i}.npy").read()))
                box = json.loads(t.extractfile(f"face_{i}_box.json").read())
                tiles.append({"tile_f32": tf, "crop_box": box})
                i += 1
        return anon_frame, tiles

    def _loop_remote_gpu(self):
        """
        저장(INN 보호본 생성)을 원격 PC GPU로 오프로드 (시험용).

        탐지·인식은 Pi(Hailo)에서, INN protect는 PC GPU에서. 매 프레임 원본을
        JPEG로 인코딩해 REMOTE_PROTECT_URL로 보내고 보호본+타일을 받아 저장한다.
        ⚠️ 익명화 前 '원본'이 네트워크로 나감 + 연속 업로드 → 시험 용도 전용.
        원격 실패 시 프레임 단위로 로컬 처리(make_protected)로 폴백한다.
        """
        import io
        import json

        import config as c
        import requests

        url = getattr(c, "REMOTE_PROTECT_URL", None)
        if not url:
            raise RuntimeError("REMOTE_PROTECT_URL 미설정")
        password = getattr(c, "DEMO_PASSWORD", "forensic2026")
        frames_per_chunk = self._frames_per_chunk()
        print(f"[Recorder] 원격 GPU 저장 시작 [v2: 단계별 시간측정] → {url} "
              f"(청크당 {frames_per_chunk}프레임)")

        frame_id = 0
        last_chunk = None
        save_count = 0
        t_report = time.time()
        chunk_start_time = time.time()
        sess = requests.Session()
        # 단계별 시간 측정용 (5초 리포트에서 프레임당 평균 출력)
        acc_detect = acc_post = acc_save = 0.0

        while self._running:
            popped = self._camera.pop_pending()
            if popped is None:
                time.sleep(0.05)
                continue
            raw, ts = popped

            _td = time.time()
            faces = self._camera.analyze_frame(raw)   # 탐지·인식(Hailo, Pi)
            acc_detect += time.time() - _td

            _tp = time.time()
            _ok, enc = cv2.imencode(".jpg", raw)
            _jpg = enc.tobytes()
            anon_frame, tiles = None, None
            while self._running:
                try:
                    resp = sess.post(
                        url,
                        files={"frame": ("f.jpg", _jpg, "image/jpeg")},
                        data={"faces": json.dumps(faces), "password": password,
                              "ts": str(ts)},
                        timeout=120,
                    )
                    if resp.status_code == 503:
                        # 서버가 복원을 우선 처리 중 → 저장 잠시 멈추고 재시도
                        time.sleep(0.3)
                        continue
                    resp.raise_for_status()
                    ctype = resp.headers.get("content-type", "")
                    if "gzip" not in ctype and "octet" not in ctype:
                        raise RuntimeError(f"원격 응답 오류: {resp.text[:200]}")
                    anon_frame, tiles = self._unpack_protected(resp.content)
                    break
                except Exception as e:
                    print(f"[Recorder] 원격 GPU 보호 실패 → 로컬 폴백: {e}")
                    anon_frame, tiles = self._camera.make_protected(raw)
                    break
            if anon_frame is None:   # 중단됨(_running False)
                break
            acc_post += time.time() - _tp

            last_chunk, frame_id, chunk_start_time = self._roll_chunk(
                last_chunk, frame_id, frames_per_chunk, chunk_start_time, ts)
            frame_id += 1
            _tsv = time.time()
            self._save_snapshot(last_chunk, frame_id, anon_frame, tiles, ts)
            acc_save += time.time() - _tsv

            save_count += 1
            if time.time() - t_report >= 5.0:
                el = time.time() - t_report
                n = max(1, save_count)
                fps = save_count / el
                print(f"[Recorder] 저장 {fps:.1f}fps (원격 GPU) | "
                      f"탐지 {acc_detect / n * 1000:.0f}ms "
                      f"전송+GPU {acc_post / n * 1000:.0f}ms "
                      f"저장 {acc_save / n * 1000:.0f}ms | "
                      f"대기큐 {self._camera.pending_size()}장")
                save_count = 0
                acc_detect = acc_post = acc_save = 0.0
                t_report = time.time()

    @staticmethod
    def _tar_bytes(tar, name, data):
        import io
        import tarfile
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    def _unpack_batch(self, content: bytes) -> dict:
        """/protect_batch 응답(tar)을 {index: (anon_frame, tiles)}로 되돌린다."""
        import io
        import json
        import tarfile
        out = {}
        with tarfile.open(fileobj=io.BytesIO(content)) as t:
            names = set(t.getnames())
            prefixes = sorted({nm.split("/")[0] for nm in names if "/" in nm})
            for idx, pfx in enumerate(prefixes):
                fp = f"{pfx}/frame.jpg"
                if fp not in names:
                    continue
                anon = cv2.imdecode(
                    np.frombuffer(t.extractfile(fp).read(), np.uint8),
                    cv2.IMREAD_COLOR)
                tiles = []
                j = 0
                while f"{pfx}/face_{j}.npy" in names:
                    tf = np.load(io.BytesIO(t.extractfile(f"{pfx}/face_{j}.npy").read()))
                    box = json.loads(t.extractfile(f"{pfx}/face_{j}_box.json").read())
                    tiles.append({"tile_f32": tf, "crop_box": box})
                    j += 1
                out[idx] = (anon, tiles)
        return out

    def _loop_remote_gpu_batch(self, batch_size: int):
        """
        여러 프레임을 한 번에 원격 GPU로 보내 보호 (배치 저장, 실험).
        왕복/포장 오버헤드를 줄인다. 탐지·인식은 Pi(Hailo), INN은 PC GPU.
        """
        import io
        import json
        import tarfile

        import config as c
        import requests

        url = getattr(c, "REMOTE_PROTECT_URL", None)
        if not url:
            raise RuntimeError("REMOTE_PROTECT_URL 미설정")
        batch_url = url.rsplit("/", 1)[0] + "/protect_batch"
        password = getattr(c, "DEMO_PASSWORD", "forensic2026")
        frames_per_chunk = self._frames_per_chunk()
        print(f"[Recorder] 원격 GPU 배치 저장 시작 → {batch_url} "
              f"(배치 {batch_size}, 청크당 {frames_per_chunk}프레임)")

        frame_id = 0
        last_chunk = None
        save_count = 0
        t_report = time.time()
        chunk_start_time = time.time()
        sess = requests.Session()

        while self._running:
            # 1) 배치 수집: (i, jpg_bytes, faces, ts)
            batch = []
            while len(batch) < batch_size and self._running:
                popped = self._camera.pop_pending()
                if popped is None:
                    if batch:
                        break            # 큐 비면 모인 것만 전송
                    time.sleep(0.05)
                    continue
                raw, ts = popped
                faces = self._camera.analyze_frame(raw)
                _ok, enc = cv2.imencode(".jpg", raw)
                batch.append((len(batch), enc.tobytes(), faces, ts))
            if not batch:
                continue

            # 2) 요청 tar(무압축) 구성
            rbuf = io.BytesIO()
            with tarfile.open(fileobj=rbuf, mode="w") as t:
                for i, jpg, faces, ts in batch:
                    self._tar_bytes(t, f"{i:04d}/frame.jpg", jpg)
                    self._tar_bytes(t, f"{i:04d}/faces.json",
                                    json.dumps(faces).encode("utf-8"))
            payload = rbuf.getvalue()

            # 3) 전송 (503이면 복원 우선 → 대기 후 재시도)
            results = None
            while self._running:
                try:
                    resp = sess.post(
                        batch_url,
                        files={"file": ("batch.tar", io.BytesIO(payload),
                                        "application/x-tar")},
                        data={"password": password},
                        timeout=300,
                    )
                    if resp.status_code == 503:
                        time.sleep(0.3)
                        continue
                    resp.raise_for_status()
                    results = self._unpack_batch(resp.content)
                    break
                except Exception as e:
                    print(f"[Recorder] 원격 배치 실패 → 로컬 폴백: {e}")
                    results = None
                    break

            # 4) 저장 (배치 순서대로)
            for i, jpg, faces, ts in batch:
                if results is not None and i in results:
                    anon_frame, tiles = results[i]
                else:
                    img = cv2.imdecode(np.frombuffer(jpg, np.uint8),
                                       cv2.IMREAD_COLOR)
                    anon_frame, tiles = self._camera.make_protected(img)
                last_chunk, frame_id, chunk_start_time = self._roll_chunk(
                    last_chunk, frame_id, frames_per_chunk, chunk_start_time, ts)
                frame_id += 1
                self._save_snapshot(last_chunk, frame_id, anon_frame, tiles, ts)
                save_count += 1

            if time.time() - t_report >= 5.0:
                fps = save_count / (time.time() - t_report)
                print(f"[Recorder] 저장 {fps:.1f}fps (원격 GPU 배치 {batch_size}) "
                      f"대기큐 {self._camera.pending_size()}장")
                save_count = 0
                t_report = time.time()

    def _mark_complete(self, chunk_path: str):
        """청크의 10분이 끝나 다음 청크로 넘어갈 때 완료 표시 (복원 허용)."""
        mpath = os.path.join(chunk_path, "manifest.json")
        m = _load_json(mpath) or {}
        m["complete"] = True
        _save_json(mpath, m)
        print(f"[Recorder] 청크 완료: {_path_to_id(chunk_path)}")

    @staticmethod
    def _chunk_end_ts(chunk_id: str) -> float | None:
        """청크의 끝 시각(epoch). chunk_id에서 파싱."""
        import config as c
        from datetime import timedelta
        try:
            parts = chunk_id.split("__")
            year, month = (int(x) for x in parts[0].split("-"))
            day = int(parts[1])
            hms = parts[4].split("-")   # "14-00-20"
            hh = int(hms[0]); mm = int(hms[1]); ss = int(hms[2]) if len(hms) > 2 else 0
            start = datetime(year, month, day, hh, mm, ss)
            end = start + timedelta(seconds=getattr(c, "CHUNK_SECONDS", 60))
            return end.timestamp()
        except Exception:
            return None

    def _is_complete(self, chunk_id: str) -> bool:
        """
        청크가 완료됐는지 판정 (프레임 수 기준 청크).
        청크가 목표 프레임 수를 다 채워 닫히면 _mark_complete로 manifest에
        complete=True가 기록된다. 진행 중인 마지막 청크만 미완성으로 남는다.
        보조: manifest의 frame_count가 목표치 이상이면 완료로 간주.
        """
        import config as c
        mpath = os.path.join(_id_to_path(chunk_id), "manifest.json")
        m = _load_json(mpath) or {}
        if m.get("complete"):
            return True
        target = getattr(c, "FRAMES_PER_CHUNK", 0) or int(
            getattr(c, "CHUNK_SECONDS", 20) * getattr(c, "SAVE_FPS", 15)
        )
        return int(m.get("frame_count", 0)) >= target

    # ── 공개 API ──────────────────────────────────────────────────────────

    def list_chunks(self) -> list[dict]:
        """
        RAM 디스크와 SD 카드를 모두 스캔하여 중복 없이 하나의 통합 청크 목록을 반환.
        """
        merged_chunks = {}
        ram_base = getattr(c, "RECORD_RAM_DIR", "recordings")
        sd_base = getattr(c, "RECORD_SD_DIR", "recordings")

        # 헬퍼 함수: 특정 디렉토리를 긁어 중복 제거하며 딕셔너리에 추가
        def scan_directory(base_dir: str):
            if not os.path.exists(base_dir):
                return
            
            # 계층 폴더 스캔 시작
            for month_dir in sorted(os.listdir(base_dir), reverse=True):
                m_path = os.path.join(base_dir, month_dir)
                if not os.path.isdir(m_path): continue
                
                for day_dir in sorted(os.listdir(m_path), reverse=True):
                    d_path = os.path.join(m_path, day_dir)
                    if not os.path.isdir(d_path): continue

                    for ampm in ["오후", "오전"]:
                        a_path = os.path.join(d_path, ampm)
                        if not os.path.exists(a_path): continue
                        
                        for hour_dir in sorted(os.listdir(a_path), reverse=True):
                            h_path = os.path.join(a_path, hour_dir)
                            if not os.path.isdir(h_path): continue
                            
                            for chunk_name in sorted(os.listdir(h_path), reverse=True):
                                c_path = os.path.join(h_path, chunk_name)
                                mpath = os.path.join(c_path, "manifest.json")
                                if os.path.exists(mpath):
                                    m = _load_json(mpath) or {}
                                    cid = _path_to_id(c_path)
                                    m["chunk_id"] = cid
                                    m["has_thumb"] = _first_frame_jpg(c_path) is not None
                                    m["complete"] = m.get("complete", False) or self._is_complete(cid)
                                    
                                    # 이미 딕셔너리에 동일한 cid가 등록되어 있다면 건너뛰거나 덮어쓰기
                                    # (RAM 스캔을 나중에 돌릴 것이므로 RAM 데이터가 자연스럽게 우선 적용됩니다)
                                    merged_chunks[cid] = m

        # 1단계: 영구 보관용 SD 카드 먼저 스캔 (과거 데이터 로드)
        if ram_base != sd_base:
            scan_directory(sd_base)
            
        # 2단계: 실시간 RAM 디스크 스캔 (오늘 데이터로 덮어쓰기 및 추가)
        scan_directory(ram_base)

        # 결과 리스트 변환 및 최신순 정렬
        result = list(merged_chunks.values())
        result.sort(key=lambda x: x.get("chunk_id", ""), reverse=True)
        
        # 넉넉하게 최근 200개 반환 (성능을 조율하기 위해 필요에 따라 조절)
        return result

    def get_chunk_detail(self, chunk_id: str) -> dict | None:
        path = _id_to_path(chunk_id)
        mpath = os.path.join(path, "manifest.json")
        if not os.path.exists(mpath):
            return None
        m = _load_json(mpath) or {}
        m["chunk_id"] = chunk_id
        m["complete"] = m.get("complete", False) or self._is_complete(chunk_id)
        frames = []
        if os.path.isdir(path):
            for fname in sorted(os.listdir(path)):
                fdir = os.path.join(path, fname)
                if not (os.path.isdir(fdir) and fname.isdigit()):
                    continue
                files = os.listdir(fdir)
                npys = [f for f in files if f.endswith(".npy")]
                frames.append({
                    "frame_id": fname,
                    "face_count": len(npys),
                    "has_faces": len(npys) > 0,
                })
        m["frames"] = frames
        return m

    def get_thumb_jpeg(self, chunk_id: str) -> bytes | None:
        path = _id_to_path(chunk_id)
        jpg = _first_frame_jpg(path)
        if jpg is None:
            return None
        with open(jpg, "rb") as f:
            return f.read()

    def get_frame_jpeg(self, chunk_id: str, frame_id: str) -> bytes | None:
        p = os.path.join(_id_to_path(chunk_id), frame_id, "frame.jpg")
        if not os.path.exists(p):
            return None
        with open(p, "rb") as f:
            return f.read()

    def restore_chunk_video(self, chunk_id: str, password: str) -> str | None:
        """
        청크 내 모든 프레임을 복원해 mp4로 합쳐 경로 반환.
        원본 이미지는 사용하지 않고 저장된 tile_f32(보호본)에서 복원.
        """
        import config as c
        from core.anonymizer import INNAnonymizer

        t_restore = time.time()   # 💡 복원 시간 측정 시작
        path = _id_to_path(chunk_id)
        if not os.path.isdir(path):
            return None
        frame_dirs = sorted(d for d in os.listdir(path) if d.isdigit())
        if not frame_dirs:
            return None
        print(f"⏱  [복원 시작] {chunk_id}: {len(frame_dirs)}프레임")

        anon = None
        if c.INN_CHECKPOINT is not None:
            anon = INNAnonymizer(checkpoint_path=c.INN_CHECKPOINT)

        # 출력 재생 fps (부드러움용). 실제 시간 길이는 타임스탬프로 맞춤.
        out_fps = max(1, getattr(c, "RESTORE_VIDEO_FPS", 10))
        raw_path = os.path.join(path, "restored_raw.mp4")  # mp4v (브라우저 비호환)
        out_path = os.path.join(path, "restored.mp4")       # H.264 (브라우저 호환)
        writer = None

        # 각 프레임의 촬영 시각(ts) 수집 → 프레임 간 실제 간격 계산
        ts_list = []
        for fid in frame_dirs:
            meta = _load_json(os.path.join(path, fid, "meta.json"))
            ts_list.append(meta.get("ts", 0.0) if meta else 0.0)

        for idx, fid in enumerate(frame_dirs):
            snap = os.path.join(path, fid)
            frame = cv2.imread(os.path.join(snap, "frame.jpg"))
            if frame is None:
                continue

            if anon is not None:
                for i in range(20):
                    npy_path = os.path.join(snap, f"face_{i}.npy")
                    box_path = os.path.join(snap, f"face_{i}_box.json")
                    if not os.path.exists(npy_path):
                        break
                    try:
                        tile_f32 = np.load(npy_path)  # 깨진 파일이면 예외
                        crop_box = _load_json(box_path)
                        frame = anon.restore_roi(frame, tile_f32, crop_box, password)
                    except Exception as e:
                        print(f"[RestoreVideo] {fid} face_{i} 건너뜀: {e}")
                        continue
            else:
                cv2.putText(frame, "INN checkpoint required", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)

            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    raw_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h)
                )

            # 실제 시간 반영: 다음 프레임까지의 간격만큼 이 화면을 유지(hold)
            if idx < len(frame_dirs) - 1 and ts_list[idx] > 0 and ts_list[idx + 1] > 0:
                dur = ts_list[idx + 1] - ts_list[idx]
            else:
                chunk_sec = getattr(c, "CHUNK_SECONDS", 600)
                if len(ts_list) > 0 and ts_list[0] > 0:
                    elapsed = ts_list[-1] - ts_list[0]
                    dur = max(1.0 / out_fps, chunk_sec - elapsed)
                else:
                    dur = 1.0 / out_fps
            
            hold = max(1, round(dur * out_fps))
            for _ in range(hold):
                writer.write(frame)

        if writer is None:
            return None
        writer.release()

        # 💡 복원 소요 시간 출력
        elapsed = time.time() - t_restore
        n = len(frame_dirs)
        print(f"✅ [복원완료] {chunk_id}: {n}프레임 {elapsed:.1f}초 "
              f"({elapsed / max(1, n) * 1000:.0f}ms/frame)")

        # ffmpeg으로 H.264 + yuv420p 변환 (브라우저 재생 호환)
        import shutil
        import subprocess
        ffmpeg = getattr(c, "FFMPEG_PATH", None)
        if not ffmpeg or not os.path.exists(ffmpeg):
            ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            try:
                subprocess.run(
                    [ffmpeg, "-y", "-i", raw_path,
                     "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                     "-movflags", "+faststart", out_path],
                    check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                print(f"[RestoreVideo] {chunk_id} H.264 변환 완료: {out_path}")
                return out_path
            except Exception as e:
                print(f"[RestoreVideo] ffmpeg 변환 실패 → mp4v 반환: {e}")
                return raw_path
        print("[RestoreVideo] ffmpeg 없음 → mp4v 반환 (브라우저 재생 안 될 수 있음)")
        return raw_path

    def restore_frame(
        self, chunk_id: str, frame_id: str, password: str
    ) -> bytes | None:
        """INN 역변환으로 익명화 얼굴 복원 → JPEG bytes 반환."""
        import config as c
        from core.anonymizer import INNAnonymizer

        t0 = time.time()   # 💡 프레임 복원 시간 측정
        snap_dir = os.path.join(_id_to_path(chunk_id), frame_id)
        frame_path = os.path.join(snap_dir, "frame.jpg")
        if not os.path.exists(frame_path):
            return None

        frame = cv2.imread(frame_path)
        if frame is None:
            return None

        if c.INN_CHECKPOINT is None:
            # 체크포인트 없음 → 원본 익명화 프레임 그대로 반환 + 워터마크
            cv2.putText(frame, "INN checkpoint required for restoration",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
            ok, buf = cv2.imencode(".jpg", frame)
            return buf.tobytes() if ok else None

        anon = INNAnonymizer(checkpoint_path=c.INN_CHECKPOINT)
        for i in range(20):
            npy_path = os.path.join(snap_dir, f"face_{i}.npy")
            box_path = os.path.join(snap_dir, f"face_{i}_box.json")
            if not os.path.exists(npy_path):
                break
            tile_f32 = np.load(npy_path)
            crop_box = _load_json(box_path)
            try:
                frame = anon.restore_roi(frame, tile_f32, crop_box, password)
            except Exception as e:
                print(f"[Restore] face_{i} 복원 실패: {e}")

        print(f"⏱  [복원] 프레임 {frame_id}: {(time.time() - t0) * 1000:.0f}ms")
        ok, buf = cv2.imencode(".jpg", frame)
        return buf.tobytes() if ok else None
