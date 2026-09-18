from __future__ import annotations
from pathlib import Path
import os
import json
import time
from typing import Any
import numpy as np
from utils.aes_crypto import encrypt_and_remove_to
import config as c


class RawRecorder:
    """
    원본 프레임을 CHUNK_SECONDS 단위의 raw 파일로 저장하고,
    시작 시각( HH:MM:SS ) 폴더별로 각 청크를 독립적으로 AES-256 암호화한다.
    """

    def __init__(
        self,
        save_root: str | Path,
        fps: float,
        width: int,
        height: int,
        channels: int = 3,
        dtype: str = "uint8",
    ):
        
        from datetime import datetime, timedelta

        self.fps = float(fps)
        self.width = int(width)
        self.height = int(height)
        self.channels = int(channels)
        self.dtype = np.dtype(dtype)
        self.save_root = Path(save_root).resolve()

        self._file = None
        self._closed = False

        self.chunk_seconds = max(1, int(getattr(c, "CHUNK_SECONDS", 20)))
        self.frames_per_chunk = max(1, round(self.fps * self.chunk_seconds))
        self.recording_started_at = datetime.now()
        self.chunk_sequence = 0
        self.chunk_frame_count = 0
        self.chunk_started_at = None
        self.chunk_id = None
        self.chunk_dir = None
        self.chunk_bin = None
        self.chunk_meta = None
        self._open_chunk()

    def _open_chunk(self):
        """청크 시작 시각을 기준으로 새 raw 파일을 연다."""
        if self._file and not self._file.closed:
            self._file.close()

        from datetime import timedelta
        self.chunk_started_at = self.recording_started_at + timedelta(
            seconds=self.chunk_sequence * self.chunk_seconds
        )
        self.chunk_id = self.chunk_started_at.strftime("%H:%M:%S")
        date_dir = self.chunk_started_at.strftime("%Y-%m-%d")
        hour_dir = self.chunk_started_at.strftime("%H")
        base_folder = self.save_root / date_dir / hour_dir
        self.chunk_dir = base_folder / self.chunk_id
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_bin = self.chunk_dir / f"{self.chunk_id}.bin"
        self.chunk_meta = self.chunk_dir / f"{self.chunk_id}.json"
        self.path = self.chunk_bin
        self.meta_path = self.chunk_meta
        self._file = self.chunk_bin.open("wb")
        self.chunk_frame_count = 0
        self._write_chunk_metadata(False)
        print("[AES] 청크 생성 시작")
        print(f"시작 시각: {self.chunk_id}")
        print(f"청크 길이: {self.chunk_seconds // 60}분")

    def _write_chunk_metadata(self, complete: bool):
        data = {
            "chunk_id": self.chunk_id,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "dtype": str(self.dtype),
            "format": "raw",
            "chunk_seconds": self.chunk_seconds,
            "target_frame_count": self.frames_per_chunk,
            "frame_count": self.chunk_frame_count,
            "complete": bool(complete),
            "updated_at": time.time(),
        }
        tmp = self.chunk_meta.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self.chunk_meta)

    def _finish_chunk(self, complete: bool):
        if self._file is None or self._file.closed:
            return
        from datetime import timedelta
        actual_end = self.chunk_started_at + timedelta(
            seconds=(self.chunk_seconds if complete else
                     self.chunk_frame_count / max(self.fps, 1e-9))
        )
        range_text = f"{self.chunk_id} ~ {actual_end.strftime('%H:%M:%S')}"
        save_started = time.perf_counter()
        try:
            self._file.flush()
            self._file.close()
            # 암호화가 끝나기 전에는 완료로 표시하지 않는다.
            self._write_chunk_metadata(False)
            encrypted = self.chunk_dir / f"{self.chunk_id}.enc"
            encrypt_and_remove_to(str(self.chunk_bin), str(encrypted))
            if complete:
                self._write_chunk_metadata(True)
            elapsed = time.perf_counter() - save_started
            print("[AES] 청크 저장 완료")
            print(f"시간 범위: {range_text}")
            print(f"저장 소요시간: {elapsed:.2f}초")
            print("[AES] 저장 완료")
        except Exception as exc:
            print("[AES] 청크 저장 실패")
            print(f"시간 범위: {range_text}")
            print(f"오류: {exc}")
            raise
        finally:
            self._file = None

    def _rotate_chunk(self):
        self._finish_chunk(True)
        self.chunk_sequence += 1

    def _write_metadata(self):

        metadata = {
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "dtype": str(self.dtype),
            "format": "raw",
        }

        with self.meta_path.open(
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(metadata, f, indent=4)

    def write(self, frame: Any):

        if self._closed:
            raise RuntimeError("Recorder already closed.")

        arr = np.asarray(frame)

        if arr.dtype != self.dtype:
            arr = arr.astype(self.dtype)

        expected = (
            (self.height, self.width)
            if self.channels == 1
            else (self.height, self.width, self.channels)
        )

        if arr.shape != expected:
            raise ValueError(
                f"Expected {expected}, got {arr.shape}"
            )

        if self._file is None:
            self._open_chunk()

        self._file.write(arr.tobytes())
        self.chunk_frame_count += 1
        if self.chunk_frame_count >= self.frames_per_chunk:
            self._rotate_chunk()
        elif self.chunk_frame_count % max(1, round(self.fps)) == 0:
            self._write_chunk_metadata(False)

    def close(self):

        if self._closed:
            return

        # 마지막 청크는 20초보다 짧아도 complete=false로 보존한다.
        self._finish_chunk(False)

        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self,
                 exc_type,
                 exc_val,
                 exc_tb):
        self.close()

    @staticmethod
    def find_chunk_file(chunk_id: str):
        root = Path(c.RAW_ENCRYPT_DIR)

        # 암호화 파일 검색
        for file in root.rglob(f"{chunk_id}.enc"):
            return file

        # 혹시 암호화 전 bin이 남아있는 경우
        for file in root.rglob(f"{chunk_id}.bin"):
            return file

        return None
