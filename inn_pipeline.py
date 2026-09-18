"""
SecureFace-RX 메인 파이프라인
원본 test_tcsvt.py 의 호출 패턴을 그대로 따름.

보호 호출 패턴 (원본 기준):
    xa_out_z, xa_proc = embedder(xa, xa_obfs, skey_dwt)
    del xa_out_z   # 부산물 폐기 (FR-6)

복원 호출 패턴:
    key_rec = skey_dwt.repeat(1, 3, 1, 1)
    xa_rev, _ = embedder(key_rec, xa_proc, skey_dwt, rev=True)

함정 대응 (SRS §8):
    ① ŷ(=xa_proc)만 저장, 평범한 블러 y는 버림
    ② crop_box 메타 저장 → 복원 시 재검출 없이 재사용
    ③ PNG 무손실 강제
    ④ 메타에서 좌표 로드
    ⑤ 부산물 z 즉시 del
    ⑥ 256×256 정규화
"""

import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

import config as c
from models.embedder import ModelDWT, init_model
from models.modules import DWT
from utils.key_gen import generate_key, make_key_rec
from utils.image_processing import Obfuscator, to_tensor, to_numpy
from utils.container import save_psf, load_psf, FaceMeta, ModelMeta
from detection.yolo_detector import (
    YOLOFaceDetector,
    expand_bbox_square,
    crop_and_resize,
    paste_back,
)


class SecureFaceRX:
    """
    SecureFace-RX 파이프라인 단일 진입점.
    동일 인스턴스로 보호·복원 모두 수행.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: str | None = None,
        obf_type: str = c.DEFAULT_OBFUSCATOR,
        detector_model: str = "yolov8n-face.pt",
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # INN 모델
        self.embedder = ModelDWT(n_blocks=c.INV_BLOCKS).to(self.device)
        self.embedder.eval()

        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        else:
            # 가중치 없을 때: 원본 init_model 방식으로 초기화
            init_model(self.embedder, self.device)

        self.dwt        = DWT().to(self.device)
        self.obfuscator = Obfuscator(obf_type=obf_type)
        self.detector   = YOLOFaceDetector(model_path=detector_model)

    # ── 체크포인트 로드 ────────────────────────────────────────────

    def _load_checkpoint(self, path: str):
        state = torch.load(path, map_location=self.device)
        # state_dict 키 추출
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model" in state:
                state = state["model"]
        # DataParallel의 module. prefix 제거
        if all(k.startswith("module.") for k in state):
            state = {k[len("module."):]: v for k, v in state.items()}
        missing, unexpected = self.embedder.load_state_dict(state, strict=False)
        if missing:
            warnings.warn(f"[체크포인트] missing keys ({len(missing)}): {missing[:3]}")
        if unexpected:
            warnings.warn(f"[체크포인트] unexpected keys ({len(unexpected)}): {unexpected[:3]}")
        print(f"[SecureFace-RX] 가중치 로드: {path}")

    # ── 보호 ──────────────────────────────────────────────────────

    @torch.no_grad()
    def protect_image(
        self,
        frame: np.ndarray,          # HWC BGR uint8
        password,                   # str 또는 int (원본: int 0 사용)
        out_psf: str | Path | None = None,
        as_zip: bool = False,
    ) -> tuple[np.ndarray, Path]:
        """
        frame을 익명화하여 PSF 컨테이너 저장.
        반환: (protected_frame HWC BGR, psf_path)
        """
        H, W = frame.shape[:2]
        detections = self.detector.detect(frame)

        if not detections:
            warnings.warn("[보호] 검출된 얼굴이 없습니다.")
            if out_psf is None:
                out_psf = Path("output.psf")
            return frame.copy(), save_psf(frame, [], Path(out_psf),
                                          model_meta=ModelMeta(), as_zip=as_zip)

        result_frame  = frame.copy()
        faces_meta: list[FaceMeta] = []
        face_tiles:  list[np.ndarray] = []   # 256×256 보호 타일 (이중 리사이즈 방지)

        for idx, det in enumerate(detections):
            # ── FR-2: 정렬·크롭 ────────────────────────────────
            crop_box = expand_bbox_square(det.bbox, H, W)
            face_np, scale = crop_and_resize(frame, crop_box, c.NORM_RESOLUTION)

            # ── FR-3: 사전 난독화 ───────────────────────────────
            xa = to_tensor(face_np, device=self.device)     # (1,3,256,256)
            xa_obfs = self.obfuscator(xa)                    # y = O(x)

            # ── FR-4: KeyGen ────────────────────────────────────
            skey     = generate_key(password, bs=1,
                                    w=c.NORM_RESOLUTION,
                                    h=c.NORM_RESOLUTION).to(self.device)
            skey_dwt = self.dwt(skey.float())               # (1,4,128,128)

            # ── FR-5: 보호 (원본 패턴) ──────────────────────────
            xa_out_z, xa_proc = self.embedder(xa, xa_obfs, skey_dwt, rev=False)

            # ── FR-6: 부산물 즉시 폐기 ─────────────────────────
            del xa_out_z

            # ── FR-7: 역변환 합성 ───────────────────────────────
            ya_hat_np    = to_numpy(xa_proc.cpu())           # 256×256 uint8 BGR (시각화용)
            result_frame = paste_back(result_frame, ya_hat_np, crop_box)
            # 정밀 복원을 위해 float32 CHW [-1,1] 원본 그대로 보관
            ya_hat_f32   = xa_proc.cpu().squeeze(0).numpy()  # (3,256,256) float32 [-1,1]
            face_tiles.append(ya_hat_f32)

            faces_meta.append(FaceMeta(
                id=idx,
                bbox=det.bbox,
                crop_box=crop_box,
                scale=scale,
                obfuscator={
                    "type":   self.obfuscator.obf_type,
                    "kernel": c.BLUR_KERNEL_SIZE,
                    "sigma":  c.BLUR_SIGMA,
                },
            ))

        # ── FR-8: PSF 무손실 저장 ───────────────────────────────
        if out_psf is None:
            out_psf = Path("output.psf")
        psf_path = save_psf(
            result_frame, faces_meta, Path(out_psf),
            model_meta=ModelMeta(), as_zip=as_zip,
            face_tiles=face_tiles,   # 256×256 타일 포함 저장
        )
        return result_frame, psf_path

    # ── 복원 ──────────────────────────────────────────────────────

    @torch.no_grad()
    def restore_image(
        self,
        psf_path: str | Path,
        password,                   # str 또는 int
    ) -> np.ndarray:
        """
        PSF 컨테이너를 열어 올바른 비밀번호로 원본 복원.
        틀린 비밀번호 → 오복원 (PSNR<11dB)
        반환: restored_frame HWC BGR uint8
        """
        # ── FR-9: 메타 로드 + 무결성 검증 ─────────────────────
        protected_frame, manifest, face_tiles = load_psf(psf_path)

        if not manifest.faces:
            warnings.warn("[복원] 얼굴 메타데이터가 없습니다.")
            return protected_frame

        result_frame = protected_frame.copy()
        norm_res = manifest.model.norm_resolution

        for face_meta in manifest.faces:
            crop_box = face_meta.crop_box

            # ── 보호 타일 로드 ────────────────────────────────────
            # float32 CHW 타일이 있으면 직접 텐서 변환 (양자화 손실 없음)
            # 없으면 전체 프레임 크롭+리사이즈 폴백
            if face_meta.id in face_tiles:
                tile = face_tiles[face_meta.id]
                if tile.dtype == np.float32 and tile.ndim == 3 and tile.shape[0] == 3:
                    # (3,H,W) float32 [-1,1] → (1,3,H,W) tensor
                    xa_proc = torch.from_numpy(tile).unsqueeze(0).to(self.device)
                else:
                    # uint8 HWC BGR (폴백)
                    xa_proc = to_tensor(tile, device=self.device)
            else:
                face_np, _ = crop_and_resize(protected_frame, crop_box, norm_res)
                xa_proc = to_tensor(face_np, device=self.device)

            # ── FR-10: 키 재생성 (보호와 동일) ─────────────────
            skey = generate_key(password, bs=1, w=norm_res, h=norm_res).to(self.device)
            skey_dwt = self.dwt(skey.float())

            # ── FR-11: 복원 (원본 패턴) ──────────────────────────
            key_rec = make_key_rec(skey_dwt)                 # (1,12,128,128)
            xa_rev, _ = self.embedder(key_rec, xa_proc, skey_dwt, rev=True)

            # ── 합성 ───────────────────────────────────────────
            x_rec_np     = to_numpy(xa_rev.cpu())
            result_frame = paste_back(result_frame, x_rec_np, crop_box)

        return result_frame

    # ── 영상 파일 처리 (FR-13.2) ──────────────────────────────────

    def protect_video(
        self,
        video_path: str,
        password,
        out_dir: str | Path = "protected_video",
    ) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"영상 열기 실패: {video_path}")
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.protect_image(frame, password,
                                out_psf=out_dir / f"frame_{idx:06d}.psf")
            idx += 1
            if idx % 100 == 0:
                print(f"  보호 완료: {idx} 프레임")
        cap.release()
        print(f"[영상 보호] 총 {idx} 프레임 → {out_dir}")
        return out_dir

    def restore_video(
        self,
        psf_dir: str | Path,
        password,
        out_path: str = "restored.mp4",
        fps: float = 30.0,
    ) -> str:
        psf_dir = Path(psf_dir)
        psf_files = sorted(psf_dir.glob("*.psf"))
        if not psf_files:
            raise FileNotFoundError(f"PSF 파일 없음: {psf_dir}")
        writer = None
        for psf in psf_files:
            frame = self.restore_image(psf, password)
            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            writer.write(frame)
        if writer:
            writer.release()
        print(f"[영상 복원] → {out_path}")
        return out_path
