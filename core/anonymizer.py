"""
INN 익명화 래퍼 — SCRFD가 미리 탐지한 bbox를 받아 처리.
자체 탐지기를 실행하지 않음 (detect_realsys.py의 SCRFD가 탐지 담당).
"""
import warnings

import numpy as np
import torch

import config as c
from models.embedder import ModelDWT, init_model
from models.modules import DWT
from utils.key_gen import generate_key, make_key_rec
from utils.image_processing import Obfuscator, to_tensor, to_numpy
from detection.yolo_detector import expand_bbox_square, crop_and_resize, paste_back


class INNAnonymizer:
    def __init__(self, checkpoint_path=None, device=None, obf_type="blur"):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.embedder = ModelDWT(n_blocks=c.INV_BLOCKS).to(self.device)
        self.embedder.eval()

        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        else:
            init_model(self.embedder, self.device)

        self.dwt = DWT().to(self.device)
        self.obfuscator = Obfuscator(obf_type=obf_type)
        print(f"[INNAnonymizer] 준비 완료 (device={self.device})")

    def _load_checkpoint(self, path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(state, dict):
            state = state.get("state_dict", state.get("model", state))
        if state and all(k.startswith("module.") for k in state):
            state = {k[7:]: v for k, v in state.items()}
        missing, _ = self.embedder.load_state_dict(state, strict=False)
        if missing:
            warnings.warn(f"[INNAnonymizer] missing keys: {missing[:3]}")
        print(f"[INNAnonymizer] 가중치 로드: {path}")

    @torch.no_grad()
    def protect_roi(self, frame: np.ndarray, bbox: list, password) -> tuple:
        """
        SCRFD가 탐지한 단일 얼굴 ROI에 INN 익명화 적용.

        Args:
            frame:    HWC BGR uint8
            bbox:     [x1, y1, x2, y2] (SCRFD 출력)
            password: str or int

        Returns:
            modified_frame: HWC BGR uint8 (익명화 합성 완료)
            tile_f32:       (3,256,256) float32 [-1,1]  — PSF 저장용
            crop_box:       [x1,y1,x2,y2] — 복원 시 재사용
        """
        H, W = frame.shape[:2]
        crop_box = expand_bbox_square(list(bbox), H, W)
        face_np, _ = crop_and_resize(frame, crop_box, c.NORM_RESOLUTION)

        xa = to_tensor(face_np, device=self.device)
        xa_obfs = self.obfuscator(xa)

        skey = generate_key(
            password, bs=1, w=c.NORM_RESOLUTION, h=c.NORM_RESOLUTION
        ).to(self.device)
        skey_dwt = self.dwt(skey.float())

        xa_out_z, xa_proc = self.embedder(xa, xa_obfs, skey_dwt, rev=False)
        del xa_out_z

        ya_hat_np = to_numpy(xa_proc.cpu())
        modified_frame = paste_back(frame, ya_hat_np, crop_box)
        tile_f32 = xa_proc.cpu().squeeze(0).numpy()

        return modified_frame, tile_f32, crop_box

    @torch.no_grad()
    def restore_roi(
        self, frame: np.ndarray, tile_f32: np.ndarray, crop_box: list, password
    ) -> np.ndarray:
        """
        float32 타일에서 단일 얼굴 원본 복원.

        Args:
            frame:    HWC BGR uint8 (익명화된 배경 프레임)
            tile_f32: (3,256,256) float32 [-1,1]
            crop_box: [x1,y1,x2,y2]
            password: str or int

        Returns:
            restored_frame: HWC BGR uint8
        """
        norm_res = c.NORM_RESOLUTION
        xa_proc = torch.from_numpy(tile_f32).unsqueeze(0).to(self.device)

        skey = generate_key(
            password, bs=1, w=norm_res, h=norm_res
        ).to(self.device)
        skey_dwt = self.dwt(skey.float())
        key_rec = make_key_rec(skey_dwt)

        xa_rev, _ = self.embedder(key_rec, xa_proc, skey_dwt, rev=True)
        x_rec_np = to_numpy(xa_rev.cpu())
        return paste_back(frame, x_rec_np, crop_box)
