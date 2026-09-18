"""
PSF 컨테이너 — Protected-Secure-Face 포맷 (SRS §6)
무손실 저장 강제 + 메타데이터 포함 (FR-8, FR-9, 함정 ②③④ 해결)

컨테이너 구조:
    <name>.psf/            (또는 <name>.psf.zip)
    ├── protected.png      보호본 ŷ가 합성된 전체 프레임 (PNG, 무손실)
    └── manifest.json      메타데이터

manifest.json 핵심 필드:
    schema      : "psf/0.1"
    model       : 모델 설정 (n_blocks, wrong_recover_type, ...)
    faces[]     : 얼굴별 bbox + align_params + obfuscator 정보
    integrity   : frame_sha256
    secrets_stored: false  ← 비밀번호/K/부산물 미저장 (FR-8.3)
"""

import hashlib
import json
import os
import shutil
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import config


# ── 데이터 클래스 ──────────────────────────────────────────────────

@dataclass
class FaceMeta:
    id: int
    bbox: list[int]                    # [x, y, w, h] in original frame
    crop_box: list[int]                # [x1, y1, x2, y2] 실제 크롭 영역
    scale: float                       # crop_size → NORM_RESOLUTION
    obfuscator: dict[str, Any] = field(default_factory=dict)  # type, params
    tile_sha256: str = ""              # 256×256 보호 타일 무결성 (있으면)


@dataclass
class ModelMeta:
    name: str = "PRO-Face S"
    n_blocks: int = config.INV_BLOCKS
    secret_key_as_noise: bool = config.SECRET_KEY_AS_NOISE
    wrong_recover_type: str = config.WRONG_RECOVER_TYPE
    norm_resolution: int = config.NORM_RESOLUTION
    checkpoint_id: str = config.CHECKPOINT_ID


@dataclass
class Manifest:
    schema: str = "psf/0.1"
    model: ModelMeta = field(default_factory=ModelMeta)
    faces: list[FaceMeta] = field(default_factory=list)
    integrity: dict[str, str] = field(default_factory=dict)
    secrets_stored: bool = False       # 항상 False (FR-8.3)


# ── SHA-256 ────────────────────────────────────────────────────────

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


# ── 저장 ───────────────────────────────────────────────────────────

def save_psf(
    protected_frame: np.ndarray,    # HWC BGR uint8 (보호본이 합성된 전체 프레임)
    faces: list[FaceMeta],
    out_path: str | Path,
    model_meta: ModelMeta | None = None,
    as_zip: bool = False,
    face_tiles: list[np.ndarray] | None = None,  # 각 얼굴의 256×256 보호 타일
) -> Path:
    """
    PSF 컨테이너를 저장한다.
    out_path   : '.psf' 디렉터리 경로 (확장자 포함)
    as_zip     : True 이면 .psf.zip으로 압축
    face_tiles : 각 얼굴의 256×256 보호 타일 (이중 리사이즈 방지)
                 제공되면 복원 시 full-frame 크롭 대신 타일을 직접 사용
    """
    out_path = Path(out_path)
    if out_path.exists():
        shutil.rmtree(out_path) if out_path.is_dir() else out_path.unlink()
    out_path.mkdir(parents=True)

    # 1. 보호본 PNG 무손실 저장 (FR-8.1) — 시각화·전체 프레임 보관용
    img_path = out_path / "protected.png"
    success = cv2.imwrite(str(img_path), protected_frame)
    if not success:
        raise IOError(f"PNG 저장 실패: {img_path}")

    # 2. 256×256 타일 저장 (복원 시 이중 리사이즈 방지)
    #    ① uint8 PNG  — 시각화용 (보호 영역 미리보기)
    #    ② float32 npy — 정밀 복원용 (양자화 손실 제거)
    if face_tiles:
        for i, (tile, face) in enumerate(zip(face_tiles, faces)):
            # 시각화용 uint8 PNG
            tile_png = out_path / f"face_{i}_tile.png"
            cv2.imwrite(str(tile_png), tile)
            # 복원용 float32: tile이 이미 float32 (CHW) 또는 uint8 (HWC)
            tile_npy = out_path / f"face_{i}_tile.npy"
            if isinstance(tile, np.ndarray) and tile.dtype == np.float32:
                np.save(str(tile_npy), tile)   # CHW float32 [-1,1]
            else:
                # uint8 BGR HWC → float32 [-1,1] CHW
                import cv2 as _cv2
                rgb = _cv2.cvtColor(tile, _cv2.COLOR_BGR2RGB)
                f32 = (rgb.astype(np.float32) / 127.5) - 1.0    # [0,255]→[-1,1]
                np.save(str(tile_npy), f32.transpose(2, 0, 1))   # CHW
            face.tile_sha256 = _sha256_file(tile_npy)

    # 3. manifest.json 생성
    manifest = Manifest(
        model=model_meta or ModelMeta(),
        faces=faces,
        integrity={"frame_sha256": _sha256_file(img_path)},
    )
    manifest_path = out_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(_manifest_to_dict(manifest), fp, indent=2, ensure_ascii=False)

    if as_zip:
        zip_path = out_path.with_suffix(".psf.zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in out_path.iterdir():
                zf.write(f, f.name)
        shutil.rmtree(out_path)
        return zip_path

    return out_path


# ── 로드 ───────────────────────────────────────────────────────────

def load_psf(
    psf_path: str | Path,
) -> tuple[np.ndarray, Manifest, dict[int, np.ndarray]]:
    """
    PSF 컨테이너를 읽는다.
    반환: (protected_frame HWC BGR uint8, Manifest, face_tiles)
      face_tiles: {face_id: 256×256 BGR tile} — 저장된 경우에만 채워짐
    무결성 검증 실패 시 RuntimeError (NFR-REL-2)
    """
    psf_path = Path(psf_path)

    # zip 압축 해제
    if psf_path.suffix == ".zip":
        tmp_dir = psf_path.with_suffix("").with_suffix("")
        with zipfile.ZipFile(psf_path) as zf:
            zf.extractall(tmp_dir)
        psf_path = tmp_dir

    img_path = psf_path / "protected.png"
    manifest_path = psf_path / "manifest.json"

    if not img_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"PSF 컨테이너 손상: {psf_path}")

    with open(manifest_path, "r", encoding="utf-8") as fp:
        raw = json.load(fp)

    manifest = _dict_to_manifest(raw)

    # 무결성 검증 (FR-9.2)
    actual_hash = _sha256_file(img_path)
    expected_hash = manifest.integrity.get("frame_sha256", "")
    if expected_hash and actual_hash != expected_hash:
        raise RuntimeError(
            f"[PSF] 무결성 검증 실패!\n"
            f"  기대: {expected_hash}\n"
            f"  실제: {actual_hash}\n"
            "파일이 손상되었거나 변조되었습니다."
        )

    frame = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise IOError(f"이미지 로드 실패: {img_path}")

    # 256×256 타일 로드 (저장된 경우)
    # float32 .npy 우선 로드 — 없으면 uint8 PNG 폴백
    face_tiles: dict[int, np.ndarray] = {}
    for face in manifest.faces:
        npy_path = psf_path / f"face_{face.id}_tile.npy"
        png_path = psf_path / f"face_{face.id}_tile.png"

        if npy_path.exists():
            # 무결성 검증 (float32 npy)
            if face.tile_sha256:
                tile_hash = _sha256_file(npy_path)
                if tile_hash != face.tile_sha256:
                    raise RuntimeError(
                        f"[PSF] 타일 무결성 실패 (face_{face.id}): "
                        f"기대={face.tile_sha256[:16]}... 실제={tile_hash[:16]}..."
                    )
            tile = np.load(str(npy_path))   # CHW float32 [-1,1]
            face_tiles[face.id] = tile
        elif png_path.exists():
            tile = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
            if tile is not None:
                face_tiles[face.id] = tile  # HWC uint8 BGR (폴백)

    return frame, manifest, face_tiles


# ── 직렬화 헬퍼 ───────────────────────────────────────────────────

def _manifest_to_dict(m: Manifest) -> dict:
    return {
        "schema": m.schema,
        "model": asdict(m.model),
        "faces": [asdict(f) for f in m.faces],
        "integrity": m.integrity,
        "secrets_stored": m.secrets_stored,
    }


def _dict_to_manifest(d: dict) -> Manifest:
    faces = [
        FaceMeta(**{k: v for k, v in f.items()})
        for f in d.get("faces", [])
    ]
    model_d = d.get("model", {})
    model = ModelMeta(**{k: v for k, v in model_d.items() if k in ModelMeta.__dataclass_fields__})
    return Manifest(
        schema=d.get("schema", "psf/0.1"),
        model=model,
        faces=faces,
        integrity=d.get("integrity", {}),
        secrets_stored=d.get("secrets_stored", False),
    )
