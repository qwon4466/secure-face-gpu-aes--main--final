# utils 패키지 — 지연 임포트(lazy import)로 선택적 의존성 오류 방지
# 각 서브모듈을 직접 임포트하거나 아래 함수를 사용할 것

def get_key_gen():
    from utils.key_gen import generate_key, make_key_rec
    return generate_key, make_key_rec

def get_obfuscator():
    from utils.image_processing import Obfuscator, to_tensor, to_numpy
    return Obfuscator, to_tensor, to_numpy

def get_container():
    from utils.container import save_psf, load_psf, FaceMeta, ModelMeta, Manifest
    return save_psf, load_psf, FaceMeta, ModelMeta, Manifest
