import os
import numpy as np
from hailo_sdk_client import ClientRunner

# 1. 경로 자동 탐색 (현재 파일 위치를 기준으로 한 단계 상위 폴더를 루트로 잡음)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 2. 안전하게 절대 경로로 결합
onnx_path = os.path.join(parent_dir, "checkpoints", "inn_protect_sim.onnx")
hef_path = os.path.join(parent_dir, "checkpoints", "inn_protect.hef")

TARGET_HW = "hailo8l" 

def compile_model():
    print(f"\n➔ ONNX 파일 찾는 중: {onnx_path}")
    print(f"1. [{TARGET_HW}] 모델 파싱 및 Hailo Runner 초기화 중...")
    runner = ClientRunner(hw_arch=TARGET_HW)
    
    try:
        runner.translate_onnx_model(
            onnx_path, 
            "inn_protect"
        )
        print("✅ ONNX 파싱 성공!")
    except Exception as e:
        print(f"❌ 파싱 에러:\n{e}")
        return

    print("2. 8-bit 양자화를 위한 더미 데이터 생성 중...")
    def calib_dataset():
        for _ in range(10):
            yield {
                'xa': np.random.rand(1, 3, 256, 256).astype(np.float32),
                'xa_obfs': np.random.rand(1, 3, 256, 256).astype(np.float32),
                'skey_dwt': np.random.rand(1, 12, 128, 128).astype(np.float32)
            }

    print("3. NPU 최적화 및 양자화 진행 중 (CPU 모드라 시간이 꽤 걸립니다)...")
    runner.optimize(calib_dataset())

    print("4. 최종 .hef 파일 컴파일 중...")
    hef = runner.compile()
    
    with open(hef_path, "wb") as f:
        f.write(hef)
        
    print(f"🎉 성공적으로 컴파일되었습니다!\n ➔ 저장된 위치: {hef_path}")

if __name__ == "__main__":
    compile_model()