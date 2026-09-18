from core.camera_manager import CameraManager
import cv2
import time

print("=" * 50)
print("노출/밝기 진단 — 공통 V4L2 자동 선택")
print("OpenCV:", cv2.__version__)
print("=" * 50)


def test(label, setup_fn, warmup=60):
    cap = CameraManager().open()
    if not cap.isOpened():
        print(f"[{label}] 열기 실패")
        return
    if setup_fn:
        setup_fn(cap)
    time.sleep(1.0)

    print(f"\n[{label}] warmup {warmup}프레임 동안 brightness 추이:")
    last_b = 0
    for i in range(warmup):
        ret, frame = cap.read()
        if ret and frame is not None:
            last_b = frame.mean()
            if i % 10 == 0 or last_b > 10:
                print(f"   frame {i:02d}: brightness={last_b:.2f}")
            if last_b > 15:
                cv2.imwrite(f"diag_{label}.jpg", frame)
                print(f"   >>> 밝은 프레임! diag_{label}.jpg 저장, brightness={last_b:.2f}")
                break
        time.sleep(0.03)
    print(f"[{label}] 최종 brightness={last_b:.2f}")
    cap.release()
    time.sleep(0.5)


# 1) 아무 설정 없음 (기본)
test("default", None)

# 2) 자동노출 ON (V4L2: 0.75)
def auto_on(cap):
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
test("autoexp_on", auto_on)

# 3) 수동노출 + 노출값 크게 (밝게)
def manual_bright(cap):
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 수동 모드
    cap.set(cv2.CAP_PROP_EXPOSURE, -4)         # V4L2 노출 (음수=밝게)
test("manual_exp-4", manual_bright)

# 4) 노출값 더 크게
def manual_bright2(cap):
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -2)
test("manual_exp-2", manual_bright2)

# 5) 밝기 속성 직접 올림
def raise_brightness(cap):
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 200)
    cap.set(cv2.CAP_PROP_GAIN, 200)
test("brightness_gain", raise_brightness)

print("\n" + "=" * 50)
print("어느 설정에서 brightness가 올라갔는지 확인하세요.")
print("=" * 50)
