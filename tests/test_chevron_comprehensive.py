"""验证 chevron 检测在所有已采集截图上的准确性。"""
import cv2
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.tasks.base_ui.auto_enhancement_support_card import _detect_chevron_count


CAPTURE_DIR = "logs/debug/test_captures/support_card"


def test_button_crop():
    """Test on individual button crops from /tmp."""
    print("=== Button crop tests ===")
    expected = {0: 1, 1: 2, 2: 0, 3: 0}
    all_pass = True
    for i, exp in expected.items():
        path = f"/tmp/btn_{i}.png"
        frame = cv2.imread(path)
        if frame is None:
            print(f"  Btn {i}: SKIP (file not found)")
            continue
        result = _detect_chevron_count(frame)
        status = "PASS" if result == exp else "FAIL"
        if result != exp:
            all_pass = False
        print(f"  Btn {i}: expected={exp} got={result} [{status}]")
    return all_pass


def test_synthetic():
    """Generate synthetic chevron images to test edge cases."""
    print("\n=== Synthetic tests ===")
    all_pass = True

    # Create a synthetic > button (white bg + dark V shape)
    img = np.ones((80, 80, 3), dtype=np.uint8) * 230
    pts = np.array([[25, 15], [55, 40], [25, 65]], dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=False, color=(80, 80, 80), thickness=5)
    result = _detect_chevron_count(img)
    status = "PASS" if result == 1 else "FAIL"
    if result != 1:
        all_pass = False
    print(f"  Synthetic >: expected=1 got={result} [{status}]")

    # Create a synthetic >> button
    img2 = np.ones((80, 80, 3), dtype=np.uint8) * 230
    pts1 = np.array([[15, 15], [35, 40], [15, 65]], dtype=np.int32)
    pts2 = np.array([[40, 15], [60, 40], [40, 65]], dtype=np.int32)
    cv2.polylines(img2, [pts1], isClosed=False, color=(80, 80, 80), thickness=4)
    cv2.polylines(img2, [pts2], isClosed=False, color=(80, 80, 80), thickness=4)
    result = _detect_chevron_count(img2)
    status = "PASS" if result == 2 else "FAIL"
    if result != 2:
        all_pass = False
    print(f"  Synthetic >>: expected=2 got={result} [{status}]")

    # Create a plain text button (wide rectangle with text-like noise)
    img3 = np.ones((100, 350, 3), dtype=np.uint8) * 200
    cv2.putText(img3, "Cancel", (50, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (60, 60, 60), 3)
    result = _detect_chevron_count(img3)
    status = "PASS" if result == 0 else "FAIL"
    if result != 0:
        all_pass = False
    print(f"  Synthetic text button: expected=0 got={result} [{status}]")

    # Empty/solid button
    img4 = np.ones((80, 80, 3), dtype=np.uint8) * 180
    result = _detect_chevron_count(img4)
    status = "PASS" if result == 0 else "FAIL"
    if result != 0:
        all_pass = False
    print(f"  Solid color button: expected=0 got={result} [{status}]")

    return all_pass


def test_noise_tolerance():
    """Test with JPEG-compressed and noise-added versions."""
    print("\n=== Noise/compression tolerance tests ===")
    all_pass = True

    for i, expected in [(0, 1), (1, 2)]:
        path = f"/tmp/btn_{i}.png"
        frame = cv2.imread(path)
        if frame is None:
            continue

        # JPEG compression artifact simulation
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
        _, encoded = cv2.imencode('.jpg', frame, encode_param)
        jpeg_frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        result = _detect_chevron_count(jpeg_frame)
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_pass = False
        print(f"  Btn {i} JPEG Q30: expected={expected} got={result} [{status}]")

        # Gaussian noise
        noise = np.random.normal(0, 15, frame.shape).astype(np.int16)
        noisy = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        result = _detect_chevron_count(noisy)
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_pass = False
        print(f"  Btn {i} Gaussian noise: expected={expected} got={result} [{status}]")

        # Scale (resize 0.7x and 1.5x)
        for scale, label in [(0.7, "0.7x"), (1.5, "1.5x")]:
            resized = cv2.resize(frame, None, fx=scale, fy=scale)
            result = _detect_chevron_count(resized)
            status = "PASS" if result == expected else "FAIL"
            if result != expected:
                all_pass = False
            print(f"  Btn {i} scale {label}: expected={expected} got={result} [{status}]")

    return all_pass


if __name__ == "__main__":
    results = []
    results.append(("Button crops", test_button_crop()))
    results.append(("Synthetic", test_synthetic()))
    results.append(("Noise tolerance", test_noise_tolerance()))

    print("\n" + "=" * 40)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {name}: {status}")
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
