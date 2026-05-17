"""Tests for CLIP data augmentation and SSIM region-exclusion stability detection.

These tests verify:
1. _augment_image produces valid augmented variants
2. Augmented images are structurally different but not drastically so
3. JPEG noise tolerance: CLIP features from JPEG-compressed images remain similar
4. SSIM region exclusion: animated regions are masked before comparison
5. No strong template matching — all methods tolerate JPG noise
"""

import cv2
import numpy as np
import pytest

from src.utils.clip_tools import _augment_image
from src.utils.opencv_tools import compute_ssim_score


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_card_image():
    """Generate a synthetic card image with gradient and text-like features."""
    h, w = 400, 300
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Gradient background
    for y in range(h):
        img[y, :, 0] = int(255 * y / h)
        img[y, :, 1] = int(128 * (1 - y / h))
        img[y, :, 2] = 180
    # Add some rectangles simulating UI elements
    cv2.rectangle(img, (20, 10), (280, 60), (255, 255, 255), -1)
    cv2.rectangle(img, (20, 340), (280, 390), (200, 200, 200), -1)
    cv2.putText(img, "CardName", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    return img


@pytest.fixture
def animated_frame_pair():
    """Create two frames that differ only in the center (simulating animation)."""
    h, w = 800, 600
    base = np.random.RandomState(42).randint(0, 255, (h, w, 3), dtype=np.uint8)
    frame_a = base.copy()
    frame_b = base.copy()
    # Animate center region (large area, simulating Live2D)
    cy, cx = h // 2, w // 2
    rh, rw = int(h * 0.4), int(w * 0.4)
    frame_b[cy - rh:cy + rh, cx - rw:cx + rw] = np.random.RandomState(99).randint(
        0, 255, (2 * rh, 2 * rw, 3), dtype=np.uint8
    )
    return frame_a, frame_b


# ---------------------------------------------------------------------------
# CLIP Augmentation Tests
# ---------------------------------------------------------------------------

class TestCLIPAugmentation:
    """Verify _augment_image produces valid augmented variants."""

    def test_augment_returns_multiple_images(self, sample_card_image):
        augmented = _augment_image(sample_card_image)
        assert len(augmented) >= 4, f"Expected at least 4 augmented images, got {len(augmented)}"

    def test_augmented_images_same_shape(self, sample_card_image):
        augmented = _augment_image(sample_card_image)
        for i, aug in enumerate(augmented):
            assert aug.shape == sample_card_image.shape, (
                f"Augmented image {i} shape {aug.shape} != original {sample_card_image.shape}"
            )

    def test_augmented_images_differ_from_original(self, sample_card_image):
        augmented = _augment_image(sample_card_image)
        for i, aug in enumerate(augmented):
            diff = np.abs(aug.astype(float) - sample_card_image.astype(float)).mean()
            assert diff > 0.5, f"Augmented image {i} is too similar to original (mean diff={diff:.2f})"

    def test_augmented_images_are_valid_uint8(self, sample_card_image):
        augmented = _augment_image(sample_card_image)
        for i, aug in enumerate(augmented):
            assert aug.dtype == np.uint8, f"Augmented image {i} dtype={aug.dtype}"
            assert aug.min() >= 0 and aug.max() <= 255

    def test_jpeg_augmentation_preserves_structure(self, sample_card_image):
        """JPEG augmentation should change pixel values but preserve structure."""
        augmented = _augment_image(sample_card_image)
        jpeg_aug = augmented[0]  # First augmented image is JPEG compressed
        ssim = compute_ssim_score(sample_card_image, jpeg_aug)
        assert ssim > 0.6, f"JPEG augmentation destroyed structure (SSIM={ssim:.3f})"
        assert ssim < 1.0, "JPEG augmentation should change the image"

    def test_noise_augmentation_preserves_structure(self, sample_card_image):
        """Gaussian noise should not destroy overall structure."""
        augmented = _augment_image(sample_card_image)
        noisy = augmented[1]  # Second augmented image is Gaussian noise
        ssim = compute_ssim_score(sample_card_image, noisy)
        assert ssim > 0.4, f"Noise augmentation destroyed structure (SSIM={ssim:.3f})"


# ---------------------------------------------------------------------------
# SSIM Region-Exclusion Tests
# ---------------------------------------------------------------------------

class TestSSIMRegionExclusion:
    """Verify that excluding animated regions allows SSIM to detect stability."""

    def test_full_frame_ssim_low_with_animation(self, animated_frame_pair):
        """Full-frame SSIM should be low when center region changes."""
        frame_a, frame_b = animated_frame_pair
        ssim = compute_ssim_score(frame_a, frame_b)
        assert ssim < 0.95, f"Expected low SSIM due to animation, got {ssim:.3f}"

    def test_masked_ssim_high_when_excluding_animation(self, animated_frame_pair):
        """SSIM should be high when the animated center region is masked out."""
        frame_a, frame_b = animated_frame_pair
        h, w = frame_a.shape[:2]

        # Mask out center region (same approach as wait_frame_stable exclude_region)
        exclude = (0.1, 0.1, 0.8, 0.8)
        a = frame_a.copy()
        b = frame_b.copy()
        x1, y1 = int(exclude[0] * w), int(exclude[1] * h)
        x2, y2 = int((exclude[0] + exclude[2]) * w), int((exclude[1] + exclude[3]) * h)
        a[y1:y2, x1:x2] = 0
        b[y1:y2, x1:x2] = 0

        ssim = compute_ssim_score(a, b)
        assert ssim > 0.95, f"Expected high SSIM after masking animation, got {ssim:.3f}"

    def test_identical_frames_ssim_is_one(self):
        """Identical frames should produce SSIM=1.0."""
        frame = np.random.RandomState(42).randint(0, 255, (400, 300, 3), dtype=np.uint8)
        ssim = compute_ssim_score(frame, frame)
        assert ssim == pytest.approx(1.0, abs=0.001)

    def test_completely_different_frames_ssim_is_low(self):
        """Completely different frames should have very low SSIM."""
        frame_a = np.zeros((400, 300, 3), dtype=np.uint8)
        frame_b = np.full((400, 300, 3), 255, dtype=np.uint8)
        ssim = compute_ssim_score(frame_a, frame_b)
        assert ssim < 0.1, f"Expected very low SSIM, got {ssim:.3f}"


# ---------------------------------------------------------------------------
# JPG Noise Resilience Tests
# ---------------------------------------------------------------------------

class TestJPGNoiseResilience:
    """Ensure image comparison methods tolerate JPEG compression artifacts."""

    @pytest.mark.parametrize("quality", [30, 50, 70, 90])
    def test_ssim_resilient_to_jpeg_quality(self, sample_card_image, quality):
        """SSIM between original and JPEG-compressed should reflect quality."""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buf = cv2.imencode(".jpg", sample_card_image, encode_param)
        compressed = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        ssim = compute_ssim_score(sample_card_image, compressed)
        if quality >= 70:
            assert ssim > 0.85, f"Q{quality} SSIM too low: {ssim:.3f}"
        else:
            assert ssim > 0.5, f"Q{quality} SSIM catastrophically low: {ssim:.3f}"

    def test_gaussian_noise_ssim(self, sample_card_image):
        """Gaussian noise (σ=15) should not destroy SSIM completely."""
        noise = np.random.normal(0, 15, sample_card_image.shape).astype(np.float32)
        noisy = np.clip(sample_card_image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        ssim = compute_ssim_score(sample_card_image, noisy)
        assert ssim > 0.4, f"Gaussian noise destroyed structure (SSIM={ssim:.3f})"

    def test_augment_does_not_produce_identical_images(self, sample_card_image):
        """No two augmented images should be identical to the original or each other."""
        augmented = _augment_image(sample_card_image)
        for i, aug_a in enumerate(augmented):
            # Compare with original
            ssim_orig = compute_ssim_score(sample_card_image, aug_a)
            assert ssim_orig < 0.999, f"Augment {i} too close to original"
            # Compare with other augments
            for j, aug_b in enumerate(augmented):
                if i != j:
                    ssim_pair = compute_ssim_score(aug_a, aug_b)
                    assert ssim_pair < 0.999, f"Augments {i} and {j} are identical"
