"""Watermark detection.

These assert the *contract* — a (bool, reason) tuple that never raises — rather
than OCR accuracy. The heuristic's accuracy is a documented limitation, not a
guarantee (see COMPLIANCE.md).
"""

import shutil

import pytest

from utils import detect_watermark_ocr

needs_tesseract = pytest.mark.skipif(
    shutil.which("tesseract") is None,
    reason="Tesseract binary not on PATH; see README prerequisites",
)


def test_always_returns_bool_and_reason(white_bg_image):
    flagged, reason = detect_watermark_ocr(white_bg_image)
    assert isinstance(flagged, bool)
    assert isinstance(reason, str)


def test_never_raises_on_a_tiny_image():
    """Crop maths must survive degenerate dimensions."""
    from PIL import Image

    flagged, _ = detect_watermark_ocr(Image.new("RGBA", (4, 4), "white"))
    assert isinstance(flagged, bool)


def test_missing_ocr_binary_is_reported_in_the_reason(white_bg_image):
    """A fail-open path must leave a trace, so a clean run can be distinguished
    from a run where OCR was simply unavailable."""
    flagged, reason = detect_watermark_ocr(white_bg_image)
    if shutil.which("tesseract") is None:
        assert "ocr_error" in reason
        assert flagged is False


@needs_tesseract
def test_clean_image_is_not_flagged(white_bg_image):
    flagged, reason = detect_watermark_ocr(white_bg_image)
    assert "ocr_error" not in reason
    assert flagged is False


@needs_tesseract
def test_translucent_overlay_is_flagged(white_bg_image):
    """A non-opaque alpha channel is treated as a possible overlay watermark."""
    semi = white_bg_image.copy()
    semi.putalpha(200)
    flagged, reason = detect_watermark_ocr(semi)
    assert flagged is True
    assert reason == "alpha_overlay"
