"""Background quality checks."""

from PIL import Image

from utils import corner_white_check


def test_white_background_accepted(white_bg_image):
    ok, corner_means = corner_white_check(white_bg_image)
    assert ok is True
    assert len(corner_means) == 4
    assert all(m >= 245 for m in corner_means)


def test_grey_background_rejected(grey_bg_image):
    ok, _ = corner_white_check(grey_bg_image)
    assert ok is False


def test_rejection_short_circuits_on_first_dark_corner():
    """A dark top-left corner stops the scan, so fewer than four means come back."""
    img = Image.new("RGBA", (400, 400), "white")
    img.paste((0, 0, 0, 255), (0, 0, 40, 40))
    ok, corner_means = corner_white_check(img)
    assert ok is False
    assert len(corner_means) == 1


def test_threshold_is_configurable(grey_bg_image):
    """A permissive threshold accepts what the default rejects."""
    assert corner_white_check(grey_bg_image, threshold=100)[0] is True
    assert corner_white_check(grey_bg_image, threshold=245)[0] is False
