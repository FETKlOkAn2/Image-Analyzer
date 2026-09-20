"""Shared fixtures.

Every fixture here builds images in memory. The test suite never reaches the
network: fetching third-party URLs in CI would be both flaky and, per
COMPLIANCE.md, inappropriate.
"""

import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _product_on_white(size=(400, 400)):
    """A red disc centred on a white background — stands in for a catalogue shot."""
    img = Image.new("RGBA", size, "white")
    w, h = size
    ImageDraw.Draw(img).ellipse((w * 0.3, h * 0.3, w * 0.7, h * 0.7), fill=(200, 30, 30, 255))
    return img


@pytest.fixture
def white_bg_image():
    return _product_on_white()


@pytest.fixture
def near_duplicate(white_bg_image):
    """The same product, rotated slightly — what a resized supplier copy looks like."""
    return white_bg_image.copy().rotate(2, fillcolor=(255, 255, 255, 255))


@pytest.fixture
def grey_bg_image():
    return Image.new("RGBA", (400, 400), (128, 128, 128, 255))


@pytest.fixture
def unrelated_image():
    """Horizontal stripes — structurally unlike the product shot."""
    img = Image.new("RGBA", (400, 400), (20, 20, 90, 255))
    draw = ImageDraw.Draw(img)
    for y in range(0, 400, 40):
        draw.rectangle((0, y, 400, y + 20), fill=(240, 240, 40, 255))
    return img
