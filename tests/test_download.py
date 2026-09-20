"""The download path's error contract, exercised without touching the network."""

import requests

import utils
from utils import download_image


def test_failure_returns_none_and_structured_error(monkeypatch):
    """Errors must be dicts carrying a `type`, so the run report can group them."""

    def boom(*args, **kwargs):
        raise requests.ConnectionError("no route to host")

    monkeypatch.setattr(utils.requests, "get", boom)

    img, error = download_image("https://example.invalid/part.jpg")

    assert img is None
    assert error["type"] == "download_error"
    assert error["url"] == "https://example.invalid/part.jpg"
    assert "no route to host" in error["error"]


def test_success_returns_rgba_image_and_no_error(monkeypatch, white_bg_image):
    import io

    buffer = io.BytesIO()
    white_bg_image.convert("RGB").save(buffer, format="PNG")
    payload = buffer.getvalue()

    class FakeResponse:
        content = payload

        def raise_for_status(self):
            return None

    monkeypatch.setattr(utils.requests, "get", lambda *a, **k: FakeResponse())

    img, error = download_image("https://example.test/part.png")

    assert error is None
    assert img.mode == "RGBA", "downstream alpha checks depend on RGBA"


def test_non_image_payload_is_reported_as_error(monkeypatch):
    """A 404 page served as a JPEG must be caught, not crash the run."""

    class FakeResponse:
        content = b"<html>Not Found</html>"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(utils.requests, "get", lambda *a, **k: FakeResponse())

    img, error = download_image("https://example.test/missing.jpg")

    assert img is None
    assert error["type"] == "download_error"
