"""Perceptual hashing and SSIM — the primitives the deduplication relies on."""

from config import PHASH_SIZE
from utils import compute_ssim, hamming, phash


def test_identical_images_hash_identically(white_bg_image):
    assert hamming(phash(white_bg_image), phash(white_bg_image.copy())) == 0


def test_hash_length_follows_config(white_bg_image):
    assert phash(white_bg_image).hash.size == PHASH_SIZE * PHASH_SIZE


def test_near_duplicate_is_closer_than_unrelated(
    white_bg_image, near_duplicate, unrelated_image
):
    reference = phash(white_bg_image)
    near = hamming(reference, phash(near_duplicate))
    far = hamming(reference, phash(unrelated_image))
    assert near < far, f"near-duplicate distance {near} should be below {far}"


def test_ssim_against_self_is_one(white_bg_image):
    assert compute_ssim(white_bg_image, white_bg_image) == 1.0


def test_ssim_ranks_near_duplicate_above_unrelated(
    white_bg_image, near_duplicate, unrelated_image
):
    assert compute_ssim(white_bg_image, near_duplicate) > compute_ssim(
        white_bg_image, unrelated_image
    )


def test_ssim_is_bounded(white_bg_image, unrelated_image):
    for score in (
        compute_ssim(white_bg_image, unrelated_image),
        compute_ssim(white_bg_image, white_bg_image),
    ):
        assert -1.0 <= score <= 1.0


def test_ssim_returns_zero_on_bad_input(white_bg_image):
    """compute_ssim swallows failures and returns 0.0 rather than raising."""
    assert compute_ssim(white_bg_image, None) == 0.0


def test_ssim_handles_mismatched_dimensions(white_bg_image):
    """Images are resized internally, so differing sizes must not raise."""
    small = white_bg_image.resize((80, 120))
    assert compute_ssim(white_bg_image, small) > 0.5
