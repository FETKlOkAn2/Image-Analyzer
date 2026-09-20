"""Metrics aggregation — the reporting contract the README documents."""

import json

from image_analyzer import ImageAnalysisMetrics


def _part(**overrides):
    base = {
        "urls_attempted": 10,
        "successful_downloads": 8,
        "failed_downloads": 2,
        "watermarked_count": 2,
        "final_count": 4,
        "white_bg_count": 3,
        "avg_ssim": 0.9,
        "processing_time": 1.5,
    }
    base.update(overrides)
    return base


def test_rates_are_computed_from_accumulated_totals():
    m = ImageAnalysisMetrics()
    m.add_part_metrics(_part())
    m.add_part_metrics(_part())

    overview = m.get_summary()["overview"]
    assert overview["parts_processed"] == 2
    assert overview["total_urls"] == 20
    assert overview["download_success_rate"] == 16 / 20
    assert overview["total_final_images"] == 8


def test_empty_run_does_not_divide_by_zero():
    """A run with no parts must report zeros, not raise."""
    summary = ImageAnalysisMetrics().get_summary()
    assert summary["overview"]["download_success_rate"] == 0
    assert summary["quality_metrics"]["avg_ssim_score"] == 0
    assert summary["error_analysis"]["total_errors"] == 0


def test_errors_are_grouped_by_type():
    m = ImageAnalysisMetrics()
    m.add_part_metrics(
        _part(
            errors=[
                {"type": "download_error"},
                {"type": "download_error"},
                {"type": "ssim_error"},
            ]
        )
    )

    analysis = m.get_summary()["error_analysis"]
    assert analysis["total_errors"] == 3
    assert analysis["error_types"]["download_error"] == 2
    assert analysis["error_types"]["ssim_error"] == 1


def test_reset_clears_previous_run():
    m = ImageAnalysisMetrics()
    m.add_part_metrics(_part())
    m.reset()
    assert m.get_summary()["overview"]["parts_processed"] == 0


def test_report_is_valid_json_on_disk(tmp_path):
    m = ImageAnalysisMetrics()
    m.add_part_metrics(_part())
    report = tmp_path / "report.json"

    m.save_detailed_report(report)
    payload = json.loads(report.read_text())

    assert payload["summary"]["overview"]["parts_processed"] == 1
    assert "generated_at" in payload
