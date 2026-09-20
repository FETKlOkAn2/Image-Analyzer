## What & why

<!-- What does this change, and what problem does it solve? Link any related issue. -->

Closes #

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Heuristic / threshold change (see below)
- [ ] Documentation
- [ ] Refactor or tooling
- [ ] Breaking change

## How was this tested?

<!-- Required. Say what you ran. -->

- [ ] `pytest` passes with no skips, meaning Tesseract is installed
- [ ] `ruff check .` passes
- [ ] New behaviour has a test, and the test fails without the change
- [ ] `python image_analyzer.py` completes on the bundled sample data
- [ ] Tested on my own image set (describe below)

<!-- Details: -->

## For heuristic or threshold changes

<!-- Delete this section if not applicable. -->

| Metric | Before | After |
|---|---|---|
| Download success rate | | |
| Watermark detection rate | | |
| White background rate | | |
| Average SSIM | | |
| Images selected | | |

<!-- Include a concrete example the previous behaviour got wrong. -->

## Checklist

- [ ] `README.md` updated if behaviour, config or requirements changed
- [ ] New config values added to the README configuration table
- [ ] Every filter stage still has a fallback, so no part ends up with zero images
      because a threshold was too aggressive
- [ ] New errors are appended to `part_metrics['errors']` with a `type` key, not just
      printed
- [ ] No new test reaches the network. Images are built in memory, requests mocked
- [ ] No credentials, bucket names, internal URLs or real part numbers in the diff
- [ ] No pipeline output committed (`image_analysis_results/`, `metrics/`,
      `final_images/` are git-ignored)
- [ ] Change is in scope per [CONTRIBUTING.md](../blob/main/CONTRIBUTING.md#scope)
