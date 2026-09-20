# Contributing to Image Analyzer

This is a prototype, so there is plenty of low-hanging fruit in the
[roadmap](README.md#roadmap).

By participating you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

| | |
|---|---|
| **Report a bug** | [Open a bug report](https://github.com/FETKlOkAn2/Image-Analyzer/issues/new?template=bug_report.yml) |
| **Suggest a feature** | [Open a feature request](https://github.com/FETKlOkAn2/Image-Analyzer/issues/new?template=feature_request.yml) |
| **Report a vulnerability** | Do not open an issue. Follow [SECURITY.md](SECURITY.md) |
| **Improve the docs** | PRs welcome directly, no issue needed |
| **Extend the tests** | Especially welcome. See the coverage gaps below |

## Development setup

```bash
git clone https://github.com/FETKlOkAn2/Image-Analyzer.git
cd Image-Analyzer
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

You also need the Tesseract OCR binary on your `PATH`, as described under
[Prerequisites](README.md#prerequisites). Without it, watermark detection returns "no
watermark" for everything, which makes your test results misleading.

### Sanity check

```bash
pytest           # 25 tests, all offline. These should pass before you start
ruff check .
```

Two tests exercise the OCR path and skip with a message if Tesseract is missing. If you
see those skips, install the binary. Otherwise you are not testing watermark detection.

For a verbose single-image walkthrough while tuning thresholds:

```bash
python simple_analyzer.py
```

## Code style

- **Formatting and linting:** [Ruff](https://github.com/astral-sh/ruff). Run
  `ruff check . --fix` before pushing; CI runs `ruff check .`.
- **Line length:** 100 characters.
- **Docstrings:** every public function gets one. Say what it returns, and be honest
  about heuristics. If something is a guess, say so in the docstring, the way
  `detect_watermark_ocr` does.
- **Type hints:** encouraged on new code, not required retroactively.
- **No new hard dependencies** without discussing it in an issue first.

### Two project conventions

1. **Never let a filter return nothing.** Every stage has a fallback, so no part number
   ends up with zero images because a threshold was too aggressive. Keep that property.
2. **Errors go into `part_metrics['errors']`** as well as any `print`. Each entry is a
   dict carrying at least a `type` and an `error` key, so the run report can group them
   by error type.

## Pull requests

1. Fork and branch from `main`, named `feat/short-description` or `fix/short-description`.
2. Keep the change focused. One concern per PR.
3. Update `README.md` if you change behaviour, config or requirements.
4. If you add a config value, add it to the table in the README too.
5. Fill in the PR template, above all the section on how you tested it.
6. Commit messages: imperative mood, present tense
   (`add dry-run flag to S3 upload`, not `added` or `adds`).

### Testing expectations

`pytest` must pass, and new behaviour needs a test.

```
tests/conftest.py            # Synthetic image fixtures
tests/test_download.py       # Download error contract (network mocked)
tests/test_similarity.py     # pHash and SSIM ordering guarantees
tests/test_quality_checks.py # Corner-whiteness thresholds
tests/test_watermark.py      # OCR contract and fail-open behaviour
tests/test_metrics.py        # Aggregation, rates, error grouping
```

Two rules the suite depends on:

- **No network calls.** Every fixture builds its image in memory. Mock
  `utils.requests.get` rather than fetching anything, as `test_download.py` does. This
  keeps CI deterministic and avoids hammering third-party hosts
  ([COMPLIANCE.md](COMPLIANCE.md#3-fetching-third-party-content)).
- **Test the contract, not the OCR accuracy.** The watermark heuristic's precision is a
  documented limitation. Assert that it returns `(bool, str)` and never raises; don't
  assert it correctly identifies a particular watermark.

Known coverage gaps, if you're looking for somewhere to start:

- No integration test for `analyze_images_for_part()`, which needs a mocked download layer
- No test that the clustering stage picks the largest cluster
- No test for `save_image_with_analysis()` writing the three expected files

Also, if you touch a threshold or filter, run `python image_analyzer.py` on the bundled
sample data and report in the PR what changed in the resulting metrics (download success
rate, watermark detection rate, images selected).

## Changing the analysis heuristics

Threshold and heuristic changes need evidence, not just intuition. If you're proposing
a different watermark check or a new similarity metric, include in the PR:

- what the metrics looked like before and after on the same input set, and
- a concrete example image the old approach got wrong.

## Scope

Contributions that make the pipeline better at filtering images you have the right to
publish are in scope. Contributions aimed at **removing, obscuring or circumventing
watermarks in order to republish third-party images** are out of scope and will be
closed. See [Responsible use](README.md#responsible-use) and
[COMPLIANCE.md](COMPLIANCE.md).

## Questions

Open an [issue](https://github.com/FETKlOkAn2/Image-Analyzer/issues). For anything
security-related, use the process in [SECURITY.md](SECURITY.md) instead.
