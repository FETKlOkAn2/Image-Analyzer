# Compliance and Responsible Use

Image Analyzer downloads images from third-party URLs, runs OCR over them, discards
images it believes carry a watermark, and can upload the survivors to cloud storage.
Each of those steps carries a legal or data-protection obligation. This document sets
out where the boundaries are.

**This is engineering guidance, not legal advice.** If you deploy this commercially,
have your own counsel review your use of it.

---

## 1. Intended use

> Image Analyzer is intended for preparing product images that **you own**, or that a
> supplier or rights holder has **licensed you to publish**, for use in your own
> catalogue.

The pipeline reduces a noisy supplier feed down to the clean, non-duplicate,
publishable subset.

### Watermark detection rejects images

An earlier version of the README described this as "watermark removal", which invites
the wrong reading:

- **What the code does:** `detect_watermark_ocr()` in [`utils.py`](utils.py) returns a
  boolean and a reason, and `analyze_images_for_part()` drops the flagged images from
  the selection. No pixels change. The codebase holds no inpainting, no logo erasure
  and no watermark reconstruction.
- **Why it works that way:** a watermark usually signals that the image belongs to
  someone else, so discarding it is the compliant behaviour.

**Do not extend this project to remove watermarks from images you do not own.**
Stripping a watermark to republish a third-party photograph infringes copyright in most
jurisdictions. In the United States it can also violate
[17 U.S.C. § 1202](https://www.law.cornell.edu/uscode/text/17/1202) (removal of
copyright management information), which is a separate claim from infringement itself
and carries its own statutory damages. In the EU, Article 7 of the
[InfoSoc Directive (2001/29/EC)](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=celex%3A32001L0029)
covers the equivalent ground. Watermarks and embedded credits are the kind of
rights-management information these provisions protect.

Contributions in that direction fall outside the project's scope and will be closed.
See [CONTRIBUTING.md](CONTRIBUTING.md#scope).

---

## 2. Image rights

Before running the pipeline against a URL list, answer yes to one of these for every
source:

- [ ] We shot the images, or commissioned them with rights assigned.
- [ ] A supplier agreement grants us the right to reproduce their product imagery in
      our catalogue.
- [ ] The images carry a licence permitting our intended use, and we follow its
      attribution terms.
- [ ] The images are public domain.

Public accessibility on the internet grants no licence, and neither does a supplier
sending you the link.

### Provenance

Every selected image records its `original_url` in the metadata sidecar, the per-part
metrics JSON and the summary CSV. Keep those records. They form your audit trail if a
rights question comes up later, so do not strip them to save space.

### Attribution

Some permissive licences, CC BY among them, require attribution even for commercial
use. The pipeline records provenance but writes no attribution into your catalogue. If
your sources require credit, you handle that downstream.

The demo images in `docs/images/` are derived from [Unsplash](https://unsplash.com/)
photographs used under the [Unsplash License](https://unsplash.com/license).

---

## 3. Fetching third-party content

Downloading at scale carries its own etiquette and, in some jurisdictions, exposure
under computer-misuse and contract law.

| Obligation | Status in this codebase |
|---|---|
| Respect `robots.txt` | Not implemented. Your responsibility |
| Respect terms of service | Not implemented. Review before fetching |
| Rate limiting | Not implemented. Downloads run sequentially, which caps load as a side effect |
| Identify your client | Sends a generic browser `User-Agent` in [`utils.py`](utils.py) |
| Request timeouts | Yes, 12 s default |
| Cap on requests per part | Yes, `MAX_DOWNLOAD`, default 20 |

Before production use, set a descriptive `User-Agent` naming your organisation with a
contact URL, add `robots.txt` checking, and add a rate limiter. The generic browser
user-agent string is a prototype convenience that misrepresents an automated client as
a browser.

---

## 4. Data protection (GDPR and similar)

Product images rarely count as personal data, which keeps this straightforward. Two
exceptions and two housekeeping points:

- **Incidental personal data.** Product photographs sometimes show people, whether
  hands, reflections or bystanders, or identifiable information such as a serial number
  on a returned unit or a name on packaging. If your feed can contain these, you are
  processing personal data and need a lawful basis for it.
- **OCR extracts text and stores it.** `detect_watermark_ocr()` runs Tesseract over the
  bottom band of every image and, when it flags a watermark, **writes the recognised
  text into `wm_reason`**, which is persisted in metadata sidecars, metrics JSON and
  the annotated overlay images. If an image contains personal data as text, that text
  is now in your output files in plain form. Treat the `metrics/` and
  `image_analysis_results/` directories as potentially containing extracted text, not
  just pixels.
- **OCR runs locally.** Tesseract is a local binary, and this pipeline sends no image
  or extracted text to a third-party OCR service.
- **Retention.** The output directories grow without bound and nothing prunes them.
  Define a retention period and delete on that schedule.

---

## 5. Data flow

What leaves your machine:

```
Third-party URL  ──HTTP GET──▶  Your machine
                                    │
                                    ├─▶ Tesseract OCR          (local subprocess)
                                    ├─▶ Hash / SSIM / DBSCAN    (local, in-memory)
                                    ├─▶ image_analysis_results/ (local disk)
                                    ├─▶ metrics/                (local disk)
                                    │
                                    └─▶ Amazon S3               (egress, your bucket)
```

Outbound network calls come to two things: the image fetches to your source URLs, and
the S3 upload. The pipeline sends no telemetry, no analytics and no third-party API
calls.

---

## 6. AWS and credential handling

- `S3_BUCKET` is read from the environment in [`config.py`](config.py). Never hardcode
  it, and never commit a real bucket name.
- `boto3` resolves AWS credentials through its default chain. Never put access keys in
  source, in `config.py` or in a committed `.env`. The [`.gitignore`](.gitignore)
  excludes `.env`, `*.pem`, `.aws/` and `credentials`, so keep those entries in place.
- Prefer an instance profile, task role or OIDC federation over long-lived access keys.
- Scope the IAM policy down. Least privilege for this pipeline is `s3:PutObject` on one
  prefix:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "s3:PutObject",
    "Resource": "arn:aws:s3:::YOUR-BUCKET/parts-images/*"
  }]
}
```

No `s3:DeleteObject`, no `s3:*`, no bucket-level wildcard. Enable default encryption and
block public access on the bucket unless your catalogue needs public objects. If it
does, serve them through a CDN rather than opening the bucket.

See [SECURITY.md](SECURITY.md) for the wider hardening checklist.

---

## 7. Open-source licence obligations

This project is [MIT licensed](LICENSE). It requires only that the copyright notice and
permission notice travel with copies.

Its dependencies carry their own terms. The table below orients you. Verify it against
the versions you install:

| Dependency | Licence family |
|---|---|
| `pillow` | MIT-CMU |
| `numpy`, `pandas`, `scikit-image`, `scikit-learn`, `boto3`, `imagehash` | BSD / Apache-2.0 family |
| `requests` | Apache-2.0 |
| `opencv-python` | Apache-2.0 (the wheel; bundled codecs vary) |
| `pytesseract` | Apache-2.0 |
| **Tesseract OCR** (system binary) | **Apache-2.0** |

All permissive, none copyleft as configured here. Tesseract is a separate system
dependency you install yourself, so this repository neither bundles nor redistributes
it. If you package this project into a distributed artifact, run a licence audit with
`pip-licenses` or similar against your resolved dependency tree, covering transitive
dependencies and any bundled OpenCV codecs.

---

## 8. Accuracy limitations that matter for decisions

If output from this pipeline feeds an automated decision, note these limits:

- **Watermark detection counts OCR characters.** It is a heuristic rather than a
  classifier, and it produces false positives on legitimate product labelling and false
  negatives on centred translucent marks. An unflagged image is not evidence that the
  image is unwatermarked or unencumbered.
- **A missing Tesseract binary fails open.** With no OCR available, the pipeline reports
  every image as clean and records `ocr_error` as the reason in the metadata. Check for
  that string before you trust a run.
- **Corner whiteness is a proxy** for background quality and misjudges tightly cropped
  images.
- **Thresholds are untuned** against any labelled dataset. Fit them to your own feed and
  keep the metrics reports as evidence of what you fitted them to.

Keep a human in the loop for anything consequential. The pipeline preserves that
option: when it flags every image for a part as watermarked, it returns the full set
for manual review rather than nothing.

---

## Questions

Open an [issue](https://github.com/FETKlOkAn2/Image-Analyzer/issues) for anything in
this document that is unclear or out of date. For security matters, use
[SECURITY.md](SECURITY.md).
