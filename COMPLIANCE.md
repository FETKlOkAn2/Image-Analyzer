# Compliance & Responsible Use

Image Analyzer downloads images from third-party URLs, runs OCR over them, discards
images it believes carry a watermark, and can upload the survivors to cloud storage.
Each of those steps has a legal or data-protection dimension. This document sets out
where the boundaries are.

**This is engineering guidance, not legal advice.** If you're deploying this
commercially, have your own counsel review your specific use.

---

## 1. Intended use

> Image Analyzer is intended for preparing product images that **you own**, or that a
> supplier or rights holder has **licensed you to publish**, for use in your own
> catalogue.

The pipeline is a *filter*. Its purpose is to reduce a noisy supplier feed down to the
clean, non-duplicate, publishable subset.

### Watermark detection is a rejection filter, not a removal tool

This deserves to be unambiguous, because the repository history described the feature as
"watermark removal" and that phrasing invites misreading:

- **What the code does:** flags an image as watermarked and **drops it from the
  selection**. See `detect_watermark_ocr()` in [`utils.py`](utils.py) — it returns a
  boolean and a reason, and `analyze_images_for_part()` filters those images out. No
  pixels are altered. There is no inpainting, no logo erasure, no watermark
  reconstruction anywhere in this codebase.
- **Why that's the design:** a watermark is usually a signal that the image belongs to
  someone else. Discarding those images is the *compliant* behaviour.

**Do not extend this project to remove watermarks from images you don't own.** Stripping
a watermark to republish a third-party photograph is copyright infringement in most
jurisdictions. In the United States it may additionally violate
[17 U.S.C. § 1202](https://www.law.cornell.edu/uscode/text/17/1202) (removal of
copyright management information) — a separate claim from infringement itself, with its
own statutory damages. In the EU, Article 7 of the
[InfoSoc Directive (2001/29/EC)](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=celex%3A32001L0029)
covers the equivalent ground. Watermarks and embedded credits are exactly the kind of
rights-management information these provisions protect.

Contributions in that direction are out of scope and will be closed
([CONTRIBUTING.md](CONTRIBUTING.md#scope)).

---

## 2. Image rights

Before running the pipeline against a URL list, you should be able to answer yes to
one of these for every source:

- [ ] The images are ours (we shot them, or we commissioned them with rights assigned).
- [ ] A supplier agreement grants us the right to reproduce their product imagery in
      our catalogue.
- [ ] The images carry a licence that permits our intended use (and we're complying
      with its attribution terms).
- [ ] The images are public domain.

"It was publicly accessible on the internet" is **not** a licence. Neither is "the
supplier sent us the link."

### Provenance is preserved deliberately

Every selected image records its `original_url` in the metadata sidecar, the per-part
metrics JSON, and the summary CSV. Keep those records — they are your audit trail if a
rights question comes up later. Don't strip them to save space.

### Attribution

Some permissive licences (CC BY, and others) require attribution even for commercial
use. The pipeline records provenance but does **not** render attribution into your
catalogue for you. If your sources require credit, that's a downstream responsibility.

The demo images in `docs/images/` are derived from [Unsplash](https://unsplash.com/)
photographs used under the [Unsplash License](https://unsplash.com/license).

---

## 3. Fetching third-party content

Downloading at scale has its own etiquette and, in some jurisdictions, legal exposure
under computer-misuse and contract law.

| Obligation | Status in this codebase |
|---|---|
| Respect `robots.txt` | ❌ Not implemented — your responsibility |
| Respect terms of service | ❌ Not implemented — review before fetching |
| Rate limiting | ❌ Not implemented; downloads are sequential, which limits load incidentally |
| Identify your client | ⚠️ Currently sends a generic browser `User-Agent` in [`utils.py`](utils.py) |
| Request timeouts | ✅ 12 s default |
| Cap on requests per part | ✅ `MAX_DOWNLOAD`, default 20 |

**Recommended before production use:** set a descriptive `User-Agent` identifying your
organisation with a contact URL, add `robots.txt` checking, and add an explicit rate
limiter. The generic browser user-agent string is a prototype convenience, not good
practice — it misrepresents an automated client as a browser.

---

## 4. Data protection (GDPR and similar)

Product images are usually not personal data, which keeps this simple. But:

- **Incidental personal data.** Product photographs sometimes contain people
  (hands, reflections, bystanders) or identifiable information (a serial number on a
  returned unit, a name on packaging). If your feed can contain these, you have a
  personal-data processing activity and need a lawful basis for it.
- **OCR extracts text and stores it.** `detect_watermark_ocr()` runs Tesseract over the
  bottom band of every image and, when it flags a watermark, **writes the recognised
  text into `wm_reason`**, which is persisted in metadata sidecars, metrics JSON and
  the annotated overlay images. If an image contains personal data as text, that text
  is now in your output files in plain form. Treat the `metrics/` and
  `image_analysis_results/` directories as potentially containing extracted text, not
  just pixels.
- **OCR runs locally.** Tesseract is a local binary. No image or extracted text is sent
  to a third-party OCR service by this pipeline.
- **Retention.** Output directories grow without bound; nothing is ever pruned. Define
  a retention period and delete accordingly.

---

## 5. Data flow

Know what leaves your machine:

```
Third-party URL  ──HTTP GET──▶  Your machine
                                    │
                                    ├─▶ Tesseract OCR          (local subprocess)
                                    ├─▶ Hash / SSIM / DBSCAN    (local, in-memory)
                                    ├─▶ image_analysis_results/ (local disk)
                                    ├─▶ metrics/                (local disk)
                                    │
                                    └─▶ Amazon S3               (egress — your bucket)
```

**Outbound network calls:** image fetches to the source URLs, and the S3 upload. That's
all. No telemetry, no analytics, no third-party API calls.

---

## 6. AWS and credential handling

- `S3_BUCKET` is read from the environment in [`config.py`](config.py). Never hardcode
  it, and never commit a real bucket name.
- AWS credentials are resolved by `boto3`'s default chain. **Never** put access keys in
  source, in `config.py`, or in a committed `.env`. The [`.gitignore`](.gitignore)
  excludes `.env`, `*.pem`, `.aws/` and `credentials` — keep it that way.
- Prefer an instance profile, task role or OIDC federation over long-lived access keys.
- Scope the IAM policy tightly. Least privilege for this pipeline is
  `s3:PutObject` on one prefix:

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

No `s3:DeleteObject`, no `s3:*`, no bucket-level wildcard. Enable default encryption
and block public access on the bucket unless your catalogue genuinely needs public
objects — and if it does, serve them through a CDN rather than opening the bucket.

See [SECURITY.md](SECURITY.md) for the wider hardening checklist.

---

## 7. Open-source licence obligations

This project is [MIT licensed](LICENSE) — permissive, requiring only that the copyright
notice and permission notice travel with copies.

Its dependencies carry their own terms. Summarised for orientation only; verify against
the versions you actually install:

| Dependency | Licence family |
|---|---|
| `pillow` | MIT-CMU |
| `numpy`, `pandas`, `scikit-image`, `scikit-learn`, `boto3`, `imagehash` | BSD / Apache-2.0 family |
| `requests` | Apache-2.0 |
| `opencv-python` | Apache-2.0 (the wheel; bundled codecs vary) |
| `pytesseract` | Apache-2.0 |
| **Tesseract OCR** (system binary) | **Apache-2.0** |

All permissive; none copyleft as configured here. Note that Tesseract is a separate
system dependency you install yourself — it is not bundled or redistributed by this
repository. If you package this project into a distributed artifact, run a proper
licence audit (`pip-licenses` or similar) against your resolved dependency tree,
including transitive dependencies and any bundled OpenCV codecs.

---

## 8. Accuracy limitations that matter for decisions

If output from this pipeline feeds an automated decision, understand what it can't do:

- **Watermark detection is a character-count heuristic**, not a classifier. It produces
  false positives on legitimate product labelling and false negatives on centred
  translucent marks. Do not treat "not flagged" as evidence an image is unwatermarked
  or unencumbered.
- **A missing Tesseract binary fails open.** With no OCR available, every image is
  reported as clean, with the reason recorded as `ocr_error` in the metadata. Check for
  that string before trusting a run.
- **Corner whiteness is a proxy** for background quality and misjudges tightly cropped
  images.
- **Thresholds are untuned** against any labelled dataset. Tune them on your own feed
  and keep the metrics reports as evidence of what you tuned to.

Keep a human in the loop for anything consequential. The pipeline is explicitly built
to preserve that option — when every image for a part is flagged as watermarked it
returns the full set for manual review rather than returning nothing.

---

## Questions

Open an [issue](https://github.com/FETKlOkAn2/Image-Analyzer/issues) for anything in
this document that's unclear or out of date. For security matters, use
[SECURITY.md](SECURITY.md).
