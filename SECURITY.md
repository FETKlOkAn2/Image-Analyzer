# Security Policy

## Supported versions

This is a prototype under active development. Only the current `main` branch is
supported; there are no tagged releases or backported fixes yet.

| Version | Supported |
|---|---|
| `main` | Yes |
| Older commits | No |

## Reporting a vulnerability

**Please do not open a public issue for a security vulnerability.**

Report it privately through
[GitHub Security Advisories](https://github.com/FETKlOkAn2/Image-Analyzer/security/advisories/new),
which lets us discuss and fix the issue before any details become public.

Please include:

- what the issue is and roughly how severe you think it is,
- steps to reproduce (a minimal input that triggers it is ideal),
- affected files or functions, and
- any suggested fix, if you have one.

### What to expect

| | |
|---|---|
| **Acknowledgement** | Within 5 days |
| **Initial assessment** | Within 14 days |
| **Fix or mitigation plan** | Communicated with the assessment |
| **Credit** | Offered in the advisory and release notes, unless you'd rather stay anonymous |

This is a personal project maintained in spare time, so those are good-faith targets
rather than a commercial SLA.

## Scope

**In scope:**

- Arbitrary code execution or path traversal via crafted image files, URLs or part
  numbers (note that `part_number` is used directly in filesystem paths)
- Credential or secret leakage, including secrets ending up in metrics reports,
  metadata sidecars or the annotated overlay images
- Server-side request forgery through the URL download path
- Denial of service via decompression bombs or pathological image dimensions
- Vulnerable pinned dependencies with a practical exploit path here

**Out of scope:**

- Vulnerabilities in Tesseract, Pillow, OpenCV or other upstream dependencies. Report
  those to the respective projects, though do tell us if we should pin or patch around one
- The watermark heuristic producing false positives or negatives. That is a known
  accuracy limitation rather than a vulnerability, so open a normal issue
- Anything requiring the attacker to already control the machine running the pipeline
- Results from running the tool against inputs you don't have permission to fetch

## Known security considerations

These are inherent to what the tool does. Understand them before running it on
untrusted input.

### It fetches arbitrary URLs

`download_image()` issues a `GET` to whatever URL it's given, with no allowlist and no
SSRF protection. If your URL list comes from an untrusted source, it can be pointed at
internal network addresses or cloud metadata endpoints. **Validate and allowlist URLs
before passing them in**, and prefer running the pipeline in a network-restricted
environment.

### It decodes untrusted images

Pillow and OpenCV decode whatever bytes come back. Malformed or hostile images have a long
history of triggering memory-safety issues in image libraries.
Keep dependencies current (Dependabot is enabled), and consider setting
`PIL.Image.MAX_IMAGE_PIXELS` to guard against decompression bombs.

### Part numbers reach the filesystem

`part_number` is interpolated into output paths. A value containing `../` or absolute
path components could write outside `LOCAL_SAVE_DIR`. Sanitise part numbers from
untrusted sources.

### Credentials

The pipeline never asks for credentials directly. `S3_BUCKET` comes from the
environment and AWS credentials are resolved by `boto3`'s standard chain. Do not
hardcode either. Use an IAM identity scoped to a single bucket prefix with write-only
permissions. [COMPLIANCE.md](COMPLIANCE.md#6-aws-and-credential-handling) has the policy.

### OCR is a subprocess

`pytesseract` shells out to the `tesseract` binary. Check that the binary on your `PATH` is one
you installed yourself.

## Hardening checklist for production use

- [ ] Allowlist or validate every source URL before it reaches `download_image()`
- [ ] Sanitise `part_number` before it's used in a path
- [ ] Set `PIL.Image.MAX_IMAGE_PIXELS` and enforce a response size cap
- [ ] Run with an egress-filtered network policy
- [ ] Use a write-only, prefix-scoped IAM role for S3
- [ ] Keep `requirements.txt` current and review Dependabot PRs
