# image_analyzer.py
"""
Prototype image analysis pipeline:
- download images (list of URLs)
- watermark heuristic via pytesseract
- perceptual hash (imagehash) + SSIM filtering
- cluster & choose best cluster
- white-background check by sampling corners
- upload selected images to S3
"""

import io
import os
import math
import tempfile
import requests
from PIL import Image, ImageStat, ImageOps
import imagehash
import numpy as np
import boto3
from skimage.metrics import structural_similarity as ssim
import cv2
import pytesseract
from sklearn.cluster import DBSCAN

# ---- CONFIG ----
MAX_DOWNLOAD = 10
PHASH_SIZE = 16  # imagehash phash size
PHASH_SIM_THRESHOLD = 6  # hamming distance <= this => similar (tune)
SSIM_SIM_THRESHOLD = 0.55  # pairwise SSIM threshold for somewhat-similar (tune)
WHITE_BG_THRESHOLD = 245  # corner mean >= this -> treat as white
MIN_IMAGES_AFTER_FILTER = 1
S3_BUCKET = os.getenv("S3_BUCKET", "your-bucket-name")
S3_PREFIX = "parts-images/"

# Init S3 client (assumes AWS creds are configured)
s3 = boto3.client("s3")


# ---- UTILITIES ----
def download_image(url, timeout=12):
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGBA")
    except Exception as e:
        print("download error", url, e)
        return None

def to_grayscale_cv(img_pil):
    arr = np.array(img_pil.convert("L"))
    return arr

def detect_watermark_ocr(img_pil):
    """Simple OCR heuristic: if large readable text overlay exists in central area -> watermark.
    This is a heuristic — use a dedicated model for production (or the GitHub repo you linked)."""
    try:
        # crop central band where watermarks often appear
        w,h = img_pil.size
        band = img_pil.crop((int(w*0.05), int(h*0.65), int(w*0.95), int(h*0.95)))  # bottom area
        # increase contrast and convert to grayscale for OCR
        band_g = ImageOps.autocontrast(band.convert("L"))
        text = pytesseract.image_to_string(band_g, config="--psm 6")
        text = text.strip()
        if len(text) >= 6:  # tune: if many characters detected assume watermark
            return True, text
        # also look for translucent overlays: large uniform low-alpha regions
        alpha = np.array(img_pil.split()[-1])
        alpha_mean = alpha.mean()
        if alpha_mean < 250:  # if not fully opaque may be overlayed watermark
            # Not conclusive but flag as suspect
            return True, "alpha_overlay"
    except Exception as e:
        # If OCR not available or fails, return unknown (False)
        return False, ""
    return False, ""

def phash(img_pil):
    return imagehash.phash(img_pil, hash_size=PHASH_SIZE)

def hamming(a, b):
    return (a - b)

def compute_ssim(img1, img2):
    # resize to same small size to compute SSIM quickly
    try:
        size = (256,256)
        a = np.array(img1.convert("L").resize(size))
        b = np.array(img2.convert("L").resize(size))
        score = ssim(a, b)
        return float(score)
    except Exception:
        return 0.0

def corner_white_check(img_pil, threshold=WHITE_BG_THRESHOLD):
    """Check four corners (square patch) mean brightness; return True if all corners are near-white."""
    w,h = img_pil.size
    box = int(min(w,h)*0.05)  # 5% patch
    patches = [
        img_pil.crop((0,0,box,box)),
        img_pil.crop((w-box,0,w,box)),
        img_pil.crop((0,h-box,box,h)),
        img_pil.crop((w-box,h-box,w,h)),
    ]
    for p in patches:
        stat = ImageStat.Stat(p.convert("L"))
        if stat.mean[0] < threshold:
            return False
    return True

def upload_to_s3_bytes(data_bytes, key, content_type="image/jpeg"):
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=data_bytes, ContentType=content_type)
    return f"s3://{S3_BUCKET}/{key}"

# ---- MAIN PROCESS FOR A SINGLE PART ----
def analyze_images_for_part(part_number, image_urls):
    """
    image_urls: list of image urls (max MAX_DOWNLOAD)
    returns: list of dicts: {url, selected, s3_path, phash, ssim_score, white_ok, watermark_flag}
    """

    results = []
    # download up to MAX_DOWNLOAD
    for url in image_urls[:MAX_DOWNLOAD]:
        img = download_image(url)
        if img is None:
            continue
        # normalize convert
        # store basic meta
        results.append({"url": url, "img": img, "watermark": False, "wm_reason":"", "phash": None, "ssim": None, "white_ok":False, "selected": False})

    # Step 1: watermark detection (heuristic)
    for r in results:
        wm, reason = detect_watermark_ocr(r["img"])
        r["watermark"] = wm
        r["wm_reason"] = reason

    # Drop watermarked images
    non_wm = [r for r in results if not r["watermark"]]
    if len(non_wm) == 0:
        print("All images flagged as watermarked. Returning original set for manual review.")
        non_wm = results  # fallback to letting human decide

    # Step 2: compute phash
    for r in non_wm:
        try:
            r["phash"] = phash(r["img"])
        except Exception as e:
            r["phash"] = None

    # Step 3: cluster by phash hamming distance using DBSCAN (distance metric = hamming)
    # build distance matrix of phashes
    phashes = [r["phash"] for r in non_wm if r["phash"] is not None]
    if len(phashes) == 0:
        print("No phashes computed; aborting similarity.")
        clustered = non_wm
    else:
        # convert phash to numpy arrays of bits for clustering
        vectors = []
        for p in phashes:
            # imagehash.ImageHash -> numpy array of bits
            bits = np.array(list(map(int, str(p))))
            vectors.append(bits)
        vecs = np.stack(vectors).astype(int)
        # DBSCAN with Hamming-like distance (eps tuned)
        db = DBSCAN(eps=PHASH_SIM_THRESHOLD/ (PHASH_SIZE*PHASH_SIZE), min_samples=1, metric="hamming")
        labels = db.fit_predict(vecs)
        # map back to results
        idx = 0
        for r in non_wm:
            if r["phash"] is not None:
                r["cluster"] = int(labels[idx]); idx+=1
            else:
                r["cluster"] = -1

        # choose largest cluster (most images) as primary cluster
        from collections import Counter
        counts = Counter([r["cluster"] for r in non_wm])
        primary_cluster = counts.most_common(1)[0][0]
        clustered = [r for r in non_wm if r.get("cluster", -1) == primary_cluster]

    # Step 4: within clustered pick images similar by SSIM and prefer white backgrounds
    # compute pairwise SSIM to cluster center (choose first as center)
    for r in clustered:
        try:
            r["ssim_center"] = compute_ssim(clustered[0]["img"], r["img"])
        except Exception:
            r["ssim_center"] = 0.0
        r["white_ok"] = corner_white_check(r["img"])

    # filter by SSIM threshold (keep those with good similarity)
    final_candidates = [r for r in clustered if r["ssim_center"] >= SSIM_SIM_THRESHOLD]
    if len(final_candidates) < MIN_IMAGES_AFTER_FILTER:
        # fallback: take clustered set (maybe loosen)
        final_candidates = clustered

    # rank candidates: prefer white backgrounds and higher ssim
    final_candidates.sort(key=lambda x: (int(x["white_ok"]), x.get("ssim_center",0)), reverse=True)

    # keep up to 5 final
    final = final_candidates[:5]

    # upload to s3
    out = []
    for i,r in enumerate(final):
        buf = io.BytesIO()
        # save as JPEG (RGB)
        rgb = r["img"].convert("RGB")
        rgb.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        key = f"{S3_PREFIX}{part_number}/{i}.jpg"
        s3_path = upload_to_s3_bytes(buf.getvalue(), key, content_type="image/jpeg")
        out.append({
            "original_url": r["url"],
            "s3_path": s3_path,
            "phash": str(r.get("phash")),
            "ssim_center": r.get("ssim_center"),
            "white_ok": r.get("white_ok"),
            "watermark_flag": r.get("watermark"),
        })

    return out

# Example usage:
if __name__ == "__main__":
    test_urls = [
        # put real image urls here for a part
    ]
    results = analyze_images_for_part("ATRTS38000", test_urls)
    print("Selected images:", results)
