#utils.py
import requests
from PIL import Image, ImageStat, ImageOps, ImageDraw, ImageFont
from pathlib import Path
import pytesseract
import numpy as np
import io
import imagehash
import json
from skimage.metrics import structural_similarity as ssim
from config import PHASH_SIZE, WHITE_BG_THRESHOLD, LOCAL_SAVE_DIR



# ---- UTILITIES ----
def download_image(url, timeout=12):
    try:
        
        resp = requests.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
        return img, None
    except Exception as e:
        error = {"type": "download_error", "url": url, "error": str(e)}
        return None, error

def to_grayscale_cv(img_pil):
    arr = np.array(img_pil.convert("L"))
    return arr

def detect_watermark_ocr(img_pil):
    """Simple OCR heuristic: if large readable text overlay exists in central area -> watermark.
    This is a heuristic — use a dedicated model for production."""
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
        return False, f"ocr_error: {e}"
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
    corner_means = []
    for p in patches:
        stat = ImageStat.Stat(p.convert("L"))
        corner_means.append(stat.mean[0])
        if stat.mean[0] < threshold:
            return False, corner_means
    return True, corner_means

def create_analysis_overlay(img_pil, metadata):
    """Create a visualization overlay showing analysis results"""
    overlay = img_pil.copy()
    draw = ImageDraw.Draw(overlay)
    
    # Try to use a default font, fallback to PIL default
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Add metadata text
    y_offset = 10
    for key, value in metadata.items():
        if key not in ['img', 'phash']:  # Skip non-displayable items
            text = f"{key}: {value}"
            draw.text((10, y_offset), text, fill="red", font=font)
            y_offset += 25
    
    return overlay

def save_image_with_analysis(part_number, index, img_pil, metadata, base_dir=LOCAL_SAVE_DIR):
    """Save image with analysis metadata and create visualization"""
    part_dir = Path(base_dir) / part_number
    part_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original image
    original_path = part_dir / f"{index}_original.jpg"
    img_pil.convert("RGB").save(original_path, format="JPEG", quality=85)
    
    # Save metadata
    metadata_clean = {k: v for k, v in metadata.items() if k not in ['img', 'phash']}
    metadata_clean['phash'] = str(metadata.get('phash', ''))
    
    metadata_path = part_dir / f"{index}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata_clean, f, indent=2)
    
    # Create and save analysis overlay
    try:
        overlay = create_analysis_overlay(img_pil, metadata_clean)
        overlay_path = part_dir / f"{index}_analysis.jpg"
        overlay.convert("RGB").save(overlay_path, format="JPEG", quality=85)
    except Exception as e:
        print(f"Could not create overlay for {index}: {e}")
    
    return {
        'original_path': str(original_path),
        'metadata_path': str(metadata_path),
        'overlay_path': str(overlay_path) if 'overlay' in locals() else None
    }