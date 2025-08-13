"""
Enhanced prototype image analysis pipeline with comprehensive testing and metrics:
- download images (list of URLs)
- watermark heuristic via pytesseract
- perceptual hash (imagehash) + SSIM filtering
- cluster & choose best cluster
- white-background check by sampling corners
- save selected images locally with detailed metrics
"""

import io
import os
import json
import math
import time
import tempfile
import requests
from datetime import datetime
from pathlib import Path
import pandas as pd
from PIL import Image, ImageStat, ImageOps, ImageDraw, ImageFont
import imagehash
import numpy as np
import boto3
from skimage.metrics import structural_similarity as ssim
import cv2
import pytesseract
from sklearn.cluster import DBSCAN
from collections import Counter, defaultdict

# ---- CONFIG ----
MAX_DOWNLOAD = 20
PHASH_SIZE = 16  # imagehash phash size
PHASH_SIM_THRESHOLD = 6  # hamming distance <= this => similar (tune)
SSIM_SIM_THRESHOLD = 0.55  # pairwise SSIM threshold for somewhat-similar (tune)
WHITE_BG_THRESHOLD = 245  # corner mean >= this -> treat as white
MIN_IMAGES_AFTER_FILTER = 1
S3_BUCKET = os.getenv("S3_BUCKET", "your-bucket-name")
S3_PREFIX = "parts-images/"
LOCAL_SAVE_DIR = "image_analysis_results"
METRICS_DIR = "metrics"

# Init S3 client (assumes AWS creds are configured)
# s3 = boto3.client("s3")

class ImageAnalysisMetrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = {
            'total_parts_processed': 0,
            'total_urls_attempted': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'watermarked_images': 0,
            'clustering_results': [],
            'processing_times': [],
            'final_selections': 0,
            'white_background_count': 0,
            'avg_ssim_scores': [],
            'errors': []
        }
        
    def add_part_metrics(self, part_data):
        self.metrics['total_parts_processed'] += 1
        self.metrics['total_urls_attempted'] += part_data.get('urls_attempted', 0)
        self.metrics['successful_downloads'] += part_data.get('successful_downloads', 0)
        self.metrics['failed_downloads'] += part_data.get('failed_downloads', 0)
        self.metrics['watermarked_images'] += part_data.get('watermarked_count', 0)
        self.metrics['final_selections'] += part_data.get('final_count', 0)
        self.metrics['white_background_count'] += part_data.get('white_bg_count', 0)
        if part_data.get('avg_ssim'):
            self.metrics['avg_ssim_scores'].append(part_data['avg_ssim'])
        if part_data.get('processing_time'):
            self.metrics['processing_times'].append(part_data['processing_time'])
        if part_data.get('cluster_info'):
            self.metrics['clustering_results'].append(part_data['cluster_info'])
        if part_data.get('errors'):
            self.metrics['errors'].extend(part_data['errors'])
    
    def get_summary(self):
        summary = {
            'overview': {
                'parts_processed': self.metrics['total_parts_processed'],
                'total_urls': self.metrics['total_urls_attempted'],
                'download_success_rate': self.metrics['successful_downloads'] / max(self.metrics['total_urls_attempted'], 1),
                'avg_processing_time': np.mean(self.metrics['processing_times']) if self.metrics['processing_times'] else 0,
                'total_final_images': self.metrics['final_selections']
            },
            'quality_metrics': {
                'watermark_detection_rate': self.metrics['watermarked_images'] / max(self.metrics['successful_downloads'], 1),
                'white_background_rate': self.metrics['white_background_count'] / max(self.metrics['final_selections'], 1),
                'avg_ssim_score': np.mean(self.metrics['avg_ssim_scores']) if self.metrics['avg_ssim_scores'] else 0,
                'ssim_std': np.std(self.metrics['avg_ssim_scores']) if self.metrics['avg_ssim_scores'] else 0
            },
            'error_analysis': {
                'total_errors': len(self.metrics['errors']),
                'error_types': Counter([e.get('type', 'unknown') for e in self.metrics['errors']])
            }
        }
        return summary
    
    def save_detailed_report(self, filepath):
        with open(filepath, 'w') as f:
            json.dump({
                'summary': self.get_summary(),
                'detailed_metrics': self.metrics,
                'generated_at': datetime.now().isoformat()
            }, f, indent=2, default=str)

# Global metrics instance
metrics = ImageAnalysisMetrics()

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

# ---- MAIN PROCESS FOR A SINGLE PART ----
def analyze_images_for_part(part_number, image_urls, save_all_steps=True):
    """
    image_urls: list of image urls (max MAX_DOWNLOAD)
    save_all_steps: if True, save intermediate results for analysis
    returns: tuple (results_list, part_metrics)
    """
    start_time = time.time()
    part_metrics = {
        'part_number': part_number,
        'urls_attempted': len(image_urls[:MAX_DOWNLOAD]),
        'successful_downloads': 0,
        'failed_downloads': 0,
        'watermarked_count': 0,
        'final_count': 0,
        'white_bg_count': 0,
        'avg_ssim': 0,
        'cluster_info': {},
        'errors': []
    }

    results = []
    # download up to MAX_DOWNLOAD
    for i, url in enumerate(image_urls[:MAX_DOWNLOAD]):
        print(f"Downloading {i+1}/{min(len(image_urls), MAX_DOWNLOAD)}: {url[:60]}...")
        img, error = download_image(url)
        if img is None:
            part_metrics['failed_downloads'] += 1
            if error:
                part_metrics['errors'].append(error)
            continue
        
        part_metrics['successful_downloads'] += 1
        # store basic meta
        result = {
            "url": url, 
            "img": img, 
            "watermark": False, 
            "wm_reason":"", 
            "phash": None, 
            "ssim": None, 
            "white_ok":False, 
            "corner_means": [],
            "selected": False,
            "download_order": i
        }
        results.append(result)

    if len(results) == 0:
        print(f"No images downloaded for part {part_number}")
        return [], part_metrics

    # Step 1: watermark detection (heuristic)
    print("Step 1: Watermark detection...")
    for r in results:
        wm, reason = detect_watermark_ocr(r["img"])
        r["watermark"] = wm
        r["wm_reason"] = reason
        if wm:
            part_metrics['watermarked_count'] += 1

    # Drop watermarked images
    non_wm = [r for r in results if not r["watermark"]]
    if len(non_wm) == 0:
        print("All images flagged as watermarked. Returning original set for manual review.")
        non_wm = results  # fallback to letting human decide

    # Step 2: compute phash
    print("Step 2: Computing perceptual hashes...")
    for r in non_wm:
        try:
            r["phash"] = phash(r["img"])
        except Exception as e:
            r["phash"] = None
            part_metrics['errors'].append({"type": "phash_error", "error": str(e)})

    # Step 3: cluster by phash hamming distance using DBSCAN
    print("Step 3: Clustering similar images...")
    phashes = [r["phash"] for r in non_wm if r["phash"] is not None]
    if len(phashes) == 0:
        print("No phashes computed; skipping clustering.")
        clustered = non_wm
        part_metrics['cluster_info'] = {"clusters": 0, "method": "no_clustering"}
    else:
        # convert phash to numpy arrays of bits for clustering
        vectors = []
        for p in phashes:
            bits = np.array(p.hash, dtype=int)
            vectors.append(bits)
        vecs = np.stack(vectors).astype(int)
        # DBSCAN with Hamming-like distance (eps tuned)
        db = DBSCAN(eps=PHASH_SIM_THRESHOLD/ (PHASH_SIZE*PHASH_SIZE), min_samples=1, metric="hamming")
        vecs = vecs.reshape(vecs.shape[0], -1)
        labels = db.fit_predict(vecs)
        
        # map back to results
        idx = 0
        for r in non_wm:
            if r["phash"] is not None:
                r["cluster"] = int(labels[idx]); idx+=1
            else:
                r["cluster"] = -1

        # analyze clustering results
        cluster_counts = Counter([r["cluster"] for r in non_wm if r.get("cluster", -1) >= 0])
        part_metrics['cluster_info'] = {
            "clusters": len(cluster_counts),
            "cluster_sizes": dict(cluster_counts),
            "method": "dbscan_hamming"
        }

        # choose largest cluster (most images) as primary cluster
        if cluster_counts:
            primary_cluster = cluster_counts.most_common(1)[0][0]
            clustered = [r for r in non_wm if r.get("cluster", -1) == primary_cluster]
        else:
            clustered = non_wm

    # Step 4: within clustered pick images similar by SSIM and check white backgrounds
    print("Step 4: SSIM analysis and background checking...")
    if len(clustered) > 0:
        # compute pairwise SSIM to cluster center (choose first as center)
        for r in clustered:
            try:
                r["ssim_center"] = compute_ssim(clustered[0]["img"], r["img"])
            except Exception as e:
                r["ssim_center"] = 0.0
                part_metrics['errors'].append({"type": "ssim_error", "error": str(e)})
            
            white_ok, corner_means = corner_white_check(r["img"])
            r["white_ok"] = white_ok
            r["corner_means"] = corner_means
            if white_ok:
                part_metrics['white_bg_count'] += 1

        # Calculate average SSIM
        ssim_scores = [r.get("ssim_center", 0) for r in clustered]
        part_metrics['avg_ssim'] = np.mean(ssim_scores) if ssim_scores else 0

        # filter by SSIM threshold (keep those with good similarity)
        final_candidates = [r for r in clustered if r["ssim_center"] >= SSIM_SIM_THRESHOLD]
        if len(final_candidates) < MIN_IMAGES_AFTER_FILTER:
            # fallback: take clustered set (maybe loosen)
            final_candidates = clustered

        # rank candidates: prefer white backgrounds and higher ssim
        final_candidates.sort(key=lambda x: (int(x["white_ok"]), x.get("ssim_center",0)), reverse=True)

        # keep up to 5 final
        final = final_candidates[:5]
        part_metrics['final_count'] = len(final)

        # Save all results with metadata
        print(f"Step 5: Saving {len(final)} selected images...")
        
        # Save all downloaded images if save_all_steps is True
        if save_all_steps:
            all_dir = Path(LOCAL_SAVE_DIR) / part_number / "all_downloaded"
            all_dir.mkdir(parents=True, exist_ok=True)
            for i, r in enumerate(results):
                save_image_with_analysis(f"{part_number}/all_downloaded", i, r["img"], r)

        # Save final selected images
        output_results = []
        for i, r in enumerate(final):
            r["selected"] = True
            paths = save_image_with_analysis(part_number, f"selected_{i}", r["img"], r)
            
            output_results.append({
                "original_url": r["url"],
                "local_paths": paths,
                "phash": str(r.get("phash", "")),
                "ssim_center": r.get("ssim_center"),
                "white_ok": r.get("white_ok"),
                "corner_means": r.get("corner_means", []),
                "watermark_flag": r.get("watermark"),
                "watermark_reason": r.get("wm_reason", ""),
                "download_order": r.get("download_order"),
                "cluster": r.get("cluster", -1)
            })
    else:
        final = []
        output_results = []

    part_metrics['processing_time'] = time.time() - start_time
    print(f"Part {part_number} completed in {part_metrics['processing_time']:.2f}s")
    print(f"Selected {len(final)} images out of {len(results)} downloaded")
    
    return output_results, part_metrics

def run_batch_analysis(parts_data, save_all_steps=True):
    """
    Run analysis on multiple parts
    parts_data: dict {part_number: [url1, url2, ...]}
    """
    print(f"Starting batch analysis of {len(parts_data)} parts...")
    
    # Reset metrics
    metrics.reset()
    
    # Create main results directory
    results_dir = Path(LOCAL_SAVE_DIR)
    results_dir.mkdir(exist_ok=True)
    
    # Create metrics directory  
    metrics_dir = Path(METRICS_DIR)
    metrics_dir.mkdir(exist_ok=True)
    
    all_results = {}
    
    for part_num, urls in parts_data.items():
        print(f"\n{'='*50}")
        print(f"Processing part: {part_num}")
        print(f"URLs to process: {len(urls)}")
        print(f"{'='*50}")
        
        try:
            results, part_metrics = analyze_images_for_part(part_num, urls, save_all_steps)
            all_results[part_num] = {
                'results': results,
                'metrics': part_metrics
            }
            metrics.add_part_metrics(part_metrics)
            
            # Save individual part metrics
            part_metrics_file = metrics_dir / f"{part_num}_metrics.json"
            with open(part_metrics_file, 'w') as f:
                json.dump({
                    'part_number': part_num,
                    'results': results,
                    'metrics': part_metrics
                }, f, indent=2, default=str)
                
        except Exception as e:
            error_info = {
                "type": "part_processing_error",
                "part": part_num,
                "error": str(e)
            }
            metrics.metrics['errors'].append(error_info)
            print(f"Error processing part {part_num}: {e}")
    
    # Save comprehensive report
    print(f"\n{'='*50}")
    print("BATCH ANALYSIS COMPLETE")
    print(f"{'='*50}")
    
    summary = metrics.get_summary()
    print(f"Parts processed: {summary['overview']['parts_processed']}")
    print(f"Total URLs attempted: {summary['overview']['total_urls']}")
    print(f"Download success rate: {summary['overview']['download_success_rate']:.2%}")
    print(f"Average processing time per part: {summary['overview']['avg_processing_time']:.2f}s")
    print(f"Total final images selected: {summary['overview']['total_final_images']}")
    print(f"Watermark detection rate: {summary['quality_metrics']['watermark_detection_rate']:.2%}")
    print(f"White background rate: {summary['quality_metrics']['white_background_rate']:.2%}")
    print(f"Average SSIM score: {summary['quality_metrics']['avg_ssim_score']:.3f}")
    print(f"Total errors: {summary['error_analysis']['total_errors']}")
    
    # Save final comprehensive report
    final_report_path = metrics_dir / f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    metrics.save_detailed_report(final_report_path)
    
    # Save results summary as CSV for easy analysis
    csv_data = []
    for part_num, part_data in all_results.items():
        for result in part_data['results']:
            csv_row = {
                'part_number': part_num,
                'original_url': result['original_url'],
                'phash': result['phash'],
                'ssim_center': result['ssim_center'],
                'white_ok': result['white_ok'],
                'watermark_flag': result['watermark_flag'],
                'download_order': result['download_order'],
                'cluster': result['cluster']
            }
            csv_data.append(csv_row)
    
    if csv_data:
        df = pd.DataFrame(csv_data)
        csv_path = metrics_dir / f"results_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
    
    print(f"Detailed report saved to: {final_report_path}")
    print(f"All images and analysis saved to: {results_dir}")
    
    return all_results, summary

# Example usage and test data:
if __name__ == "__main__":
    # Test with multiple parts and various image types
    test_data = {
        "PART001_FRUITS": [
            "https://medilifefood.com/wp-content/uploads/2019/10/purepng.com-red-appleappleapplesfruitsweet-1701527180174lrnig.png",
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSez_skPqKDaYWkFNrBrvU8cBxnt405GYOZqw&s",
            "https://th-thumbnailer.cdn-si-edu.com/e5t4TtPeT6Y4yLTLHKxXiNBtwDQ=/470x251/https://tf-cmsv2-smithsonianmag-media.s3.amazonaws.com/filer/20110721125011banana2.jpg",
            "https://cdn.shopify.com/s/files/1/0767/4655/files/IG-FEED-0814_grande.JPG?v=1580934189",
            "https://images.unsplash.com/photo-1560472354-b33ff0c44a43?w=500",
            "https://images.unsplash.com/photo-1546173159-315724a31696?w=500"
        ],
        "PART002_MECHANICAL": [
            "https://images.unsplash.com/photo-1581092918056-0c4c3acd3789?w=500",
            "https://images.unsplash.com/photo-1487754180451-c456f719a1fc?w=500",
            "https://images.unsplash.com/photo-1581092162384-8987c1d64718?w=500",
            "https://example.com/nonexistent1.jpg",  # Test failed download
            "https://example.com/nonexistent2.jpg",  # Test failed download
        ],
        "PART003_ELECTRONICS": [
            "https://images.unsplash.com/photo-1518717758536-85ae29035b6d?w=500",
            "https://images.unsplash.com/photo-1531297484001-80022131f5a1?w=500",
            "https://images.unsplash.com/photo-1593640408182-31c70c8268f5?w=500",
        ]
    }
    
    # Run batch analysis
    results, summary = run_batch_analysis(test_data, save_all_steps=True)
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE!")
    print("Check the following directories:")
    print(f"- Images: {LOCAL_SAVE_DIR}/")
    print(f"- Metrics: {METRICS_DIR}/")
    print("="*50)