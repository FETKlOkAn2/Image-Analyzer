#!/usr/bin/env python3
"""
Simple image analysis pipeline for testing individual images.
Shows all processing steps on a single image with clear output.
"""

import requests
import io
import numpy as np
from PIL import Image, ImageStat, ImageOps
import imagehash
import pytesseract
from skimage.metrics import structural_similarity as ssim
import cv2
from pathlib import Path
import json

# Configuration
PHASH_SIZE = 8
WHITE_BG_THRESHOLD = 245
OUTPUT_DIR = "test_results"

class SimpleImageAnalyzer:
    def __init__(self, output_dir=OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def download_image(self, url, timeout=10):
        """Download image from URL"""
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            response = requests.get(url, timeout=timeout, headers=headers)
            response.raise_for_status()
            
            img = Image.open(io.BytesIO(response.content)).convert("RGBA")
            return img, True, "Downloaded successfully"
        except Exception as e:
            return None, False, f"Download failed: {str(e)}"
    
    def detect_watermark(self, img):
        """Simple watermark detection using OCR"""
        try:
            # Check bottom area where watermarks often appear
            w, h = img.size
            bottom_area = img.crop((0, int(h*0.7), w, h))
            
            # Convert to grayscale and enhance contrast for OCR
            gray = ImageOps.autocontrast(bottom_area.convert("L"))
            
            # Extract text
            text = pytesseract.image_to_string(gray, config="--psm 6").strip()
            
            # Simple heuristic: if significant text found, likely watermark
            has_watermark = len(text) > 8
            return has_watermark, text if has_watermark else "No text detected"
            
        except Exception as e:
            return False, f"OCR error: {str(e)}"
    
    def compute_perceptual_hash(self, img):
        """Compute perceptual hash for similarity comparison"""
        try:
            return imagehash.phash(img, hash_size=PHASH_SIZE)
        except Exception as e:
            return None
    
    def check_white_background(self, img):
        """Check if image has white background by examining corners"""
        try:
            w, h = img.size
            corner_size = int(min(w, h) * 0.05)  # 5% of smallest dimension
            
            # Extract corner patches
            corners = [
                img.crop((0, 0, corner_size, corner_size)),  # top-left
                img.crop((w-corner_size, 0, w, corner_size)),  # top-right
                img.crop((0, h-corner_size, corner_size, h)),  # bottom-left
                img.crop((w-corner_size, h-corner_size, w, h))  # bottom-right
            ]
            
            corner_means = []
            for corner in corners:
                stat = ImageStat.Stat(corner.convert("L"))
                corner_means.append(stat.mean[0])
            
            # All corners should be bright (near white)
            is_white = all(mean >= WHITE_BG_THRESHOLD for mean in corner_means)
            avg_brightness = sum(corner_means) / len(corner_means)
            
            return is_white, avg_brightness, corner_means
            
        except Exception as e:
            return False, 0, []
    
    def compute_similarity(self, img1, img2):
        """Compute SSIM similarity between two images"""
        try:
            # Resize to standard size for comparison
            size = (256, 256)
            gray1 = np.array(img1.convert("L").resize(size))
            gray2 = np.array(img2.convert("L").resize(size))
            
            similarity = ssim(gray1, gray2)
            return similarity
        except Exception as e:
            return 0.0
    
    def analyze_single_image(self, image_input, save_results=True):
        """
        Analyze a single image through all steps
        image_input: can be URL string or PIL Image object
        """
        results = {
            "input_type": "",
            "download": {"success": False, "message": ""},
            "watermark": {"detected": False, "text": ""},
            "perceptual_hash": "",
            "white_background": {"is_white": False, "avg_brightness": 0, "corner_means": []},
            "image_properties": {"width": 0, "height": 0, "mode": ""},
            "recommendation": "REJECT"
        }
        
        # Step 1: Get the image
        if isinstance(image_input, str):
            # It's a URL
            results["input_type"] = "URL"
            print(f"Downloading image from: {image_input[:60]}...")
            img, success, message = self.download_image(image_input)
            results["download"] = {"success": success, "message": message}
            
            if not success:
                print(f"Download failed: {message}")
                return results
            print("Downloaded successfully")
        else:
            # It's already a PIL Image
            results["input_type"] = "PIL_Image"
            img = image_input
            results["download"] = {"success": True, "message": "Direct PIL image provided"}
            print(" Using provided PIL image")
        
        # Store image properties
        results["image_properties"] = {
            "width": img.size[0],
            "height": img.size[1],
            "mode": img.mode
        }
        
        # Step 2: Watermark detection
        print("Checking for watermarks...")
        has_watermark, watermark_text = self.detect_watermark(img)
        results["watermark"] = {"detected": has_watermark, "text": watermark_text}
        
        if has_watermark:
            print(f"✗ Watermark detected: {watermark_text}")
            results["recommendation"] = "REJECT - Watermark detected"
        else:
            print("✓ No watermark detected")
        
        # Step 3: Perceptual hash
        print("Computing perceptual hash...")
        phash = self.compute_perceptual_hash(img)
        results["perceptual_hash"] = str(phash) if phash else "Failed to compute"
        print(f"✓ Perceptual hash: {results['perceptual_hash']}")
        
        # Step 4: White background check
        print("Checking background...")
        is_white, avg_brightness, corner_means = self.check_white_background(img)
        results["white_background"] = {
            "is_white": is_white,
            "avg_brightness": round(avg_brightness, 2),
            "corner_means": [round(m, 2) for m in corner_means]
        }
        
        if is_white:
            print(f"White background detected (avg brightness: {avg_brightness:.1f})")
        else:
            print(f"Non-white background (avg brightness: {avg_brightness:.1f})")
        
        # Step 5: Final recommendation
        if not has_watermark and is_white:
            results["recommendation"] = "ACCEPT - Good quality image"
            print("RECOMMENDATION: ACCEPT")
        elif not has_watermark:
            results["recommendation"] = "MAYBE - No watermark but non-white background"
            print("RECOMMENDATION: MAYBE")
        else:
            results["recommendation"] = "REJECT - Has watermark"
            print("RECOMMENDATION: REJECT")
        
        # Save results if requested
        if save_results:
            self.save_analysis_results(img, results, image_input)
        
        return results, img
    
    def save_analysis_results(self, img, results, original_input):
        """Save analysis results and image"""
        # Create filename based on input
        if isinstance(original_input, str):
            # URL - use last part as filename
            filename_base = Path(original_input).name.split('.')[0] or "image"
        else:
            filename_base = "uploaded_image"
        
        # Save the image
        img_path = self.output_dir / f"{filename_base}_analyzed.jpg"
        img.convert("RGB").save(img_path, quality=90)
        
        # Save analysis results
        json_path = self.output_dir / f"{filename_base}_analysis.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {self.output_dir}")
        print(f"  - Image: {img_path}")
        print(f"  - Analysis: {json_path}")
    
    def compare_images(self, img1, img2):
        """Compare similarity between two images"""
        phash1 = self.compute_perceptual_hash(img1)
        phash2 = self.compute_perceptual_hash(img2)
        ssim_score = self.compute_similarity(img1, img2)
        
        # Hamming distance for perceptual hashes
        hamming_distance = None
        if phash1 and phash2:
            hamming_distance = phash1 - phash2
        
        return {
            "ssim": round(ssim_score, 3),
            "hamming_distance": hamming_distance,
            "phash1": str(phash1),
            "phash2": str(phash2),
            "similar": ssim_score > 0.7 and (hamming_distance is None or hamming_distance < 5)
        }


def main():
    """Example usage"""
    analyzer = SimpleImageAnalyzer()
    
    test_urls = [
        "https://images.unsplash.com/photo-1581092918056-0c4c3acd3789?w=500",  # mechanical part
        "https://images.unsplash.com/photo-1560472354-b33ff0c44a43?w=500",   # apple
    ]
    
    print("="*60)
    print("SIMPLE IMAGE ANALYSIS PIPELINE")
    print("="*60)
    
    for i, url in enumerate(test_urls, 1):
        print(f"\n--- ANALYZING IMAGE {i} ---")
        results, img = analyzer.analyze_single_image(url)
        
        print(f"\nSUMMARY:")
        print(f"  Size: {results['image_properties']['width']}x{results['image_properties']['height']}")
        print(f"  Watermark: {'YES' if results['watermark']['detected'] else 'NO'}")
        print(f"  White BG: {'YES' if results['white_background']['is_white'] else 'NO'}")
        print(f"  Hash: {results['perceptual_hash']}")
        print(f"  Decision: {results['recommendation']}")
        
        print("-" * 50)

if __name__ == "__main__":
    main()