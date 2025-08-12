Image Analyzer
Image Analyzer is a Python prototype for smart image filtering and clustering, designed to clean and prepare product image datasets before uploading.

Features
Image download — retrieves images from a list of URLs for a single part.

Watermark removal — uses a simple OCR / overlay-text heuristic with pytesseract.

Similarity analysis — compares images using:

Perceptual hashes (imagehash)

Structural Similarity Index (SSIM via skimage)

Clustering — groups similar images with DBSCAN on Hamming distances of phash.

Quality checks — detects corner whiteness for cleaner visuals.

Cloud upload — sends the final set of approved images to Amazon S3 with boto3.

Tech Stack
Tool / Library	Purpose
Python	Core programming language
pytesseract	OCR for watermark detection
imagehash	Perceptual hashing
scikit-image	SSIM similarity analysis
scikit-learn	DBSCAN clustering
boto3	S3 upload automation

Workflow
mermaid
Kopírovať
Upraviť
flowchart TD
    A[Download Image URLs] --> B[Remove Watermarks]
    B --> C[Compute Hashes + SSIM]
    C --> D[Cluster Similar Images]
    D --> E[Check Corner Whiteness]
    E --> F[Upload to S3]
Example Usage
bash
Kopírovať
Upraviť
# Install dependencies
pip install -r requirements.txt

# Run the analyzer
python analyzer.py --input urls.txt --bucket my-s3-bucket
