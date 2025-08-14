#config.py
import os

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
