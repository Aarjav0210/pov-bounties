"""
Configuration file example for Video Validation API
Copy this to config.py and customize as needed
"""

# API Server Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 1  # Keep at 1 for GPU workloads

# Model Configuration
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
DEVICE = "cuda"  # or "cpu" if no GPU available
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 512 * 28 * 28

# Video Processing Defaults
DEFAULT_METHOD = "pelt"
DEFAULT_PENALTY = 8.0
DEFAULT_FEATURE_TYPE = "combined"
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_FRAMES = 12
DEFAULT_QUESTION_TEMPLATE = "What action(s) is happening in this scene? Be specific, not vague."

# Domain Match Configuration (Initial Quick Validation)
DEFAULT_DOMAIN_MATCH_SAMPLES = 5  # Number of frames to sample for quick validation
DEFAULT_DOMAIN_MATCH_THRESHOLD = 0.6  # Percentage of frames that must match (0.0 to 1.0)
DEFAULT_DOMAIN_MATCH_FPS = 0.2  # FPS for frame extraction (0.2 = 1 frame per 5 seconds)

# Storage Configuration
UPLOAD_DIR = "./uploaded_videos"
MAX_UPLOAD_SIZE_MB = 500  # Maximum video upload size in MB

# CORS Configuration
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
    "https://www.thisispepper.tech",
    "https://pov-bounties-frontend.vercel.app"

    # Add your frontend domain here
    # "https://your-frontend-domain.com",
]

# Logging Configuration
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Job Management
MAX_CONCURRENT_JOBS = 3
JOB_RETENTION_HOURS = 24  # How long to keep completed jobs in memory

# Performance Tuning
ENABLE_FLASH_ATTENTION = True
TORCH_DTYPE = "bfloat16"  # or "float16", "float32"

