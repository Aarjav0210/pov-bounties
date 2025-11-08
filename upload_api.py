"""
Lightweight API for bounty video uploads to S3
No ML dependencies - just handles file uploads
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
import uuid
from datetime import datetime
import shutil
import boto3
from botocore.exceptions import ClientError
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
print(f"🔐 Loaded .env file")
print(f"   AWS Key present: {bool(os.getenv('AWS_ACCESS_KEY_ID'))}")
print(f"   S3 Bucket: {os.getenv('S3_BUCKET_NAME', 'pov-bounties')}")

# Configuration
try:
    import config
    UPLOAD_DIR = Path(config.UPLOAD_DIR)
    CORS_ORIGINS = config.CORS_ORIGINS
    API_HOST = config.API_HOST
    API_PORT = config.API_PORT
    print("✅ Loaded configuration from config.py")
except ImportError:
    UPLOAD_DIR = Path("./uploaded_videos")
    CORS_ORIGINS = ["*"]
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    print("⚠️  config.py not found, using default configuration")

app = FastAPI(title="POV Bounties Upload API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create upload directory
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/")
async def root():
    return {
        "message": "POV Bounties Upload API",
        "status": "running",
        "endpoints": ["/health", "/submit-bounty-video"]
    }


@app.get("/health")
async def health_check():
    aws_configured = bool(os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('AWS_SECRET_ACCESS_KEY'))
    return {
        "status": "healthy",
        "aws_configured": aws_configured,
        "s3_bucket": os.getenv('S3_BUCKET_NAME', 'pov-bounties'),
        "storage": "s3" if aws_configured else "local"
    }


@app.post("/submit-bounty-video")
async def submit_bounty_video(
    file: UploadFile = File(...),
    name: str = Form(...),
    email: str = Form(...),
    venmo_id: str = Form(...)
):
    """
    Submit a bounty video with user information
    Uploads to S3 with VenmoID as filename (no validation)
    """
    print("\n" + "="*50)
    print("🎬 RECEIVED POST /submit-bounty-video")
    print(f"📝 Name: {name}")
    print(f"📧 Email: {email}")
    print(f"💰 Venmo ID: {venmo_id}")
    print(f"📹 File: {file.filename} ({file.content_type})")
    print("="*50 + "\n")
    
    if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid video format. Please upload MP4, AVI, MOV, or MKV.")
    
    # Clean venmo_id (remove @ if present)
    clean_venmo_id = venmo_id.strip()
    if clean_venmo_id.startswith('@'):
        clean_venmo_id = clean_venmo_id[1:]
    
    # Get file extension
    file_extension = Path(file.filename).suffix
    
    # Create S3 filename as venmoID.mp4
    s3_filename = f"{clean_venmo_id}{file_extension}"
    
    # Check if AWS credentials are configured
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    s3_bucket = os.getenv('S3_BUCKET_NAME', 'pov-bounties')
    
    file_id = str(uuid.uuid4())
    s3_url = None
    
    try:
        if aws_access_key and aws_secret_key:
            # Upload to S3
            s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=os.getenv('AWS_REGION', 'us-east-1')
            )
            
            # Upload file to S3
            file.file.seek(0)  # Reset file pointer
            s3_client.upload_fileobj(
                file.file,
                s3_bucket,
                s3_filename,
                ExtraArgs={
                    'Metadata': {
                        'name': name,
                        'email': email,
                        'venmo_id': clean_venmo_id,
                        'submission_id': file_id,
                        'submitted_at': datetime.now().isoformat()
                    }
                }
            )
            
            s3_url = f"s3://{s3_bucket}/{s3_filename}"
            print(f"✅ Video uploaded to S3: {s3_url}")
        else:
            # Fallback: Save locally if S3 not configured
            print("⚠️  AWS credentials not found, saving locally")
            local_path = UPLOAD_DIR / s3_filename
            file.file.seek(0)
            with local_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            s3_url = f"local://{local_path}"
        
        # Store submission metadata (could be saved to database)
        submission_metadata = {
            "file_id": file_id,
            "name": name,
            "email": email,
            "venmo_id": clean_venmo_id,
            "filename": s3_filename,
            "s3_url": s3_url,
            "submitted_at": datetime.now().isoformat(),
            "status": "pending_review"
        }
        
        # Save metadata to a JSON file (in production, this would go to a database)
        metadata_path = UPLOAD_DIR / f"{file_id}_metadata.json"
        with metadata_path.open("w") as f:
            json.dump(submission_metadata, f, indent=2)
        
        return {
            "message": "Video submitted successfully",
            "file_id": file_id,
            "s3_url": s3_url,
            "status": "pending_review"
        }
        
    except ClientError as e:
        print(f"❌ S3 upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload to S3: {str(e)}")
    except Exception as e:
        print(f"❌ Submission error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit video: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print(f"\n🚀 Starting Upload API on {API_HOST}:{API_PORT}")
    print(f"📁 Upload directory: {UPLOAD_DIR}")
    print(f"☁️  S3 Bucket: {os.getenv('S3_BUCKET_NAME', 'pov-bounties')}")
    print(f"🌐 CORS Origins: {CORS_ORIGINS}\n")
    uvicorn.run(app, host=API_HOST, port=API_PORT)

