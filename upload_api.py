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
    API_PORT = 8080
    print("⚠️  config.py not found, using default configuration")

app = FastAPI(title="POV Bounties Upload API", version="1.0.0")

# CORS configuration
# if os.getenv('ALLOW_ALL_CORS') == 'true':
#     CORS_ORIGINS = ["*"]
#     print("⚠️  CORS set to allow ALL origins (not recommended for production)")

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


@app.post("/generate-upload-url")
async def generate_upload_url(
    name: str = Form(...),
    email: str = Form(...),
    venmo_id: str = Form(...),
    filename: str = Form(...),
    content_type: str = Form(...)
):
    """
    Generate a presigned S3 URL for direct upload from browser
    This bypasses the backend for file upload, preventing timeouts
    """
    print("\n" + "="*50)
    print("🔗 GENERATE PRESIGNED URL REQUEST")
    print(f"📝 Name: {name}")
    print(f"📧 Email: {email}")
    print(f"💰 Venmo ID: {venmo_id}")
    print(f"📹 Filename: {filename}")
    print("="*50 + "\n")
    
    # Validate file extension
    if not filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid video format. Please upload MP4, AVI, MOV, or MKV.")
    
    # Clean venmo_id
    clean_venmo_id = venmo_id.strip()
    if clean_venmo_id.startswith('@'):
        clean_venmo_id = clean_venmo_id[1:]
    
    # Generate unique submission ID first
    file_id = str(uuid.uuid4())
    
    # Get file extension
    file_extension = Path(filename).suffix
    # Create unique filename with submission ID to allow duplicates
    s3_filename = f"{clean_venmo_id}_{file_id}{file_extension}"
    
    # Check AWS credentials
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    s3_bucket = os.getenv('S3_BUCKET_NAME', 'pov-bounties')
    
    if not aws_access_key or not aws_secret_key:
        raise HTTPException(status_code=503, detail="S3 upload not configured")
    
    try:
        # Create S3 client with signature version 4 (required for presigned URLs)
        from botocore.config import Config
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=os.getenv('AWS_REGION', 'us-east-2'),
            config=Config(signature_version='s3v4')
        )
        
        # Generate presigned URL (valid for 10 minutes)
        # Note: Removed Metadata from presigned URL to avoid CORS preflight issues
        # Metadata is stored in backend database instead
        presigned_url = s3_client.generate_presigned_url(
            'put_object',
            Params={
                'Bucket': s3_bucket,
                'Key': s3_filename,
                'ContentType': content_type
            },
            ExpiresIn=600  # 10 minutes
        )
        
        # Store submission metadata
        submission_metadata = {
            "file_id": file_id,
            "name": name,
            "email": email,
            "venmo_id": clean_venmo_id,
            "filename": s3_filename,
            "s3_url": f"s3://{s3_bucket}/{s3_filename}",
            "submitted_at": datetime.now().isoformat(),
            "status": "upload_pending"
        }
        
        # Save metadata
        metadata_path = UPLOAD_DIR / f"{file_id}_metadata.json"
        with metadata_path.open("w") as f:
            json.dump(submission_metadata, f, indent=2)
        
        print(f"✅ Presigned URL generated for: {s3_filename}")
        
        return {
            "upload_url": presigned_url,
            "file_id": file_id,
            "s3_filename": s3_filename,
            "expires_in": 600
        }
        
    except ClientError as e:
        print(f"❌ S3 error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate upload URL: {str(e)}")


@app.post("/confirm-upload")
async def confirm_upload(
    file_id: str = Form(...),
):
    """
    Confirm that upload to S3 was successful
    Updates submission status and adds metadata to S3 object
    """
    print(f"✅ Upload confirmed for file_id: {file_id}")
    
    # Update metadata
    metadata_path = UPLOAD_DIR / f"{file_id}_metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r") as f:
            metadata = json.load(f)
        
        metadata["status"] = "pending_review"
        metadata["uploaded_at"] = datetime.now().isoformat()
        
        with metadata_path.open("w") as f:
            json.dump(metadata, f, indent=2)
        
        # Update S3 object metadata
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        s3_bucket = os.getenv('S3_BUCKET_NAME', 'pepper-videos')
        
        if aws_access_key and aws_secret_key:
            try:
                from botocore.config import Config
                s3_client = boto3.client(
                    's3',
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    region_name=os.getenv('AWS_REGION', 'us-east-2'),
                    config=Config(signature_version='s3v4')
                )
                
                # Copy object to itself with updated metadata
                s3_filename = metadata["filename"]
                s3_client.copy_object(
                    Bucket=s3_bucket,
                    CopySource={'Bucket': s3_bucket, 'Key': s3_filename},
                    Key=s3_filename,
                    Metadata={
                        'name': metadata["name"],
                        'email': metadata["email"],
                        'venmo_id': metadata["venmo_id"],
                        'submission_id': file_id,
                        'submitted_at': metadata["submitted_at"],
                        'uploaded_at': metadata["uploaded_at"]
                    },
                    MetadataDirective='REPLACE'
                )
                
                print(f"✅ S3 metadata updated for: {s3_filename}")
            except ClientError as e:
                print(f"⚠️  Failed to update S3 metadata: {e}")
                # Don't fail the request if metadata update fails
        
        return {
            "message": "Upload confirmed",
            "file_id": file_id,
            "status": "pending_review"
        }
    else:
        raise HTTPException(status_code=404, detail="Submission not found")


@app.post("/submit-bounty-video")
async def submit_bounty_video(
    file: UploadFile = File(...),
    name: str = Form(...),
    email: str = Form(...),
    venmo_id: str = Form(...)
):
    """
    LEGACY: Submit a bounty video with user information (uploads through backend)
    Uploads to S3 with VenmoID as filename (no validation)
    
    NOTE: For large files or mobile uploads, use /generate-upload-url instead
    to upload directly to S3 and avoid timeouts
    """
    print("\n" + "="*50)
    print("🎬 RECEIVED POST /submit-bounty-video (LEGACY)")
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
    
    # Generate unique submission ID first
    file_id = str(uuid.uuid4())
    
    # Get file extension
    file_extension = Path(file.filename).suffix
    
    # Create unique S3 filename with UUID to allow duplicates
    s3_filename = f"{clean_venmo_id}_{file_id}{file_extension}"
    
    # Check if AWS credentials are configured
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    s3_bucket = os.getenv('S3_BUCKET_NAME', 'pov-bounties')
    s3_url = None
    
    try:
        if aws_access_key and aws_secret_key:
            # Upload to S3
            from botocore.config import Config
            s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=os.getenv('AWS_REGION', 'us-east-2'),
                config=Config(signature_version='s3v4')
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

