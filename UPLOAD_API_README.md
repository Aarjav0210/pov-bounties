# Upload API - Lightweight Video Upload Service

A minimal FastAPI service for handling bounty video uploads to S3, without the heavy ML dependencies.

## Purpose

This is a lightweight alternative to `api.py` that **only** handles video uploads. Perfect for deployment when you don't need real-time video validation.

## Features

- ✅ Upload videos to S3 with VenmoID as filename
- ✅ Store submission metadata (name, email, venmo_id)
- ✅ Automatic fallback to local storage if S3 not configured
- ✅ No ML dependencies (PyTorch, transformers, etc.)
- ✅ Fast startup and low memory usage
- ✅ Easy to deploy on Railway, Render, or Fly.io

## Installation

### Install minimal dependencies:

```bash
cd pov-bounties
pip install -r requirements_upload.txt
```

Or install manually:
```bash
pip install fastapi uvicorn boto3 python-multipart
```

## Configuration

Create a `.env` file with your AWS credentials:

```env
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
S3_BUCKET_NAME=pov-bounties
```

**Without `.env`**: Videos will be saved locally to `uploaded_videos/` directory.

## Running the API

### Option 1: Use the start script

```bash
chmod +x start_upload_api.sh
./start_upload_api.sh
```

### Option 2: Run directly

```bash
python upload_api.py
```

### Option 3: With custom port

```bash
uvicorn upload_api:app --host 0.0.0.0 --port 8001
```

## API Endpoints

### Health Check
```bash
GET /health

Response:
{
  "status": "healthy",
  "aws_configured": true,
  "s3_bucket": "pov-bounties",
  "storage": "s3"
}
```

### Submit Bounty Video
```bash
POST /submit-bounty-video

Form Data:
- file: video file (MP4, AVI, MOV, MKV)
- name: user's full name
- email: user's email
- venmo_id: user's venmo ID (with or without @)

Response:
{
  "message": "Video submitted successfully",
  "file_id": "uuid-here",
  "s3_url": "s3://pov-bounties/username.mp4",
  "status": "pending_review"
}
```

## Deployment

### Railway (Recommended)

1. Create a new project on Railway
2. Connect your GitHub repo
3. Set root directory to `pov-bounties`
4. Add environment variables (AWS credentials)
5. Set start command: `python upload_api.py`

### Render

1. Create new Web Service
2. Build command: `pip install -r requirements_upload.txt`
3. Start command: `python upload_api.py`
4. Add environment variables

### Fly.io

```bash
cd pov-bounties
fly launch
# Follow prompts and add env vars
```

## Frontend Configuration

Update your frontend `.env`:

```env
NEXT_PUBLIC_API_URL=https://your-upload-api.railway.app
```

Or in Vercel dashboard, set the environment variable to your deployed API URL.

## Differences from `api.py`

| Feature | upload_api.py | api.py |
|---------|---------------|--------|
| Video Upload | ✅ | ✅ |
| S3 Storage | ✅ | ✅ |
| ML Validation | ❌ | ✅ |
| PyTorch | ❌ | ✅ |
| Qwen2-VL Model | ❌ | ✅ |
| GPU Required | ❌ | ✅ |
| Memory Usage | ~50MB | ~8GB+ |
| Startup Time | <1s | ~30s |
| Dependencies | 4 | 15+ |

## Storage

### S3 (When configured)
- Videos stored as: `s3://bucket-name/{venmoID}.mp4`
- Metadata stored in S3 object metadata
- Local backup: `uploaded_videos/{submission_id}_metadata.json`

### Local (Fallback)
- Videos: `uploaded_videos/{venmoID}.mp4`
- Metadata: `uploaded_videos/{submission_id}_metadata.json`

## Viewing Submissions

List all submission metadata:
```bash
ls uploaded_videos/*_metadata.json
```

View specific submission:
```bash
cat uploaded_videos/{uuid}_metadata.json
```

Example metadata:
```json
{
  "file_id": "abc-123-def",
  "name": "John Doe",
  "email": "john@example.com",
  "venmo_id": "johndoe",
  "filename": "johndoe.mp4",
  "s3_url": "s3://pov-bounties/johndoe.mp4",
  "submitted_at": "2025-11-08T12:34:56.789",
  "status": "pending_review"
}
```

## Monitoring

Check API status:
```bash
curl http://localhost:8000/health
```

Watch logs:
```bash
tail -f logs/upload_api.log  # if you set up logging
```

## Security Notes

1. **CORS**: By default allows all origins. Update `CORS_ORIGINS` in `config.py` for production
2. **File Size**: No limit set by default - add `max_upload_size` if needed
3. **Rate Limiting**: Not implemented - consider adding for production
4. **Authentication**: Not implemented - add if needed for private bounties

## Troubleshooting

**S3 Upload Fails**:
- Check AWS credentials in `.env`
- Verify S3 bucket exists and you have permissions
- Check `AWS_REGION` matches bucket region

**Local Storage Issues**:
- Ensure `uploaded_videos/` directory is writable
- Check disk space

**CORS Errors**:
- Update `CORS_ORIGINS` in config or code
- Ensure frontend uses correct API URL

## Cost Estimation

**S3 Storage**: ~$0.023 per GB/month
- 100 videos @ 100MB each = ~$0.23/month

**Railway Free Tier**: 500 hours/month
- Should handle upload API easily

**Render Free Tier**: Available but with cold starts
- Good for low traffic

