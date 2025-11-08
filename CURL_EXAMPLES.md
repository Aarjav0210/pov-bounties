# Quick Curl Command Reference

Quick reference for testing the Video Validation API with curl.

## Base URL

**Local:**
```bash
API_URL="http://localhost:8000"
```

**Remote (Public IP):**
```bash
API_URL="http://184.105.3.161"
```

---

## 1. Health Check

Check if the API is running and models are loaded:

```bash
curl http://localhost:8000/health | jq '.'
```

**Expected response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true,
  "active_jobs": 0
}
```

---

## 2. Upload Video

Upload a video file to the server:

```bash
curl -X POST http://localhost:8000/upload-video \
  -F "file=@/path/to/your/video.mp4" | jq '.'
```

**Example:**
```bash
curl -X POST http://localhost:8000/upload-video \
  -F file=@~/Downloads/grilled-cheese.mp4 | jq '.'
```

**Expected response:**
```json
{
  "file_id": "uuid-here",
  "filename": "grilled-cheese.mp4",
  "path": "./uploaded_videos/uuid-here.mp4",
  "message": "Video uploaded successfully"
}
```

**Save the `path` value** - you'll need it for validation!

---

## 3. Validate Video (Polling Method)

Start a validation job and poll for results:

```bash
curl -X POST http://localhost:8000/validate-video \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "./uploaded_videos/YOUR-UUID-HERE.mp4",
    "expected_task": "Make grilled cheese"
  }' | jq '.'
```

**Expected response:**
```json
{
  "job_id": "job-uuid-here",
  "status": "processing",
  "message": "Validation job started",
  "poll_url": "/validation-status/job-uuid-here"
}
```

---

## 4. Check Job Status

Poll the status endpoint to get results:

```bash
curl http://localhost:8000/validation-status/JOB-ID-HERE | jq '.'
```

**While processing:**
```json
{
  "job_id": "...",
  "status": "processing",
  "stage": "analyzing_scenes",
  "progress": 0.6
}
```

**When complete:**
```json
{
  "job_id": "...",
  "status": "completed",
  "summary": {
    "task_confirmed": true,
    "verified_scenes": 15,
    "total_scenes": 15
  }
}
```

---

## 5. Stream Results (Real-time)

Get real-time streaming updates using Server-Sent Events:

```bash
curl -X POST http://localhost:8000/validate-video/stream \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "./uploaded_videos/YOUR-UUID-HERE.mp4",
    "expected_task": "Make grilled cheese"
  }'
```

**Pretty-print streaming output:**
```bash
curl -N -X POST http://localhost:8000/validate-video/stream \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "./uploaded_videos/YOUR-UUID-HERE.mp4",
    "expected_task": "Make grilled cheese"
  }' | while read -r line; do
    if [[ $line == data:* ]]; then
      echo "${line#data: }" | jq '.'
    fi
  done
```

---

## 6. List All Jobs

See all validation jobs and their status:

```bash
curl http://localhost:8000/validation-jobs | jq '.'
```

---

## 7. Delete Job

Remove a completed job from memory:

```bash
curl -X DELETE http://localhost:8000/validation-job/JOB-ID-HERE | jq '.'
```

---

## Complete Workflow Example

```bash
# 1. Upload video
UPLOAD=$(curl -s -X POST http://localhost:8000/upload-video \
  -F file=@~/Downloads/grilled-cheese.mp4)

echo "$UPLOAD" | jq '.'

# 2. Extract path
VIDEO_PATH=$(echo "$UPLOAD" | jq -r '.path')
echo "Video path: $VIDEO_PATH"

# 3. Start validation
VALIDATION=$(curl -s -X POST http://localhost:8000/validate-video \
  -H "Content-Type: application/json" \
  -d "{
    \"video_path\": \"$VIDEO_PATH\",
    \"expected_task\": \"Make grilled cheese\"
  }")

echo "$VALIDATION" | jq '.'

# 4. Extract job ID
JOB_ID=$(echo "$VALIDATION" | jq -r '.job_id')
echo "Job ID: $JOB_ID"

# 5. Poll for results (repeat until status != "processing")
while true; do
  STATUS=$(curl -s http://localhost:8000/validation-status/$JOB_ID)
  CURRENT_STATUS=$(echo "$STATUS" | jq -r '.status')
  
  echo "Status: $CURRENT_STATUS"
  
  if [[ "$CURRENT_STATUS" != "processing" ]]; then
    echo "$STATUS" | jq '.'
    break
  fi
  
  sleep 2
done

# 6. Clean up
curl -X DELETE http://localhost:8000/validation-job/$JOB_ID
```

---

## Advanced: Custom Parameters

You can customize validation with additional parameters:

```bash
curl -X POST http://localhost:8000/validate-video \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "./uploaded_videos/YOUR-UUID.mp4",
    "expected_task": "Make grilled cheese",
    "method": "pelt",
    "penalty": 8.0,
    "max_frames": 12,
    "max_retries": 3,
    "domain_match_threshold": 0.6,
    "domain_match_samples": 5
  }' | jq '.'
```

**Parameters:**
- `method`: Scene detection method ("pelt", "binseg", "bottomup", "window")
- `penalty`: Scene detection sensitivity (higher = fewer scenes)
- `max_frames`: Max frames to extract per scene
- `max_retries`: Retry attempts for scene analysis
- `domain_match_threshold`: Required match percentage (0.0-1.0)
- `domain_match_samples`: Frames to sample for domain matching

---

## Tips

### Using with Remote API

Replace `localhost:8000` with your public IP:

```bash
curl http://184.105.3.161/health | jq '.'
```

### No jq installed?

Omit `| jq '.'` to see raw JSON:

```bash
curl http://localhost:8000/health
```

### Save response to file

```bash
curl http://localhost:8000/validation-status/JOB-ID > result.json
```

### Check multiple jobs

```bash
for job_id in job1 job2 job3; do
  echo "Checking $job_id..."
  curl -s http://localhost:8000/validation-status/$job_id | jq '.status'
done
```

---

## Troubleshooting

**Connection refused:**
- Make sure API is running: `ps aux | grep api.py`
- Check API logs for errors

**404 Not Found:**
- Verify the endpoint URL
- Check job ID is correct

**500 Internal Server Error:**
- Check API logs for Python errors
- Verify video file exists at the path

**Model not loaded:**
- Wait for model to load on startup (1-2 minutes)
- Check `model_loaded: true` in `/health`

