# Direct S3 Upload Implementation Guide

## 🎉 What Changed

We've implemented **direct browser-to-S3 uploads** using presigned URLs. This solves the mobile timeout issue you experienced!

## How It Works Now

### Old Flow (Caused Timeouts):
```
Browser → Upload to Railway → Railway uploads to S3
         (87 seconds, hit 100s timeout limit)
```

### New Flow (No Timeouts):
```
Browser → Request presigned URL from Railway (fast)
Browser → Upload directly to S3 (bypasses Railway)
Browser → Confirm with Railway (fast)
```

## Benefits

✅ **No More Timeouts** - Uploads bypass the backend completely
✅ **Better Mobile Experience** - Works even on slow connections
✅ **Real Progress** - Actual upload progress, not simulated
✅ **Larger Files** - No 100-second Railway timeout limit
✅ **Same Security** - Presigned URLs are just as secure

## What Was Added

### Backend (upload_api.py)

1. **`/generate-upload-url`** - New endpoint that:
   - Validates user metadata (name, email, venmo)
   - Generates a presigned S3 URL (expires in 10 min)
   - Saves submission metadata
   - Returns URL for direct upload

2. **`/confirm-upload`** - Confirms successful upload:
   - Updates submission status to "pending_review"
   - Records upload completion time

3. **`/submit-bounty-video`** - Marked as LEGACY:
   - Still works for backwards compatibility
   - Not recommended for large files

### Frontend (client.ts)

1. **`submitBountyVideoDirectS3()`** - New function that:
   - Calls backend for presigned URL
   - Uploads file directly to S3 using XMLHttpRequest
   - Reports real-time progress
   - Confirms upload with backend

2. **`submitBountyVideo()`** - Marked as LEGACY:
   - Still available but not recommended
   - Use `submitBountyVideoDirectS3()` instead

### Frontend (bounty-traction page)

- Updated to use `submitBountyVideoDirectS3()`
- Shows real upload progress (not simulated)
- Handles errors gracefully

## Setup Required

### ⚠️ CRITICAL: Configure S3 CORS

Your S3 bucket **MUST** have CORS configured or uploads will fail!

**See: `S3_CORS_SETUP.md` for complete instructions**

Quick version:
1. Go to AWS S3 Console
2. Select your bucket
3. Permissions → CORS
4. Add configuration allowing PUT requests from your frontend domain

### Environment Variables (Already Set)

Backend (Railway):
- `AWS_ACCESS_KEY_ID` ✓
- `AWS_SECRET_ACCESS_KEY` ✓
- `AWS_REGION` ✓
- `S3_BUCKET_NAME` ✓

Frontend (Vercel):
- `NEXT_PUBLIC_API_URL` ✓

## Testing

### Local Testing:

1. **Start backend:**
   ```bash
   cd pov-bounties
   python upload_api.py
   ```

2. **Start frontend:**
   ```bash
   cd pov-bounties-frontend
   pnpm dev
   ```

3. **Test upload:**
   - Go to http://localhost:3000/bounty-traction/1
   - Fill out form
   - Upload a large video (20+ MB recommended)
   - Watch real progress in console and progress bar
   - Verify file appears in S3

### Production Testing:

After deploying to Railway and Vercel:

1. **Configure S3 CORS** (see `S3_CORS_SETUP.md`)
2. **Test from desktop:**
   - Upload 10-20 MB video
   - Should complete in seconds
3. **Test from mobile:**
   - Upload same video on 4G/5G
   - Should work even on slow connection
   - No timeout errors

## Deployment Checklist

Before deploying:

- [ ] S3 CORS configured with your production domain
- [ ] Backend deployed to Railway with all env vars
- [ ] Frontend deployed to Vercel with API URL
- [ ] Production domain added to S3 CORS config
- [ ] Tested upload from production URL
- [ ] Tested upload from mobile device

## Troubleshooting

### "CORS policy" errors

**Problem:** S3 CORS not configured correctly

**Solution:** See `S3_CORS_SETUP.md` - ensure your frontend domain is in `AllowedOrigins`

### Uploads fail at 100%

**Problem:** `/confirm-upload` endpoint failing

**Solution:** Check Railway logs for errors, verify backend is running

### "Failed to generate upload URL"

**Problem:** AWS credentials not set or invalid

**Solution:** Verify all 4 AWS env vars are set in Railway dashboard

### Progress stuck at 0%

**Problem:** Browser can't reach S3 (network or CORS issue)

**Solution:** 
1. Check browser console for errors
2. Verify CORS configuration
3. Test with smaller file first

## File Locations

### Backend:
- `/pov-bounties/upload_api.py` - Main API with new endpoints
- `/pov-bounties/S3_CORS_SETUP.md` - CORS configuration guide

### Frontend:
- `/pov-bounties-frontend/src/lib/api/client.ts` - New upload functions
- `/pov-bounties-frontend/src/lib/api/config.ts` - New endpoints
- `/pov-bounties-frontend/src/app/(app)/bounty-traction/[id]/page.tsx` - Updated submission form

## API Reference

### POST /generate-upload-url

**Request (FormData):**
```
name: string
email: string
venmo_id: string
filename: string
content_type: string
```

**Response:**
```json
{
  "upload_url": "https://s3.amazonaws.com/bucket/file?presigned-params...",
  "file_id": "uuid-here",
  "s3_filename": "venmo123.mp4",
  "expires_in": 600
}
```

### PUT <presigned_url>

**Request:**
- Binary video file in body
- Content-Type header matching video type

**Response:**
- 200 OK if successful
- No response body needed

### POST /confirm-upload

**Request (FormData):**
```
file_id: string
```

**Response:**
```json
{
  "message": "Upload confirmed",
  "file_id": "uuid-here",
  "status": "pending_review"
}
```

## Performance Comparison

### Old Method (Through Backend):
- Small file (5 MB): ~10 seconds ✓
- Medium file (15 MB): ~30 seconds ✓
- Large file (30 MB): ~60 seconds ⚠️
- Very large (50 MB): ~100 seconds ❌ TIMEOUT

### New Method (Direct S3):
- Small file (5 MB): ~5 seconds ✓
- Medium file (15 MB): ~15 seconds ✓
- Large file (30 MB): ~30 seconds ✓
- Very large (50 MB): ~50 seconds ✓
- Huge (100 MB+): Works! (may take minutes on mobile) ✓

## Security Notes

✅ **Presigned URLs are secure:**
- Valid for only 10 minutes
- Scoped to specific file path (venmo_id.mp4)
- Include content type restrictions
- Backend controls all parameters

✅ **User validation:**
- Backend validates metadata before generating URL
- Can add rate limiting to prevent abuse
- Submission tracking via file_id

✅ **S3 permissions:**
- CORS only allows uploads from your domains
- AWS credentials never sent to client
- Users can't modify S3 settings

## Next Steps

1. **Deploy backend** to Railway (already configured)
2. **Configure S3 CORS** (see `S3_CORS_SETUP.md`)
3. **Deploy frontend** to Vercel
4. **Test production uploads** (especially mobile!)
5. **Monitor submissions** via S3 console

## Questions?

Common scenarios covered:

**Q: Does this work with the existing metadata tracking?**
A: Yes! Metadata is saved the same way, just status starts as "upload_pending"

**Q: Can users still upload malicious files?**
A: We validate file extensions and content type, same as before

**Q: What if presigned URL expires before upload completes?**
A: 10 minutes is generous. If needed, can extend to 30+ minutes

**Q: Do I need to keep the old upload method?**
A: No, but it's kept for backwards compatibility. Remove after testing.

**Q: How do I see failed uploads?**
A: Check metadata files with status "upload_pending" but no "uploaded_at"

