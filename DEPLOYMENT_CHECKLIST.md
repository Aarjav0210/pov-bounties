# 🚀 Deployment Checklist - Direct S3 Upload

## ✅ What's Been Implemented

- ✅ Backend: Presigned S3 URL generation (`/generate-upload-url`)
- ✅ Backend: Upload confirmation endpoint (`/confirm-upload`)
- ✅ Frontend: Direct S3 upload with real-time progress
- ✅ Security: Metadata validation and presigned URL expiry
- ✅ Documentation: Complete setup guides

## 📋 Pre-Deployment Checklist

### 1. S3 CORS Configuration (CRITICAL!)

**Status:** ⚠️ MUST BE DONE BEFORE DEPLOYMENT

**Action Required:**
1. Go to [AWS S3 Console](https://console.aws.amazon.com/s3/)
2. Select your bucket (`pepper-videos`)
3. Permissions → CORS → Edit
4. Add this configuration:

```json
[
    {
        "AllowedHeaders": ["*"],
        "AllowedMethods": ["GET", "PUT", "POST"],
        "AllowedOrigins": [
            "http://localhost:3000",
            "https://pov-bounties-frontend.vercel.app",
            "https://*.vercel.app"
        ],
        "ExposeHeaders": ["ETag"],
        "MaxAgeSeconds": 3000
    }
]
```

5. **Update `AllowedOrigins`** with your actual Vercel domain!

**See:** `S3_CORS_SETUP.md` for detailed instructions

### 2. Backend Environment Variables

**Status:** ✅ Already set in Railway

Confirm these are set:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `S3_BUCKET_NAME`

### 3. Frontend Environment Variable

**Status:** ✅ Already set in Vercel

Confirm this is set:
- `NEXT_PUBLIC_API_URL` = `https://pov-bounties-production.up.railway.app`

### 4. Railway Configuration

**Status:** ✅ Already configured

Verify:
- Root Directory: `pov-bounties`
- Start Command: `uvicorn upload_api:app --host 0.0.0.0 --port $PORT`
- Build Command: `pip install -r requirements_upload.txt`

## 🧪 Testing Before Going Live

### Local Testing (Optional but Recommended)

1. **Test backend:**
   ```bash
   cd pov-bounties
   python test_direct_upload.py
   ```
   
   Expected: All checks pass ✅

2. **Test frontend locally:**
   ```bash
   # Terminal 1 - Backend
   cd pov-bounties
   python upload_api.py
   
   # Terminal 2 - Frontend
   cd pov-bounties-frontend
   pnpm dev
   ```
   
   Expected: Upload works, progress bar shows real progress

### Production Testing (After Deployment)

1. **Test from desktop:**
   - Go to your Vercel URL
   - Navigate to any bounty-traction page
   - Upload a 10-20 MB video
   - Verify: Upload completes, success page shows
   - Check: File appears in S3 bucket

2. **Test from mobile:**
   - Open Vercel URL on phone
   - Upload a 20+ MB video on 4G/5G
   - Verify: No timeout errors
   - Verify: Real progress updates
   - Check: File in S3

3. **Check backend logs:**
   - Railway Dashboard → Your Service → Logs
   - Look for: "🔗 GENERATE PRESIGNED URL REQUEST"
   - Look for: "✅ Upload confirmed for file_id: ..."

## 🚀 Deployment Steps

### Step 1: Deploy Backend to Railway

**If already deployed:**
```bash
git add .
git commit -m "Add direct S3 upload with presigned URLs"
git push origin main
```
Railway will auto-deploy.

**If first deploy:**
1. Push code to GitHub
2. Railway → New Project → Deploy from GitHub
3. Select repo and configure as above

### Step 2: Configure S3 CORS

⚠️ **DO THIS BEFORE TESTING!**

See `S3_CORS_SETUP.md` for complete instructions.

### Step 3: Deploy Frontend to Vercel

**If already deployed:**
```bash
cd pov-bounties-frontend
git add .
git commit -m "Update to use direct S3 upload"
git push origin main
```
Vercel will auto-deploy.

**If first deploy:**
```bash
cd pov-bounties-frontend
vercel
```

### Step 4: Verify Environment Variables

**Railway:**
- Dashboard → Service → Variables
- Confirm all 4 AWS vars are set

**Vercel:**
- Dashboard → Project → Settings → Environment Variables
- Confirm `NEXT_PUBLIC_API_URL` is set with `https://` prefix

### Step 5: Test Production

Follow "Production Testing" steps above.

## 🐛 Common Issues & Fixes

### Issue: CORS errors in browser

**Symptoms:** Console shows "blocked by CORS policy"

**Fix:**
1. Verify S3 CORS is configured
2. Check your frontend domain is in `AllowedOrigins`
3. Make sure to include `https://` in domain
4. Wait 30 seconds for CORS to propagate

### Issue: "Failed to generate upload URL"

**Symptoms:** Upload fails immediately

**Fix:**
1. Check Railway logs for errors
2. Verify AWS credentials are set
3. Test with `curl https://your-railway-url.railway.app/health`

### Issue: Upload gets stuck at 0%

**Symptoms:** Progress bar doesn't move

**Fix:**
1. Check CORS configuration
2. Open browser console for errors
3. Verify presigned URL was generated (check Network tab)

### Issue: Upload reaches 100% but fails

**Symptoms:** Progress completes but shows error

**Fix:**
1. Check `/confirm-upload` endpoint in Railway logs
2. Verify backend is still running
3. Check S3 permissions

## 📊 Success Metrics

After deployment, verify:

- ✅ Desktop uploads complete in < 30 seconds
- ✅ Mobile uploads don't timeout
- ✅ Progress bar shows real progress (0% → 100%)
- ✅ Files appear in S3 with correct venmo_id names
- ✅ Metadata JSON files created in backend
- ✅ No 499 errors in Railway logs
- ✅ No CORS errors in browser console

## 📚 Reference Documentation

- `DIRECT_S3_UPLOAD_GUIDE.md` - Complete implementation details
- `S3_CORS_SETUP.md` - S3 CORS configuration
- `UPLOAD_API_README.md` - Original upload API docs
- `test_direct_upload.py` - Backend testing script

## 🎯 Post-Deployment

After successful deployment:

1. **Monitor submissions:**
   - Check S3 bucket regularly
   - Review metadata JSON files
   - Look for any failed uploads (status: "upload_pending")

2. **User feedback:**
   - Test on various devices
   - Monitor for error reports
   - Track completion rates

3. **Optional improvements:**
   - Add rate limiting to `/generate-upload-url`
   - Set up S3 lifecycle policies for old files
   - Add email notifications on successful upload
   - Implement admin dashboard for submissions

## 🎉 You're Ready!

Once all checkboxes above are ✅, you're ready to deploy!

The direct S3 upload will solve your mobile timeout issues and provide a much better user experience.

**Questions?** See the reference docs above or check the implementation in:
- Backend: `upload_api.py`
- Frontend: `src/lib/api/client.ts`
- UI: `src/app/(app)/bounty-traction/[id]/page.tsx`

