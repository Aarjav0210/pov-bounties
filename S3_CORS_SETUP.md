# S3 CORS Configuration for Direct Uploads

## Overview

For direct browser-to-S3 uploads to work, your S3 bucket **MUST** have CORS (Cross-Origin Resource Sharing) configured properly.

Without CORS, browsers will block the upload requests as a security measure.

## Setup Instructions

### 1. Go to AWS S3 Console

1. Navigate to [AWS S3 Console](https://console.aws.amazon.com/s3/)
2. Click on your bucket (`pepper-videos` or whatever name you're using)
3. Go to the **Permissions** tab
4. Scroll down to **Cross-origin resource sharing (CORS)**
5. Click **Edit**

### 2. Add CORS Configuration

Paste the following JSON configuration:

```json
[
    {
        "AllowedHeaders": [
            "*"
        ],
        "AllowedMethods": [
            "GET",
            "PUT",
            "POST"
        ],
        "AllowedOrigins": [
            "http://localhost:3000",
            "https://pov-bounties-frontend.vercel.app",
            "https://*.vercel.app"
        ],
        "ExposeHeaders": [
            "ETag"
        ],
        "MaxAgeSeconds": 3000
    }
]
```

### 3. Customize for Your Domain

**Important:** Update the `AllowedOrigins` array to include:
- Your production frontend domain(s)
- Your preview/staging domains
- `http://localhost:3000` for local development

**Security Note:** Using `"*"` (allow all origins) works but is NOT recommended for production.

### 4. Save Changes

Click **Save changes** in the AWS console.

## Testing CORS Configuration

### Quick Test:

1. Open your frontend in the browser
2. Open Developer Console (F12)
3. Try to submit a video
4. Check the **Network** tab for the PUT request to S3
5. If you see a **CORS error**, double-check your configuration

### Common CORS Errors:

**Error:** `Access to XMLHttpRequest at 'https://s3.amazonaws.com/...' has been blocked by CORS policy`

**Solution:** Make sure:
- Your frontend domain is in `AllowedOrigins`
- Include the protocol (`https://` or `http://`)
- CORS config has been saved and applied (can take a few seconds)

## How It Works

1. Frontend requests presigned URL from backend
2. Backend generates URL valid for 10 minutes
3. Frontend uploads directly to S3 using PUT request
4. Browser checks CORS policy before allowing request
5. S3 allows request if origin matches CORS config
6. Upload succeeds, frontend confirms with backend

## Security Benefits

✅ **Presigned URLs are secure:**
- Expire after 10 minutes
- Scoped to one specific file path
- Include content type restrictions
- AWS credentials never exposed to client

✅ **CORS prevents unauthorized access:**
- Only specified domains can upload
- Browser enforces the restrictions
- Can't be bypassed from legitimate browsers

## Troubleshooting

### "CORS policy: No 'Access-Control-Allow-Origin' header is present"

**Cause:** CORS not configured or wrong origin

**Fix:**
1. Check that CORS is configured in S3
2. Verify your frontend domain is in `AllowedOrigins`
3. Clear browser cache and try again

### "CORS policy: Method PUT is not allowed"

**Cause:** PUT method not in `AllowedMethods`

**Fix:** Add `"PUT"` to the `AllowedMethods` array in your CORS config

### Upload works locally but fails in production

**Cause:** Production domain not in `AllowedOrigins`

**Fix:** Add your Vercel domain to the CORS configuration:
```json
"AllowedOrigins": [
    "https://your-app.vercel.app",
    "https://*.vercel.app"  // For preview deployments
]
```

## Additional Resources

- [AWS S3 CORS Documentation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/cors.html)
- [MDN CORS Guide](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS)
- [Presigned URL Documentation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/PresignedUrlUploadObject.html)

## Production Checklist

Before deploying to production:

- [ ] CORS configured in S3 bucket
- [ ] Production domain added to `AllowedOrigins`
- [ ] Preview/staging domains included (if using Vercel)
- [ ] Tested upload from production domain
- [ ] Confirmed metadata is being saved correctly
- [ ] Backend environment variables set correctly
- [ ] File size limits appropriate for your use case

## Support

If you encounter issues:
1. Check browser console for specific error messages
2. Verify CORS configuration matches this guide
3. Test with a different browser
4. Check AWS S3 bucket permissions
5. Ensure presigned URL hasn't expired (10 min limit)

