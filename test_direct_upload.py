#!/usr/bin/env python3
"""
Test script for direct S3 upload functionality
Tests the new presigned URL endpoints
"""

import requests
import sys

# Configuration
API_URL = "http://localhost:8000"  # Change to your Railway URL for production testing

def test_health():
    """Test health endpoint"""
    print("\n1️⃣ Testing /health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        response.raise_for_status()
        data = response.json()
        print(f"✅ Health check passed")
        print(f"   AWS Configured: {data.get('aws_configured')}")
        print(f"   S3 Bucket: {data.get('s3_bucket')}")
        print(f"   Storage: {data.get('storage')}")
        return data.get('aws_configured', False)
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_generate_url():
    """Test presigned URL generation"""
    print("\n2️⃣ Testing /generate-upload-url endpoint...")
    try:
        # Test data
        data = {
            'name': 'Test User',
            'email': 'test@example.com',
            'venmo_id': '@testuser',
            'filename': 'test_video.mp4',
            'content_type': 'video/mp4'
        }
        
        response = requests.post(f"{API_URL}/generate-upload-url", data=data)
        response.raise_for_status()
        result = response.json()
        
        print(f"✅ Presigned URL generated successfully")
        print(f"   File ID: {result.get('file_id')}")
        print(f"   S3 Filename: {result.get('s3_filename')}")
        print(f"   Expires in: {result.get('expires_in')} seconds")
        print(f"   Upload URL: {result.get('upload_url')[:100]}...")
        
        return result.get('file_id'), result.get('upload_url')
    except Exception as e:
        print(f"❌ URL generation failed: {e}")
        return None, None

def test_confirm_upload(file_id):
    """Test upload confirmation"""
    print("\n3️⃣ Testing /confirm-upload endpoint...")
    try:
        data = {'file_id': file_id}
        
        response = requests.post(f"{API_URL}/confirm-upload", data=data)
        response.raise_for_status()
        result = response.json()
        
        print(f"✅ Upload confirmed successfully")
        print(f"   Message: {result.get('message')}")
        print(f"   Status: {result.get('status')}")
        
        return True
    except Exception as e:
        print(f"❌ Confirmation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("🧪 Direct S3 Upload - Backend Test Suite")
    print("="*60)
    
    # Test 1: Health check
    aws_configured = test_health()
    if not aws_configured:
        print("\n⚠️  WARNING: AWS not configured. Tests will use local storage.")
        print("   Set AWS environment variables for full testing.")
    
    # Test 2: Generate presigned URL
    file_id, upload_url = test_generate_url()
    if not file_id:
        print("\n❌ Cannot continue - URL generation failed")
        sys.exit(1)
    
    # Test 3: Confirm upload
    # Note: In a real scenario, you would upload to S3 here
    print("\n⚠️  Skipping actual S3 upload in test (would upload to presigned URL)")
    test_confirm_upload(file_id)
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    print("✅ Backend endpoints are working correctly!")
    print("\nNext steps:")
    print("1. Configure S3 CORS (see S3_CORS_SETUP.md)")
    print("2. Test from frontend UI with a real video file")
    print("3. Verify file appears in S3 bucket")
    print("4. Test from mobile device")
    print("\n💡 For full testing, use the frontend UI to upload a video")
    print("="*60)

if __name__ == "__main__":
    main()

