import requests
import json
import sys

API_URL = "http://184.105.3.161"

def test_streaming(video_path, expected_task):
    url = f"{API_URL}/validate-video/stream"
    payload = {
        "video_path": video_path,
        "expected_task": expected_task
    }
    
    print(f"🎬 Starting streaming validation...")
    print(f"   Video: {video_path}")
    print(f"   Task: {expected_task}")
    print("-" * 60)
    
    with requests.post(url, json=payload, stream=True) as response:
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data = json.loads(line_str[6:])  # Remove 'data: ' prefix
                    
                    # Pretty print key info
                    status = data.get('status', '?')
                    stage = data.get('stage', '?')
                    progress = data.get('progress', 0)
                    step = data.get('current_step', '')
                    
                    print(f"[{progress*100:5.1f}%] {status:12} | {stage:20} | {step}")
                    
                    # Print full data for completed/failed
                    if status in ['completed', 'failed', 'rejected']:
                        print("\n" + "="*60)
                        print("FINAL RESULT:")
                        print(json.dumps(data, indent=2))
                        print("="*60)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_stream.py <video_path> <expected_task>")
        print("Example: python test_stream.py './uploaded_videos/uuid.mp4' 'Make grilled cheese'")
        sys.exit(1)
    
    test_streaming(sys.argv[1], sys.argv[2])