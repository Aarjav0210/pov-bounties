import torch
import gc
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import json
import asyncio
import uuid
from typing import Optional, Dict, Any
import shutil
from datetime import datetime
import numpy as np
import boto3
from botocore.exceptions import ClientError
import os

# Import from existing video-validation module
from video_validation import (
    load_model,
    load_text_model,
    detect_task_changes,
    analyze_scenes_with_retry_loop,
    validate_task_with_llm,
    analyze_scene_with_smart_retry,
    randomized_domain_match,
    extract_frames
)


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj

# Try to import config, fall back to defaults if not found
try:
    import config
    UPLOAD_DIR = Path(config.UPLOAD_DIR)
    CORS_ORIGINS = config.CORS_ORIGINS
    API_HOST = config.API_HOST
    API_PORT = config.API_PORT
    DEFAULT_METHOD = config.DEFAULT_METHOD
    DEFAULT_PENALTY = config.DEFAULT_PENALTY
    DEFAULT_FEATURE_TYPE = config.DEFAULT_FEATURE_TYPE
    DEFAULT_MAX_RETRIES = config.DEFAULT_MAX_RETRIES
    DEFAULT_MAX_FRAMES = config.DEFAULT_MAX_FRAMES
    DEFAULT_QUESTION_TEMPLATE = config.DEFAULT_QUESTION_TEMPLATE
    DEFAULT_DOMAIN_MATCH_SAMPLES = config.DEFAULT_DOMAIN_MATCH_SAMPLES
    DEFAULT_DOMAIN_MATCH_THRESHOLD = config.DEFAULT_DOMAIN_MATCH_THRESHOLD
    DEFAULT_DOMAIN_MATCH_FPS = config.DEFAULT_DOMAIN_MATCH_FPS
    print("✅ Loaded configuration from config.py")
except ImportError:
    # Fallback to hardcoded defaults
    UPLOAD_DIR = Path("./uploaded_videos")
    CORS_ORIGINS = ["*"]
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    DEFAULT_METHOD = "pelt"
    DEFAULT_PENALTY = 8.0
    DEFAULT_FEATURE_TYPE = "combined"
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_MAX_FRAMES = 12
    DEFAULT_QUESTION_TEMPLATE = "What action(s) is happening in this scene? Be specific, not vague."
    DEFAULT_DOMAIN_MATCH_SAMPLES = 5
    DEFAULT_DOMAIN_MATCH_THRESHOLD = 0.6
    DEFAULT_DOMAIN_MATCH_FPS = 0.2
    print("⚠️  config.py not found, using default configuration")

app = FastAPI(title="Video Validation API", version="1.0.0")

# CORS configuration - uses config.py or defaults
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
model_cache = {
    "model": None,          # Vision-Language model
    "processor": None,      # VL processor
    "text_model": None,     # Text-only reasoning model
    "text_tokenizer": None  # Text tokenizer
}

# Store for ongoing validation jobs
validation_jobs: Dict[str, Dict[str, Any]] = {}

# Create upload directory
UPLOAD_DIR.mkdir(exist_ok=True)


class VideoValidationRequest(BaseModel):
    video_path: Optional[str] = None
    expected_task: str
    method: str = DEFAULT_METHOD
    penalty: float = DEFAULT_PENALTY
    feature_type: str = DEFAULT_FEATURE_TYPE
    max_retries: int = DEFAULT_MAX_RETRIES
    max_frames: int = DEFAULT_MAX_FRAMES
    question_template: str = DEFAULT_QUESTION_TEMPLATE
    domain_match_samples: int = DEFAULT_DOMAIN_MATCH_SAMPLES
    domain_match_threshold: float = DEFAULT_DOMAIN_MATCH_THRESHOLD
    domain_match_fps: float = DEFAULT_DOMAIN_MATCH_FPS


class ValidationStatus(BaseModel):
    job_id: str
    status: str  # "processing", "completed", "failed", "rejected"
    stage: str  # "initializing", "domain_matching", "parsing_scenes", "analyzing_scenes", "validating_task", "done"
    progress: float  # 0.0 to 1.0
    current_step: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    print("🚀 Loading Qwen2-VL model...")
    model, processor = load_model()
    model_cache["model"] = model
    model_cache["processor"] = processor
    print("✅ Vision model loaded successfully")
    
    print("\n🚀 Loading text reasoning model...")
    text_model, text_tokenizer = load_text_model()
    model_cache["text_model"] = text_model
    model_cache["text_tokenizer"] = text_tokenizer
    if text_model is not None:
        print("✅ Text model loaded successfully")
    else:
        print("⚠️  Text model not available, will use vision model for text tasks")
    
    print("\n✅ All models loaded and ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if model_cache["model"] is not None:
        del model_cache["model"]
        del model_cache["processor"]
    if model_cache["text_model"] is not None:
        del model_cache["text_model"]
        del model_cache["text_tokenizer"]
    torch.cuda.empty_cache()
    gc.collect()


@app.get("/")
async def root():
    return {
        "message": "Video Validation API",
        "status": "running",
        "model_loaded": model_cache["model"] is not None
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_cache["model"] is not None,
        "gpu_available": torch.cuda.is_available(),
        "active_jobs": len(validation_jobs)
    }


@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file for validation"""
    if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid video format")
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix
    file_path = UPLOAD_DIR / f"{file_id}{file_extension}"
    
    # Save uploaded file
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {
        "file_id": file_id,
        "filename": file.filename,
        "path": str(file_path),
        "message": "Video uploaded successfully"
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


@app.post("/validate-video/stream")
async def validate_video_stream(request: VideoValidationRequest):
    """
    Stream validation progress and results in real-time using Server-Sent Events (SSE)
    """
    
    async def event_generator():
        job_id = str(uuid.uuid4())
        
        def emit_json(data):
            """Helper to emit SSE data with numpy type conversion"""
            return f"data: {json.dumps(convert_numpy_types(data))}\n\n"
        
        try:
            # Determine video path
            if request.video_path:
                video_path = Path(request.video_path)
            else:
                raise HTTPException(status_code=400, detail="video_path is required")
            
            if not video_path.exists():
                error_data = {
                    "job_id": job_id,
                    "status": "failed",
                    "stage": "initializing",
                    "progress": 0.0,
                    "error": f"Video file not found: {video_path}"
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                return
            
            # Initialize
            yield f"data: {json.dumps({'job_id': job_id, 'status': 'processing', 'stage': 'initializing', 'progress': 0.0, 'current_step': 'Loading model...'})}\n\n"
            await asyncio.sleep(0.1)
            
            model = model_cache["model"]
            processor = model_cache["processor"]
            
            if model is None or processor is None:
                error_data = {
                    "job_id": job_id,
                    "status": "failed",
                    "stage": "initializing",
                    "progress": 0.0,
                    "error": "Model not loaded"
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                return
            
            # Stage 0: Domain Matching (Quick Initial Validation)
            domain_match_result = None
            domain_match_error = None
            
            try:
                yield f"data: {json.dumps({'job_id': job_id, 'status': 'processing', 'stage': 'domain_matching', 'progress': 0.05, 'current_step': 'Extracting frames for quick validation...'})}\n\n"
                await asyncio.sleep(0.1)
                
                # Extract frames from entire video
                frames = extract_frames(video_path, fps=request.domain_match_fps)
                
                yield f"data: {json.dumps({'job_id': job_id, 'status': 'processing', 'stage': 'domain_matching', 'progress': 0.1, 'current_step': f'Running domain match on {len(frames)} frames...'})}\n\n"
                await asyncio.sleep(0.1)
                
                # Run randomized domain match
                domain_match_result = randomized_domain_match(
                    frames=frames,
                    task_description=request.expected_task,
                    n=min(request.domain_match_samples, len(frames)),
                    model=model,
                    processor=processor
                )
                
                # Send domain match result
                domain_match_data = {
                    "job_id": job_id,
                    "status": "processing",
                    "stage": "domain_matching",
                    "progress": 0.15,
                    "current_step": f"Domain match: {'✓ PASS' if domain_match_result['overall_match'] else '✗ FAIL'} ({domain_match_result['match_percentage']:.1%} match)",
                    "result": {
                        "domain_match": domain_match_result,
                        "passed": domain_match_result['overall_match'],
                        "match_percentage": domain_match_result['match_percentage'],
                        "confidence": domain_match_result['confidence']
                    }
                }
                yield emit_json(domain_match_data)
                await asyncio.sleep(0.1)
                
                # If domain match legitimately fails (rejection), stop early
                if not domain_match_result['overall_match']:
                    rejection_data = {
                        "job_id": job_id,
                        "status": "rejected",
                        "stage": "domain_matching",
                        "progress": 1.0,
                        "current_step": "Video rejected: Does not match expected task domain",
                        "result": {
                            "domain_match": domain_match_result,
                            "passed": False,
                            "match_percentage": domain_match_result['match_percentage'],
                            "confidence": domain_match_result['confidence'],
                            "reason": f"Only {domain_match_result['match_percentage']:.1%} of sampled frames matched the task description (threshold: {request.domain_match_threshold:.1%})",
                            "expected_task": request.expected_task
                        }
                    }
                    yield emit_json(rejection_data)
                    return
                    
            except Exception as e:
                # Domain match function failed - log error and continue with validation
                domain_match_error = str(e)
                error_msg = f"Domain match error: {domain_match_error} - Bypassing and continuing validation..."
                
                yield f"data: {json.dumps({'job_id': job_id, 'status': 'processing', 'stage': 'domain_matching', 'progress': 0.15, 'current_step': error_msg, 'result': {'domain_match_error': domain_match_error, 'bypassed': True}})}\n\n"
                await asyncio.sleep(0.1)
            
            # Stage 1: Parse scenes (detect_task_changes)
            yield f"data: {json.dumps({'job_id': job_id, 'status': 'processing', 'stage': 'parsing_scenes', 'progress': 0.2, 'current_step': 'Detecting scene changes...'})}\n\n"
            await asyncio.sleep(0.1)
            
            scenes = detect_task_changes(
                video_path,
                method=request.method,
                pen=request.penalty,
                feature_type=request.feature_type
            )
            
            # Send scenes result
            scenes_data = {
                "job_id": job_id,
                "status": "processing",
                "stage": "parsing_scenes",
                "progress": 0.3,
                "current_step": f"Found {len(scenes)} scenes",
                "result": {
                    "scenes": scenes,
                    "num_scenes": len(scenes)
                }
            }
            yield emit_json(scenes_data)
            await asyncio.sleep(0.1)
            
            # Stage 2: VLM Classification with retry loop
            yield f"data: {json.dumps({'job_id': job_id, 'status': 'processing', 'stage': 'analyzing_scenes', 'progress': 0.35, 'current_step': 'Starting scene analysis...'})}\n\n"
            await asyncio.sleep(0.1)
            
            # Analyze scenes with progress updates
            verified_results = []
            failed_results = []
            previous_description = None
            
            for idx, scene in enumerate(scenes):
                scene_num = scene['scene_num']
                scene_progress = 0.35 + (0.45 * (idx / len(scenes)))
                
                # Update progress for this scene
                yield f"data: {json.dumps({'job_id': job_id, 'status': 'processing', 'stage': 'analyzing_scenes', 'progress': scene_progress, 'current_step': f'Analyzing scene {scene_num}/{len(scenes)}'})}\n\n"
                await asyncio.sleep(0.1)
                
                # Extract frames
                from video_validation import extract_scene_frames
                frames = extract_scene_frames(video_path, scene['start'], scene['end'], request.max_frames)
                
                # Build question with context
                if previous_description:
                    question = f"{request.question_template}\n\nPrevious scene: {previous_description}"
                else:
                    question = request.question_template
                
                # Analyze with verification
                analysis = analyze_scene_with_smart_retry(
                    frames, question, processor, model, request.max_retries, max_tokens=200
                )
                
                scene_result = {
                    **scene,
                    **analysis,
                    'previous_context': previous_description
                }
                
                # Categorize result
                if analysis['verified']:
                    verified_results.append(scene_result)
                    previous_description = analysis['description']
                else:
                    failed_results.append(scene_result)
                
                # Send scene analysis result
                scene_analysis_data = {
                    "job_id": job_id,
                    "status": "processing",
                    "stage": "analyzing_scenes",
                    "progress": scene_progress,
                    "current_step": f"Scene {scene_num} analyzed: {analysis['description'][:50]}...",
                    "result": {
                        "scene_num": scene_num,
                        "scene_result": scene_result,
                        "verified": analysis['verified'],
                        "verified_count": len(verified_results),
                        "failed_count": len(failed_results)
                    }
                }
                yield emit_json(scene_analysis_data)
                await asyncio.sleep(0.1)
            
            # Stage 2 complete
            analysis_complete_data = {
                "job_id": job_id,
                "status": "processing",
                "stage": "analyzing_scenes",
                "progress": 0.85,
                "current_step": f"Scene analysis complete: {len(verified_results)} verified, {len(failed_results)} failed",
                "result": {
                    "verified_results": verified_results,
                    "failed_results": failed_results,
                    "total_verified": len(verified_results),
                    "total_failed": len(failed_results)
                }
            }
            yield emit_json(analysis_complete_data)
            await asyncio.sleep(0.1)
            
            # Stage 3: LLM Task Validation
            yield f"data: {json.dumps({'job_id': job_id, 'status': 'processing', 'stage': 'validating_task', 'progress': 0.9, 'current_step': 'Validating task with LLM...'})}\n\n"
            await asyncio.sleep(0.1)
            
            validation_result = validate_task_with_llm(
                verified_results,
                expected_task=request.expected_task,
                processor=processor,
                model=model,
                text_model=model_cache["text_model"],
                text_tokenizer=model_cache["text_tokenizer"]
            )
            
            # Final result
            summary = {
                "total_scenes": len(scenes),
                "verified_scenes": len(verified_results),
                "failed_scenes": len(failed_results),
                "task_confirmed": validation_result['confirmed'],
                "expected_task": request.expected_task
            }
            
            # Add domain match results if available
            if domain_match_result:
                summary.update({
                    "domain_match_passed": domain_match_result['overall_match'],
                    "domain_match_percentage": domain_match_result['match_percentage'],
                    "domain_match_confidence": domain_match_result['confidence']
                })
            elif domain_match_error:
                summary["domain_match_error"] = domain_match_error
                summary["domain_match_bypassed"] = True
            
            final_data = {
                "job_id": job_id,
                "status": "completed",
                "stage": "done",
                "progress": 1.0,
                "current_step": "Validation complete",
                "result": {
                    "domain_match": domain_match_result,
                    "domain_match_error": domain_match_error,
                    "scenes": scenes,
                    "verified_results": verified_results,
                    "failed_results": failed_results,
                    "validation": validation_result,
                    "summary": summary
                }
            }
            yield emit_json(final_data)
            
        except Exception as e:
            error_data = {
                "job_id": job_id,
                "status": "failed",
                "stage": "error",
                "progress": 0.0,
                "error": str(e)
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/validate-video")
async def validate_video(request: VideoValidationRequest, background_tasks: BackgroundTasks):
    """
    Start a video validation job (non-streaming, returns job_id for polling)
    """
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    validation_jobs[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "stage": "initializing",
        "progress": 0.0,
        "created_at": datetime.now().isoformat(),
        "request": request.dict()
    }
    
    # Start background task
    background_tasks.add_task(process_validation_job, job_id, request)
    
    return {
        "job_id": job_id,
        "status": "processing",
        "message": "Validation job started",
        "poll_url": f"/validation-status/{job_id}"
    }


async def process_validation_job(job_id: str, request: VideoValidationRequest):
    """Background task to process validation"""
    try:
        # Determine video path
        if request.video_path:
            video_path = Path(request.video_path)
        else:
            validation_jobs[job_id].update({
                "status": "failed",
                "error": "video_path is required"
            })
            return
        
        if not video_path.exists():
            validation_jobs[job_id].update({
                "status": "failed",
                "error": f"Video file not found: {video_path}"
            })
            return
        
        model = model_cache["model"]
        processor = model_cache["processor"]
        
        # Stage 0: Domain Matching
        domain_match_result = None
        domain_match_error = None
        
        try:
            validation_jobs[job_id].update({
                "stage": "domain_matching",
                "progress": 0.05,
                "current_step": "Running domain match..."
            })
            
            frames = extract_frames(video_path, fps=request.domain_match_fps)
            domain_match_result = randomized_domain_match(
                frames=frames,
                task_description=request.expected_task,
                n=min(request.domain_match_samples, len(frames)),
                model=model,
                processor=processor
            )
            
            validation_jobs[job_id].update({
                "progress": 0.15,
                "domain_match": domain_match_result,
                "current_step": f"Domain match: {'PASS' if domain_match_result['overall_match'] else 'FAIL'} ({domain_match_result['match_percentage']:.1%})"
            })
            
            # If domain match legitimately fails (rejection), stop early
            if not domain_match_result['overall_match']:
                validation_jobs[job_id].update({
                    "status": "rejected",
                    "stage": "domain_matching",
                    "progress": 1.0,
                    "current_step": "Video rejected: Does not match expected task domain",
                    "domain_match": domain_match_result,
                    "rejection_reason": f"Only {domain_match_result['match_percentage']:.1%} of sampled frames matched"
                })
                return
                
        except Exception as e:
            # Domain match function failed - log error and continue with validation
            domain_match_error = str(e)
            validation_jobs[job_id].update({
                "progress": 0.15,
                "domain_match_error": domain_match_error,
                "current_step": f"Domain match error: {domain_match_error} - Bypassing and continuing..."
            })
        
        # Stage 1: Parse scenes
        validation_jobs[job_id].update({
            "stage": "parsing_scenes",
            "progress": 0.2,
            "current_step": "Detecting scene changes..."
        })
        
        scenes = detect_task_changes(
            video_path,
            method=request.method,
            pen=request.penalty,
            feature_type=request.feature_type
        )
        
        validation_jobs[job_id].update({
            "progress": 0.3,
            "scenes": scenes
        })
        
        # Stage 2: Analyze scenes
        validation_jobs[job_id].update({
            "stage": "analyzing_scenes",
            "progress": 0.35,
            "current_step": "Starting scene analysis..."
        })
        
        verified_results, failed_results = analyze_scenes_with_retry_loop(
            video_path=video_path,
            scenes=scenes,
            question_template=request.question_template,
            retry_function=analyze_scene_with_smart_retry,
            processor=processor,
            model=model,
            max_retries=request.max_retries,
            max_frames=request.max_frames
        )
        
        validation_jobs[job_id].update({
            "progress": 0.85,
            "verified_results": verified_results,
            "failed_results": failed_results
        })
        
        # Stage 3: Validate task
        validation_jobs[job_id].update({
            "stage": "validating_task",
            "progress": 0.9,
            "current_step": "Validating task with LLM..."
        })
        
        validation_result = validate_task_with_llm(
            verified_results,
            expected_task=request.expected_task,
            processor=processor,
            model=model,
            text_model=model_cache["text_model"],
            text_tokenizer=model_cache["text_tokenizer"]
        )
        
        # Complete
        summary = {
            "total_scenes": len(scenes),
            "verified_scenes": len(verified_results),
            "failed_scenes": len(failed_results),
            "task_confirmed": validation_result['confirmed']
        }
        
        # Add domain match results if available
        if domain_match_result:
            summary.update({
                "domain_match_passed": domain_match_result['overall_match'],
                "domain_match_percentage": domain_match_result['match_percentage'],
                "domain_match_confidence": domain_match_result['confidence']
            })
        elif domain_match_error:
            summary["domain_match_error"] = domain_match_error
            summary["domain_match_bypassed"] = True
        
        validation_jobs[job_id].update({
            "status": "completed",
            "stage": "done",
            "progress": 1.0,
            "validation_result": validation_result,
            "domain_match": domain_match_result,
            "domain_match_error": domain_match_error,
            "summary": summary,
            "completed_at": datetime.now().isoformat()
        })
        
    except Exception as e:
        validation_jobs[job_id].update({
            "status": "failed",
            "error": str(e)
        })


@app.get("/validation-status/{job_id}")
async def get_validation_status(job_id: str):
    """Get the status of a validation job"""
    if job_id not in validation_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return convert_numpy_types(validation_jobs[job_id])


@app.delete("/validation-job/{job_id}")
async def delete_validation_job(job_id: str):
    """Delete a completed validation job"""
    if job_id in validation_jobs:
        del validation_jobs[job_id]
        return {"message": "Job deleted successfully"}
    raise HTTPException(status_code=404, detail="Job not found")


@app.get("/validation-jobs")
async def list_validation_jobs():
    """List all validation jobs"""
    return convert_numpy_types({
        "jobs": list(validation_jobs.values()),
        "total": len(validation_jobs)
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)

