import torch
import gc
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
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

# Import from existing video-validation module
from video_validation import (
    load_model,
    detect_task_changes,
    analyze_scenes_with_retry_loop,
    validate_task_with_llm,
    analyze_scene_with_smart_retry
)

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
model_cache = {"model": None, "processor": None}

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


class ValidationStatus(BaseModel):
    job_id: str
    status: str  # "processing", "completed", "failed"
    stage: str  # "initializing", "parsing_scenes", "analyzing_scenes", "validating_task", "done"
    progress: float  # 0.0 to 1.0
    current_step: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    print("🚀 Loading Qwen2-VL model...")
    model, processor = load_model()
    model_cache["model"] = model
    model_cache["processor"] = processor
    print("✅ Model loaded successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if model_cache["model"] is not None:
        del model_cache["model"]
        del model_cache["processor"]
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


@app.post("/validate-video/stream")
async def validate_video_stream(request: VideoValidationRequest):
    """
    Stream validation progress and results in real-time using Server-Sent Events (SSE)
    """
    
    async def event_generator():
        job_id = str(uuid.uuid4())
        
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
            
            # Stage 1: Parse scenes (detect_task_changes)
            yield f"data: {json.dumps({'job_id': job_id, 'status': 'processing', 'stage': 'parsing_scenes', 'progress': 0.1, 'current_step': 'Detecting scene changes...'})}\n\n"
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
                "progress": 0.25,
                "current_step": f"Found {len(scenes)} scenes",
                "result": {
                    "scenes": scenes,
                    "num_scenes": len(scenes)
                }
            }
            yield f"data: {json.dumps(scenes_data)}\n\n"
            await asyncio.sleep(0.1)
            
            # Stage 2: VLM Classification with retry loop
            yield f"data: {json.dumps({'job_id': job_id, 'status': 'processing', 'stage': 'analyzing_scenes', 'progress': 0.3, 'current_step': 'Starting scene analysis...'})}\n\n"
            await asyncio.sleep(0.1)
            
            # Analyze scenes with progress updates
            verified_results = []
            failed_results = []
            previous_description = None
            
            for idx, scene in enumerate(scenes):
                scene_num = scene['scene_num']
                scene_progress = 0.3 + (0.5 * (idx / len(scenes)))
                
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
                yield f"data: {json.dumps(scene_analysis_data)}\n\n"
                await asyncio.sleep(0.1)
            
            # Stage 2 complete
            analysis_complete_data = {
                "job_id": job_id,
                "status": "processing",
                "stage": "analyzing_scenes",
                "progress": 0.8,
                "current_step": f"Scene analysis complete: {len(verified_results)} verified, {len(failed_results)} failed",
                "result": {
                    "verified_results": verified_results,
                    "failed_results": failed_results,
                    "total_verified": len(verified_results),
                    "total_failed": len(failed_results)
                }
            }
            yield f"data: {json.dumps(analysis_complete_data)}\n\n"
            await asyncio.sleep(0.1)
            
            # Stage 3: LLM Task Validation
            yield f"data: {json.dumps({'job_id': job_id, 'status': 'processing', 'stage': 'validating_task', 'progress': 0.85, 'current_step': 'Validating task with LLM...'})}\n\n"
            await asyncio.sleep(0.1)
            
            validation_result = validate_task_with_llm(
                verified_results,
                expected_task=request.expected_task,
                processor=processor,
                model=model
            )
            
            # Final result
            final_data = {
                "job_id": job_id,
                "status": "completed",
                "stage": "done",
                "progress": 1.0,
                "current_step": "Validation complete",
                "result": {
                    "scenes": scenes,
                    "verified_results": verified_results,
                    "failed_results": failed_results,
                    "validation": validation_result,
                    "summary": {
                        "total_scenes": len(scenes),
                        "verified_scenes": len(verified_results),
                        "failed_scenes": len(failed_results),
                        "task_confirmed": validation_result['confirmed'],
                        "expected_task": request.expected_task
                    }
                }
            }
            yield f"data: {json.dumps(final_data)}\n\n"
            
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
        
        # Stage 1: Parse scenes
        validation_jobs[job_id].update({
            "stage": "parsing_scenes",
            "progress": 0.1,
            "current_step": "Detecting scene changes..."
        })
        
        scenes = detect_task_changes(
            video_path,
            method=request.method,
            pen=request.penalty,
            feature_type=request.feature_type
        )
        
        validation_jobs[job_id].update({
            "progress": 0.25,
            "scenes": scenes
        })
        
        # Stage 2: Analyze scenes
        validation_jobs[job_id].update({
            "stage": "analyzing_scenes",
            "progress": 0.3,
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
            "progress": 0.8,
            "verified_results": verified_results,
            "failed_results": failed_results
        })
        
        # Stage 3: Validate task
        validation_jobs[job_id].update({
            "stage": "validating_task",
            "progress": 0.85,
            "current_step": "Validating task with LLM..."
        })
        
        validation_result = validate_task_with_llm(
            verified_results,
            expected_task=request.expected_task,
            processor=processor,
            model=model
        )
        
        # Complete
        validation_jobs[job_id].update({
            "status": "completed",
            "stage": "done",
            "progress": 1.0,
            "validation_result": validation_result,
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
    
    return validation_jobs[job_id]


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
    return {
        "jobs": list(validation_jobs.values()),
        "total": len(validation_jobs)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)

