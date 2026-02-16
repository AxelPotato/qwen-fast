from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Security, Depends, status
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import shutil
import os
import asyncio
import uuid
import httpx
import ulid
from typing import List, Optional, Dict

from engine import QwenTTSEngine
from video_concat import concat_videos, merge_video_audio

# -- Configuration --
VOICE_DIR = "/voices"
OUTPUT_DIR = "/output"
FINAL_OUTPUT_DIR = "/final_outputs"
os.makedirs(VOICE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

# -- Security Configuration --
API_KEY_NAME = "x-api-key"
API_KEY = os.getenv("API_KEY") 
if not API_KEY:
    print("WARNING: API_KEY environment variable not set. Security is disabled!")

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    """Validates the x-api-key header against the env var."""
    if API_KEY and api_key_header == API_KEY:
        return api_key_header
    elif not API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server misconfigured: API_KEY not set"
        )
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Could not validate credentials"
    )

app = FastAPI(title="Qwen3-TTS RTX 3060 Service")

# -- Global State --
inference_lock = asyncio.Lock()
tts_engine = None

# In-memory Job Store
# Structure: { task_id: { "status": str, "filename": str|None, "error": str|None } }
JOBS: Dict[str, dict] = {}

@app.on_event("startup")
async def startup_event():
    global tts_engine
    tts_engine = QwenTTSEngine()

# -- Data Models --
class GenerationRequest(BaseModel):
    text: str
    voice_id: str
    language: Optional[str] = "auto"

class GenerationResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TaskStatus(BaseModel):
    task_id: str
    status: str
    download_url: Optional[str] = None
    file_path: Optional[str] = None
    error: Optional[str] = None

class VoiceMetadata(BaseModel):
    voice_id: str
    filename: str
    size_bytes: int

class VideoDownloadRequest(BaseModel):
    url: str
    project_folder: str

class VideoDownloadResponse(BaseModel):
    status: str
    filename: str
    file_path: str
    size_bytes: int

class VideoConcatRequest(BaseModel):
    project_folder: str

class VideoConcatResponse(BaseModel):
    status: str
    file_path: str
    size_bytes: int

class MergeVideoAudioRequest(BaseModel):
    video_path: str
    audio_path: str

class MergeVideoAudioResponse(BaseModel):
    status: str
    file_path: str
    download_url: str
    size_bytes: int

# -- Background Worker --
async def process_generation_task(task_id: str, text: str, voice_id: str, language: str):
    """
    Background worker that acquires the GPU lock and runs inference.
    """
    # 1. Resolve Voice File
    ref_audio_path = None
    for filename in os.listdir(VOICE_DIR):
        if filename.startswith(voice_id):
            ref_audio_path = os.path.join(VOICE_DIR, filename)
            break
            
    if not ref_audio_path:
        JOBS[task_id]["status"] = "failed"
        JOBS[task_id]["error"] = "Voice ID not found"
        return

    # 2. Wait for GPU Availability
    async with inference_lock:
        try:
            JOBS[task_id]["status"] = "processing"
            
            # Run blocking inference in threadpool
            loop = asyncio.get_running_loop()
            output_filename = await loop.run_in_executor(
                None,
                tts_engine.clone_voice,
                text,
                ref_audio_path,
                OUTPUT_DIR,
                language
            )
            
            JOBS[task_id]["status"] = "completed"
            JOBS[task_id]["filename"] = output_filename
            
        except Exception as e:
            JOBS[task_id]["status"] = "failed"
            JOBS[task_id]["error"] = str(e)

# -- Endpoints --

@app.post("/generate", response_model=GenerationResponse, dependencies=[Depends(get_api_key)])
async def generate_audio(req: GenerationRequest, background_tasks: BackgroundTasks):
    """
    Submits a generation job. Returns immediately with a task_id.
    """
    task_id = str(uuid.uuid4())
    
    # Initialize Job State
    JOBS[task_id] = {
        "status": "queued",
        "filename": None,
        "error": None
    }
    
    # Add to background queue
    background_tasks.add_task(
        process_generation_task, 
        task_id, 
        req.text, 
        req.voice_id, 
        req.language
    )
    
    return GenerationResponse(
        task_id=task_id,
        status="queued",
        message="Job submitted successfully. Poll /status/{task_id} for results."
    )

@app.get("/status/{task_id}", response_model=TaskStatus, dependencies=[Depends(get_api_key)])
async def get_task_status(task_id: str):
    """
    Checks the status of a generation task.
    """
    if task_id not in JOBS:
        raise HTTPException(status_code=404, detail="Task ID not found")
        
    job = JOBS[task_id]
    
    response = TaskStatus(
        task_id=task_id,
        status=job["status"],
        error=job["error"]
    )
    
    if job["status"] == "completed":
        response.download_url = f"/download/{task_id}"
        response.file_path = os.path.join(OUTPUT_DIR, job["filename"])
        
    return response

@app.get("/download/{task_id}", dependencies=[Depends(get_api_key)])
async def download_audio_endpoint(task_id: str):
    """
    Downloads the audio for a completed task.
    """
    if task_id not in JOBS:
        raise HTTPException(status_code=404, detail="Task ID not found")
        
    job = JOBS[task_id]
    
    if job["status"]!= "completed":
        raise HTTPException(status_code=400, detail="Task not completed yet")
        
    file_path = os.path.join(OUTPUT_DIR, job["filename"])
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File lost on server")
    
    return FileResponse(
        path=file_path,
        media_type="audio/wav",
        filename=f"qwen_tts_{task_id}.wav"
    )

@app.post("/voices/upload", response_model=VoiceMetadata, dependencies=[Depends(get_api_key)])
async def upload_voice_sample(file: UploadFile = File(...)):
    """Uploads a voice sample for cloning."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    if not file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    voice_id = str(uuid.uuid4())
    extension = os.path.splitext(file.filename)[1]
    safe_filename = f"{voice_id}{extension}"
    file_path = os.path.join(VOICE_DIR, safe_filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail="File write failed")
        
    return VoiceMetadata(
        voice_id=voice_id,
        filename=file.filename,
        size_bytes=os.path.getsize(file_path)
    )

@app.get("/voices/list", response_model=List[VoiceMetadata], dependencies=[Depends(get_api_key)])
async def list_voices():
    """Lists all available voice samples."""
    results = []
    for filename in os.listdir(VOICE_DIR):
        path = os.path.join(VOICE_DIR, filename)
        if os.path.isfile(path):
            vid = os.path.splitext(filename)[0]
            results.append(VoiceMetadata(
                voice_id=vid,
                filename=filename,
                size_bytes=os.path.getsize(path)
            ))
    return results

@app.delete("/voices/{voice_id}", dependencies=[Depends(get_api_key)])
async def delete_voice_sample(voice_id: str):
    """Deletes a specific voice sample."""
    target_path = None
    for filename in os.listdir(VOICE_DIR):
        if filename.startswith(voice_id):
            target_path = os.path.join(VOICE_DIR, filename)
            break
    
    if target_path and os.path.exists(target_path):
        os.remove(target_path)
        return {"status": "deleted", "voice_id": voice_id}
    
    raise HTTPException(status_code=404, detail="Voice ID not found")

@app.post("/videos/download", response_model=VideoDownloadResponse, dependencies=[Depends(get_api_key)])
async def download_video(req: VideoDownloadRequest):
    """Downloads a video file from a URL into the specified project folder."""
    project_path = os.path.join(OUTPUT_DIR, req.project_folder)
    os.makedirs(project_path, exist_ok=True)

    # Prefix with ULID to prevent overwrites and ensure correct sort order
    url_basename = req.url.rsplit("/", 1)[-1].split("?")[0]
    if not url_basename or "." not in url_basename:
        url_filename = f"{ulid.new()}.mp4"
    else:
        url_filename = f"{ulid.new()}_{url_basename}"

    file_path = os.path.join(project_path, url_filename)

    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(req.url)
            response.raise_for_status()

            with open(file_path, "wb") as f:
                f.write(response.content)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Remote server returned {e.response.status_code}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Download failed: {str(e)}")

    return VideoDownloadResponse(
        status="downloaded",
        filename=url_filename,
        file_path=file_path,
        size_bytes=os.path.getsize(file_path)
    )

@app.post("/videos/concat", response_model=VideoConcatResponse, dependencies=[Depends(get_api_key)])
async def concat_video_endpoint(req: VideoConcatRequest):
    """Concatenates all video files in a project folder into one."""
    project_path = os.path.join(OUTPUT_DIR, req.project_folder)

    if not os.path.isdir(project_path):
        raise HTTPException(status_code=404, detail="Project folder not found")

    output_path = os.path.join(OUTPUT_DIR, f"{req.project_folder}_final.mp4")

    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, concat_videos, project_path, output_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Concat failed: {str(e)}")

    return VideoConcatResponse(
        status="completed",
        file_path=output_path,
        size_bytes=os.path.getsize(output_path)
    )

@app.post("/videos/merge", response_model=MergeVideoAudioResponse, dependencies=[Depends(get_api_key)])
async def merge_video_audio_endpoint(req: MergeVideoAudioRequest):
    """Merges a video and audio file. Video duration matches the audio. Output goes to final_outputs."""
    if not os.path.isfile(req.video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    if not os.path.isfile(req.audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    filename = f"{ulid.new()}.mp4"
    output_path = os.path.join(FINAL_OUTPUT_DIR, filename)

    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, merge_video_audio, req.video_path, req.audio_path, output_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Merge failed: {str(e)}")

    return MergeVideoAudioResponse(
        status="completed",
        file_path=output_path,
        download_url=f"/final/{filename}",
        size_bytes=os.path.getsize(output_path)
    )

@app.get("/final/{filename}", dependencies=[Depends(get_api_key)])
async def download_final_video(filename: str):
    """Downloads a file from final_outputs."""
    file_path = os.path.join(FINAL_OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=filename
    )