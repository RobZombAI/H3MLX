"""
Fal.ai MiniMax H3-Max Text-to-Video Compatible Local API Server
==============================================================
Drop-in local replacement for https://fal.run/minimax/h3-max/text-to-video
Powered by native MiniMax-H3 C engine + Metal 4 NAX on Apple Silicon M5 Max.
"""

import os
import time
import uuid
import base64
import subprocess
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Constants & Paths
ROOT_DIR = Path("/Users/robzomb/Documents/antigravity/cool-hopper/h3-lora-lab")
MODEL_DIR = Path("/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step")
OUTPUTS_DIR = ROOT_DIR / "outputs" / "fal_api_outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="MiniMax H3-Max API (Fal.ai Local Drop-in)",
    description="Local high-throughput implementation of MiniMax H3-Max Text-to-Video API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResolutionEnum(str, Enum):
    res_480p = "480P"
    res_768p = "768P"

class AspectRatioEnum(str, Enum):
    ar_21_9 = "21:9"
    ar_16_9 = "16:9"
    ar_4_3 = "4:3"
    ar_1_1 = "1:1"
    ar_3_4 = "3:4"
    ar_9_16 = "9:16"

class PromptExpansionModeEnum(str, Enum):
    disabled = "disabled"
    balanced = "balanced"
    quality = "quality"

class H3MaxInput(BaseModel):
    prompt: str = Field(..., description="Text prompt for video generation", example="A white kitten chases a butterfly across a sunlit garden. Gentle camera tracking, natural movement, soft afternoon light filtering through the leaves.")
    duration: int = Field(default=5, ge=5, le=15, description="Duration in seconds (5-15)")
    resolution: ResolutionEnum = Field(default=ResolutionEnum.res_768p, description="Generation resolution (480P or 768P)")
    seed: Optional[int] = Field(default=None, description="Random seed")
    enable_safety_checker: bool = Field(default=True, description="Safety checker toggle")
    sync_mode: bool = Field(default=False, description="Return base64 instead of URL")
    prompt_expansion_mode: PromptExpansionModeEnum = Field(default=PromptExpansionModeEnum.balanced, description="Prompt expansion mode")
    aspect_ratio: AspectRatioEnum = Field(default=AspectRatioEnum.ar_16_9, description="Aspect ratio of the generated video")

def map_resolution_and_aspect(resolution: ResolutionEnum, aspect_ratio: AspectRatioEnum) -> tuple[int, int]:
    """Computes exact width x height canvas for H3 engine."""
    if resolution == ResolutionEnum.res_768p:
        if aspect_ratio == AspectRatioEnum.ar_16_9:
            return (960, 544)
        elif aspect_ratio == AspectRatioEnum.ar_1_1:
            return (768, 768)
        elif aspect_ratio == AspectRatioEnum.ar_9_16:
            return (544, 960)
        elif aspect_ratio == AspectRatioEnum.ar_21_9:
            return (1024, 432)
        elif aspect_ratio == AspectRatioEnum.ar_4_3:
            return (896, 672)
        elif aspect_ratio == AspectRatioEnum.ar_3_4:
            return (672, 896)
        return (960, 544)
    else: # 480P
        if aspect_ratio == AspectRatioEnum.ar_16_9:
            return (864, 480)
        elif aspect_ratio == AspectRatioEnum.ar_1_1:
            return (512, 512)
        elif aspect_ratio == AspectRatioEnum.ar_9_16:
            return (480, 864)
        return (864, 480)

def expand_prompt(prompt: str, mode: PromptExpansionModeEnum) -> str:
    """Prompt expansion mirroring fal balanced / quality mode."""
    if mode == PromptExpansionModeEnum.disabled:
        return prompt
    elif mode == PromptExpansionModeEnum.balanced:
        return f"{prompt.rstrip('.')}, cinematic 35mm Master Prime lighting, fluid photorealistic motion, rich dynamic range, clean depth of field bokeh, atmospheric synchronized soundtrack."
    elif mode == PromptExpansionModeEnum.quality:
        return f"{prompt.rstrip('.')}, hyper-detailed 8K textures, volumetric morning sunlight rays, pristine physical motion dynamics, award-winning cinematography, authentic high-fidelity acoustic soundscape."
    return prompt

@app.get("/health")
def health_check():
    return {"status": "healthy", "engine": "MiniMax-H3-Max", "backend": "Apple Silicon M5 Max Metal 4 NAX"}

@app.get("/minimax/h3-max/text-to-video/llms.txt", response_class=PlainTextResponse)
def get_llms_txt():
    doc_path = ROOT_DIR / "h3_max_suite" / "RECIPE_GUIDE.md"
    if doc_path.exists():
        return doc_path.read_text()
    return "# MiniMax H3 Max Text to Video API"

@app.get("/files/{file_name}")
def serve_file(file_name: str):
    file_path = OUTPUTS_DIR / file_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="video/mp4")

@app.post("/minimax/h3-max/text-to-video")
@app.post("/fal.run/minimax/h3-max/text-to-video")
async def generate_text_to_video(input_data: H3MaxInput, request: Request):
    t_start = time.time()
    
    # 1. Expand prompt
    final_prompt = expand_prompt(input_data.prompt, input_data.prompt_expansion_mode)
    
    # 2. Map resolution
    width, height = map_resolution_and_aspect(input_data.resolution, input_data.aspect_ratio)
    
    # 3. Compute duration & frame count
    # 5 seconds at 24fps = 107 frames (17n+5 chunk aligned)
    # 10 seconds = 243 frames
    frames = 56 if input_data.duration <= 3 else (107 if input_data.duration <= 5 else 243)
    steps = 6 if input_data.duration > 5 else 8
    seed = input_data.seed if input_data.seed is not None else int(time.time()) % 100000
    
    req_id = f"{uuid.uuid4().hex[:12]}"
    raw_filename = f"h3_max_{req_id}_raw.mp4"
    master_filename = f"h3_max_{req_id}.mp4"
    
    raw_path = OUTPUTS_DIR / raw_filename
    master_path = OUTPUTS_DIR / master_filename
    
    env = os.environ.copy()
    env.update({
        "H3_PROFILE": "1",
        "H3_NAX": "1",
        "H3_ZERO_COPY_WEIGHTS": "1",
        "H3_REUSE_MPS_COMMAND": "1",
        "H3_GPU_SAMPLER": "1",
        "OMP_NUM_THREADS": "12",
        "METAL_DEVICE_WRAPPER_TYPE": "0",
        "MTL_DEBUG_LAYER": "0",
        "MTL_SHADER_VALIDATION": "0",
        "METAL_CAPTURE_ENABLED": "0",
    })
    
    cmd_h3 = [
        "caffeinate", "-dimsu", "nice", "-n", "-20",
        str(ROOT_DIR / "h3"),
        "-d", str(MODEL_DIR),
        "-p", final_prompt,
        "--width", str(width),
        "--height", str(height),
        "--frames", str(frames),
        "--steps", str(steps),
        "--layers", "50",
        "--use-int8-row-fc2",
        "--seed", str(seed),
        "-o", str(raw_path)
    ]
    
    t_infer_start = time.time()
    res = subprocess.run(cmd_h3, cwd=str(ROOT_DIR), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise HTTPException(status_code=500, detail=f"H3 Engine Error: {res.stderr[-500:]}")
    t_infer = time.time() - t_infer_start
    
    # 4. Hardware Master 1080p 10-Bit Main10 Apple Native
    cmd_master = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(raw_path),
        "-filter_complex", "[0:v]scale=1920:1080:flags=lanczos+accurate_rnd+full_chroma_int+full_chroma_inp,cas=0.30,format=yuv420p10le[v];[0:a]aresample=48000,loudnorm=I=-14:TP=-1.0:LRA=11[a]",
        "-map", "[v]", "-map", "[a]",
        "-c:v", "hevc_videotoolbox", "-profile:v", "main10", "-pix_fmt", "p010le", "-b:v", "80M", "-tag:v", "hvc1", "-r", "24",
        "-c:a", "aac", "-b:a", "320k", "-ar", "48000",
        "-movflags", "+faststart",
        str(master_path)
    ]
    subprocess.run(cmd_master, check=True)
    
    # Clean raw intermediate
    if raw_path.exists():
        raw_path.unlink()
        
    t_total = time.time() - t_start
    file_size = master_path.stat().st_size
    
    base_url = str(request.base_url).rstrip("/")
    file_url = f"{base_url}/files/{master_filename}"
    
    video_response = {
        "content_type": "video/mp4",
        "file_name": master_filename,
        "url": file_url,
        "file_size": file_size
    }
    
    if input_data.sync_mode:
        with open(master_path, "rb") as f:
            video_response["base64"] = base64.b64encode(f.read()).decode("utf-8")
            
    return {
        "video": video_response,
        "expanded_prompt": final_prompt if input_data.prompt_expansion_mode != PromptExpansionModeEnum.disabled else None,
        "timings": {
            "inference": round(t_infer, 2),
            "total": round(t_total, 2)
        }
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print("==========================================================")
    print("🚀 FAL.AI MINI-MAX H3-MAX COMPATIBLE LOCAL SERVER RUNNING")
    print(f"  Endpoint: http://127.0.0.1:{port}/minimax/h3-max/text-to-video")
    print(f"  Documentation: http://127.0.0.1:{port}/docs")
    print("==========================================================")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
