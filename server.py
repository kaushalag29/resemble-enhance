import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from resemble_enhance.enhancer.inference import enhance, load_enhancer
from resemble_enhance.denoiser.inference import load_denoiser
import os
import logging
import signal
from contextlib import asynccontextmanager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Detect device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
logger.info(f"Using device: {device}")

# Global model instances (loaded at startup)
enhancer_model = None
denoiser_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle management"""
    global enhancer_model, denoiser_model

    # Startup - preload models to avoid CUDA busy errors during first request
    logger.info("Preloading Resemble Enhance models at startup...")
    try:
        enhancer_model = load_enhancer(None, device)
        logger.info(f"Enhancer model loaded successfully on {device}")

        denoiser_model = load_denoiser(None, device)
        logger.info(f"Denoiser model loaded successfully on {device}")
    except Exception as e:
        logger.error(f"Failed to preload models: {e}", exc_info=True)
        # Continue startup even if preload fails - models will load on first request

    yield

    # Shutdown cleanup
    logger.info("Cleaning up models...")
    enhancer_model = None
    denoiser_model = None

app = FastAPI(title="Resemble Enhance API", lifespan=lifespan)

class EnhanceRequest(BaseModel):
    input_path: str
    output_path: str
    solver: str = "rk4"
    nfe: int = 128
    tau: float = 0.5

@app.post("/enhance-audio")
async def enhance_audio(request: EnhanceRequest):
    try:
        logger.info(f"Enhancing audio from {request.input_path} to {request.output_path}")

        if not os.path.exists(request.input_path):
            raise HTTPException(status_code=400, detail=f"Input file not found at {request.input_path}")
        
        # Load audio
        dwav, sr = torchaudio.load(request.input_path)
        dwav = dwav.mean(dim=0)

        # Enhance (without denoising)
        # lambd is 0.1 when not denoising, according to existing code.
        enhanced_wav, new_sr = enhance(
            dwav, 
            sr, 
            device, 
            nfe=request.nfe, 
            solver=request.solver.lower(), 
            lambd=0.1, 
            tau=request.tau
        )
        
        # Save audio
        torchaudio.save(request.output_path, enhanced_wav.unsqueeze(0).cpu(), new_sr)

        if not os.path.exists(request.output_path) or os.path.getsize(request.output_path) == 0:
            raise HTTPException(status_code=500, detail="Enhancement failed to produce an output file.")

        return {"status": "success", "message": f"Enhanced audio saved to {request.output_path}"}
    
    except Exception as e:
        logger.error(f"Error enhancing audio: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error enhancing audio: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": "resemble-enhance", "device": device}

@app.post("/shutdown")
async def shutdown():
    logger.info("Shutdown request received")
    os.kill(os.getpid(), signal.SIGTERM)
    return {"status": "shutdown", "message": "Server shutting down"}

if __name__ == "__main__":
    import uvicorn
    # Port is configurable via environment variable
    port = int(os.environ.get("PORT", 8014))
    logger.info(f"Starting Resemble Enhance server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port) 