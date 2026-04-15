"""
ComposerIQ — main.py
---------------------
Local:   uvicorn main:app --reload --port 8000
Railway: automatically uses $PORT env variable
"""
import os
import uuid
import shutil
import tempfile
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

load_dotenv()

if not os.getenv("ANTHROPIC_API_KEY"):
    os.environ["DEMO_MODE"] = "true"

from agent.visibility_agent import run_analysis

app = FastAPI(title="ComposerIQ", docs_url="/api/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Stub job store — kept for /analyze/status endpoint
jobs: dict[str, dict] = {}

static_path = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.get("/", response_class=HTMLResponse)
@app.get("/analyze", response_class=HTMLResponse)
@app.get("/reports", response_class=HTMLResponse)
async def root():
    return FileResponse(str(static_path / "index.html"))


@app.get("/mode")
async def get_mode():
    demo = os.getenv("DEMO_MODE", "true").lower() == "true"
    has_key = bool(os.getenv("ANTHROPIC_API_KEY", ""))
    return {"demo_mode": demo or not has_key, "has_anthropic_key": has_key, "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "composeriq-visibility-engine", "demo_mode": os.getenv("DEMO_MODE", "true")}


@app.get("/debug/pinecone")
async def debug_pinecone():
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "composeriq-tracks"))
        stats = index.describe_index_stats()
        results = index.query(
            vector=[0.5] * 768,
            top_k=10,
            include_metadata=True
        )
        return {
            "total_vectors": stats.total_vector_count,
            "sample_tracks": [
                {
                    "name": m.metadata.get("name"),
                    "artist": m.metadata.get("artist"),
                    "popularity": m.metadata.get("popularity"),
                    "genre": m.metadata.get("genre"),
                    "mood": m.metadata.get("mood"),
                }
                for m in results.matches
            ]
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/analyze")
async def analyze_track(
    audio_file: UploadFile = File(...),
    genre: str = Form(default="ambient"),
    mood: str = Form(default=""),
):
    MAX_SIZE = 50 * 1024 * 1024
    file_bytes = await audio_file.read()
    if len(file_bytes) > MAX_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Max 50MB.")

    request_id = str(uuid.uuid4())[:8]
    # Clamp suffix to avoid Windows MAX_PATH issues with giant filenames
    raw_suffix = Path(audio_file.filename or "track.mp3").suffix
    suffix = raw_suffix[:10] if raw_suffix else ".mp3"
    tmp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(tmp_dir, f"track_{request_id}{suffix}")

    try:
        with open(audio_path, "wb") as f:
            f.write(file_bytes)

        result = run_analysis(
            audio_path=audio_path,
            genre=genre,
            mood=mood or "unknown",
            user_id=request_id,
        )

        # Extract MIR percentiles for the frontend
        mir = result.get("mir_analysis", {})
        result["hook_score"]      = mir.get("hook_percentile", 0)
        result["skip_risk_score"] = mir.get("skip_risk_percentile", 0)

        if "error" in result and result["error"] and not result.get("report"):
            raise HTTPException(status_code=500, detail=result["error"])

        return JSONResponse(content=make_json_safe(result))

    except HTTPException:
        raise
    except Exception as e:
        detail = str(e) or type(e).__name__
        raise HTTPException(status_code=500, detail=detail)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/analyze/status/{job_id}")
async def analyze_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JSONResponse(content=job)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
