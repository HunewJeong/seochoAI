from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional
import uuid
import os
import logging
from model import AIModelHandler

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MusicRequest(BaseModel):
    user_input: str = Field(..., min_length=1, max_length=500)
    duration: int = Field(10, ge=5, le=30)

class TaskStatus(BaseModel):
    task_id: str
    status: str
    message: Optional[str] = None
    progress: Optional[int] = None

tasks = {}
ai_model_handler = AIModelHandler()

@app.post("/api/generate-music")
async def generate_music(request: MusicRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "pending", "progress": 0}
    background_tasks.add_task(process_music_generation, task_id, request.user_input, request.duration)
    return {"task_id": task_id}

@app.get("/api/task/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskStatus(task_id=task_id, **tasks[task_id])

async def process_music_generation(task_id: str, user_input: str, duration: int):
    try:
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["progress"] = 10

        optimized_prompt, audio_segment = await ai_model_handler.process_music_generation(user_input, duration)
        tasks[task_id]["progress"] = 90

        file_path = f"generated_audio/{task_id}.wav"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        audio_segment.export(file_path, format="wav")

        tasks[task_id]["status"] = "completed"
        tasks[task_id]["message"] = optimized_prompt
        tasks[task_id]["progress"] = 100
    except Exception as e:
        logger.error(f"Error in process_music_generation: {str(e)}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["message"] = str(e)

@app.get("/api/download/{task_id}")
async def download_music(task_id: str):
    if task_id not in tasks or tasks[task_id]["status"] != "completed":
        raise HTTPException(status_code=404, detail="Generated music not found")
    file_path = f"generated_audio/{task_id}.wav"
    return FileResponse(file_path, media_type="audio/wav", filename=f"generated_music_{task_id}.wav")

@app.get("/api/sampling-rate")
async def get_sampling_rate():
    return {"sampling_rate": ai_model_handler.get_sampling_rate()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)