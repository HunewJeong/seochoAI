from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import logging
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import numpy as np
from scipy.io import wavfile
import uuid
from pydub import AudioSegment
import asyncio
import soundfile as sf
from prompt import MUSIC_GENERATION_PROMPT
from type import MusicGenerationRequest, MusicGenerationResponse, TaskStatusResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AUDIO_DIR = "generated_music"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MusicGen 모델 및 프로세서 초기화
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 샘플링 레이트 설정
SAMPLE_RATE = model.config.audio_encoder.sampling_rate

# 태스크 상태를 저장할 딕셔너리
tasks = {}

@app.post("/api/generate-music", response_model=MusicGenerationResponse)
async def generate_music(request: MusicGenerationRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "pending", "progress": 0}
    background_tasks.add_task(generate_music_task, task_id, request.prompt, request.duration, request.num_generations)
    return MusicGenerationResponse(task_id=task_id)

@app.get("/api/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    task = tasks[task_id]
    return TaskStatusResponse(
        status=task["status"],
        message=task.get("message", ""),
        progress=task["progress"],
        file_url=task.get("file_url", "")
    )

@app.get("/download/{file_name}")
async def download_music(file_name: str):
    file_path = os.path.join("generated_music", file_name)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/wav", filename=file_name)
    raise HTTPException(status_code=404, detail="File not found")

def apply_fade(audio, fade_length=1000):
    return audio.fade_in(fade_length).fade_out(fade_length)

def normalize_audio(audio):
    return audio.normalize()

"""
async def generate_single_music(prompt: str, duration: int):
    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
    audio_values = model.generate(**inputs, max_new_tokens=256)
    audio_data = audio_values[0, 0].cpu().numpy()

    audio_segment = AudioSegment(
        audio_data.tobytes(),
        frame_rate=SAMPLE_RATE,
        sample_width=audio_data.dtype.itemsize,
        channels=1
    )
    
    #audio_segment = normalize_audio(audio_segment)
    #audio_segment = apply_fade(audio_segment)

    target_duration_ms = duration * 1000
    if len(audio_segment) < target_duration_ms:
        audio_segment = audio_segment * (target_duration_ms // len(audio_segment) + 1)
    audio_segment = audio_segment[:target_duration_ms]

    return audio_segment
"""
async def generate_single_music(prompt, duration):
    try:
        # Tokenize the prompt
        inputs = processor(text=prompt, return_tensors="pt").to(device)

        total_tokens = int(duration * 50)

        # Generate the full music segment in one go
        generated_audio = model.generate(inputs.input_ids, max_new_tokens=total_tokens, do_sample=True)

        # Convert the model output to a NumPy array
        audio_array = generated_audio.cpu().numpy().squeeze()

        return audio_array  # Return the generated audio array
    except Exception as e:
        logger.error(f"Error generating music: {str(e)}")
        raise


"""
async def generate_single_music(prompt, duration):
    try:
        # Tokenize the prompt
        inputs = processor(text=prompt, return_tensors="pt").to(device)

        # Define chunk size (number of tokens to generate per segment)
        chunk_size = 256  # Adjust based on memory capacity
        total_steps = duration * 2  # Assuming each chunk represents 0.5 seconds of music

        combined_audio = []

        # Generate multiple chunks and concatenate them (continuation, not variation)
        for step in range(total_steps):
            logger.info(f"Generating chunk {step + 1}/{total_steps}")
            generated_audio = model.generate(inputs.input_ids, max_new_tokens=chunk_size, do_sample=True)

            # Convert to NumPy array
            audio_array = generated_audio.cpu().numpy().squeeze()

            # Append the chunk to create one continuous audio output
            combined_audio.append(audio_array)

        # Concatenate the generated audio chunks into one continuous track
        combined_audio = np.concatenate(combined_audio)

        return combined_audio  # Return the continuous audio array
    except Exception as e:
        logger.error(f"Error generating music: {str(e)}")
        raise
"""


"""
async def generate_music_task(task_id: str, prompt: str, duration: int, num_generations: int):
    try:
        tasks[task_id]["status"] = "processing"
        combined_audio = AudioSegment.empty()

        for i in range(num_generations):
            tasks[task_id]["progress"] = int((i / num_generations) * 100)
            varied_prompt = f"{prompt}, variation {i+1}"
            audio_segment = await generate_single_music(varied_prompt, duration)
            combined_audio += audio_segment
            await asyncio.sleep(0.1)

        file_name = f"generated_music_{task_id}.wav"
        file_path = os.path.join("generated_music", file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # 고품질 WAV 파일로 저장
        audio_array = np.array(combined_audio.get_array_of_samples()).astype(np.float32)
        max_val = np.max(np.abs(audio_array))
        audio_array /= max_val

        audio_array = (audio_array * 32767).astype(np.int16)

        print(audio_array.shape, audio_array.dtype, np.max(audio_array), np.min(audio_array))

        SAMPLE_RATE = 32000
        sf.write(file_path, audio_array, SAMPLE_RATE, subtype='PCM_16')

        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 100
        tasks[task_id]["file_url"] = f"/download/{file_name}"
        tasks[task_id]["message"] = f"Music generated successfully ({num_generations} times)"

        logger.info(f"Music generated successfully: {file_path}")
    except Exception as e:
        logger.error(f"Error generating music: {str(e)}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["message"] = f"Error generating music: {str(e)}"
"""
async def generate_music_task(task_id: str, prompt: str, duration: int, num_generations: int):
    try:
        tasks[task_id]["status"] = "processing"

        # Generate the raw audio
        audio_array = await generate_single_music(prompt, duration)

        # Ensure the audio array is correctly normalized (between -1.0 and 1.0)
        max_val = np.max(np.abs(audio_array))
        if max_val > 0:
            audio_array /= max_val  # Normalize between -1.0 and 1.0

        # Convert to PCM_16 format (-32768 to 32767)
        audio_array = (audio_array * 32767).astype(np.int16)

        # Save the normalized audio in PCM_16 format
        file_name = f"generated_music_{task_id}.wav"
        file_path = os.path.join("generated_music", file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        sf.write(file_path, audio_array, SAMPLE_RATE, subtype='PCM_16')

        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 100
        tasks[task_id]["file_url"] = f"/download/{file_name}"

        logger.info(f"Music generated successfully: {file_path}")
    except Exception as e:
        logger.error(f"Error generating music: {str(e)}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["message"] = f"Error generating music: {str(e)}"


@app.get("/")
async def root():
    return {"message": "Welcome to the Music Generation API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)