from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import requests
from transformers import pipeline
from io import BytesIO
from scipy.io.wavfile import write
import numpy as np
import torch
import os

# Initialize FastAPI app
app = FastAPI()

# Set up the MusicGen model (facebook/musicgen-melody)
model_name = "facebook/musicgen-melody"  # Can be changed to 'facebook/musicgen-small' as needed
device = 0 if torch.cuda.is_available() else -1

# Initialize pipeline for text-to-audio generation
audio_pipeline = pipeline("text-to-audio", model=model_name, device=device)

# Directory to store generated audio files
output_dir = "generated_audio"
os.makedirs(output_dir, exist_ok=True)

# Define input model for the prompt
class UserRequest(BaseModel):
    text: str  # This is the text input from the user about what kind of music they want

# Health check route
@app.get("/")
def read_root():
    return {"message": "Welcome to the Music Generation API"}

# Function to call the LLM module (GPT)
def process_text_with_llm(user_text: str) -> str:
    try:
        # Send the user text to the LLM module via a POST request
        llm_endpoint = "http://localhost:9000/process_text"  # Assuming the LLM module is running on localhost:9000
        response = requests.post(llm_endpoint, json={"text": user_text})
        
        if response.status_code == 200:
            processed_text = response.json().get("processed_text", "")
            return processed_text
        else:
            raise Exception(f"LLM module returned an error: {response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process text with LLM: {str(e)}")

# Route to generate music from text
@app.post("/generate_music")
async def generate_music(request: UserRequest):
    try:
        user_text = request.text
        print(f"Received user request: {user_text}")

        # Step 1: Send user text to LLM to process it into a more specific format
        processed_text = process_text_with_llm(user_text)
        print(f"Processed text from LLM: {processed_text}")

        # Step 2: Send processed text to MusicGen to generate the music
        with torch.no_grad():
            output = audio_pipeline(processed_text)

        if not output or 'audio' not in output:
            raise ValueError("MusicGen did not generate any audio.")

        # Extract the generated audio data
        audio_tensor = output['audio']
        audio_flattened = audio_tensor.flatten()

        # Convert to int16 format (assuming the audio is in [-1, 1] range)
        audio_np = np.int16(audio_flattened * 32767)

        # Define the output WAV file path
        output_file_path = os.path.join(output_dir, "generated_music.wav")

        # Save the generated audio as a WAV file
        write(output_file_path, output['sampling_rate'], audio_np)

        # Step 3: Return the generated WAV file as a response
        return FileResponse(output_file_path, media_type="audio/wav", filename="generated_music.wav")

    except Exception as e:
        print(f"Error during music generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate music: {str(e)}")

# Start the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)