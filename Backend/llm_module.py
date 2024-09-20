from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Load a GPT-based model (or any other language model that can process user text)
gpt_pipeline = pipeline("text-generation", model="gpt2")  # Assuming OpenAI GPT or similar

# Define input model for the LLM request
class LLMRequest(BaseModel):
    text: str

@app.post("/process_text")
async def process_text(request: LLMRequest):
    try:
        user_text = request.text
        print(f"Received user text: {user_text}")

        # Provide a more specific prompt to the LLM
        prompt = (f"Take the following description of a desired song and turn it into a music prompt with details about "
                  f"the mood, genre, tempo, and instruments: {user_text}. "
                  f"Make sure it is concise and suitable for generating music.")
        
        response = gpt_pipeline(prompt, max_length=100)
        processed_text = response[0]['generated_text']
        print(f"Processed text: {processed_text}")

        # Return the processed text
        return {"processed_text": processed_text}

    except Exception as e:
        print(f"Error during LLM processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process text with LLM: {str(e)}")

# Start the LLM FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)