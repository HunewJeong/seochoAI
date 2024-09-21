import os
import logging
from dotenv import load_dotenv
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import numpy as np
from pydub import AudioSegment
import io
import aiohttp
from typing import Tuple

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIModelHandler:
    def __init__(self):
        self.gpt4o_api_key = os.getenv("OPENAI_API_KEY")
        self.gpt4o_endpoint = "https://api.openai.com/v1/chat/completions"
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        self.processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        self.model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(self.device)
        
        # 올바른 샘플링 레이트 접근 방법 확인
        self.sampling_rate = self.model.config.audio_codec_kwargs.get("sample_rate", 32000)
        logger.info(f"Model initialized with sampling rate: {self.sampling_rate}")

    async def generate_optimized_prompt(self, user_input: str) -> str:
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a music description optimizer. Convert the user's input into a concise, keyword-rich description for music generation."},
                    {"role": "user", "content": user_input}
                ],
                "max_tokens": 100
            }
            headers = {
                "Authorization": f"Bearer {self.gpt4o_api_key}",
                "Content-Type": "application/json"
            }
            try:
                async with session.post(self.gpt4o_endpoint, json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data['choices'][0]['message']['content'].strip()
                    else:
                        raise Exception(f"GPT-4o-mini API error: {response.status}")
            except Exception as e:
                logger.error(f"Error in generate_optimized_prompt: {str(e)}")
                raise

    async def generate_music(self, prompt: str, duration: int = 10, num_samples: int = 1) -> AudioSegment:
        try:
            inputs = self.processor(
                text=[prompt],
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            audio_values = self.model.generate(
                **inputs,
                do_sample=True,
                guidance_scale=3,
                max_new_tokens=256,
                num_return_sequences=num_samples
            )
            
            # Convert to numpy array
            audio_data = audio_values[0, 0].cpu().numpy()
            
            # Normalize audio data
            audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
            
            # Create AudioSegment
            audio_segment = AudioSegment(
                audio_data.tobytes(),
                frame_rate=self.sampling_rate,
                sample_width=2,
                channels=1
            )

            # Trim or extend to desired duration
            if len(audio_segment) < duration * 1000:
                audio_segment = audio_segment * (duration * 1000 // len(audio_segment) + 1)
            audio_segment = audio_segment[:duration * 1000]

            return audio_segment
        except Exception as e:
            logger.error(f"Error in generate_music: {str(e)}")
            raise

    def get_sampling_rate(self) -> int:
        return self.sampling_rate

    async def process_music_generation(self, user_input: str, duration: int = 10) -> Tuple[str, AudioSegment]:
        try:
            optimized_prompt = await self.generate_optimized_prompt(user_input)
            logger.info(f"Optimized prompt: {optimized_prompt}")
            
            audio_segment = await self.generate_music(optimized_prompt, duration)
            logger.info(f"Generated audio of length: {len(audio_segment)}ms")
            
            return optimized_prompt, audio_segment
        except Exception as e:
            logger.error(f"Error in process_music_generation: {str(e)}")
            raise