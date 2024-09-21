from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class MusicGenre(str, Enum):
    POP = "pop"
    ROCK = "rock"
    JAZZ = "jazz"
    CLASSICAL = "classical"
    ELECTRONIC = "electronic"
    HIPHOP = "hiphop"

class Mood(str, Enum):
    HAPPY = "happy"
    SAD = "sad"
    ENERGETIC = "energetic"
    CALM = "calm"
    ROMANTIC = "romantic"
    ANGRY = "angry"

class EditType(str, Enum):
    TEMPO = "tempo"
    PITCH = "pitch"
    VOLUME = "volume"

class KeywordToMusicRequest(BaseModel):
    keyword: str = Field(..., min_length=1, max_length=100, description="Keyword for generating lyrics")
    genre: MusicGenre = Field(..., description="Music genre")
    duration: float = Field(30.0, ge=10.0, le=120.0, description="Duration of the music in seconds")

class LyricsRequest(BaseModel):
    theme: str = Field(..., min_length=1, max_length=100, description="Theme for lyrics generation")
    genre: MusicGenre = Field(..., description="Music genre")
    mood: Mood = Field(..., description="Mood of the lyrics")
    length: int = Field(16, ge=4, le=32, description="Number of lines in the lyrics")

class MusicRequest(BaseModel):
    lyrics: str = Field(..., min_length=1, description="Lyrics for music generation")
    genre: MusicGenre = Field(..., description="Music genre")
    tempo: int = Field(120, ge=60, le=200, description="Tempo in BPM")
    mood: Mood = Field(Mood.ENERGETIC, description="Mood of the music")
    duration: float = Field(30.0, ge=10.0, le=120.0, description="Duration of the music in seconds")

class EditMusicRequest(BaseModel):
    edit_type: EditType = Field(..., description="Type of edit to perform")
    edit_value: str = Field(..., description="Value for the edit")

class AnalyzeMusicRequest(BaseModel):
    analysis_type: str = Field(..., description="Type of analysis to perform")

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    message: Optional[str] = None

class ErrorResponse(BaseModel):
    detail: str

class SuccessResponse(BaseModel):
    message: str
    data: Optional[dict] = None

# WebSocket message models
class WebSocketLyricsRequest(BaseModel):
    theme: str
    genre: MusicGenre
    mood: Mood
    length: int = 16

class AudioAnalysis(BaseModel):
    duration: float
    sample_rate: int
    num_channels: int
    rms: float
    peak: float
    dominant_frequency: Optional[float]

class CombineMusicAndLyricsRequest(BaseModel):
    lyrics: str = Field(..., min_length=1, description="Lyrics to combine with music")

class CreateProjectRequest(BaseModel):
    user_id: int
    project_name: str
    description: Optional[str] = None

class AddFileRequest(BaseModel):
    project_id: int
    file_name: str
    file_type: str  # 'txt' or 'mp3'
    file_content: bytes

class MusicGenerationRequest(BaseModel):
    user_input: str = Field(..., min_length=1, max_length=500, description="User input for music generation")
