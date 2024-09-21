import pandas as pd
import numpy as np
import librosa
import logging
from typing import List, Dict, Any, Tuple
from pydub import AudioSegment
from base64 import b64encode
import json
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import io

class MusicDataLoader:
    def __init__(self):
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            filename='music_data_loading.log',
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
        )

    def load_audio(self, file_content: bytes, file_format: str) -> Tuple[np.ndarray, int]:
        try:
            audio = AudioSegment.from_file(io.BytesIO(file_content), format=file_format)
            samples = np.array(audio.get_array_of_samples())
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
            sr = audio.frame_rate
            logging.info(f"Audio file loaded successfully")
            return samples, sr
        except Exception as err:
            logging.exception(f"An error occurred while loading the audio file: {err}")
            raise

    def get_audio_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        features = {
            'tempo': librosa.beat.tempo(y=y, sr=sr)[0],
            'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            'spectral_rolloff': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(y=y)),
            'rms': np.sqrt(np.mean(np.square(y))),
            'peak': np.max(np.abs(y))
        }
        logging.info("Audio features extracted successfully")
        return features

    def get_lyrics_features(self, lyrics: str) -> Dict[str, Any]:
        features = {
            'word_count': len(lyrics.split()),
            'char_count': len(lyrics),
            'line_count': len(lyrics.split('\n')),
            'unique_words': len(set(lyrics.lower().split()))
        }
        logging.info("Lyrics features extracted successfully")
        return features

    def audio_to_base64(self, audio_segment: AudioSegment) -> str:
        try:
            buffer = io.BytesIO()
            audio_segment.export(buffer, format="mp3")
            b64 = b64encode(buffer.getvalue()).decode()
            logging.info("Audio converted to base64")
            return b64
        except Exception as err:
            logging.exception(f"An error occurred while converting audio to base64: {err}")
            raise

    def get_waveform_plot(self, y: np.ndarray, sr: int) -> go.Figure:
        times = librosa.times_like(y, sr=sr)
        fig = px.line(x=times, y=y, title='Waveform')
        fig.update_xaxes(title_text='Time')
        fig.update_yaxes(title_text='Amplitude')
        return fig

    def get_spectrogram_plot(self, y: np.ndarray, sr: int) -> go.Figure:
        D = librosa.stft(y)  # STFT of y
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        fig = px.imshow(S_db, aspect='auto', origin='lower', 
                        x=librosa.times_like(S_db), y=librosa.fft_frequencies(sr=sr),
                        title='Spectrogram')
        fig.update_xaxes(title_text='Time')
        fig.update_yaxes(title_text='Frequency')
        return fig

    def get_combined_plot(self, y: np.ndarray, sr: int) -> go.Figure:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05,
                            subplot_titles=('Waveform', 'Spectrogram'))
        
        # Waveform
        times = librosa.times_like(y, sr=sr)
        fig.add_trace(go.Scatter(x=times, y=y, name='Waveform'), row=1, col=1)
        
        # Spectrogram
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        fig.add_trace(go.Heatmap(z=S_db, x=times, y=librosa.fft_frequencies(sr=sr), 
                                 colorscale='Viridis', name='Spectrogram'), 
                      row=2, col=1)
        
        fig.update_xaxes(title_text='Time', row=2, col=1)
        fig.update_yaxes(title_text='Amplitude', row=1, col=1)
        fig.update_yaxes(title_text='Frequency', row=2, col=1)
        
        fig.update_layout(height=800, title_text="Waveform and Spectrogram")
        return fig

    def analyze_audio(self, audio_segment: AudioSegment, analysis_type: str) -> Dict[str, Any]:
        y = np.array(audio_segment.get_array_of_samples())
        if audio_segment.channels == 2:
            y = y.reshape((-1, 2))
        sr = audio_segment.frame_rate

        analysis_result = {}

        if analysis_type == 'basic' or analysis_type == 'all':
            analysis_result.update(self.get_audio_features(y, sr))
        
        if analysis_type == 'spectral' or analysis_type == 'all':
            spectral_features = {
                'spectral_contrast': np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)),
                'chroma_stft': np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
            }
            analysis_result.update(spectral_features)
        
        if analysis_type == 'rhythm' or analysis_type == 'all':
            rhythm_features = {
                'tempo': librosa.beat.tempo(y=y, sr=sr)[0],
                'beat_frames': len(librosa.beat.beat_track(y=y, sr=sr)[1])
            }
            analysis_result.update(rhythm_features)

        logging.info(f"Audio analysis completed for type: {analysis_type}")
        return analysis_result

def get_stats_table(data: Dict[str, Any]) -> go.Table:
    headers = list(data.keys())
    values = [[str(round(value, 2)) if isinstance(value, float) else str(value) 
               for value in data.values()]]
    
    table = go.Table(
        header=dict(values=headers, font={"size": 14, "color": "white"}, 
                    fill_color="rgb(127, 127, 127)"),
        cells=dict(values=values, font_size=16, height=30, 
                   fill_color="rgb(244, 244, 246)"),
    )
    return table

def convert_to_title_case(snake_str):
    components = snake_str.split('_')
    return ' '.join(x.capitalize() for x in components)