import soundfile as sf
from abc import ABC, abstractmethod

class BaseTTSModel(ABC):
    @abstractmethod
    def synthesize_speech(self, text: str, voice: str):
        pass

    @staticmethod
    def save_audio(audio_data, file_name='output.wav', sampling_rate=22050):
        # Save audio to a .wav file
        sf.write(file_name, audio_data, samplerate=sampling_rate)
