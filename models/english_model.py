from transformers import pipeline
from datasets import load_dataset
import torch
from .base_model import BaseTTSModel

class EnglishTTSModel(BaseTTSModel):
    def __init__(self):
        self.synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
        self.embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speakers = {
            "voice1": torch.tensor(self.embeddings_dataset[512]["xvector"]).unsqueeze(0),
            "voice2": torch.tensor(self.embeddings_dataset[1500]["xvector"]).unsqueeze(0),
            "voice3": torch.tensor(self.embeddings_dataset[7306]["xvector"]).unsqueeze(0)
        }

    def synthesize_speech(self, text: str, voice: str):
        if voice not in self.speakers:
            raise ValueError("Invalid voice choice. Choose from 'voice1', 'voice2', or 'voice3'.")
        speaker_embedding = self.speakers[voice]
        output = self.synthesiser(text, forward_params={"speaker_embeddings": speaker_embedding})
        return output["audio"], output["sampling_rate"]
