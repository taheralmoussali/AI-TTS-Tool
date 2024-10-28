from transformers import VitsModel, AutoTokenizer
import torch
from .base_model import BaseTTSModel

class ArabicTTSModel(BaseTTSModel):
    def __init__(self):
        self.model = VitsModel.from_pretrained("facebook/mms-tts-ara")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ara")

    def synthesize_speech(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = self.model(**inputs).waveform
        return output.squeeze().numpy(), self.model.config.sampling_rate
