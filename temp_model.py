import torch
import torchaudio
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
import IPython.display as ipd
from datasets import load_dataset

# Load FastSpeech model
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

# Load speaker embedding
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = embeddings_dataset[7306]["xvector"]
speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)

# Load HiFi-GAN vocoder using torch.hub with CPU mapping
vocoder = torch.hub.load(
    'bshall/hifigan', 'hifigan', source='github',
    map_location=torch.device('cpu')  # Ensures model loads on CPU
)

# Define a linear transformation layer to map 80 channels to 128 channels
linear_transform = torch.nn.Linear(80, 128)

def text_to_speech(text, output_path="output.wav"):
    """
    Convert input text to speech and save as an audio file.
    
    Parameters:
    text (str): Input text to synthesize.
    output_path (str): Path to save the generated audio.
    """
    # Tokenize and generate mel spectrogram
    inputs = processor(text=text, return_tensors="pt")
    with torch.no_grad():
        mel_output = model.generate(inputs["input_ids"], speaker_embeddings=speaker_embeddings)

    # Transform mel_output to 128 channels using the linear layer
    mel_output_128 = linear_transform(mel_output.squeeze(0)).unsqueeze(0)

    # Use HiFi-GAN vocoder to convert mel spectrogram to waveform on CPU
    with torch.no_grad():
        waveform = vocoder(mel_output_128).squeeze(0).cpu()

    # Save waveform as a .wav file
    torchaudio.save(output_path, waveform.unsqueeze(0), sample_rate=22050)
    print(f"Audio saved to {output_path}")

    # Play audio
    ipd.display(ipd.Audio(output_path))


text = "Hello, this is a sample text to speech conversion using FastSpeech."
text_to_speech(text)
