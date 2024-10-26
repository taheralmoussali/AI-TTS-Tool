from transformers import FastSpeech2ConformerTokenizer, FastSpeech2ConformerWithHifiGan
import soundfile as sf

# Load Spanish model
tokenizer = FastSpeech2ConformerTokenizer.from_pretrained("espnet/kan-bayashi_css10_spanish_fastspeech2")
model = FastSpeech2ConformerWithHifiGan.from_pretrained("espnet/kan-bayashi_css10_spanish_fastspeech2")

# Convert text to speech
text = "Hola, bienvenido a la demostración de texto a voz en español."
inputs = tokenizer(text, return_tensors="pt")
output_dict = model(inputs["input_ids"], return_dict=True)
waveform = output_dict["waveform"]

# Save output
sf.write("spanish_speech.wav", waveform.squeeze().numpy(), samplerate=22050)
