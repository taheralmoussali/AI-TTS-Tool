import nltk

# Download required NLTK resources for g2p_en
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('cmudict')  # Ensures CMU dictionary is available for G2P


from transformers import FastSpeech2ConformerTokenizer, FastSpeech2ConformerWithHifiGan
import soundfile as sf

# Load tokenizer and model
tokenizer = FastSpeech2ConformerTokenizer.from_pretrained("espnet/fastspeech2_conformer")
model = FastSpeech2ConformerWithHifiGan.from_pretrained("espnet/fastspeech2_conformer_with_hifigan")

def text_to_speech(text, output_path="speech.wav"):
    """
    Convert input text to speech and save as an audio file.
    
    Parameters:
    text (str): Input text to synthesize.
    output_path (str): Path to save the generated audio.
    """
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Generate speech waveform
    output_dict = model(input_ids, return_dict=True)
    waveform = output_dict["waveform"]

    # Save waveform as a .wav file
    sf.write(output_path, waveform.squeeze().detach().numpy(), samplerate=22050)
    print(f"Audio saved to {output_path}")




text = "We are excited to have you work on this assignment to assess your skills in developing AI-powered tools."
text_to_speech(text)
