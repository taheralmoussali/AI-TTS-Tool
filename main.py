from models.english_model import EnglishTTSModel
from models.arabic_model import ArabicTTSModel
from models.spanish_model import SpanishTTSModel
from utils.audio_utils import save_audio

def main():
    language = input("Select language (1: english, 2: arabic, 3: spanish): ").strip().lower()
    text = input("Enter the text: ").strip()
    
    if language == "english" or language == str(1):
        voice = input("Choose a voice (1, 2, 3): ").strip()
        model = EnglishTTSModel()
        audio_data, sampling_rate = model.synthesize_speech(text, f"voice{voice}")
        save_audio(audio_data, f"results/english_{voice}.wav", sampling_rate)
    
    elif language == "arabic" or language == str(2):
        model = ArabicTTSModel()
        audio_data, sampling_rate = model.synthesize_speech(text)
        save_audio(audio_data, "results/arabic_voice.wav", sampling_rate)

    elif language == "spanish" or language == str(3):
        model = SpanishTTSModel()
        audio_data, sampling_rate = model.synthesize_speech(text)
        save_audio(audio_data, "results/spanish_voice.wav", sampling_rate)
    
    else:
        print("Invalid language selected. Please choose 'english' or 'arabic'.")

if __name__ == "__main__":
    main()
