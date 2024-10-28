def save_audio(audio_data, file_name, sampling_rate):
    import soundfile as sf
    sf.write(file_name, audio_data, samplerate=sampling_rate)
