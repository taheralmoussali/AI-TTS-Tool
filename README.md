# AI Text-to-Speech (TTS) Tool

## Overview
This project is an AI-powered Text-to-Speech (TTS) tool that supports multiple languages (English, Arabic, and Spanish) with different voices. Using pretrained models from Hugging Face's `transformers` library, the tool converts input text into natural-sounding audio files, with support for various voices in English.

## Features
- **Multi-language Support**: Generates speech in English, Arabic, and Spanish.
- **Multiple Voices**: Provides three different voices for English text.
- **Audio Output**: Saves generated audio in `.wav` format.


## Prerequisites
Ensure that you have Python 3.7 or later installed on your system.

## Setup Instructions

### Step 1: Create a Virtual Environment
It’s recommended to work within a virtual environment to manage dependencies:

- For Windows:
  ```bash
  python -m venv env
  ```

- For Linux/macOS:
  ```bash
  python3 -m venv env
  ```

### Step 2: Activate the Virtual Environment
- For Windows:
  ```bash
  .\env\Scripts\activate
  ```

- For Linux/macOS:
  ```bash
  source env/bin/activate
  ```

### Step 3: Install Dependencies
Install the required packages using `pip`:
```bash
pip install -r requirements.txt
```

### Step 4: Run the Tool
Run the main script to start generating speech:

```bash
python main.py
```

## Usage

1. **Choose a Language**: When prompted, select a language (`english`, `arabic`, or `spanish`).
2. **Enter Text**: Input the text you want to convert to speech.
3. **Select a Voice (for English only)**: If `english` is selected, choose a voice (`voice1`, `voice2`, or `voice3`).
4. **Listen to the Output**: The audio file will be saved in the current directory as a `.wav` file (e.g., `english_voice1.wav`, `arabic_voice.wav`, `spanish_voice.wav`).

### Example Commands
1. Select language:
   ```
   Select language (english/spanish/arabic): english
   ```

2. Enter the text:
   ```
   Enter the text: Welcome to the AI Text-to-Speech tool!
   ```

3. Choose a voice (for English):
   ```
   Choose a voice (voice1, voice2, voice3): voice1
   ```

4. The output will be saved in the current directory, for example, as `english_voice1.wav`.

## Supported Languages and Voices
- **English**: Three voices (`voice1`, `voice2`, `voice3`)
- **Arabic**: One default voice
- **Spanish**: One default voice

## Customization
- **Add New Languages or Voices**: To add more languages or voices, create a new class in the `models/` directory following the structure of `english_model.py`, `arabic_model.py`, or `spanish_model.py`.
- **Modify Output Audio**: To change output file names or formats, edit the `save_audio` function in `audio_utils.py`.


## Voice Testing
You can find the following sample audio files in the project directory:

- **English**:
  - [english_voice1.wav](./results/english_3.wav)
  
- **Arabic**:
  - [arabic_voice.wav](./results/arabic_voice.wav)

- **Spanish**:
  - [spanish_voice.wav](./results/spanish_voice.wav)