# B2AI Voice Trimkit

A Streamlit-based audio trimming tool for acoustic researchers with Praat-Parselmouth integration.

## Features

- **Multiple Task Types:**
  - **Sustained Vowel** - Single segment extraction per file
  - **Cough** - Detect and isolate cough segments
  - **Speech** - Praat-based trimming to isolate voiced region
  - **Breathing** - Trim leading and trailing silence
  - **Everything Fused** - Multi-segment detection (cough, breathing, speech, vowel)
  - **General-manual** - Manual region selection and trimming

- **Supported Formats:** `.wav` and `.mp3` (auto-converted to WAV)
- **Interactive Waveform:** Visual selection with Plotly
- **Batch Processing:** Process multiple files via CSV upload


## Installation

```bash
pip install b2ai-voice-trimkit
```

## Usage

Launch the dashboard:

```bash
b2ai_voice_trimkit
```

### CSV Format

Upload a CSV file with a column named `audio_file_path`:

```csv
audio_file_path
/path/to/audio1.wav
/path/to/audio2.mp3
```

## Output Location

Trimmed files are saved in a `Final_trim` folder in the same directory as the original audio file.

## Quitting

- Click **Quit Application** in the sidebar or
- Press `Ctrl+C` in the terminal

## Troubleshooting

**"Could not find ffmpeg or avconv"** - Install ffmpeg:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
choco install ffmpeg
```

## Dependencies

- numpy, pandas, streamlit
- praat-parselmouth
- plotly, soundfile, librosa, pydub

## License

MIT