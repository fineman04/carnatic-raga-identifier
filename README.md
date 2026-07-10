# Carnatic Raga Identifier

A Python audio-classification pipeline for identifying Carnatic ragas from
recorded or live vocal audio. The project extracts a fundamental-frequency
(F0) contour with librosa's pYIN algorithm, normalizes the notes relative to a
tonic, converts the contour into a pitch-class histogram, and classifies it
with a Random Forest model.

The repository includes the complete preprocessing and training pipeline, a
saved model, file-based prediction, and one-shot microphone prediction.

## How It Works

```text
Audio recording
      |
      v
Harmonic/percussive separation (HPSS)
      |
      v
15-second harmonic audio chunks
      |
      v
pYIN pitch tracking -> tonic normalization -> pitch histogram
      |
      v
Random Forest classifier -> predicted raga and confidence scores
```

The current pipeline assumes a tonic of **C-sharp (`138.59 Hz`)**. Training
and prediction must use the same tonic value for the features to be
comparable.

## Features

- Converts video and other media files to WAV audio
- Reduces percussion with harmonic-percussive source separation (HPSS)
- Splits recordings into fixed 15-second training samples
- Tracks vocal pitch using pYIN
- Represents pitch relative to the tonic in cents
- Trains and evaluates a Random Forest classifier
- Predicts from an existing audio file or a live microphone recording
- Includes plotting helpers for pitch contours, histograms, and confusion
  matrices

## Project Structure

```text
carnatic-raga-identifier/
|-- models/
|   `-- saved_models/
|       `-- raga_rf_model.pkl       # Included trained Random Forest model
|-- notebooks/
|   `-- 01_data_exploration.ipynb   # Experimentation notebook
|-- src/
|   |-- audio_processing.py         # HPSS, silence trimming, and chunking
|   |-- convert_to_wav.py           # Media-to-WAV conversion
|   |-- feature_extract.py          # Pitch and histogram features
|   |-- live_predict.py             # One-shot microphone prediction
|   |-- predict.py                  # Prediction from an audio file
|   `-- train.py                    # Training and evaluation
|-- requirements.txt
`-- README.md
```

The `data/` directory is created locally and is not included in the
repository. Use this layout:

```text
data/
|-- raw/                 # Original recordings and test audio
`-- processed_chunks/    # Generated 15-second WAV chunks
```

## Requirements

- Python 3.10 or newer is recommended
- FFmpeg for media conversion through MoviePy
- A working microphone and PortAudio-compatible audio setup for live
  prediction

Create and activate a virtual environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install joblib moviepy sounddevice
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

`joblib`, `moviepy`, and `sounddevice` are used directly by the scripts but
are not currently listed in `requirements.txt`, so the extra installation
command is required.

## Prepare a Dataset

Run all preparation and training commands from the repository root.

### 1. Add recordings

Create `data/raw/` and add one or more recordings for each raga:

```bash
mkdir -p data/raw
```

The filename must start with the raga label followed by an underscore. The
training script uses everything before the first underscore as the class:

```text
mayamalavagowla_concert_01.mp4
mayamalavagowla_vocal_02.wav
kalyani_concert_01.mp4
```

Use single-word labels because underscores delimit the label.

### 2. Convert recordings to WAV

WAV inputs can be used as-is. Convert MP4 and other supported media files
with:

```bash
python3 src/convert_to_wav.py
```

The converted WAV files are written beside the originals in `data/raw/`.

### 3. Separate and chunk the audio

```bash
python3 src/audio_processing.py
```

This keeps the harmonic component, trims silence, and writes complete
15-second chunks to `data/processed_chunks/`. Audio shorter than 15 seconds
after trimming does not produce a chunk.

## Train the Model

```bash
python3 src/train.py
```

Training performs an 80/20 stratified split, fits a 100-tree Random Forest,
prints accuracy and a classification report, displays a confusion matrix,
and saves the resulting model to:

```text
models/saved_models/raga_rf_model.pkl
```

Each class needs enough chunks for a stratified train/test split. For a more
meaningful evaluation, use several independent recordings per raga instead of
chunks from only one concert.

## Predict an Audio File

1. Open `src/predict.py`.
2. Set `TEST_AUDIO_FILE` to the path of a WAV recording that was not used for
   training.
3. Run the script from the `src/` directory because its current model and
   sample paths are relative to that directory:

```bash
cd src
python3 predict.py
```

The command prints the predicted raga and the model's confidence score for
each known class.

## Predict From a Microphone

Run the live predictor from the repository root:

```bash
python3 src/live_predict.py
```

The script records 15 seconds from the default microphone, trims silence,
extracts the same pitch-histogram features used during training, and prints a
single prediction. Sing or play phrases in the same tonic used to train the
model.

## Configuration

The main constants are defined near the top of each script:

| Setting | Default | Location |
|---|---:|---|
| Tonic frequency | `138.59 Hz` | `train.py`, `predict.py`, `live_predict.py` |
| Chunk length | `15 seconds` | `audio_processing.py` |
| Live recording length | `15 seconds` | `live_predict.py` |
| Sample rate | `22050 Hz` | feature extraction and live prediction |
| Model path | `models/saved_models/raga_rf_model.pkl` | training/live prediction |

If you change the tonic, retrain the model and use the identical value for
all prediction modes.

## Current Limitations

- The classifier learns only the ragas represented in the training dataset.
- Tonic detection is not automatic; the tonic is configured manually.
- Pitch histograms capture note distribution but do not fully represent phrase
  order or detailed gamaka movement.
- Confidence scores are Random Forest class probabilities, not a calibrated
  measure of musical certainty.
- HPSS reduces percussion but does not perform full vocal source separation.



## Acknowledgements

This project uses [librosa](https://librosa.org/) for audio analysis and
[scikit-learn](https://scikit-learn.org/) for classification.
