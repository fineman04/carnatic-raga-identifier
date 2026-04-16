import os
import joblib
import numpy as np
import sounddevice as sd
import librosa

# Import your unified feature extraction functions
from feature_extract import extract_pitch_contour, normalize_to_cents, create_pitch_histogram

# --- CONFIGURATION ---
MODEL_PATH = "models/saved_models/raga_rf_model.pkl"
TONIC_HZ = 138.59
SAMPLE_RATE = 22050 
# Bumped up to 15 seconds so you have time to sing a full phrase!
RECORD_SECONDS = 15 

def record_live_audio(duration, fs):
    """Captures live audio from the default microphone."""
    print(f"\n Listening for {duration} seconds... (Sing now!)")
    
    # Record audio
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait() # The script pauses here until the 10 seconds are up
    
    print("Recording complete. Processing the Gamakas...")
    return recording.flatten()

def run_single_prediction():
    """Listens ONCE, processes, predicts, and exits cleanly."""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Please train it first!")
        return
        
    
    model = joblib.load(MODEL_PATH)
    classes = model.classes_
    
    
    print("ONE-SHOT RAGA RECOGNITION")

    
    # 1. Capture microphone input exactly once
    live_audio = record_live_audio(RECORD_SECONDS, SAMPLE_RATE)
    
    # Trim silence in case you started singing late
    y_trimmed, _ = librosa.effects.trim(live_audio, top_db=30)
    
    # If the audio is mostly silence, exit gracefully
    if len(y_trimmed) < (SAMPLE_RATE * 1):
        print("Audio was too quiet or too short. Please run the script again and sing louder!")
        return
    
    # 2. Extract Features
    f0, _, _ = librosa.pyin(
        y_trimmed, 
        fmin=librosa.note_to_hz('C2'), 
        fmax=librosa.note_to_hz('C6'),
        sr=SAMPLE_RATE,
        frame_length=2048, 
        hop_length=512     
    )
    
    # 3. Normalize and Histogram
    normalized_cents = normalize_to_cents(f0, TONIC_HZ)
    hist, _ = create_pitch_histogram(normalized_cents)
    
    # 4. Predict
    features = hist.reshape(1, -1)
    probabilities = model.predict_proba(features)[0]
    
    # Get the highest scoring raga
    best_idx = np.argmax(probabilities)
    best_raga = classes[best_idx]
    
    # --- Display Results ---
    
    print(f"PREDICTED RAGA: {best_raga.upper()} ")
    
    
    print("\nConfidence Scores:")
    sorted_indices = np.argsort(probabilities)[::-1]
    for i in sorted_indices:
        raga_name = classes[i]
        prob_score = probabilities[i] * 100
        if prob_score > 0: 
            print(f" - {raga_name.capitalize()}: {prob_score:.1f}%")

if __name__ == "__main__":
    run_single_prediction()