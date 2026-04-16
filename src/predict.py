import joblib
import numpy as np
import os

# Import the feature extraction logic
from feature_extract import extract_pitch_contour, normalize_to_cents, create_pitch_histogram

# --- CONFIGURATION ---
MODEL_PATH = "../models/saved_models/raga_rf_model.pkl"
# CRITICAL: This MUST match the exact tonic you used during training!
TONIC_HZ = 138.59  

def predict_raga(audio_path):
    """
    Takes an unseen audio file, processes it, and asks the trained model to guess the raga.
    """
    # 1. Check if the model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Please run train.py first!")
        return
        
    # 2. Load the trained Random Forest model
    print("Loading the trained AI model...")
    model = joblib.load(MODEL_PATH)
    
    # 3. Process the new audio file
    print(f"Analyzing new audio: {os.path.basename(audio_path)}...")
    try:
        # Extract -> Normalize -> Histogram
        raw_f0, sr = extract_pitch_contour(audio_path)
        normalized_cents = normalize_to_cents(raw_f0, TONIC_HZ)
        hist, _ = create_pitch_histogram(normalized_cents)
        
        # Scikit-learn expects a 2D array for predictions (even if it's just one sample)
        # We reshape our 1D histogram array: [a, b, c] becomes [[a, b, c]]
        features = hist.reshape(1, -1)
        
        # 4. Make the prediction
        predicted_raga = model.predict(features)[0]
        
        # 5. Get the confidence percentages
        # This tells us how "sure" the model is, and what its second choice was.
        probabilities = model.predict_proba(features)[0]
        classes = model.classes_ # The list of ragas the model knows
        
        # --- Print the Results ---
        print("\n" + "="*40)
        print(f"🎵 PREDICTED RAGA: {predicted_raga.upper()} 🎵")
        print("="*40)
        
        print("\nConfidence Scores:")
        # Sort the probabilities from highest to lowest
        sorted_indices = np.argsort(probabilities)[::-1]
        for i in sorted_indices:
            raga_name = classes[i]
            prob_score = probabilities[i] * 100
            # Only print scores greater than 0%
            if prob_score > 0: 
                print(f" - {raga_name.capitalize()}: {prob_score:.1f}%")
            
    except Exception as e:
        print(f"An error occurred while analyzing the audio: {e}")

if __name__ == "__main__":
    # --- TEST THE SCRIPT ---
    # Point this to a BRAND NEW recording that was NOT in your training folder.
    # You can create a new folder: ../data/test_samples/
    TEST_AUDIO_FILE = "../data/raw/my_new_test_recording.wav" 
    
    if os.path.exists(TEST_AUDIO_FILE):
        predict_raga(TEST_AUDIO_FILE)
    else:
        print(f"File not found: {TEST_AUDIO_FILE}")
        print("Please record a new sample, update the file path, and run again!")