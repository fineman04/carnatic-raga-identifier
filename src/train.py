import os
import glob
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import the unified functions
from feature_extract import extract_pitch_contour, normalize_to_cents, create_pitch_histogram

# --- CONFIGURATION (Fixed Paths!) ---
DATA_DIR = "data/processed_chunks/"
MODEL_SAVE_PATH = "models/saved_models/raga_rf_model.pkl"

# Your C# Tonic
TONIC_HZ = 138.59 

def load_and_process_data():
    """Loops through the processed chunks and extracts features."""
    X = [] 
    y = [] 
    
    audio_files = glob.glob(os.path.join(DATA_DIR, "*.wav"))
    
    if not audio_files:
        print(f"No .wav files found in {DATA_DIR}. Did you run audio_processing.py?")
        return None, None

    print(f"Found {len(audio_files)} audio chunks. Starting feature extraction...")
    print("This might take a couple of minutes depending on your CPU...\n")
    
    for file_path in audio_files:
        filename = os.path.basename(file_path)
        # Extract raga name (e.g., 'mayamalavagowla' from 'mayamalavagowla_01_full.wav')
        raga_label = filename.split('_')[0].lower() 
        
        try:
            # Extract -> Normalize -> Histogram
            raw_f0, sr = extract_pitch_contour(file_path)
            normalized_cents = normalize_to_cents(raw_f0, TONIC_HZ)
            hist, _ = create_pitch_histogram(normalized_cents)
            
            X.append(hist)
            y.append(raga_label)
            print(f"Processed: {filename} -> Label: {raga_label}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    return np.array(X), np.array(y)

def plot_confusion_matrix(y_true, y_pred, labels):
    """Creates a visual heatmap of the model's accuracy."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Raga Classification Confusion Matrix')
    plt.ylabel('Actual Raga')
    plt.xlabel('Predicted Raga')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    X, y = load_and_process_data()
    
    if X is not None and len(X) > 0:
        print("\nDataset built successfully!")
        print(f"Total samples: {X.shape[0]}")
        
        # Split into Training (80%) and Testing (20%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print("\nTraining the Random Forest model...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        print("\nEvaluating model on test data...")
        y_pred = rf_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nOverall Accuracy: {accuracy * 100:.2f}%\n")
        
        print("Detailed Classification Report:")
        print(classification_report(y_test, y_pred))
        
        unique_labels = np.unique(y)
        plot_confusion_matrix(y_test, y_pred, unique_labels)
        
        # Save the model
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        joblib.dump(rf_model, MODEL_SAVE_PATH)
        print(f"\nModel successfully saved to: {MODEL_SAVE_PATH}")