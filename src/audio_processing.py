import os
import glob
import librosa
import soundfile as sf
import numpy as np

# --- CONFIGURATION ---
RAW_DATA_DIR = "data/raw/"
PROCESSED_DIR = "data/processed_chunks/" 

def separate_and_chunk_math(audio_path, segment_length_sec=15):
    """
    Uses pure signal processing (HPSS) to remove drums/mridangam,
    then chops the melodic track into 15-second training chunks.
    """
    filename = os.path.basename(audio_path)
    base_name = os.path.splitext(filename)[0]
    print(f"\nProcessing Concert: {filename}...")
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    try:
        # 1. Load the audio
        print(" Loading audio file...")
        y, sr = librosa.load(audio_path, sr=22050)
        
        # 2. Mathematical Separation (Remove the Mridangam)
        print(" Running Harmonic Separation (Removing percussion)...")
        # margin>1.0 pushes more of the audio into the percussive bucket, 
        # leaving an incredibly clean, isolated harmonic (vocal) track!
        y_harmonic, _ = librosa.effects.hpss(y, margin=(1.0, 5.0))
        
        # 3. Trim Silence from the new clean track
        y_trimmed, _ = librosa.effects.trim(y_harmonic, top_db=30)
        
        # 4. Chunking
        print(f"  -> Slicing melodic track into {segment_length_sec}s chunks...")
        samples_per_segment = segment_length_sec * sr
        total_samples = len(y_trimmed)
        num_segments = total_samples // samples_per_segment
        
        saved_chunks = 0
        for i in range(num_segments):
            start_sample = i * samples_per_segment
            end_sample = start_sample + samples_per_segment
            chunk = y_trimmed[start_sample:end_sample]
            
            chunk_filename = f"{base_name}_chunk_{i+1:03d}.wav"
            chunk_path = os.path.join(PROCESSED_DIR, chunk_filename)
            
            sf.write(chunk_path, chunk, sr)
            saved_chunks += 1
            
        print(f"Created {saved_chunks} clean harmonic chunks for training.")
        
    except Exception as e:
        print(f" Error processing {filename}: {e}")

def process_all_concerts():
    audio_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.*"))
    # HPSS can handle converted .wav files flawlessly
    audio_files = [f for f in audio_files if f.lower().endswith('.wav')]

    if not audio_files:
        print(f"No .wav files found in {RAW_DATA_DIR}. Please run convert_to_wav.py first!")
        return

    for file_path in audio_files:
        separate_and_chunk_math(file_path, segment_length_sec=15)
        
    print("\n All concert processing complete! You can now run train.py.")

if __name__ == "__main__":
    process_all_concerts()