import os
# We changed this line! We just import directly from moviepy now.
from moviepy import AudioFileClip

# --- CONFIGURATION ---
RAW_DATA_DIR = "data/raw/"

def convert_all_to_wav():
    """
    Scans the raw data folder for .mp4 (or other formats) and extracts 
    the pure audio track to a high-quality .wav file using moviepy.
    """
    if not os.path.exists(RAW_DATA_DIR):
        print(f"Error: Could not find the directory '{RAW_DATA_DIR}'.")
        return

    # List all files in the directory
    files = os.listdir(RAW_DATA_DIR)
    converted_count = 0

    print(f"Scanning '{RAW_DATA_DIR}' for files to convert...\n")

    for filename in files:
        file_path = os.path.join(RAW_DATA_DIR, filename)
        
        # Skip folders and files that are already .wav
        if not os.path.isfile(file_path) or filename.lower().endswith('.wav'):
            continue

        print(f"Converting: {filename}...")
        
        try:
            # 1. Load the file (moviepy can handle .mp4 easily)
            audio_clip = AudioFileClip(file_path)
            
            # 2. Create the new filename
            name_without_ext = os.path.splitext(filename)[0]
            new_file_path = os.path.join(RAW_DATA_DIR, f"{name_without_ext}.wav")
            
            # 3. Export as .wav (logger=None stops it from spamming the console)
            audio_clip.write_audiofile(new_file_path, logger=None)
            
            # 4. Close the clip to free up memory
            audio_clip.close()
            
            print(f"  -> Success! Saved as {name_without_ext}.wav")
            converted_count += 1
            
        except Exception as e:
            print(f"  -> Error converting {filename}.")
            print(f"  -> Details: {e}")

    print(f"\nDone! Successfully converted {converted_count} files to .wav.")

if __name__ == "__main__":
    convert_all_to_wav()