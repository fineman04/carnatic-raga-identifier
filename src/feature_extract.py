import librosa
import numpy as np
import matplotlib.pyplot as plt

def extract_pitch_contour(audio_path):
    """
    Extracts the fundamental frequency (F0) contour using the pYIN algorithm.
    """
    # sr=22050 is usually sufficient for pitch tracking and saves memory.
    y, sr = librosa.load(audio_path, sr=22050)
    
    # Fmin and Fmax define the human vocal range (roughly C2 to C6)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, 
        fmin=librosa.note_to_hz('C2'), 
        fmax=librosa.note_to_hz('C6'),
        sr=sr,
        frame_length=2048, 
        hop_length=512     
    )
    
    return f0, sr

def normalize_to_cents(f0_array, tonic_hz):
    """
    Converts raw Hertz values to relative cents based on a given tonic.
    """
    cents = np.zeros_like(f0_array)
    valid_idx = ~np.isnan(f0_array)
    
    # Formula: 1200 * log2(f / f_tonic)
    cents[valid_idx] = 1200 * np.log2(f0_array[valid_idx] / tonic_hz)
    cents[~valid_idx] = np.nan
    
    return cents

def create_pitch_histogram(cents_array, bins_per_octave=120):
    """
    Converts a time-series pitch contour into a fixed-size pitch histogram.
    """
    min_cents = -1200
    max_cents = 1200
    total_bins = int((max_cents - min_cents) / 10) 
    
    valid_cents = cents_array[~np.isnan(cents_array)]
    
    hist, bin_edges = np.histogram(valid_cents, bins=total_bins, range=(min_cents, max_cents))
    
    if np.sum(hist) > 0:
        hist = hist / np.sum(hist)
        
    return hist, bin_edges

def plot_gamakas(cents_contour, time_array):
    """Visualizes the normalized pitch contour."""
    plt.figure(figsize=(12, 6))
    plt.plot(time_array, cents_contour, label="Pitch Contour", color='b')
    for i in range(0, 1300, 100):
        plt.axhline(y=i, color='gray', linestyle='--', alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.8, label="Tonic (Sa)")
    plt.title("Pitch Contour (Normalized)")
    plt.xlabel("Time (s)")
    plt.ylabel("Cents")
    plt.legend()
    plt.show()

def plot_histogram(hist, bin_edges):
    """Visualizes the pitch histogram."""
    plt.figure(figsize=(10, 4))
    plt.bar(bin_edges[:-1], hist, width=10, color='purple', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', label='Tonic (Sa)')
    plt.axvline(x=702, color='green', linestyle='--', label='Perfect 5th (Pa)')
    plt.title("Raga Pitch Histogram")
    plt.xlabel("Cents Relative to Tonic")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()