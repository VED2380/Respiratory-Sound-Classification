import librosa
import numpy as np
import os
import pandas as pd
from scipy.signal import get_window
from skimage.transform import resize

# Parameters
TARGET_SR = 4000  # Resample to 4000 Hz
FIXED_DURATION = 2.7  # Fixed length of each cycle in seconds
SAMPLES_PER_CYCLE = int(TARGET_SR * FIXED_DURATION)

def preprocess_audio(audio_path, annotation_path):
    # Load audio (original sample rate preserved)
    y, sr = librosa.load(audio_path, sr=None)
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

    # Load annotations (tab-separated)
    annotations = pd.read_csv(annotation_path, delimiter='\t', 
                              names=['start', 'end', 'crackles', 'wheezes'])
    
    cycles = []
    labels = []
    for _, row in annotations.iterrows():
        start_sample = int(row['start'] * TARGET_SR)
        end_sample = int(row['end'] * TARGET_SR)
        cycle = y_resampled[start_sample:end_sample]

        # Pad or truncate cycle to fixed length
        if len(cycle) > SAMPLES_PER_CYCLE:
            cycle = cycle[:SAMPLES_PER_CYCLE]
        elif len(cycle) < SAMPLES_PER_CYCLE:
            cycle = np.pad(cycle, (0, SAMPLES_PER_CYCLE - len(cycle)), 'constant')
        
        cycles.append(cycle)
        
        # Label encoding: 0=normal,1=crackles,2=wheezes,3=both
        if row['crackles'] == 1 and row['wheezes'] == 1:
            labels.append(3)
        elif row['crackles'] == 1:
            labels.append(1)
        elif row['wheezes'] == 1:
            labels.append(2)
        else:
            labels.append(0)
    
    return np.array(cycles), np.array(labels)

# Set your actual directories here
audio_files_dir = r"C:\Dataset\ICBHI_final_database"  # your audio files folder
annotation_files_dir = r"C:\Dataset\ICBHI_final_database"  # your annotation files folder (same folder if .txt with .wav)

cycles, labels = [], []

for audio_file in os.listdir(audio_files_dir):
    if audio_file.endswith('.wav'):
        audio_path = os.path.join(audio_files_dir, audio_file)
        annotation_path = os.path.join(annotation_files_dir, audio_file.replace('.wav', '.txt'))
        if not os.path.exists(annotation_path):
            print(f"Missing annotation for {audio_file}, skipping.")
            continue
        
        c, l = preprocess_audio(audio_path, annotation_path)
        cycles.append(c)
        labels.append(l)
        print(f"Processed {audio_file} with {len(c)} cycles.")

cycles = np.concatenate(cycles, axis=0)
labels = np.concatenate(labels, axis=0)

def compute_spectrogram(cycle):
    window = get_window('hann', 256)
    stft = librosa.stft(cycle, n_fft=256, hop_length=128, window=window)
    spectrogram = np.abs(stft) ** 2
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    return spectrogram_db

spectrograms = np.array([compute_spectrogram(c) for c in cycles])
spectrograms_resized = np.array([resize(spec, (75, 50), mode='constant') for spec in spectrograms])
spectrograms_resized = np.expand_dims(spectrograms_resized, axis=-1)

print("Spectrograms shape:", spectrograms_resized.shape)
print("Labels shape:", labels.shape)

