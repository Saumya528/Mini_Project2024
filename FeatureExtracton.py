import os
import librosa
import numpy as np
import pandas as pd
import pickle

# Define emotion mapping
EMOTION_MAP = {
    'ANG': 'negative',
    'DIS': 'negative',
    'FEA': 'negative',
    'HAP': 'positive',
    'NEU': 'neutral',
    'SAD': 'negative'
}

# Function to extract audio features from a given file
def extract_audio_features(file_path):
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    
    # Extract features
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
    pitch, mag = librosa.piptrack(y=audio_data, sr=sample_rate)

    features = np.hstack([
        np.mean(mfccs, axis=1), 
        np.mean(chroma, axis=1), 
        np.mean(spectral_contrast, axis=1), 
        np.mean(zero_crossing_rate),
        np.mean(pitch[pitch > 0])
    ])
    return features

# Load CREMA-D dataset and extract features
def load_crema_d_data(dataset_folder):
    features_list = []
    labels = []

    for file_name in os.listdir(dataset_folder):
        if file_name.endswith(".wav"):
            file_path = os.path.join(dataset_folder, file_name)
            emotion_code = file_name.split('_')[2]  # Extract emotion code from filename
            if emotion_code in EMOTION_MAP:
                features = extract_audio_features(file_path)
                features_list.append(features)
                labels.append(EMOTION_MAP[emotion_code])  # Map emotion to sentiment
                
    return np.array(features_list), np.array(labels)

# Example usage
dataset_folder = 'D://Downloads/SAUMYA MITRA/IIT Jodhpur/Mini_Project2024/CremaD'  # Path to the folder containing the CREMA-D audio files
# X, y = load_crema_d_data(dataset_folder)/*
# print(f"Extracted {len(X)} feature vectors from CREMA-D.")


# Main execution
if __name__ == "__main__":
    data_directory = dataset_folder  # Replace with your directory
    X, y = load_crema_d_data(data_directory)

    # Save extracted features and labels as a pickle file
    with open("features_labels.pkl", 'wb') as f:
        pickle.dump((X, y), f)

    print("Feature extraction completed. Saved to 'features_labels.pkl'.")

