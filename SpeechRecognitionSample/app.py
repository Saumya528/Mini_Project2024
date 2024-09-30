from flask import Flask, request, jsonify, render_template
import librosa
import numpy as np
import pickle
import os

# Load the trained model.
#model_path = "D:\Downloads\SAUMYA MITRA\IIT Jodhpur\Mini_Project2024\SpeechRecognitionSample\model.pkl"
with open("model.pkl", 'rb') as f:
    model = pickle.load(f)

# Initialize the Flask application
app = Flask(__name__)

# Route to serve the HTML frontend
@app.route('/')
def index():
    return render_template('index.html')

# Function to extract audio features from an uploaded audio file
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    pitch, _ = librosa.piptrack(y=y, sr=sr)

    # Aggregate into a single feature vector
    features = np.hstack([
        np.mean(mfccs, axis=1), 
        np.mean(chroma, axis=1), 
        np.mean(spectral_contrast, axis=1), 
        np.mean(zero_crossing_rate),
        np.mean(pitch[pitch > 0]) if len(pitch[pitch > 0]) > 0 else 0
    ])
    return features

# Define an endpoint to accept audio files and return predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({"error": "Please provide an audio file using the 'audio' key."}), 400

    # Read the uploaded file
    file = request.files['audio']

    # Check for valid file types (optional)
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    # Process the file (e.g., feature extraction and prediction)
    features = extract_features(file)  # Your feature extraction function
    features = features.reshape(1, -1)  # Reshape for the model

    # Predict using the loaded model
    prediction = model.predict(features)

    return jsonify({'predicted_class': prediction[0]})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
