import numpy as np
import librosa
import joblib
import os

# Load your model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

def extract_features(audio_file_path):
    y, sr = librosa.load(audio_file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

def predict_pitch(audio_file_path):
    # Extract features
    features = extract_features(audio_file_path)
    
    # Scale features using loaded scaler
    scaled_features = scaler.transform([features])
    
    # Predict pitch
    predicted_pitch = model.predict(scaled_features)
    
    return predicted_pitch[0]

# Example usage for single file pitch detection
if __name__ == "__main__":
    audio_file_path = r'C:\pitch_detection_11\music\C#.wav'  # Replace with your audio file path
    predicted_pitch = predict_pitch(audio_file_path)
    print(f"Predicted Pitch: {predicted_pitch}")
