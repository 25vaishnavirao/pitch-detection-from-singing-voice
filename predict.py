import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
import joblib

# Load the pre-trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

def extract_features(audio_file_path):
    y, sr = librosa.load(audio_file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

def predict_pitch(audio_file_path):
    print("Extracting features...")
    features = extract_features(audio_file_path)
    print("Features extracted:", features)

    # Ensure the scaler is fitted before transforming
    print("Scaling features...")
    scaled_features = scaler.transform([features])
    print("Scaled features:", scaled_features)

    print("Reshaping for model input...")
    input_data = scaled_features.reshape(1, 13, 1, 1)
    print("Input data:", input_data)

    print("Predicting pitch...")
    prediction = model.predict(input_data)
    print("Prediction:", prediction)

    return prediction[0]
