import numpy as np
import librosa
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# Function to extract features from an audio file
def extract_features(audio_file_path):
    y, sr = librosa.load(audio_file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

# Function to load data
def load_data(data_dir):
    features = []
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for audio_file in os.listdir(label_dir):
                audio_file_path = os.path.join(label_dir, audio_file)
                feature = extract_features(audio_file_path)
                features.append(feature)
                labels.append(label)
    return np.array(features), np.array(labels)

# Path to your data directory
data_dir = 'D:\pitch_detection_12\dataset'
X, y = load_data(data_dir)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = MLPClassifier()
model.fit(X_train, y_train)

# Save the trained model and scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler have been saved.")
