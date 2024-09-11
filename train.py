import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from feature_extraction import extract_features
from model import create_model

def load_dataset(data_dir, max_length=44):
    features = []
    labels = []
    pitch_labels = {'A#': 0, 'B': 1, 'C': 2, 'C#': 3, 'D': 4, 'D#': 5, 'E': 6, 'F': 7, 'F#': 8, 'G': 9, 'G#': 10, 'A': 11}

    for pitch in pitch_labels:
        pitch_dir = os.path.join(data_dir, pitch)
        for file_name in os.listdir(pitch_dir):
            file_path = os.path.join(pitch_dir, file_name)
            feature = extract_features(file_path, max_length)
            features.append(feature)
            labels.append(pitch_labels[pitch])
    
    features = np.array(features)
    labels = np.array(labels)
    
    return features, labels

data_dir = 'dataset'
X, y = load_dataset(data_dir)

print("Shape of X before scaling:", X.shape)  # Debug print

scaler = StandardScaler()
X = scaler.fit_transform(X.reshape(X.shape[0], -1))

# Verify the shape of the data before reshaping
print("Shape of X after scaling:", X.shape)  # Debug print

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure the reshaping step is correct
num_samples, num_features = X_train.shape
expected_features = 13 * 44  # Based on your feature extraction

if num_features != expected_features:
    raise ValueError(f"Unexpected number of features. Expected {expected_features}, but got {num_features}")

X_train = X_train.reshape(num_samples, 13, 44, 1)
X_test = X_test.reshape(X_test.shape[0], 13, 44, 1)

model = create_model((13, 44, 1))
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

model.save('pitch_detection_model.h5')
