import librosa
import numpy as np

def extract_features(file_name, max_length=44):
    # Load the audio file
    y, sr = librosa.load(file_name, sr=None)
    
    # Extract MFCC features with 13 coefficients
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Pad or truncate to ensure fixed length
    if mfccs.shape[1] > max_length:
        mfccs = mfccs[:, :max_length]
    elif mfccs.shape[1] < max_length:
        pad_width = max_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    
    # Print the shape of the feature for debugging purposes
    print(f"Feature shape for {file_name}: {mfccs.shape}")  # Debug print
    
    return mfccs
