import os
import numpy as np
import librosa
import joblib
from flask import Flask, request, render_template, redirect, url_for, abort
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
import data_preperation

# Load your model and scaler
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    raise e

def extract_features(audio_file_path):
    try:
        y, sr = librosa.load(audio_file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except Exception as e:
        print(f"Error extracting features: {e}")
        raise e

def predict_pitch(audio_file_path):
    try:
        # Extract features
        features = extract_features(audio_file_path)
        
        # Scale features using loaded scaler
        scaled_features = scaler.transform([features])
        
        # Predict pitch
        predicted_pitch = model.predict(scaled_features)
        
        return predicted_pitch[0]
    except Exception as e:
        print(f"Error predicting pitch: {e}")
        raise e

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print("No file part in the request.")
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        print("No selected file.")
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return render_template('index.html', file_path=file_path)

@app.route('/predict', methods=['POST'])
def predict():
    file_path = request.form['file_path']
    try:
        predicted_pitch = predict_pitch(file_path)
        return render_template('result.html', pitch=predicted_pitch, file_path=file_path)
    except Exception as e:
        print(f"Error during pitch prediction: {e}")
        return abort(500)

@app.route('/show_accuracy', methods=['POST'])
def show_accuracy():
    try:
        data_dir = 'C:/pitch_detection_project/dataset'  # Replace with the actual path to your data directory
        X, y = data_preparation.load_data(data_dir)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        accuracy = model.score(X_test_scaled, y_test) * 100

        file_path = request.form['file_path']
        return render_template('result.html', pitch=None, file_path=file_path, accuracy=accuracy)
    except Exception as e:
        print(f"Error calculating accuracy: {e}")
        return abort(500)

@app.errorhandler(500)
def internal_error(error):
    return "500 Internal Server Error: An error occurred while processing your request.", 500

if __name__ == '__main__':
    app.run(debug=True)
