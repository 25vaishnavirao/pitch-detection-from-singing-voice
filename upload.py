import requests

url = "http://127.0.0.1:5000/predict"
file_path = "D:/pitch_detection_12/music/nagumotest.wav"

with open(file_path, 'rb') as f:
    files = {'file': f}
    response = requests.post(url, files=files)

print("Status Code:", response.status_code)
print("Response Text:", response.text)
