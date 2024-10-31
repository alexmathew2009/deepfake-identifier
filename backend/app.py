from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from deepfake_detection_model import Model, predict, ValidationDataset, transforms
import torch
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configure a folder to store uploaded videos
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the deepfake detection model
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])

model = Model(2).cuda()
path_to_model = r'E:\Class Files\MajorProject\ProjectDir\deepfake_detection\backend\models\Copy of checkpoint2.pt'
model.load_state_dict(torch.load(path_to_model))
model.eval()

# Endpoint for deepfake detection
@app.route('/detect_deepfake', methods=['POST'])
def detect_deepfake():
    if 'video' not in request.files:
        return jsonify({'error': 'No file part'})

    video_file = request.files['video']

    if video_file.filename == '':
        return jsonify({'error': 'No selected file'})

    allowed_extensions = {'mp4', 'mov', 'avi'}
    if not '.' in video_file.filename or video_file.filename.split('.')[-1].lower() not in allowed_extensions:
        return jsonify({'error': 'Invalid file extension'})

    # Save the uploaded video file
    filename = secure_filename(video_file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(file_path)

    # Perform deepfake detection analysis using the model
    video_dataset = ValidationDataset([file_path], sequence_length=20, transform=train_transforms)
    prediction, confidence = predict(model, video_dataset[0])

    if prediction == 1:
        result = "REAL"
    else:
        result = "FAKE"

    return jsonify({'result': result, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
