import torch
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
from torch import nn
from torchvision import models

class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

def predict(model, img):
    fmap, logits = model(img.to('cuda'))
    sm = nn.Softmax(dim=1)
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    return [int(prediction.item()), confidence]

class ValidationDataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        vidObj = cv2.VideoCapture(video_path)
        success = 1
        while success and len(frames) < self.count:
            success, image = vidObj.read()
            if success:
                face_image = self.detect_and_crop_face(image)
                if face_image is not None:
                    if self.transform:
                        face_image = self.transform(face_image)
                    frames.append(face_image)
        vidObj.release()
        frames = torch.stack(frames)
        return frames.unsqueeze(0)

    def detect_and_crop_face(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
        if len(faces) == 0:
            return None
        x, y, w, h = faces[0]
        cropped_face = frame[y:y+h, x:x+w]
        return cropped_face
    

def load_deepfake_model(model_path):
    model = Model(2).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
