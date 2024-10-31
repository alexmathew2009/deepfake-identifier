
# Deepfake Detection System

## Overview

The **Deepfake Detection System** is a machine learning application that utilizes computer vision techniques to analyze videos and detect manipulated content. This project aims to provide users with a reliable tool for identifying deepfake media, addressing the growing concerns about misinformation and the authenticity of digital content.

## Features

- Upload and analyze video files to detect deepfakes.
- User-friendly web interface built with HTML, CSS, and JavaScript.
- Utilizes a pre-trained deep learning model for accurate predictions.

## Technologies Used

- **Python**: Main programming language for backend development.
- **Flask**: Web framework for building the application.
- **PyTorch**: Deep learning framework for model development and predictions.
- **OpenCV**: Library for video processing and face detection.
- **HTML/CSS**: Frontend technologies for user interface.
- **JavaScript**: Client-side scripting for interactivity.
- **Bootstrap**: Responsive design framework for frontend styling.


##Project Structure
deepfake-identifier/
│
├── app.py                   # Main Flask application file
├── deepfake_detection_model.py  # Model architecture and prediction logic
├── requirements.txt         # List of project dependencies
│
├── templates/
│   ├── index.html           # Landing page for the application
│   └── detect.html          # Video analysis interface
│
├── static/
│   ├── styles.css           # Styles for the main application
│   └── stylesdetect.css      # Styles for detection page
│   └── script.js            # JavaScript for client-side interactivity
│
└── uploads/                 # Directory to store uploaded video files


## Installation

To set up the project locally, follow these steps:

1. **Clone the repository**:
   git clone https://github.com/alexmathew2009/deepfake-identifier.git

3. **Navigate to the project directory**:
      cd deepfake-identifier
   
3. **Create a virtual environment (optional but recommended)**:
      python -m venv venv
      source venv/bin/activate  # For Windows, use `venv\Scripts\activate`

4. **Install required dependencies:**:
    ```bash
    pip install -r requirements.txt


## Usage

After setting up the project and running the application, you can follow these steps to analyze a video for deepfake content:

### Step 1: Run the Application
      ```bash
          python app.py

### Step 2:Open your web browser and navigate to http://127.0.0.1:5000.

### Step 3:Upload a video: Select a video file to analyze for deepfake content.

### Step 4:Analyze the video: Click the "Analyze" button to perform deepfake detection. The result will be displayed on the interface.



