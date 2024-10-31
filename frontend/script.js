function displayVideo() {
    const videoFile = document.getElementById('videoInput').files[0];
    const uploadedVideo = document.getElementById('uploadedVideo');
    const videoContainer = document.getElementById('videoContainer');

    if (!videoFile) {
        alert('Please select a video.');
        return;
    }

    // Display the uploaded video with controls
    uploadedVideo.src = URL.createObjectURL(videoFile);
    videoContainer.classList.remove('d-none');
}

function analyzeVideo() {
    const videoFile = document.getElementById('videoInput').files[0];
    const resultMessage = document.getElementById('result');

    if (!videoFile) {
        alert('Please select a video.');
        return;
    }

    // Display analysis in progress message
    resultMessage.textContent = "Analysis in progress...";

    // Create form data and append the video file
    const formData = new FormData();
    formData.append('video', videoFile);

    // Send POST request to backend API
    fetch('http://localhost:5000/detect_deepfake', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Update result message with analysis result from backend
        resultMessage.textContent = `Result: ${data.result}, Confidence: ${data.confidence}%`;
    })
    .catch(error => {
        console.error('Error:', error);
        resultMessage.textContent = 'An error occurred during analysis.';
    });
}
