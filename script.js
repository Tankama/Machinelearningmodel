// Handle the form submission to upload an image and get the prediction
document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault();
    
    var formData = new FormData();
    var fileInput = document.getElementById('imageUpload');
    formData.append('file', fileInput.files[0]);
    
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())  // Handle JSON response
    .then(data => {
        if (data.error) {
            throw new Error(data.error);  // Throw error if present in response
        }
        // Display the prediction result
        document.getElementById('stroke-type').textContent = data.stroke_type;
        
        // Convert base64 to image and display it
        var maskImage = document.getElementById('mask-image');
        maskImage.src = 'data:image/png;base64,' + data.mask_image;
        maskImage.style.display = 'block';  // Ensure image is visible
        
        // Show result div
        document.getElementById('result').classList.remove('hidden');
        document.getElementById('error').classList.add('hidden');
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('error-message').textContent = error.message;
        document.getElementById('error').classList.remove('hidden');
        document.getElementById('result').classList.add('hidden');
    });
});

// Handle chatbot interaction
function askQuestion() {
    var question = document.getElementById('questionInput').value;

    fetch('/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ question: question })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            throw new Error(data.error);  // Throw error if present in response
        }
        document.getElementById('chatbotResponse').textContent = data.answer;
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('chatbotResponse').textContent = 'Error asking the question.';
    });
}