document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault();
    
    var formData = new FormData();
    var fileInput = document.getElementById('imageUpload');
    formData.append('file', fileInput.files[0]);
    
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.blob())
    .then(blob => {
        var url = window.URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = 'mask.png';
        document.body.appendChild(a);
        a.click();
        a.remove();
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('result').textContent = 'Error processing the image.';
    });
});
