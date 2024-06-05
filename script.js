const form = document.getElementById('upload-form');
const fileInput = document.getElementById('file-input');
const imagePreview = document.getElementById('image-preview');
const resultDiv = document.getElementById('result');

form.addEventListener('submit', function(event) {
    event.preventDefault();
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('image', file);

    // Make API request to PPE detection model
    fetch('/best.pt', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Display the result
        resultDiv.textContent = data.result;
        resultDiv.style.display = 'block';
    })
    .catch(error => {
        console.error('Error:', error);
    });

    // Display the uploaded image
    const reader = new FileReader();
    reader.onload = function(e) {
        imagePreview.src = e.target.result;
    }
    reader.readAsDataURL(file);
});
