// static/script.js
document.getElementById('uploadForm').addEventListener('submit', async function(e) {
    e.preventDefault();

    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    const errorDiv = document.getElementById('error');
    const loadingDiv = document.getElementById('loading');
    const resultImg = document.getElementById('transformedImage');

    if (!file) {
        errorDiv.textContent = 'No file selected!';
        errorDiv.style.display = 'block';
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    errorDiv.style.display = 'none';
    loadingDiv.style.display = 'block';

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('File upload failed');
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);

        resultImg.src = url;
        resultImg.style.display = 'block';
    } catch (error) {
        errorDiv.textContent = error.message;
        errorDiv.style.display = 'block';
    } finally {
        loadingDiv.style.display = 'none';
    }
});
