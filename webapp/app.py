# app.py
import torch 
from flask import Flask, request, jsonify, send_file, render_template, url_for
import io
from model import load_model, process_image, denormalize

app = Flask(__name__)
model, device = load_model()

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify(error="No file part"), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No selected file"), 400
    
    if file and allowed_file(file.filename):
        try:
            image_bytes = file.read()
            input_image = process_image(image_bytes).to(device)
            
            with torch.no_grad():
                output_image = model(input_image)
            
            output_image = denormalize(output_image.cpu())
            img_byte_arr = io.BytesIO()
            output_image.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            
            return send_file(img_byte_arr, mimetype='image/jpeg')
        
        except Exception as e:
            return jsonify(error=str(e)), 500
    
    return jsonify(error="Invalid file format"), 400

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
