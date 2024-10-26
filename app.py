from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import time 

from imgFeed import detect_most_probable_disease
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/upload')
def upload_page():
    return render_template('upload.html') 

@app.route('/chat')
def chat_page():
    # Retrieve prediction from the query parameters
    prediction = request.args.get('pred', 'No prediction')
    return render_template('chat.html', prediction=prediction)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        model_path = './best.pt'
        model = YOLO(model_path)  # Load the model once
        image_path =  file_path
        prediction = detect_most_probable_disease(model, image_path)
        time.sleep(5)

        # Redirect to chat page with prediction as a query parameter
        return redirect(url_for('chat_page', pred=prediction))

if __name__ == '__main__':
    app.run(port=5501, debug=True)
