from flask import Flask, request, render_template, send_from_directory
import cv2
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Ensure upload and processed directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)



# Load the YOLOv8 model
model = YOLO('D://web developement/project3/yolov8/best.pt')

def process_image_with_yolov8(image_path):
     # Load and resize the image
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (416, 361))

    # Save the resized image to a temporary path for YOLO processing
    resized_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'resized_' + os.path.basename(image_path))
    cv2.imwrite(resized_image_path, resized_image)
    # Run inference
    results = model([resized_image_path])
    
    # Process results
    for result in results:
        image = result.orig_img  # Original image from the result
        boxes = result.boxes  # Bounding box outputs
        class_names = result.names  # Class names
        
        # Draw bounding boxes and class names
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls)
            label = class_names[cls_id]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        processed_image_path = os.path.join(app.config['PROCESSED_FOLDER'], os.path.basename(image_path))
        cv2.imwrite(processed_image_path, image)
    
    return processed_image_path



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return "No file part"
    file = request.files['image']
    if file.filename == '':
        return "No selected file"
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        processed_image_path = process_image_with_yolov8(file_path)
        return render_template('index.html', image_url=f'/processed/{filename}')

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
