import cv2
from ultralytics import YOLO

def detect_most_probable_disease(model, image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: Could not load image.")

    # Perform inference
    results = model(image)

    # Process results to find the most probable disease
    highest_confidence = 0
    most_probable_disease = None

    for result in results:
        scores = result.boxes.conf.numpy()  # Confidence scores
        labels = result.names  # Class labels
        
        # Find the disease with the highest confidence score
        for i, score in enumerate(scores):
            if score > highest_confidence:
                highest_confidence = score
                most_probable_disease = labels[int(result.boxes.cls[i])]

    return most_probable_disease

# Usage example:
model_path = './best.pt'
model = YOLO(model_path)  # Load the model once
image_path = r'./images.jpg'
most_probable_disease = detect_most_probable_disease(model, image_path)

print(f"Most probable disease: {most_probable_disease}")
