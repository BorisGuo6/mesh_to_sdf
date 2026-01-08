import torch
import cv2
import numpy as np

# Load a pretrained YOLOv5 model from ultralytics
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load an image using OpenCV (replace with your image path)
img_path = 'test_image.jpg'
img = cv2.imread(img_path)

# Get the image dimensions
image_height, image_width = img.shape[:2]

# Perform inference
results = model(img_path)

# Save the results to an image file instead of showing it
results.save(save_dir='yolov5_results')  # The results will be saved in the 'yolov5_results' folder

# Get the bounding boxes of the detected objects
# Each box is in the format (x1, y1, x2, y2, confidence, class_id)
bounding_boxes = results.xyxy[0].cpu().numpy()

# Loop through bounding boxes and calculate corner points and orientation
for box in bounding_boxes:
    x1, y1, x2, y2, confidence, class_id = box
    # Convert y1 and y2 based on new coordinate system (bottom-left is (0, 0))
    y1_new = image_height - y1
    y2_new = image_height - y2

    # Corners of the bounding box in the new coordinate system
    corners = np.array([[x1, y1_new], [x2, y1_new], [x2, y2_new], [x1, y2_new]])

    # Calculate center points
    center_x = (x1 + x2) / 2
    center_y = (y1_new + y2_new) / 2

    # Assume the object is axis-aligned; orientation would be based on the aspect ratio
    width = x2 - x1
    height = y1_new - y2_new
    if width >= height:
        orientation = 0  # Horizontal orientation
    else:
        orientation = 90  # Vertical orientation

    # Print corner points and orientation
    print(f'Object {class_id} detected:')
    print(f'  Corners: {corners}')
    print(f'  Center: ({center_x}, {center_y})')
    print(f'  Orientation: {orientation} degrees')
    print(f'  Confidence: {confidence}')
