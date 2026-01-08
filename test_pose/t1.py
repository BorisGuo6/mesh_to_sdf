import torch

# Load a pretrained YOLOv5 model from ultralytics
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load an image (replace with your image path)
img = 'test_image.jpg'

# Perform inference
results = model(img)

# Save the results to an image file instead of showing it
results.save(save_dir='yolov5_results')  # The results will be saved in the 'yolov5_results' folder

# Get the bounding boxes of the detected objects
# Each box is in the format (x1, y1, x2, y2, confidence, class_id)
bounding_boxes = results.xyxy[0].cpu().numpy()

# Loop through bounding boxes and calculate center points
for box in bounding_boxes:
    x1, y1, x2, y2, confidence, class_id = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    print(f'Object detected at ({center_x}, {center_y}) with confidence {confidence}')
