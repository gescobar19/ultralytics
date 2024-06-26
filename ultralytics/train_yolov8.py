from ultralytics import YOLO
# Initialize the YOLO model
model = YOLO('yolov8n.yaml')  # You can choose a different model configuration (e.g., yolov8n.yaml, yolov8s.yaml)

# Train the model
model.train(data=r'C:\Users\EEsco\OneDrive\Documents\GitHub\ultralytics\ultralytics\cfg\datasets\StrawberryFixedCam4.v2i.yolov8\data.yaml', epochs=50, imgsz=640)

