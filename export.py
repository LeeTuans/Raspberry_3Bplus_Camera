from ultralytics import YOLO

model = YOLO("./models/yolov8n-pose.pt")  # Thay yolov8n.pt bằng mô hình của bạn

model.export(format="tflite")