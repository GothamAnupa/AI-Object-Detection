"""Re-create model/yolov8n.onnx from the PyTorch weights.  Usage: pip install ultralytics onnx && python export_model.py"""
from ultralytics import YOLO
import shutil
path = YOLO("yolov8n.pt").export(format="onnx", imgsz=640, opset=12, simplify=False, dynamic=False)
shutil.move(path, "model/yolov8n.onnx")
print("saved model/yolov8n.onnx")
