from ultralytics import YOLO
import cv2

model=YOLO('../YOLO-Weights/yolov8n.pt')
results=model("../Images/2.png", show=True)

cv2.waitKey(0)