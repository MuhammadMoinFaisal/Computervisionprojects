import cv2
import torch
from super_gradients.training import models
from super_gradients.common.object_names import  Models
img = cv2.imread("images/image.jpg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#Note that currently YOLOX and PPYOLOE are supported

model = models.get(Models.YOLO_NAS_S, pretrained_weights='coco')

model = model.to("cuda" if torch.cuda.is_available() else 'cpu')

#model.predict_webcam()
#outputs = model.predict(img)
#outputs.show()

models.convert_to_onnx(model = model, input_shape = (3,640,640), out_path = "yolo_nas_s.onnx")


