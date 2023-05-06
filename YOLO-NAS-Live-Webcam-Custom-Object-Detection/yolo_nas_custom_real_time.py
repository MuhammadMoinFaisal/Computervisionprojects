import cv2
import torch
from super_gradients.training import models
from super_gradients.common.object_names import  Models
img = cv2.imread("images/image.jpg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
model = models.get('yolo_nas_s', num_classes= 7, checkpoint_path='weights/ckpt_best.pth')

#model.predict_webcam()
#outputs = model.predict(img)
#outputs.show()

models.convert_to_onnx(model = model, input_shape = (3,640,640), out_path = "custom.onnx")
