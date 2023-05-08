import cv2
from super_gradients.training import models
from super_gradients.common.object_names import Models
model = models.get('yolo_nas_s', num_classes=26, checkpoint_path = 'model_weights/ckpt_best.pth')

#output = model.predict_webcam()
models.convert_to_onnx(model=model, input_shape=(3,640,640), out_path='custom.onnx')