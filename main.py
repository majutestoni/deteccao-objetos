import cv2
import numpy as np
from ultralytics import YOLO
from matplotlib import pyplot as plt

# modelo escolhido (ele já é treinado)
model = YOLO('yolov8n.pt') 

# imagem original
img  = "img/000002.jpg"
result_predict = model.predict(source = img, imgsz=(640))

# apresetando resultado
plot = result_predict[0].plot()
plot_rgb = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)
cv2.imshow("Resultado da Detecção", plot_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
