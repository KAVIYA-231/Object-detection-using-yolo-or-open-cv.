# Object-detection-using-yolo-or-open-cv.
This model is used to detect objects in real time from web cam
//Importing libraries

import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
model=torch.hub.load('ultralytics/yolov5','yolov5s')
model

//for loding image

img = 'C:\\Users\\kaviya\\Downloads\\los-angeles-1396606_960_720.jpg'
results = model(img)
results.print()
print(results)
print(type(results))
rendered_images=results.render()
print(type(rendered_images))
print(len(rendered_images))
if len(rendered_images)>0:
    print(type(rendered_images[0]))
plt.imshow(np.squeeze(results.render()))

//for real time object detection
cap=cv2.VideoCapture(0)
while cap.isOpened():
    ret,frame = cap.read()
    results = model(frame)
    
    cv2.imshow('YOLO',np.squeeze(results.render()))
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

    
