import cv2
import numpy as np
from ultralytics import YOLO
import serial
import time
from openport import PortOpen

model = YOLO(r"yolov8n.pt")
initialize = PortOpen()
cap = cv2.VideoCapture(0)
ws, hs = 1920, 1080
servoPos = [90, 90]
cap.set(3, ws)
cap.set(4, hs)

while True:
    ret, frame = cap.read()
    results = model.predict(frame,verbose=False)
    img = frame.copy()
    for box in results[0].boxes:
        x1, y1, x2, y2, = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        cls = box.cls[0].item()
        label = results[0].names[int(cls)]
        if label == "person":
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            fx,fy = x1,y1

            servoX = np.interp(fx, [0, ws], [10, 160])
            servoY = np.interp(fy, [0, hs], [20, 160])
            # print(servoX,end="\r")

            if servoX < 0:
                servoX = 0
            elif servoX > 180:
                servoX = 180
            if servoY < 0:
                servoY = 0
            elif servoY > 180:
                servoY = 180
            
            servoPos[0] = servoX
            servoPos[1] = servoY

            initialize.move_servo(180 - int(servoPos[0]))
            print(int(servoPos[0]), end="\r")
            # print(servoX, end='\r')

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(img, f"{results[0].names[int(cls)]}: Confidence - {conf*100:.2f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow("frame", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()