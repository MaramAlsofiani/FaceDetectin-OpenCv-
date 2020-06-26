import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # This file should be in your project's directory.
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') # This file should be in your project's directory.

cap = cv2.VideoCapture(0) #Load webcamera

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        #Draw rectangular
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0,255), 2) #(imge, starting point, ending point of the rec., color)
        roi_gray = gray[y:y + h, x:x + w] #location of face
        roi_color = img[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
