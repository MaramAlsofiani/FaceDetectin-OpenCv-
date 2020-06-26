import cv2

# Load the cascade(Should be in the project's directory)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#face
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')#eye

# Read the input image
img = cv2.imread('test1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# Convert into grayscale
faces = face_cascade.detectMultiScale(gray, 1.1, 3)

for (x, y, w, h) in faces:# Draw rec.
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 225), 2)
    roi_gray = gray[y:y + h, x:x + w]  # location of face
    roi_color = img[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 225, 0), 2)

cv2.imshow('img', img)# Display the output
cv2.waitKey()
