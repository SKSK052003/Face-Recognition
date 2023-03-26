import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_faces.xml')

people = ['Sarath Kumar S K']

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')
capture = cv.VideoCapture(0)
while True:
    isTrue, frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #cv.imshow('Person', gray)
    faces_rect = haar_cascade.detectMultiScale(frame, 1.1, 4)

    for (x,y,w,h) in faces_rect:
        faces_roi = gray[y:y+h,x:x+w]

        label, confidence = face_recognizer.predict(faces_roi)
        print(f'Label = {people[label]} with a confidence of {confidence}')
        if confidence<55:
            cv.putText(frame, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
            cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)
        else:
            cv.putText(frame, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
            cv.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), thickness=2)

    cv.imshow('Detected Face', frame)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()
