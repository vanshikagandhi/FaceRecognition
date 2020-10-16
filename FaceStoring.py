import numpy as np
import cv2

dataset=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

facedata=[]
cap=cv2.VideoCapture(0)
while True:
    response,frame=cap.read()
    
    if response:
        faces=dataset.detectMultiScale(frame)
        for x,y,w,h in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),4)
            face=frame[x:x+w,y:y+h,:]
            face=cv2.resize(face,(50,50))
            if len(facedata)<50:
                facedata.append(face)
                print(len(facedata))
        frame=cv2.resize(frame,None,fx=1.2,fy=1.2)
        cv2.imshow('result',frame)
        if cv2.waitKey(10) == ord('q') or len(facedata) >= 50:
            break
    else:
        print("Camera not working")
        break
        
cap.release()
cv2.destroyAllWindows()

facedata=np.asarray(facedata)
np.save('User2.npy',facedata)



