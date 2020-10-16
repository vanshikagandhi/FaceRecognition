import cv2
import numpy as np
import os

cap=cv2.VideoCapture(0)
folder='faces'

facelist=os.listdir(folder)
facedata=[]
datalength=[]

for i in range(len(facelist)):
    face=np.load(folder+'/'+facelist[i])
    datalength.append(len(face))
    facedata.append(face)
    
facedata=np.asarray(facedata)

facedata=np.vstack(facedata)
facedata=facedata.reshape(facedata.shape[0],-1)

names={}
for i in range(len(facelist)):
    name=facelist[i].split('.')[0]
    names[i]=name
    
labels=np.zeros((facedata.shape[0],1))

x = 0
y = 0

datalength.insert(0, 0)

for j in range(len(datalength)-1):
    x += datalength[j]
    y += datalength[j + 1]
    labels[x:y] = float(j)
    
def distance(x1,x2):
    return np.sqrt(((x2-x1)**2).sum())

def knn(data,target,k=5):
    n=data.shape[0]
    d=[]
    for i in range(n):
        d.append(distance(data[i],target))
    d=np.asarray(d)
    indexes=np.argsort(d)
    sorted_labels=labels[indexes][:k]
    count=np.unique(sorted_labels,return_counts=True)
    max_index=np.argmax(count[1])
    result=count[0][max_index]
    return result

#os.chdir(current_dir)
dataset=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font=cv2.FONT_HERSHEY_COMPLEX

try:
    
    while True:
        flag,frame=cap.read()
        if flag:
            #gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces=dataset.detectMultiScale(frame)
            for x,y,w,h, in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),4)
                face=frame[y:y+h,x:x+w]
                face=cv2.resize(face,(50,50))
                print(face.shape)
                index=knn(facedata,face.flatten())
                name=names[int(index)]
                cv2.putText(frame,name,(x,y),font,1,(203,123,154),2)
                
            cv2.putText(frame,'Press ESC to close',(2,22),font,1,(132,213,190),1)
            cv2.imshow('Recognition System',frame)
            if cv2.waitKey(10) == 27:
                break
        '''
        else:
            print("Camera not working")
        '''
except BaseException as ex:
    print("Exception okay",ex)
finally:
    cap.release()
    cv2.destroyAllWindows()
                






