import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time


mixer.init()
sound = mixer.Sound(r"D:\GIT\Drowsiness-Research\beep-02.wav")

face = cv2.CascadeClassifier('D:\GIT\Drowsiness-Research\Cascade Files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('D:\GIT\Drowsiness-Research\Cascade Files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('D:\GIT\Drowsiness-Research\Cascade Files\haarcascade_righteye_2splits.xml')
mth = cv2.CascadeClassifier('D:\GIT\Drowsiness-Research\Cascade Files\haarcascade_mouth.xml')

lbl=['Close','Open']

model = load_model('D:\GIT\Drowsiness-Research\Eyes Open.h5')
model1 = load_model('D:\GIT\Drowsiness-Research\Yawning.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 35)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]

while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # cv2.imshow('frame',gray)
    # cv2.waitKey(0)

    faces = face.detectMultiScale(gray, 1.3, 5)
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)
    mouth = mth.detectMultiScale(gray)

    cv2.rectangle(frame, (0,height-50) , (280,height) , (0,0,0) , thickness=cv2.FILLED )
    cv2.rectangle(frame, (10,20) , (180,70) , (0,0,0) , thickness=cv2.FILLED )
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (225,0,0) , 1 )

    for (x,y,w,h) in left_eye:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,225,0) , 1 )

    for (x,y,w,h) in right_eye:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,225,0) , 1 )

    for (x,y,w,h) in mouth:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,0, 255) , 1 )

    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        # r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = model.predict(r_eye)
        rpred = np.argmax(rpred)
        if(rpred==1):
            lbl='Open' 
        if(rpred==0):
            lbl='Closed'
        break

    for (x,y,w,h) in mouth:
        Yawn=frame[y:y+h,x:x+w]
        count=count+1
        # r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        Yawn = cv2.resize(Yawn,(24,24))
        Yawn= Yawn/255
        Yawn=  Yawn.reshape(24,24,-1)
        Yawn = np.expand_dims(Yawn,axis=0)
        yawnpred = model1.predict(Yawn)
        yawnpred = np.argmax(yawnpred)
        if(yawnpred==1):
            lbl='Not_Yawning' 
        if(yawnpred==0):
            lbl='Yawning'
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        # l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = model.predict(l_eye)
        lpred = np.argmax(lpred)
        if(lpred==1):
            lbl='Open'   
        if(lpred==0):
            lbl='Closed'
        break

    if((rpred==0 and lpred==0) or yawnpred==0):
        score=score+1
        cv2.putText(frame,"Closed",(20,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(frame,"Not_Yawning",(20,50), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        score=score-1
        cv2.putText(frame,"Open",(20,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(frame,"Yawning",(20,50), font, 1,(255,255,255),1,cv2.LINE_AA)
    
        
    if(score<0):
        score=0   
    cv2.putText(frame,'Drowsiness:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score>25):
        #person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            sound.play()
            
        except:  # isplaying = False
            pass
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
