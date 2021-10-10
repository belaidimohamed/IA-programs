import cv2 ,os
import numpy as np
os.chdir(r"C:\Users\user16\Desktop\IA programs")
face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0)
#face_recognition
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read(r"recognizer\trainningdata.yml")

id=0
while 1 :
    id=0
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face.detectMultiScale(gray,1.3,5)
    for (x,y,z,h) in faces :
        cv2.rectangle(img,(x,y),(x+z,y+h),(0,150,0),2)
        id,conf=rec.predict(gray[y:y+h,x:x+z])
        if id==1 :
            id="Mohamed"
        elif id==3:
            id="helmi"
        elif id==2 :
            id="hamza"
        elif id==15:
            id="wael"
        elif id==4:
            id="nizar"
        elif id==6:
            id="ampoula"
        elif id==7:
            id="ibrahim"
        elif id==11:
            id='eya'
        elif id==12:
            id="ameni"
        elif id==16:
            id="yesmine"
        cv2.putText(img,str(id),(y,y+h),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

    cv2.imshow("face",img)
    if cv2.waitKey(1)==ord('p') :
        break
cam.release()
cv2.destroyAllWindows()