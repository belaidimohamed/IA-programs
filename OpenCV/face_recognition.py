import cv2 ,os
import numpy as np
os.chdir(r"C:\Users\user16\Desktop\IA programs")
face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0)
id=input("input id : ")
num=0
while 1 :
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face.detectMultiScale(gray,1.3,5)
    for (x,y,z,h) in faces :
        print(ret,num)

        if ret==True :
            num+=1
            cv2.imwrite("dataSet1/User."+id+"."+str(num)+".jpg",gray[y:y+h,x:x+z])
            cv2.rectangle(img,(x,y),(x+z,y+h),(0,150,0),2)
            cv2.waitKey(50)
    cv2.imshow("face",img)
    if num>60:
        break
cam.release()
cv2.destroyAllWindows()