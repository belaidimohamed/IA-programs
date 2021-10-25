import cv2
import numpy as np
cam=cv2.VideoCapture(0)
lowerBound=np.array([33,80,40])
upperBound=np.array([102,255,255])
kernelopen=np.ones((7,7))
kernelclose=np.ones((13,13))

while 1:
    ret,img=cam.read()
    img=cv2.resize(img,(340,220))
    #convert BGR to HSV
    imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #create mask
    mask=cv2.inRange(imgHSV,lowerBound,upperBound)
    #morphology (delete small irrelevent pixels)
    maskopen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelopen)
    maskclose=cv2.morphologyEx(maskopen,cv2.MORPH_CLOSE,kernelclose)
    finalmask=maskclose
    cont,h=cv2.findContours(finalmask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img,cont,-1,(0,0,255),3)
    for i in range(len(cont)):
        x,y,w,h=cv2.boundingRect(cont[i])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
        cv2.cv2.putText(img,str(i+1),(x,y+h),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
    cv2.imshow('1',img)
    cv2.imshow('2',maskclose)
    cv2.imshow('3',maskopen)
    cv2.imshow('4', mask)
    if cv2.waitKey(1)==ord('q') :
        break
cam.release()
cv2.destroyAllWindows()