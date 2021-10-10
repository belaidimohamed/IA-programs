import cv2
import numpy as np
from pynput.mouse import Button , Controller
import wx
mouse =Controller()
app=wx.App(False)
(sx,sy)=wx.GetDisplaySize()
(camx,camy)=(320,240)


cam=cv2.VideoCapture(0)
cam.set(3,camx)
cam.set(4,camy)
lowerBound=np.array([33,80,40])
upperBound=np.array([102,255,255])
kernelopen=np.ones((10,10))
kernelclose=np.ones((11,11))
mLocOld=np.array([0,0])
mouseLoc=np.array([0,0])
damping_factor=2
#mouseLoc=mLocOld+(targetloc-mLocOld)/damping_factor
flag=0
openx,openy,openw,openh=(0,0,0,0)
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

    if len(cont)==2 :
        if flag ==1:
            flag=0
            mouse.release(Button.left)
        x1,y1,w1,h1=cv2.boundingRect(cont[0])
        x2,y2,w2,h2=cv2.boundingRect(cont[1])
        cv2.rectangle(img,(x1,y1),(x1+w1,y1+h1),(255,0,0),2)
        cv2.rectangle(img,(x2,y2),(x2+w2,y2+h2),(255,0,0),2)
        cx1=round(x1+w1/2)
        cx2=round(x2+w2/2)
        cy1=round(y1+h1/2)
        cy2=round(y2+h2/2)
        cx=(cx1+cx2)//2
        cy=(cy1+cy2)//2
        cv2.line(img,(cx1,cy1),(cx2,cy2),(255,0,0),2)
        cv2.circle(img,(cx,cy),2,(0,0,255),2)
        mouseLoc=mLocOld+((cx,cy)-mLocOld)/damping_factor

        mouse.position=(sx-(mouseLoc[0]*sx/camx),mouseLoc[1]*sy/camy)
        while mouse.position==(sx-(mouseLoc[0]*sx/camx),mouseLoc[1]*sy/camy):
            pass
        mLocOld=mouseLoc
        openx,openy,openw,openh=cv2.boundingRect(np.array([[[x1,y1],[x1+w1,y1+h1],[x2,y2],[x2+w2,x2+h2]]]))
        #cv2.rectangle(img,(openx,openy),(openx+openw,openy+openh),(255,0,0),2)
    elif len(cont)==1:
        x,y,w,h=cv2.boundingRect(cont[0])
        if flag ==0 :
            if (abs((w*h-openw*openh)*100/w*h)<=20) :
                flag=1
                mouse.press(Button.left)
                openx,openy,openw,openh=(0,0,0,0)
        else :
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cx=x+w//2
            cy=y+h//2
            cv2.circle(img,(cx,cy),round((w+h)/4),(0,0,255),2)
            mouseLoc=mLocOld+((cx,cy)-mLocOld)/damping_factor
            mouse.position=(sx-(mouseLoc[0]*sx/camx),mouseLoc[1]*sy/camy)
            while mouse.position==(sx-(mouseLoc[0]*sx/camx),mouseLoc[1]*sy/camy):
                pass
            mLocOld=mouseLoc

    cv2.imshow('hello',img)

    if cv2.waitKey(1)==ord('q') :
        break
cam.release()
cv2.destroyAllWindows()
del app
print("mohammed bhim")

