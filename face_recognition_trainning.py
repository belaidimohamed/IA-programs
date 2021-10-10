import os , cv2
import numpy as np
from PIL import Image
os.chdir(r"C:\Users\user16\Desktop\IA programs")
recognizer=cv2.face.LBPHFaceRecognizer_create()
path="dataSet1"
def getpaths(path):
    l=[os.path.join(path,f) for f in os.listdir(path)] #list of images directories
    ids=[]
    faces=[]
    for i in l :
        faceimg=Image.open(i).convert('L')
        facenp=np.array(faceimg,'uint8')
        ID=int(os.path.split(i)[-1].split('.')[1])
        faces.append(facenp)
        ids.append(ID)
        cv2.imshow("trainning",facenp)
        cv2.waitKey(10)
    return ids,faces
ids, faces=getpaths(path)
print(ids,faces)
recognizer.train(faces,np.array(ids))
recognizer.save("recognizer/trainningdata.yml")
cv2.destroyAllWindows()