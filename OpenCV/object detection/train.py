import cv2
import numpy as np
detector=cv2.ORB_create()
flannparam=dict(algorithm=0,tree=5)
flan=cv2.FlannBasedMatcher(flannparam,{})
trainImg=cv2.imread(r'C:\Users\user16\Desktop\IA programs\object detctition\training_images\bounya.png',0)
trainkp,traindesc =detector.detectAndCompute(trainImg,None)
cam=cv2.VideoCapture(0)
MAX_FEATURES=30
GOOD_MATCH_PERCENT=0.75
def alignImages(im1, im2):

  # Convert images to grayscale
  im1Gray = im1
  im2Gray = im2

  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create()
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)

  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)

  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]

  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  cv2.imwrite("matches.jpg", imMatches)

  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)

  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

  # Use homography
  height, width = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))

  return h
while 1 :
    ret,QueryImgRGB=cam.read()
    QueryImg=cv2.cvtColor(QueryImgRGB,cv2.COLOR_BGR2GRAY)
    queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)
    if(len(queryKP)>=2) and (len(trainkp)>=2) :
        matches=flan.knnMatch(np.asarray(queryDesc,np.float32),np.asarray(traindesc,np.float32), 2) #2
        goodMatch=[]
        for m,n  in matches : # n  train matches , m query matches
            if (m.distance<GOOD_MATCH_PERCENT*n.distance):
                goodMatch.append(m)
        if (len(goodMatch)>MAX_FEATURES) :
            tp=[]
            qp=[]
            for i in goodMatch :
                tp.append(trainkp[m.trainIdx].pt)
                qp.append(queryKP[m.queryIdx].pt)
            tp,qp=np.float32((tp,qp))
            H=alignImages(trainImg,QueryImg)

            h,w = trainImg.shape
            trainborder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
            queryborder=cv2.perspectiveTransform(trainborder,H)
            cv2.polylines(QueryImgRGB,[np.int32(queryborder)],True,(0,255,0),5)

        else :
            print('not enough {}/30 '.format(len(goodMatch)))
    cv2.imshow('result',QueryImgRGB)
    if cv2.waitKey(1)==ord('q') :
        break
cam.release()
cv2.destroyAllWindows()
