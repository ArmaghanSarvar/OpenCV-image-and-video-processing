import cv2
import imutils as imutils
import numpy as np
import os

myImage = cv2.imread("test.jpg")
cv2.imshow("Original Image", myImage)
cv2.waitKey()
cv2.destroyAllWindows()

gray_image = cv2.cvtColor(myImage, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray image", gray_image)
cv2.waitKey()
cv2.destroyAllWindows()

#########################################

b = myImage.copy()
# set green and red channels to 0
b[:, :, 1] = 0
b[:, :, 2] = 0
cv2.imshow('Blue-RGB', b)
cv2.waitKey()
cv2.destroyAllWindows()

#########################################

blurred = cv2.GaussianBlur(myImage, (5, 5), 0)
cv2.imshow("Smoothed image" , blurred)
cv2.waitKey()
cv2.destroyAllWindows()

#########################################

angle = 90
rotated = imutils.rotate_bound(myImage, angle)
cv2.imshow("Rotated Image", rotated)
cv2.waitKey()
cv2.destroyAllWindows()

#########################################

resized = cv2.resize(myImage, (0,0), fx=0.5 , fy=1)
cv2.imshow("Resized Image" , resized)
cv2.waitKey()
cv2.destroyAllWindows()

#########################################
# canny edge detector
edges = cv2.Canny(myImage , 100 , 200)
cv2.imshow("Edge Detection" , edges)
cv2.waitKey()
cv2.destroyAllWindows()

#########################################

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
numOfClusters = 100
ret, label, centers = cv2.kmeans(np.float32(myImage), numOfClusters, None, criteria, 50, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(centers)
shift = cv2.pyrMeanShiftFiltering(myImage, sp=8, sr=16, maxLevel=1, termcrit=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1))
gray = cv2.cvtColor(myImage, cv2.COLOR_BGR2GRAY)
arg1, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
fg = cv2.erode(thresh, None, iterations=1)
bgt = cv2.dilate(thresh, None, iterations=1)
arg1, bg = cv2.threshold(bgt, 1, 128, 1)
marker = cv2.add(fg, bg)
canny = cv2.Canny(marker, 110, 150)
contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
marker32 = np.int32(marker)
cv2.watershed(myImage, marker32)
arg1, thresh = cv2.threshold(cv2.convertScaleAbs(marker32), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
thresh_inv = cv2.bitwise_not(thresh)
firstres = cv2.bitwise_and(myImage, myImage, mask=thresh)
secondres = cv2.bitwise_and(myImage, myImage, mask=thresh_inv)
thirdres = cv2.addWeighted(firstres, 1, secondres, 1, 0)
out = cv2.drawContours(thirdres, contours, -1, (0, 255, 0), 1)
cv2.imshow("Segmentation", out)
cv2.waitKey()
cv2.destroyAllWindows()

#########################################

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(myImage, cv2.COLOR_BGR2GRAY)
face_detect = face_cascade.detectMultiScale(gray, scaleFactor=1.1 , minNeighbors=5)
print('num of faces found :', len(face_detect))
for(x,y,w,h) in face_detect:
    cv2.rectangle(myImage, (x,y) , (x+w , y+h), (0,0,255) , 2)

cv2.imshow("Face Detection", myImage)
cv2.waitKey()
cv2.destroyAllWindows()

#########################################

cap = cv2.VideoCapture("test.avi")
frames = []
for i in range(5):
    success, img = cap.read()
    if success:
        frames.append(img)

for i in range(5):
    cv2.imshow('frame {0}'.format(i + 1), frames[i])
    cv2.waitKey(500)
    pathOut = 'videoFrames'
    cv2.imwrite(os.path.join(pathOut, "frame{:d}.jpg".format(i + 1)), frames[i])
    cv2.destroyAllWindows()
