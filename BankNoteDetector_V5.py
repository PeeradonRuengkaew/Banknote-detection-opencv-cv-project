import cv2
import numpy as np
import os
running = True
path = 'Image_Temp'
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
orb = cv2.ORB_create(nfeatures= 1250)
# Import Images
images = []                                                                               
className = []
myList = os.listdir(path)
print("Total Class",len(myList))

for cl in myList:
    imgCur = cv2.imread(f'{path}/{cl}',0)
    images.append(imgCur)
    className.append(os.path.splitext(cl)[0])
print(className)

def findDes(images):
    desList = []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        desList.append(des)
    return desList

def findID(img, desList, thres = 14):
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher()
    #loop descriptor 1 by 1
    matchList = []
    finalVal = -1
    #incase didnt match anything 
    try:
        for des in desList:
            matches = bf.knnMatch(des, des2, k = 2)
            good = []
            for m, n in matches:
                # Confidence
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            matchList.append(len(good))
    except:
        pass
    #print(matchList)
    if len(matchList) != 0:
        if max(matchList) > thres:
            #find max value and index
            finalVal =  matchList.index(max(matchList))
    return finalVal

desList = findDes(images)
print(len(desList))

while running:
    _, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width, _ = frame.shape
    imgOriginal = frame.copy()
    cx = int(width / 2)
    cy = int(height / 2)
    # Pick pixel value
    pixel_center = hsv_frame[cy, cx]
    hue_value = pixel_center[0]
    pixel_center_bgr = imgOriginal[cy, cx]
    b, g, r = int(pixel_center_bgr[0]), int(pixel_center_bgr[1]), int(pixel_center_bgr[2])
    cv2.circle(imgOriginal, (cx, cy), 5, (25, 25, 25), 2)
    cv2.circle(frame, (cx, cy), 5, (25, 25, 25), 2)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    id = findID(frame,desList)

    color = "Undefined"
    if hue_value < 5:
        color = "100"
    elif hue_value < 18:
        color = "1000"
    elif hue_value < 35:
        color = "20"
    elif hue_value < 95:
        color = "50"
    elif hue_value < 130:
        color = "500"
    elif hue_value < 169:
        color = "100"
    else:
        color = "500"
    cv2.putText(frame, color, (cx - 200, 100), 0, 3, (b, g, r), 5)
    # IF Colordetector detect 20
    

    
    # if color = 20 or 50 baht so id can't be 100 baht and 500
    if color == "20" or color == "50":
        if id == 2 or id == 3 or id == 7 == id == 8:
            id = -1
    if color == "100" or color == "500":
        if id == 4 or id == 5 or id == 6 or id == 9 or id == 10:
            id = -1
    
#If equal nothing dont print name
    if id != -1:
        cv2.putText(imgOriginal, className[id], (50,50), cv2.FONT_HERSHEY_DUPLEX, 1,(0,0,255), 2)
    # Color camera
    cv2.imshow('color', frame)
    cv2.imshow('knn', imgOriginal)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()