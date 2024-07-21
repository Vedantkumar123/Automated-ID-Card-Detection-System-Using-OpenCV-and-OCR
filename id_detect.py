import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import matplotlib.pyplot as plt
import os

class ImageConstantROI():
    class CCCD(object):
        ROIS = {
            "id": [(200, 1040, 380, 70)],
            "name": [(20, 890, 750, 165)]
        }
        CHECK_ROI = [(313, 174, 597, 63)]

def display_img(cvImg):
    cvImg = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6,4))
    plt.imshow(cvImg)
    plt.show()  

def cropImageRoi(image, roi):
    roi_cropped = image[
        int(roi[1]) : int(roi[1] + roi[3]), int(roi[0]) : int(roi[0] + roi[2])
    ]
    return roi_cropped

def preprocessing_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.multiply(gray, 1.5)
    blured1 = cv2.medianBlur(gray,3)
    blured2 = cv2.medianBlur(gray,51)
    divided = np.ma.divide(blured1, blured2).data
    normed = np.uint8(255*divided/divided.max())
    th, threshed = cv2.threshold(normed, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    return threshed

def extractDataFromIdCard(img):
    MODEL_CONFIG = '-l eng --oem 1 --psm 6'
    l=[]
    for key, roi in ImageConstantROI.CCCD.ROIS.items():
        data = ''
        for r in roi:
            crop_img = cropImageRoi(img, r)
            crop_img = preprocessing_image(crop_img)
            # display_img(crop_img)
            data += pytesseract.image_to_string(crop_img, config = MODEL_CONFIG) + ' '
        print(f"{key} : {data.strip()}")
        l.append(data.strip())
    return (l)

def finalrun(baseImg,img2):
    baseH, baseW, baseC = baseImg.shape
    # display_img(img2)
    orb=cv2.ORB_create(1000)
    kp,des=orb.detectAndCompute(baseImg,None)
    imgkp=cv2.drawKeypoints(baseImg,kp,None)
    PER_MATCH=0.25
    kp1,des1=orb.detectAndCompute(img2,None)
    bf=cv2.BFMatcher(cv2.NORM_HAMMING)
    matches=list(bf.match(des1,des))
    matches.sort(key=lambda x:x.distance)
    best_matches = matches[:int(len(matches)*PER_MATCH)]
    imgMatch=cv2.drawMatches(img2,kp1,baseImg,kp,best_matches,None,flags=2)
    srcPoints=np.float32([kp1[m.queryIdx].pt for m in best_matches]).reshape(-1,1,2)
    dstPoints=np.float32([kp[m.trainIdx].pt for m in best_matches]).reshape(-1,1,2)
    matrix_relationship,_=cv2.findHomography(srcPoints,dstPoints,cv2.RANSAC,10.0)
    img_final=cv2.warpPerspective(img2,matrix_relationship,(baseW,baseH))
    l=extractDataFromIdCard(img_final)
    return l

# baseImg = cv2.imread("base_img.jpg")
# img2=cv2.imread("test_img.jpg")
# k=finalrun(baseImg,img2)
#print(k)
