import cv2
import numpy as np
import imutils

import argparse
import re

parser = argparse.ArgumentParser(description='Syllable-based word segmentation for Myanmar language')
parser.add_argument('-i', '--input', type=str, help='input file', required=True)
parser.add_argument('-o', '--output', type=str, default='processed.jpg', help='output file', required=True)
parser.add_argument('-c', '--crop', type=str, default='on', help='one digit per one image use on or else off', required=True)
parser.add_argument('-p', '--path', type=str, default='./', help='path if you want to crop', required=False)

args = parser.parse_args()

inputFile = getattr(args, 'input')
outFile = getattr(args, 'output')
crop = getattr(args, 'crop') 
path = getattr(args, 'path') 

def bounding(img):
    image = cv2.imread(img)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    
    ret,out_binary=cv2.threshold(gray ,127,255,cv2.THRESH_BINARY_INV)

    img_dilation = cv2.dilate(out_binary, None, iterations=2)
    
    contours, h = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for i, ctr in enumerate(contours):

        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        M = cv2.moments(ctr)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        if crop == 'on':
            im = image[y:y+h, x:x+w]
            cv2.imwrite(path, im)

        cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
    return image


img = bounding(inFile)

cv2.imwrite(outFile,img)
