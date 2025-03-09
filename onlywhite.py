import numpy as np
import cv2
from Threshholding import X_threshholding




def binit(frame, thresh):
    outbin = np.zeros_like(frame)
    outbin[(frame >= thresh[0]) & (frame <= thresh[1])] = 1
    return outbin


def white_tresh(frame):
    hls = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)
    kernel = np.ones((1,1), np.uint8)
    lower = np.array([0,180,0], dtype="uint8")
    upper = np.array([255,255,255], dtype="uint8") 
    mask = cv2.inRange(hls, lower, upper)
    eroded = cv2.erode(mask, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    #rgbw = cv2.bitwise_and(hls,hls, mask=mask).astype(np.uint8)
    #rgbw = cv2.cvtColor(rgbw, cv2.COLOR_RGB2GRAY)
    #rgbw = binit(rgbw, thresh=(20,255))
    return mask
    
