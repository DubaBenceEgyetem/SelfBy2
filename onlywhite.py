import numpy as np
import cv2

def binit(frame, thresh):
    outbin = np.zeros_like(frame)
    outbin[(frame >= thresh[0]) & (frame <= thresh[1])] = 1
    return outbin

def white_tresh(frame):
    lower = np.array([180,180,200], dtype="uint8")
    upper = np.array([255,255,255], dtype="uint8")
    mask = cv2.inRange(frame, lower, upper)
    rgbw = cv2.bitwise_and(frame, frame, mask=mask).astype(np.uint8)
    rgbw = cv2.cvtColor(rgbw, cv2.COLOR_RGB2GRAY)
    rgbw = binit(rgbw, thresh=(100,255))
    return rgbw 