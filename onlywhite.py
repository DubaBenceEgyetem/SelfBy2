import numpy as np
import cv2

def binit(frame, thresh):
    outbin = np.zeros_like(frame)
    outbin[(frame >= thresh[0]) & (frame <= thresh[1])] = 1
    return outbin