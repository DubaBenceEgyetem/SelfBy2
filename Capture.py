import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numpy import pi
from Threshholding import  combined2, Y_threshholding, LUV_colorspace, LAB_colorspace,combined, tested,  mag_thresh, GradDirThresh, CombinedSobelThresh, HSV_colorspace, HLS_colorspace


image = cv2.VideoCapture("test/videos/example.mp4")
fps = image.get(cv2.CAP_PROP_FPS)

slow_down_factor = 2
frame_delay = 1 / (fps / slow_down_factor)
paused = False


while image.isOpened():
    try:
        if not paused:
                ret, frame = image.read()
                #sobel = mag_thresh(frame, sobel_kernel=0, thresh=(20, 100))
                c = combined2(frame)
                cv2.imshow('combined', c * 255)
        
        key = cv2.waitKey(int(frame_delay * 100)) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
    except Exception as e:
        print(f"hiba: {e}")



image.release()
cv2.destroyAllWindows()


