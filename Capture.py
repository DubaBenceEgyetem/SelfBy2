import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numpy import pi
from Threshholding import  s_threshhold, HLS_colorspace, Y_threshholding, combined, mag_thresh, GradDirThresh, CombinedSobelThresh, HLS_colorspace
from onlywhite import white_tresh
from birdEyeView import birdEyeView
image = cv2.VideoCapture("test/videos/example.mp4")
fps = image.get(cv2.CAP_PROP_FPS)

slow_down_factor = 2
frame_delay = 1 / (fps / slow_down_factor)
paused = False


while image.isOpened():
    try:
        if not paused:
                ret, frame = image.read()
                if frame is not None:
                    '''
                    H,L,S = HLS_colorspace(frame)
                    onlychannel = combined(frame)
                    cv2.imshow('combined', onlychannel * 255)
                    '''
                    white = white_tresh(frame)
                    cv2.imshow('white', white * 255)
                    biw = birdEyeView(frame)
                    cv2.imshow('biw', biw * 255)

        key = cv2.waitKey(int(frame_delay * 100)) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
    except Exception as e:
        print(f"hiba: {e}")



image.release()
cv2.destroyAllWindows()


