import numpy as np
import cv2
from onlywhite import white_tresh
def birdEyeView(frame):
    size = (frame.shape[1], frame.shape[0])
    src  = np.float32([[545,480], [835, 480], [310, 640], [990,640]])
    dst = np.float32([[310,350], [1075, 350], [310, 640], [1075, 640]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, M, size)
    blurred = cv2.GaussianBlur(warped, (3,3), 0)
    biw = white_tresh(blurred)
    return biw 