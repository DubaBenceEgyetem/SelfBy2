import cv2
import numpy as np
from birdEyeView import birdEyeView


def sobel_thresh(frame, orient='x', thresh=(0,255)):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    if orient == 'x':
        dx, dy = 1, 0 
    else:
        dx, dy = 0, 1

    sobel = cv2.Sobel(gray, cv2.CV_64F, dx, dy)
    
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sxbinary


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    mag_sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
    
    scaled_sobel = np.uint8(255*mag_sobel/np.max(mag_sobel))
    
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sxbinary 



def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    
    binary_output =  np.zeros_like(grad_dir)
    binary_output[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    return binary_output


def X_threshholding(frame):
    blured = cv2.GaussianBlur(frame, (5,5), 0)
    gradX = sobel_thresh(blured, 'x', (50,255))
    return gradX
def Y_threshholding(frame):
    blured = cv2.GaussianBlur(frame, (5,5), 0)
    gradY = sobel_thresh(blured, 'y', (30,250))
    return gradY
def GradMag(frame):
    blured = cv2.GaussianBlur(frame, (5,5), 0)
    gradmagthresh = mag_thresh(blured, 3, (20,255))
    return gradmagthresh
def GradDirThresh(frame):
    blured = cv2.GaussianBlur(frame, (5,5), 0)
    a = 0.95
    b = .1
    graddirthresh = dir_thresh(blured, 3, (a-b, a+b))
    return graddirthresh


def CombinedSobelThresh(frame):
    blurred = cv2.GaussianBlur(frame, (5,5), 0)
    grad_x_thresh = sobel_thresh(blurred, 'x', (20, 255))
    grad_y_thresh = sobel_thresh(blurred, 'y', (40, 150))
    grad_mag_thresh = mag_thresh(blurred, 3, (20, 255))
    grad_dir_thresh = dir_thresh(blurred, 3, (0.85, 1.05))
    combined = np.zeros_like(grad_x_thresh)
    combination = ((grad_mag_thresh == 1) & (grad_dir_thresh == 1) | (grad_x_thresh ==1 ) | (grad_y_thresh == 1))
    combined[combination]=1
    return combined


##color thresh holding

def HLS_colorspace(frame):
    blurred = cv2.GaussianBlur(frame, (3,3), 0)
    hls = cv2.cvtColor(blurred, cv2.COLOR_RGB2HLS)
    H = hls[:,:,0]
    L = hls[:,:,1] #eselyes
    S = hls[:,:,2]
    return (H,L,S)



def r_threshhold(frame, thresh):
    R = frame[:,:,0]

    binout = np.zeros_like(R)
    binout[(R >= thresh[0]) & (R <= thresh[1])] = 1
    return np.copy(binout);

def b_threshhold(frame, thresh):
    B = frame[:,:,2]

    binout = np.zeros_like(B)
    binout[(B >= thresh[0]) & (B <= thresh[1])] = 1
    return np.copy(binout);


def l_threshhold(frame, thresh):
    hls = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS) 
    l = hls[:,:,1]

    binout = np.zeros_like(l)
    binout[(l >= thresh[0]) & (l <= thresh[1])] = 1
    return np.copy(binout);



def s_threshhold(frame, thresh):
    hls = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS) 
    S = hls[:,:,2]

    binout = np.zeros_like(S)
    binout[(S >= thresh[0]) & (S <= thresh[1])] = 1
    return np.copy(binout);



def combined(frame):
    blured = cv2.GaussianBlur(frame, (5,5),0)

    r_tresh = r_threshhold(blured, (200, 255))
    b_tresh = b_threshhold(blured, (200, 255))
    s_tresh = s_threshhold(blured, (200, 255))
    l_tresh = l_threshhold(blured, (200,255))
    combination = ((s_tresh == 1) | (l_tresh == 1) | ((r_tresh == 1) & (b_tresh ==1 )))
    color_treshold = (combination)
    combined = np.zeros_like(s_tresh)
    combined[color_treshold] = 1
    xsize = frame.shape[1]
    ysize = frame.shape[0]
    verticles = np.array([[(0, ysize), (xsize / 2 -1, 380), (xsize/2 +1, 380), (xsize, ysize)]], dtype=np.int32)
    
    masked_img = region_of_interests(combined, verticles)
    return masked_img
 

def region_of_interests(frame, vertices):
    mask = np.zeros_like(frame)
    if len(frame.shape) > 2:
        channel = frame.shape[2]
        imaskcollor = (255,) * channel
    else:
        imaskcollor = 255
    cv2.fillPoly(mask, vertices, imaskcollor)
    mask_img = cv2.bitwise_and(frame, mask)
    return mask_img


