import cv2
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from skimage.feature import local_binary_pattern
from skimage import io ,filters,feature,transform


#  Feature Extraction LBP
def LBP(greyscale):
    lbp = local_binary_pattern(greyscale, 8, 4, method='default')
    n_bins =256 
    imgHist = histogram(lbp, n_bins)
    
    return imgHist[0];
	
	
def DiskFractal(img, loops=25):
    arr = np.zeros((loops, 2))
    arr[1] = ([np.log(1), np.log(np.sum(255 - img) / 255) - np.log(1)])
    for x in range(2, loops):
        img_dilate = cv2.erode(img.copy(), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * x - 1, 2 * x - 1)),
                               iterations=1)
        arr[x] = ([np.log(x), np.log(np.sum(255 - img_dilate) / 255) - np.log(x)])

    error = 999
    slope = [0, 0, 0]
    loops = int(loops)
    for x in range(2, loops - 2):
        for y in range(x + 2, loops - 1):
            first = arr[1:x + 1, :]
            second = arr[x + 1:y + 1, :]
            third = arr[y + 1:loops, :]
            slope1, _, _, _, std_err1 = stats.linregress(x=first[:, 0], y=first[:, 1])
            slope2, _, _, _, std_err2 = stats.linregress(x=second[:, 0], y=second[:, 1])
            slope3, _, _, _, std_err3 = stats.linregress(x=third[:, 0], y=third[:, 1])

            if error > std_err1 + std_err2 + std_err3:
                error = std_err1 + std_err2 + std_err3
                slope = [slope1, slope2, slope3]

    return slope


def AnglesHistogram(image):
    values, count = np.unique(image, return_counts=True)
    countBlack = count[0]

    sob_img_v = np.multiply(filters.sobel_v(image), 255)
    sob_img_h = np.multiply(filters.sobel_h(image), 255)

    # Getting angles in radians
    angles = np.arctan2(sob_img_v, sob_img_h)
    angles = np.multiply(angles, (180 / math.pi))
    angles = np.round(angles)

    anglesHist = []
    angle1 = 10
    angle2 = 40

    while angle2 < 180:
        anglesCopy = angles.copy()
        anglesCopy[np.logical_or(anglesCopy < angle1, anglesCopy > angle2)] = 0
        anglesCopy[np.logical_and(anglesCopy >= angle1, anglesCopy <= angle2)] = 1
        anglesHist.append(np.sum(anglesCopy))
        angle1 += 30
        angle2 += 30

    return np.divide(anglesHist, countBlack)  


def get_slant_angle(img):
    edges = feature.canny(img, sigma=0.6)
    hspace, angles, distances=transform.hough_line(edges)
    accum, angles, dists = transform.hough_line_peaks(hspace, angles, distances)
    angle = np.rad2deg(np.median(angles))
    return angle    
    
    
def deskew(img):
    thresh=img
    edges = cv2.Canny(thresh,50,200,apertureSize = 3)

    lines = cv2.HoughLines(edges,1,np.pi/1000, 55)

    d1 = OrderedDict()
    if lines is None:
        return 0
    for i in range(len(lines)):
        for rho,theta in lines[i]:
            deg = np.rad2deg(theta)
            if deg in d1:
                d1[deg] += 1
            else:
                d1[deg] = 1
                   
    t1 = OrderedDict(sorted(d1.items(), key=lambda x:x[1] , reverse=False))
    angle =list(t1.keys())[0]
    
    return angle
    
def extract_features(greyscale1, binarized1, segmentsBinarized1, segmentsGrey1):
    features=[]
    features.extend(LBP(segmentsGrey1))
#     features.extend(AnglesHistogram(segmentsGrey1))
#     features.extend(DiskFractal(segmentsGrey1))
#     features.append(get_slant_angle(segmentsGrey1))
#     features.append(deskew(segmentsGrey1))
    
    return features       