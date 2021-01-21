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
	
	

def get_slant_angle(img):
    edges = feature.canny(img, sigma=0.6)
    hspace, angles, distances=transform.hough_line(edges)
    accum, angles, dists = transform.hough_line_peaks(hspace, angles, distances)
    angle = np.rad2deg(np.median(angles))
    return angle    
    
    
def extract_features(greyscale1, binarized1, segmentsBinarized1, segmentsGrey1):
    features=[]
    features.extend(LBP(segmentsGrey1))
#     features.append(get_slant_angle(segmentsGrey1))

    
    return features       