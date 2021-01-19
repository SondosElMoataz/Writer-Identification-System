import glob
import cv2
import numpy as np
import math
import glob
import csv
import os
import time

from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from natsort import natsorted


def calculate_accuracy(true_values, SVM_prediction):
    
    SVM_prediction= np.asarray(SVM_prediction)
    true_values= np.asarray(true_values)
 
    results = np.array([SVM_prediction == true_values])
    
    accuracy = results[results==True].shape[0]/true_values.shape[0]
    print('Accuracy = ' + str(round(accuracy,4) * 100) + '%')
    
    return  
