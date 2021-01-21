import glob
import cv2
import numpy as np
import math
import glob
import csv
import os
import time
from sklearn.svm import SVC

from PreprocessingAndSegmentation import *
from ExtractingFeatures import *
from ModelsAndTraining import *
from PerformanceAnalysis import *


def main(directory,true_values,l):
    
    # Read test image
    test_images = sorted(glob.glob(directory+'/*.png'))
    img_original = cv2.imread(test_images[0],0)
    
    # Get training features and labels
    training_data,trainTime = train(directory)
    
    start = time.time()
    
    training_data = np.asarray(training_data)
    training_features = training_data[:,1:]
    labels = training_data[:,0]
    
    
    #Train SVM
    clf= SVM(training_features,labels)  
    
    # Preprocess test image
    greyscale1, binarized1, segmentsBinarized1, segmentsGrey1 = Preprocess(img_original)
    
    maxOccurSegmentSVM=[]
    
    for j in range(segmentsGrey1.shape[0]):
        
        # Extract Features
        test_point = extract_features(greyscale1, binarized1, segmentsBinarized1, segmentsGrey1[j])
        
        #Get prediction of each segment for the SVM Model
        maxOccurSegmentSVM.append(int(clf.predict([test_point])[0]))  #SVM
  
    maxOccurSegmentSVM=np.asarray(maxOccurSegmentSVM)
    
    # Append the prediction
    SVM_prediction = np.bincount(maxOccurSegmentSVM).argmax()

    end = time.time()
    predictionTime= end-start
    totalTime = trainTime+ predictionTime

    # Print time and result
    
    f = open("time.txt", "a")  
    f.write(str(round(totalTime,2)))
    f.write("\n")
    f.close()
    
    f2= open("results.txt","a")
    f2.write(str(SVM_prediction))
    f2.write("\n")
    f2.close()

    #print("SVM_prediction class :", SVM_prediction)
    
    return SVM_prediction, totalTime    
    
def predict():
    direcs = natsorted(glob.glob ("data/*"))
    true_values = []
    SVM_prediction=[]
    totalTime=[]

    for l in range(len(direcs)):
        SVM_predict, timeTest  = main(direcs[l],true_values,l)
        SVM_prediction.append(SVM_predict)
        totalTime.append(timeTest)

    return true_values, SVM_prediction  


f = open("time.txt", "w") 
f.close()
f2= open("results.txt","w")
f2.close()

true_values, SVM_prediction = predict()
#calculate_accuracy(true_values, SVM_prediction)