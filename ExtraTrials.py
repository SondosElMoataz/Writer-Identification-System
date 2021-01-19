import cv2
import numpy as np
import math
import glob
import csv
import os
import time
from collections import OrderedDict
import matplotlib.pyplot as plt
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from skimage.feature import local_binary_pattern
from skimage import io ,filters,feature,transform

from scipy import stats

from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D

from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics

from PreprocessingAndSegmentation import *
from ExtractingFeatures import *
from ModelsAndTraining import *


def main_all_models(directory,true_values,l):
    
    # Get training features and labels
    training_data,trainTime = train(directory)
    training_data = np.asarray(training_data)

    training_features = training_data[:,1:]
    labels = training_data[:,0]
    
    test_images = sorted(glob.glob(directory+'/*.png'))
    
    f3 = open(directory+'/ids.txt')
    ids = f3.readlines()
    ids=np.asarray(ids,dtype=int)
    xtest=ids[-1]
    ytest=np.where(ids==xtest)[0][0]+1 
    true_values.append(ytest)
    f3.close()
    
    k = 5

    f = open("time.txt", "a")
    f2= open("results.txt","a")
    
    clf= SVM(training_features,labels)  #SVM
    
    ada = AdaBoostClassifier(n_estimators=50, learning_rate=1)    #ADA
    ADA_model = ada.fit(training_features, labels)
    
    alpha_t = adaboost_classifier(labels, training_features,clf, ADA_model)    # ADA2
    
    neural_model = build_neural_model( 15, training_features, labels, num_passes=200, print_loss=True) #Neural Network 
   
    # Read test image
    img_original = cv2.imread(test_images[0],0)
    
    start = time.time()
    # Preprocess test image
    greyscale1, binarized1, segmentsBinarized1, segmentsGrey1 = Preprocess(img_original)
    
    maxOccurSegment=[]
    maxOccurSegmentSVM=[]
    maxOccurSegmentADA=[]
    maxOccurSegmentADA2=[]
    maxOccurSegmentNN=[]
    
    
    for j in range(segmentsGrey1.shape[0]):
        
        # Extract Features
        test_point = extract_features(greyscale1, binarized1, segmentsBinarized1, segmentsGrey1[j])
        
        #Get prediction of each segment for each model
        maxOccurSegment.append(int(KNN(test_point, training_features, labels, k))) #KNN

        maxOccurSegmentSVM.append(int(clf.predict([test_point])[0]))  #SVM

        maxOccurSegmentADA.append(int(ADA_model.predict([test_point])[0]))  #ADA

        predictions = [maxOccurSegmentSVM[j], maxOccurSegmentADA[j]]
        maxOccurSegmentADA2.append(adaboost_predict(test_point, alpha_t, predictions)) #ADA2 

        maxOccurSegmentNN.extend(neural_predict(neural_model, test_point)) #Neural Network 


    maxOccurSegment=np.asarray(maxOccurSegment)
    maxOccurSegmentSVM=np.asarray(maxOccurSegmentSVM)
    maxOccurSegmentADA=np.asarray(maxOccurSegmentADA)
    maxOccurSegmentADA2=np.asarray(maxOccurSegmentADA2)
    maxOccurSegmentNN=np.asarray(maxOccurSegmentNN)

    
    # Append the prediction
    knn_prediction = np.bincount(maxOccurSegment).argmax()
    SVM_prediction = np.bincount(maxOccurSegmentSVM).argmax()
    ADA_prediction = np.bincount(maxOccurSegmentADA).argmax()
    ADA2_prediction = np.bincount(maxOccurSegmentADA2).argmax()
    NN_prediction = np.bincount(maxOccurSegmentNN).argmax()


    end = time.time()
    predictionTime= end-start
    totalTime = trainTime+ predictionTime


    f.write(str(round(totalTime,2)))
    f.write("\n")

    f2.write(str(knn_prediction))
    f2.write("\n")


    print("knn_prediction class :", knn_prediction)
    print("SVM_prediction class :", SVM_prediction)
    print("ADA_prediction class :", ADA_prediction)
    print("ADA2_prediction class :", ADA2_prediction)
    print("NN_prediction class :", NN_prediction)

    f.close()
    f2.close()

    return knn_prediction, SVM_prediction, ADA_prediction, ADA2_prediction, NN_prediction, totalTime
    
    
def predict_all_models():
    direcs = natsorted(glob.glob ("data/*"))
    true_values = []
    knn_prediction = []
    SVM_prediction=[]
    ADA_prediction=[]
    ADA2_prediction=[]
    NN_prediction=[]
    totalTime=[]

    for l in range(len(direcs)):
        knn_predict, SVM_predict, ADA_predict, ADA2_predict, NN_predict, timeTest  = main_all_models(direcs[l],true_values,l)
        knn_prediction.append(knn_predict)
        SVM_prediction.append(SVM_predict)
        ADA_prediction.append(ADA_predict)
        ADA2_prediction.append(ADA2_predict)
        NN_prediction.append(NN_predict)
        totalTime.append(timeTest)

    f4 = open("time.txt", "a")
    totalTime=np.asarray(totalTime)
    f4.write(str(np.mean(totalTime)))   
    f4.close()
    return true_values, knn_prediction, SVM_prediction, ADA_prediction, ADA2_prediction, NN_prediction   



def calculate_accuracy_all_models(true_values, knn_prediction, SVM_prediction, ADA_prediction, ADA2_prediction, NN_prediction):
    
    knn_prediction= np.asarray(knn_prediction)
    SVM_prediction= np.asarray(SVM_prediction)
    ADA_prediction= np.asarray(ADA_prediction)
    ADA2_prediction= np.asarray(ADA2_prediction)
    NN_prediction= np.asarray(NN_prediction)
    true_values= np.asarray(true_values)
 
    results_KNN = np.array([knn_prediction == true_values])  
    accuracy_knn= results_KNN[results_KNN==True].shape[0]/true_values.shape[0]
    
    results_SVM = np.array([SVM_prediction == true_values])  
    accuracy_SVM= results_SVM[results_SVM==True].shape[0]/true_values.shape[0]
    
    results_ADA = np.array([ADA_prediction == true_values])  
    accuracy_ADA= results_ADA[results_ADA==True].shape[0]/true_values.shape[0]
    
    results_ADA2 = np.array([ADA2_prediction == true_values])  
    accuracy_ADA2= results_ADA2[results_ADA2==True].shape[0]/true_values.shape[0]
    
    results_NN = np.array([NN_prediction == true_values])  
    accuracy_NN= results_NN[results_NN==True].shape[0]/true_values.shape[0]

    print("K-Nearest Neighbour Classifier Accuracy: ", accuracy_knn*100, "%")
    print("SVM Classifier Accuracy: ", accuracy_SVM*100, "%")
    print("ADA Classifier Accuracy: ", accuracy_ADA*100, "%")
    print("ADA2 Classifier Accuracy: ", accuracy_ADA2*100, "%")
    print("Neural Network Classifier Accuracy: ", accuracy_NN*100, "%")
    return   

true_values, knn_prediction, SVM_prediction, ADA_prediction, ADA2_prediction, NN_prediction = predict_all_models() 
calculate_accuracy_all_models(true_values, knn_prediction, SVM_prediction, ADA_prediction, ADA2_prediction, NN_prediction)    