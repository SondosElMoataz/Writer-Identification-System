import cv2
import numpy as np
import math
import glob
import csv
import os
import time
from skimage import io ,filters,feature,transform
from scipy import stats
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics

from PreprocessingAndSegmentation import *
from ExtractingFeatures import *


def train(folder):
    trainTime=0
    feature_vector_all=[]
    direcs = sorted(glob.glob (folder+"/*"))
    for direc in direcs:
        if os.path.isdir(direc):
            files = sorted(glob.glob (direc+'/*'))
            for file in files:
                img = cv2.imread(file,0) 
                tempStart =time.time()

                #Preprocessing
                greyscale1, binarized1, segmentsBinarized1, segmentsGrey1 = Preprocess(img)
                
                #Extract features of each segment
                for i in range(segmentsGrey1.shape[0]):
                    feature_vector=[]
     
                    z = direc.split('\\')
                    y= int(z[1])
                    if(y< 100):
                        feature_vector.append(int(direc[8]))
                    if(y>=100 and y<1000):
                        feature_vector.append(int(direc[9]))
                    if(y>=1000):
                        feature_vector.append(int(direc[10]))

                    features =extract_features(greyscale1, binarized1, segmentsBinarized1, segmentsGrey1[i])
                    feature_vector.extend(features)
                    feature_vector_all.append(feature_vector)
                    
                tempEnd =time.time()
                trainTime+= tempEnd-tempStart 

    return feature_vector_all,trainTime
    

def calculateDistance(x1, x2):

    distance =np.linalg.norm(x1-x2)
    return distance
	
    
def KNN(test_point, training_features, y_train, k):
    class1=0
    class2=0
    class3=0
    dist=[]
    indexs=[]
    
    for i in range(training_features.shape[0]):
        dist.append(calculateDistance(test_point,training_features[i]))
        
    dist2=np.argsort(dist)

    for i in range(k):
        if(y_train[dist2[i]]==1):
            class1=class1+1
        elif(y_train[dist2[i]]==2):
            class2=class2+1
        else:
            class3=class3+1

    if(max(class1,class2,class3)==class1):
        classification=1
    elif(max(class1,class2,class3)==class2):
        classification=2
    else:
        classification=3
    return classification


def SVM(training_features, y_train):
    clf = SVC(kernel='linear', C=6.0)    #linear sigmoid
    clf.fit(training_features, y_train)  
    return clf
    
    
# Adaboostof the 3 models: knn, svm, DecisionTree with adaboost
def adaboost_classifier_3(Y_train, training_features,clf,k,ADA_model):     

    M = training_features.shape[0]
    w = np.full((M,), (1/M))
    
    classValue = []
    knnPredicts = []
    svmPredicts = []
    adaPredicts = []
    
    for i in range(M):
        knnPredicts.append((KNN(training_features[i], training_features, Y_train, k)))
        svmPredicts.append((clf.predict([training_features[i]])[0]))
        adaPredicts.append(ADA_model.predict([training_features[i]])[0])
        
      
    predictions = [knnPredicts,svmPredicts,adaPredicts]
    alpha_t =np.zeros(len(predictions))

    for i in range(len(predictions)):  
        
        miss=np.array(1*(predictions[i]==Y_train))
        err_t = (w*(1-miss)).sum()/(w.sum())

        if(err_t==0):
            alpha_t[i] = (1)
            break
        elif(err_t==1):
            alpha_t[i]=np.min(w)
        else:
            alpha_t[i] = (np.log((1-err_t)/err_t) + np.log(2))

        w=w*np.exp(alpha_t[i]*(1-miss))
        w= w/np.linalg.norm(w)    

        
    return alpha_t 


# Adaboost of both: svm and DecisionTree with adaboost
def adaboost_classifier(Y_train, training_features,clf,ADA_model):     

    M = training_features.shape[0]
    w = np.full((M,), (1/M))
    
    classValue = []
    svmPredicts = []
    adaPredicts = []
    
    # Get the alpha of each model by training it on the given training features we have
    for i in range(M):
        svmPredicts.append((clf.predict([training_features[i]])[0]))
        adaPredicts.append(ADA_model.predict([training_features[i]])[0])
        
      
    predictions = [svmPredicts,adaPredicts]
    alpha_t =np.zeros(len(predictions))
    for i in range(len(predictions)):  
        
        miss=np.array(1*(predictions[i]==Y_train))
        err_t = (w*(1-miss)).sum()/(w.sum())

        if(err_t==0):
            alpha_t[i] = (1)
            break
        elif(err_t==1):
            alpha_t[i]=np.min(w)
        else:
            alpha_t[i] = (np.log((1-err_t)/err_t) + np.log(2))

        w=w*np.exp(alpha_t[i]*(1-miss))
        w= w/np.linalg.norm(w)  
        
    return alpha_t     
    
def adaboost_predict (X_test, alpha_t, predictions): 
    classValue = []
    for k in range(3):
        value=0
        for j in range(len(predictions)):
            value+=alpha_t[j]*(predictions[j]==k+1)
        classValue.append(value)
        

    return np.argmax(classValue)+1    
    
    
def sigmoid(x):
    sig = 1/(1+ np.exp(-x))
    return sig


def softmax(x):
    num = np.exp(x - np.max(x))
    num /= num.mean(axis=0, keepdims=True)
    return num    
    
    
def build_neural_model(nn_hdim, training_features, labels, num_passes=20000, print_loss=False):
    
    m = training_features.shape[0]  # training set size
    nn_input_dim = training_features.shape[1]  # input layer dimensionality (we have two input features)
    nn_output_dim = 3  # output layer dimensionality (we have one output)
    
    alpha = 0.1  # learning rate for gradient descent
   
    np.random.seed(0)
    W1 = np.random.randn(nn_hdim, nn_input_dim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((nn_hdim, 1))
    W2 = np.random.randn(nn_output_dim, nn_hdim) / np.sqrt(nn_hdim)
    b2 = np.zeros((nn_output_dim, 1))

    model = {}

    for i in range(0, num_passes):
        DW1 = 0
        DW2 = 0
        Db1 = 0
        Db2 = 0
        cost = 0
        # Loop on every training example...
        for j in range(0, m):
            a0 = training_features[j, :].reshape(-1, 1)  # Every training example is a column vector.
            y = labels[j]
            
            Y = [0,0,0]
            Y[int(y-1)]=1
            
            # Forward propagation
            z1 = W1@a0 + b1
            a1 = np.tanh(z1)
            z2 = W2@a1 + b2
            a2 = softmax(z2)
            
            # Loss
            cost_j = -np.mean(Y * np.log(a2.T + 1e-8))
            
            # Backward propagation
            da2 = (a2-y)/(a2*(1-a2))
            dz2 = (a2-y)
            dW2 = dz2@a1.T
            db2 = dz2
            
            da1 = (dz2.T@W2).T               # added transpose
            dz1 = da1*(1-pow(a1,2))
            dW1 = dz1@a0.T
            db1 = dz1
           
            DW1 += dW1
            DW2 += dW2
            Db2 += db2
            Db1 += db1
            cost += cost_j
        
        # Averaging DW1, DW2, Db1, Db2 and cost over the m training examples. 
        DW1 /= m
        DW2 /= m
        Db1 /= m
        Db2 /= m
        cost /= m

        # Gradient descent parameter update
        W1 += -alpha*DW1
        b1 += -alpha*Db1
        W2 += -alpha*DW2
        b2 += -alpha*Db2
        
        # Assign new parameters to the model
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    return model    
    
    
def neural_predict(model, x):

    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    a0 = np.empty((len(x),1))
    for i in range(len(x)):
        a0[i]=x[i]
        
    z1 = W1@a0 + b1
    a1 = np.tanh(z1)
    z2 = W2@a1 + b2
    a2 = softmax(z2)
    
    prediction = np.argmax(a2, axis=0)
    
    return prediction+1    
    
	