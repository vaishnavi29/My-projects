from utils import *
from svm_training import *
import numpy as np
import sys
sys.path.append('/libsvm-3.21/python')
import matplotlib.pyplot as plt
import scipy
from svmutil import *
import warnings
import cv2
from skimage.feature import hog
import skimage.color
from skimage import data, exposure
import glob
import pickle
warnings.filterwarnings('ignore')

#-------------------For Governor-------------------------------------------#
training_file = "stat-sen.mat"
training_file2 = "stat-gov.mat"

print('Loading load_votingscores..')
vote_diff = load_votingscores(training_file2)

print('Loading training annotations..')
landmarks = load_landmarks(training_file2)
print('training landmarks shape is ', landmarks.shape)

landmarks = Normalize(landmarks)
hog_features = extract_hog_feaures('img-elec/governor', 'hog_features_governor.pkl', False)
hog_features = np.array(hog_features)
print("shape of hog_features",hog_features.shape)

hog_features = Normalize(hog_features)
hog_and_landmarks  = np.concatenate((landmarks,hog_features),axis=1)
print("shape of hog_and_landmarks",hog_and_landmarks.shape)

labels = []
subtracted_feaures = []
i = 0
while(i <len(vote_diff)/2):
    if(vote_diff[i] > vote_diff[i+1]):
        subtracted_feaures.append(hog_and_landmarks[i]-hog_and_landmarks[i+1])
        labels.append(-1)
    else:
        subtracted_feaures.append(hog_and_landmarks[i+1]-hog_and_landmarks[i])
        labels.append(1)
    i=i+2

while(i <len(vote_diff)):
    if(vote_diff[i] > vote_diff[i+1]):
        subtracted_feaures.append(hog_and_landmarks[i]-hog_and_landmarks[i+1])
        labels.append(1)
    else:
        subtracted_feaures.append(hog_and_landmarks[i+1]-hog_and_landmarks[i])
        labels.append(-1)
    i=i+2

subtracted_feaures = np.array(subtracted_feaures)
labels = np.array(labels)

train_accuracies,test_accuracies= train_and_plot_part21(subtracted_feaures,labels,'part21','governor')
#train_accuracies,test_accuracies= train_and_plot_part21(subtracted_feaures,labels,'part21','senator')
print("train_accuracies for gorvernor",train_accuracies)
print("test_accuracies governor",test_accuracies)

#-------------------For senator-------------------------------------------#
training_file = "stat-sen.mat"
training_file2 = "stat-gov.mat"

print('Loading load_votingscores..')
vote_diff = load_votingscores(training_file)

print('Loading training annotations..')
landmarks = load_landmarks(training_file)
print('training landmarks shape is ', landmarks.shape)

landmarks = Normalize(landmarks)
hog_features = extract_hog_feaures('img-elec/senator', 'hog_features_senator.pkl', False)
hog_features = np.array(hog_features)
print("shape of hog_features",hog_features.shape)

hog_features = Normalize(hog_features)
hog_and_landmarks  = np.concatenate((landmarks,hog_features),axis=1)
print("shape of hog_and_landmarks",hog_and_landmarks.shape)

labels = []
subtracted_feaures = []
i = 0
while(i <len(vote_diff)/2):
    if(vote_diff[i] > vote_diff[i+1]):
        subtracted_feaures.append(hog_and_landmarks[i]-hog_and_landmarks[i+1])
        labels.append(-1)
    else:
        subtracted_feaures.append(hog_and_landmarks[i+1]-hog_and_landmarks[i])
        labels.append(1)
    i=i+2

while(i <len(vote_diff)):
    if(vote_diff[i] > vote_diff[i+1]):
        subtracted_feaures.append(hog_and_landmarks[i]-hog_and_landmarks[i+1])
        labels.append(1)
    else:
        subtracted_feaures.append(hog_and_landmarks[i+1]-hog_and_landmarks[i])
        labels.append(-1)
    i=i+2

subtracted_feaures = np.array(subtracted_feaures)
labels = np.array(labels)

#train_accuracies,test_accuracies= train_and_plot_part21(subtracted_feaures,labels,'part21','governor')
train_accuracies,test_accuracies= train_and_plot_part21(subtracted_feaures,labels,'part21','senator')
print("train_accuracies for senator ",train_accuracies)
print("test_accuracies for senator",test_accuracies)
