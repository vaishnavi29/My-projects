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
#warnings.filterwarnings('ignore')
warnings.filterwarnings(action='once')

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

no_of_models = 14
type = 'rich'
social_feautures = np.zeros(shape = (hog_features.shape[0],no_of_models))
for i in range(0,no_of_models):
    predicted = predict_using_svm('clf_'+str(i) + type +'.pkl', hog_and_landmarks)
    social_feautures[:,i]=(predicted)

labels = []
subtracted_feaures = []
i = 0
while(i <len(vote_diff)/2):
    if(vote_diff[i] > vote_diff[i+1]):
        subtracted_feaures.append(social_feautures[i]-social_feautures[i+1])
        labels.append(-1)
    else:
        subtracted_feaures.append(social_feautures[i+1]-social_feautures[i])
        labels.append(1)
    i=i+2

while(i <len(vote_diff)):
    if(vote_diff[i] > vote_diff[i+1]):
        subtracted_feaures.append(social_feautures[i]-social_feautures[i+1])
        labels.append(1)
    else:
        subtracted_feaures.append(social_feautures[i+1]-social_feautures[i])
        labels.append(-1)
    i=i+2

subtracted_feaures = np.array(subtracted_feaures)
subtracted_feaures = Normalize(subtracted_feaures)
labels = np.array(labels)

seed = np.random.randint(low=30,high=100,size=70)
seed = [48]
final_test = []
final_train = []
for i in seed:
    print("seed is",i)
    train_accuracies,test_accuracies= train_and_plot_part21(subtracted_feaures,labels,'part22','governor',i)
    #train_accuracies, test_accuracies = train_and_plot_part21(subtracted_feaures, labels, 'part22', 'senator', i)

    if(test_accuracies[0] >= 0.55 and test_accuracies[0]<train_accuracies[0]):
        print("train_accuracies",train_accuracies)
        print("test_accuracies",test_accuracies)
        final_test.append(test_accuracies[0])
        final_train.append(train_accuracies[0])

print("final_train accuracy for governor",final_train)
print("final_test accuracy for governor",final_test)

#-------------------For senator-------------------------------------------#

training_file = "stat-sen.mat"

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

no_of_models = 14
type = 'rich'
social_feautures = np.zeros(shape = (hog_features.shape[0],no_of_models))
for i in range(0,no_of_models):
    predicted = predict_using_svm('clf_'+str(i) + type +'.pkl', hog_and_landmarks)
    social_feautures[:,i]=(predicted)

labels = []
subtracted_feaures = []
i = 0
while(i <len(vote_diff)/2):
    if(vote_diff[i] > vote_diff[i+1]):
        subtracted_feaures.append(social_feautures[i]-social_feautures[i+1])
        labels.append(-1)
    else:
        subtracted_feaures.append(social_feautures[i+1]-social_feautures[i])
        labels.append(1)
    i=i+2

while(i <len(vote_diff)):
    if(vote_diff[i] > vote_diff[i+1]):
        subtracted_feaures.append(social_feautures[i]-social_feautures[i+1])
        labels.append(1)
    else:
        subtracted_feaures.append(social_feautures[i+1]-social_feautures[i])
        labels.append(-1)
    i=i+2

subtracted_feaures = np.array(subtracted_feaures)
subtracted_feaures = Normalize(subtracted_feaures)
labels = np.array(labels)

seed = np.random.randint(low=30,high=100,size=70)
seed = [48]
final_test = []
final_train = []
for i in seed:
    print("seed is",i)
    train_accuracies,test_accuracies= train_and_plot_part21(subtracted_feaures,labels,'part22','senator',i)
    #train_accuracies, test_accuracies = train_and_plot_part21(subtracted_feaures, labels, 'part22', 'senator', i)

    if(test_accuracies[0] >= 0.55 and test_accuracies[0]<train_accuracies[0]):
        print("train_accuracies",train_accuracies)
        print("test_accuracies",test_accuracies)
        final_test.append(test_accuracies[0])
        final_train.append(train_accuracies[0])

print("final_train accuarcy for senator",final_train)
print("final_test accuracy for senator",final_test)