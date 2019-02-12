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



training_file = "train-anno.mat"
print('Loading training landmarks..')
landmarks = load_landmarks(training_file)
print('training landmarks shape is ', landmarks.shape)

print('Loading training annotations..')
annotations = load_annotations(training_file)

landmarks = Normalize(landmarks)

param_grid = {'C': np.linspace(2 ** 5, 2 ** 13, num=20),
              'gamma': np.linspace(2 ** -17, 2 ** 5, num=20),
              'epsilon':np.linspace(2 ** -10, 2 ** 1, num=20)}


hog_features = extract_hog_feaures('img', 'hog_features.pkl', False)


hog_features = np.array(hog_features)
print("shape of hog_features",hog_features.shape)

train_accuracies_landmarks = []
test_accuracies_landmarks = []
train_mse_landmarks = []
test_mse_landmarks = []
train_precision_landmarks = []
test_precision_landmarks = []

seeds = [None,None,None,None,None,81,None,62,None,None,37,37,None,None]

for i in range(0,14):
  train_acc, test_acc, train_mse,test_mse,train_prec,test_pres= train_and_plot_svm3(landmarks,annotations,param_grid,'poor',i,seeds[i],computeagain=False)
  train_accuracies_landmarks.append(train_acc)
  test_accuracies_landmarks.append(test_acc)
  train_mse_landmarks.append(train_mse)
  test_mse_landmarks.append(test_mse)
  train_precision_landmarks.append(train_prec)
  test_precision_landmarks.append(test_pres)

type = 'poor'
plot_graphs(14, train_accuracies_landmarks, test_accuracies_landmarks, 'mean train-',
            'mean test-', 'mean accuracies-', 'mean accuracies-', type)
plot_graphs(14, train_mse_landmarks, test_mse_landmarks, 'mean train-', 'mean test-',
            'mean mse-', 'mean mse-', type)
plot_graphs(14, train_precision_landmarks, test_precision_landmarks, 'mean train-',
            'mean test-', 'mean precision-', 'mean precision-', type)


