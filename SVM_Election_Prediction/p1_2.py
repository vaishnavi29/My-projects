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
#for data in annotations[:,0]:
    #plt.scatter(data,0, c='r')

#plt.show()
#new_annotations = get_one_hot_annotations(annotations)
#print("new_annotations",new_annotations.shape)

landmarks = Normalize(landmarks)
param_grid = {'C': [1,10,100,1000,10000,100000],
              'gamma': [0.1,0.01,0.001,0.0001,0.005,1],
              'epsilon': [0.1,0.01,0.001,0.0001,0.005,1,0.00001,0.000001,0.0000001,0.00000001,0.00000000001]}

param_grid = {'C': [0.1,1,10,100,1000,2,4,6,8,16,32],
              'gamma': [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1],
              'epsilon': [0,0.1,0.01,0.5,1,2,4,0.0001,0.005,1,0.00001,0.000001,0.0000001,0.00000001,0.00000000001]}
param_grid = {'C': [0.1,1,10,100,1000,2,4,6,8,16,32,0.01,0.5,0.00001,0.000001],
              'gamma':  np.linspace(2 ** -12, 2 ** -5, num=10),
              'epsilon': [0,0.1,0.01,0.5,1,0.0001,0.005,1,0.00001,0.000001,0.0000001,0.00000001,0.00000000001]}
param_grid = {'C': np.linspace(2 ** 5, 2 ** 13, num=20),
              'gamma': np.linspace(2 ** -17, 2 ** 5, num=20),
              'epsilon':np.linspace(2 ** -10, 2 ** 1, num=20)}

arr1 = [2,2,2,2,2,2,2,2,2]
#c = np.power(arr1, [2,3,4,5,6,7,8,9,10,11,12,13])
g = []
e = []
c=[]
for i in range(-9,-5):
    e.append(2**i)

for i in range(-17,5):
    g.append(2**i)
for i in range(1,13):
    c.append(2**i)
param_grid = {'C': [2**3,2**-1,2**1,2**-5,10,1],
              'gamma': [2**-13,2**-15,2**17,2**11,1e-3,1e-2,1e-1,2**-17],
              'epsilon': [2**-5,2**-3,2**-7,2**9,0.01,2**-9,2**4,2**-9,2**1]}
param_grid = {'C': c,
              'gamma': g,
              'epsilon': e}
param_grid = {'C': [1,10,100,1000,10000,100000,32],
              'gamma': [0.1,0.01,0.001,0.0001,0.005,1],
              'epsilon': [0.1,0.01,0.001,0.0001,0.005,1,0.00001,0.000001,0.0000001,0.00000001,0.00000000001]}
param_grid = {'C': [32],
              'gamma': [0.01],
              'epsilon': [0.1]}
hog_features = extract_hog_feaures('img', 'hog_features.pkl', False)

#pickle.dump(hog_features, open('hog_features.pkl', 'wb'))
#hog_features = np.load('hog_features.pkl')
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


hog_features = Normalize(hog_features)
hog_and_landmarks  = np.concatenate((landmarks,hog_features),axis=1)
print("shape of hog_and_landmarks",hog_and_landmarks.shape)
param_grid = {'C': [0.1,1,10,100,1000],
              'gamma': [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1],
              'epsilon': [0,0.1,0.01,0.5,1,2,4]}

param_grid = {'C': np.linspace(2 ** 5, 2 ** 13, num=10),
              'gamma': np.linspace(2 ** -18, 2 ** -5, num=10),
              'epsilon': np.linspace(2 ** -8, 2 ** 5, num=6)}

arr1 = [2,2,2,2,2,2,2,2,2,2]
c = np.power(arr1, [1,2,3,4,5,6,7,8,9,10])
g = []
e = []
for i in range(-8,6):
    e.append(2**i)

for i in range(-20,-10):
    g.append(2**i)
param_grid = {'C': c,
              'gamma': g,
              'epsilon': e}
param_grid = {'C': np.linspace(2 ** 1, 2 ** 5, num=10),
              'gamma': np.linspace(2 ** -20, 2 ** -18, num=10),
              'epsilon': np.linspace(2 ** -8, 2 ** 5, num=6)}

arr1 = [2,2,2,2,2,2,2,2,2,2]
c = np.power(arr1, [1,2,3,4,5,6,7,8,9,10])
g = []
e = []
for i in range(-8,6):
    e.append(2**i)

for i in range(-20,-10):
    g.append(2**i)
param_grid = {'C': c,
              'gamma': g,
              'epsilon': e}

train_accuracies_hog,test_accuracies_hog,train_mse_hog,test_mse_hog,train_precision_hog, test_precision_hog= train_and_plot_svm(hog_and_landmarks,annotations,param_grid,'rich',ylim= 0.55)

no_of_models = 14
plt.figure()
no_of_models = list(range(no_of_models))
plt.plot(no_of_models, train_accuracies_landmarks, 'o-', color="r",
         label='train accuracy poor')
plt.plot(no_of_models, test_accuracies_landmarks, 'o-', color="g",
         label='test accuracy poor')
plt.plot(no_of_models, train_accuracies_hog, 'o-', color="b",
         label='train accuracy rich')
plt.plot(no_of_models, test_accuracies_hog, 'o-', color="m",
         label='test accuracy rich')
plt.xlabel("Model No")
plt.ylabel("Classification accuracy")
plt.title("Classification accuracy")
plt.legend(loc="best")
plt.savefig("Classification accuracy_compare" + '.png')
#plt.show()

plt.figure()
#no_of_models = list(range(no_of_models))
plt.plot(no_of_models, train_mse_landmarks, 'o-', color="r",
         label='train mse poor')
plt.plot(no_of_models, test_mse_landmarks, 'o-', color="g",
         label='test mse poor')
plt.plot(no_of_models, train_mse_hog, 'o-', color="b",
         label='train mse rich')
plt.plot(no_of_models, test_mse_hog, 'o-', color="m",
         label='test mse rich')
plt.xlabel("Model No")
plt.ylabel("mean square error")
plt.title("mean square error")
plt.legend(loc="best")
plt.savefig("mean square error_compare" + '.png')
#plt.show()

plt.figure()
plt.plot(no_of_models, train_precision_landmarks, 'o-', color="r",
         label='train precision poor')
plt.plot(no_of_models, test_precision_landmarks, 'o-', color="g",
         label='test precision poor')
plt.plot(no_of_models, train_precision_hog, 'o-', color="b",
         label='train precision rich')
plt.plot(no_of_models, test_precision_hog, 'o-', color="m",
         label='test precision rich')
plt.xlabel("Model No")
plt.ylabel("precision")
plt.title("precision")
plt.legend(loc="best")
plt.savefig("precision_compare" + '.png')
