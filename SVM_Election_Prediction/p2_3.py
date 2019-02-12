from svm_training import *
import numpy as np
import sys
sys.path.append('/libsvm-3.21/python')
import warnings
from scipy.stats import linregress

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

no_of_models = 14
type = 'rich'
social_feautures = np.zeros(shape = (hog_features.shape[0],no_of_models))
for i in range(0,no_of_models):
    predicted = predict_using_svm('clf_'+str(i) + type +'.pkl', hog_and_landmarks)
    social_feautures[:,i]=(predicted)


subtracted_feaures = []
i = 0
while(i <len(vote_diff)):
    if(vote_diff[i] > vote_diff[i+1]):
        subtracted_feaures.append(social_feautures[i]-social_feautures[i+1])
    else:
        subtracted_feaures.append(social_feautures[i+1]-social_feautures[i])
    i=i+2

subtracted_feaures = np.array(subtracted_feaures)

vote_diff_absolute = []
i = 0
while(i <len(vote_diff)):
    vote_diff_absolute.append(abs(vote_diff[i]))
    i=i+2

subtracted_feaures = np.array(subtracted_feaures)
vote_diff_absolute = np.array(vote_diff_absolute)
correlation_coeffecient = []
for i in range(0,no_of_models):
     x = list(subtracted_feaures[:, i])
     y = list(np.reshape(vote_diff_absolute,vote_diff_absolute.shape[0]))
     #r = np.corrcoef(x, y)
     r = linregress(x, y)
     rvalue = r.rvalue
     correlation_coeffecient.append(rvalue)

print(correlation_coeffecient)

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

no_of_models = 14
type = 'rich'
social_feautures = np.zeros(shape = (hog_features.shape[0],no_of_models))
for i in range(0,no_of_models):
    predicted = predict_using_svm('clf_'+str(i) + type +'.pkl', hog_and_landmarks)
    social_feautures[:,i]=(predicted)


subtracted_feaures = []
i = 0
while(i <len(vote_diff)):
    if(vote_diff[i] > vote_diff[i+1]):
        subtracted_feaures.append(social_feautures[i]-social_feautures[i+1])
    else:
        subtracted_feaures.append(social_feautures[i+1]-social_feautures[i])
    i=i+2

subtracted_feaures = np.array(subtracted_feaures)

vote_diff_absolute = []
i = 0
while(i <len(vote_diff)):
    vote_diff_absolute.append(abs(vote_diff[i]))
    i=i+2

subtracted_feaures = np.array(subtracted_feaures)
vote_diff_absolute = np.array(vote_diff_absolute)
correlation_coeffecient = []
for i in range(0,no_of_models):
     x = list(subtracted_feaures[:, i])
     y = list(np.reshape(vote_diff_absolute,vote_diff_absolute.shape[0]))
     #r = np.corrcoef(x, y)
     r = linregress(x, y)
     rvalue = r.rvalue
     correlation_coeffecient.append(rvalue)

print("correlation_coeffecients are")
print(correlation_coeffecient)