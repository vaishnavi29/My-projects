import os
import scipy.io
import numpy as np
import cv2
from skimage.feature import hog
import skimage.color
from skimage import data, exposure
import glob
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def load_landmarks(landmarks_file):

        train_landmarks = scipy.io.loadmat(landmarks_file)
        landmarks = train_landmarks.get('face_landmark')
        print(landmarks.shape)
        return landmarks


def load_annotations(annotations_file):
    train_annotations = scipy.io.loadmat(annotations_file)
    annotations = train_annotations.get('trait_annotation')
    print(annotations.shape)
    return annotations

def load_votingscores(annotations_file):
    train_annotations = scipy.io.loadmat(annotations_file)
    vot_diff = train_annotations.get('vote_diff')
    print(vot_diff.shape)
    return vot_diff

def get_one_hot_annotations(annotations):
    no_of_traits = annotations.shape[1]
    no_of_training_samples = annotations.shape[0]
    new_annotations = np.zeros(shape=(no_of_training_samples, no_of_traits))
    for i in range(0, no_of_traits):
        a = annotations[:, i]
        #b = np.sort(a)
        mean = np.mean(a)
        #maxval = b.max()
        #threshold = maxval * 0.5
        threshold = mean
        for j in range(0, no_of_training_samples):
            if (annotations[j, i] >= threshold):
                label = 1
            else:
                label = -1
            new_annotations[j, i] = label
    return new_annotations

def Normalize(data):
    '''
    scaled_data = np.zeros(shape=data.shape)
    for i in range (0, data.shape[0]):
        min = np.min(data[i,:])
        max = np.max(data[i,:])
        scaled_data[i,:] = (data[i,:] - min)/(max-min)
    '''
    scaled_data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)

    return scaled_data

def extract_hog_feaures(dir_name, save_dir_hog_features, display_hog = False):
    hog_features = []
    if save_dir_hog_features is not None and os.path.exists(save_dir_hog_features):
        print('[Find cached hog_features, %s loading...]' % save_dir_hog_features)
        hog_features = np.load(save_dir_hog_features)
    else:
        for filename in os.listdir(dir_name):
            original_img = cv2.imread(os.path.join(dir_name, filename))
            # cv2.imshow('Image', original_img)

            fd, hog_image = hog(original_img, orientations=32, pixels_per_cell=(16, 16),
                                cells_per_block=(1, 1), visualize=True, multichannel=True, feature_vector=True)
            hog_features.append(fd)
            if(display_hog):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

                ax1.axis('off')
                ax1.imshow(original_img, cmap=plt.cm.gray)
                ax1.set_title('Input image')

                # Rescale histogram for better display
                hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

                ax2.axis('off')
                ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
                ax2.set_title('Histogram of Oriented Gradients')
                plt.show()
        if save_dir_hog_features is not None:
                pickle.dump(hog_features, open(save_dir_hog_features, 'wb'))

    return hog_features