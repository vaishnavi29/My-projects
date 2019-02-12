import numpy as np
import time
import cv2
from boosting_classifier import Boosting_Classifier
from visualizer import Visualizer
from im_process import normalize
from utils import *


def main():
    # flag for debugging
    flag_subset = False
    boosting_type = 'Real'  # 'Real' or 'Ada'
    training_epochs = 100 if not flag_subset else 20
    act_cache_dir = 'wc_activations.npy' if not flag_subset else 'wc_activations_subset.npy'
    chosen_wc_cache_dir = 'chosen_wcs.pkl' if not flag_subset else 'chosen_wcs_subset.pkl'
    # data configurations
    pos_data_dir = '/home/vaishnaviravindran29/Desktop/prml/Release/newface16'
    neg_data_dir = '/home/vaishnaviravindran29/Desktop/prml/nonface16/nonface16'
    image_w = 16
    image_h = 16
    data, labels = load_data(pos_data_dir, neg_data_dir, image_w, image_h, flag_subset)
    data = integrate_images(normalize(data))

    # number of bins for boosting
    num_bins = 25

    # number of cpus for parallel computing
    num_cores = 1 if not flag_subset else 1  # always use 1 when debugging

    # create Haar filters
    filters = generate_Haar_filters(4, 4, 16, 16, image_w, image_h, flag_subset)

    # create visualizer to draw histograms, roc curves and best weak classifier accuracies
    drawer = Visualizer([10, 20, 50, 100], [1, 10, 20, 50, 100])

    # create boost classifier with a pool of weak classifier
    boost = Boosting_Classifier(filters, data, labels, training_epochs, num_bins, drawer, num_cores, boosting_type)

    # calculate filter values for all training images
    start = time.clock()
    wc_activations = boost.calculate_training_activations(act_cache_dir, act_cache_dir)
    end = time.clock()
    print('%f seconds for activation calculation' % (end - start))

    weak_classifiers_all =  np.load(chosen_wc_cache_dir)
    sc_scores= []
    T = [10,50,100]
    for j in range(0, len(T)):
        a = T[j]
        weak_classifiers = weak_classifiers_all[0:T[j]]

        for i in range(0,T[j]):
            boost.weak_classifiers[i].id = (weak_classifiers[i])[1].id
            #boost.weak_classifiers = [Real_Weak_Classifier(i, filt[0], filt[1], self.num_bins) \
                                     #for i, filt in enumerate(self.filters)]
        sc_scores.append(boost.train_real_boost(wc_activations, T[j], num_bins))


    drawer.labels = labels
    drawer.strong_classifier_scores = {10:sc_scores[0], 50:sc_scores[1], 100:sc_scores[2] }
    drawer.draw_histograms()
    drawer.draw_rocs()


if __name__ == '__main__':
    main()
