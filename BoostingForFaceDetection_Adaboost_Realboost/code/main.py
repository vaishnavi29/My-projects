import numpy as np
import time
import cv2
from boosting_classifier import Boosting_Classifier
from visualizer import Visualizer
from im_process import normalize
from utils import *
import pickle

def main():
    # flag for debugging
    flag_subset = False
    boosting_type = 'Ada'  # 'Real' or 'Ada'
    training_epochs = 100 if not flag_subset else 20
    act_cache_dir = 'wc_activations.npy' if not flag_subset else 'wc_activations_subset.npy'
    chosen_wc_cache_dir = 'chosen_wcs.pkl' if not flag_subset else 'chosen_wcs_subset.pkl'
    save_dir_sc_error = 'sc_error.pkl' if not flag_subset else 'sc_error_subset.pkl'
    save_dir_current_weights = 'current_weights.pkl' if not flag_subset else 'current_weights_subset.pk1'
    saved_wrong_patches_dir_image1 = 'wrong_patches_without_normalise3.pkl'
    saved_wrong_patchespredict= 'wrong_patches_predict.pkl'
    saved_patches_dir_image1 = 'saved_patches_image1.pkl'

    # data configurations (Change to pos and neg folder paths)
    pos_data_dir = '/home/vaishnaviravindran29/Desktop/prml/Release/newface16'
    neg_data_dir = '/home/vaishnaviravindran29/Desktop/prml/nonface16/nonface16'
    #neg_data_dir =  '/home/vaishnaviravindran29/Desktop/prml/Release/subsample_non/nonface16'
    image_w = 16
    image_h = 16
    data1, labels = load_data(pos_data_dir, neg_data_dir, image_w, image_h, flag_subset)
    data = integrate_images(normalize(data1))

    # number of bins for boosting
    num_bins = 25

    # number of cpus for parallel computing
    num_cores = 16 if not flag_subset else 1  # always use 1 when debugging

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

    print("Performing Adaboost..")

    wc_errors_1, sc_errors_1 = boost.train(wc_activations, 1, chosen_wc_cache_dir, save_dir_sc_error,save_dir_current_weights,hnm=False)
    sc_scores_1 = boost.GetScoresAndLabels(1, wc_activations)

    wc_errors_10, sc_errors_10 = boost.train(wc_activations, 10, chosen_wc_cache_dir, save_dir_sc_error,
                                             save_dir_current_weights,hnm=False)
    sc_scores_10 = boost.GetScoresAndLabels(10, wc_activations)

    wc_errors_50, sc_errors_50 = boost.train(wc_activations, 50, chosen_wc_cache_dir, save_dir_sc_error,
                                             save_dir_current_weights,hnm=False)
    sc_scores_50 = boost.GetScoresAndLabels(50, wc_activations)

    wc_errors_100, sc_errors_100 = boost.train(wc_activations, 100, chosen_wc_cache_dir, save_dir_sc_error,
                                               save_dir_current_weights,hnm=False)
    sc_scores_100 = boost.GetScoresAndLabels(100, wc_activations)

    drawer.labels = labels
    drawer.strong_classifier_scores = {10:sc_scores_10, 50:sc_scores_50, 100:sc_scores_100 }
    drawer.weak_classifier_accuracies = {1:wc_errors_1, 10:wc_errors_10, 55:wc_errors_50,100:wc_errors_100}
    drawer.strong_classifier_errors = {10: sc_errors_10, 50: sc_errors_50, 100: sc_errors_100}
    drawer.draw_histograms()
    drawer.draw_rocs()
    drawer.draw_wc_accuracies()
    drawer.draw_sc_training_errors()

    chosen_wcs = np.load(chosen_wc_cache_dir)
    i= 0
    for wc in chosen_wcs[0:20]:
        print("alpha(voting weight) of top Haar Filter Number %d = %f" %(i ,wc[0]))
        i = i+1

    original_img = cv2.imread('./Testing_Images/Face_1.jpg', cv2.IMREAD_GRAYSCALE)
    original_img_color = cv2.imread('./Testing_Images/Face_1.jpg')
    result_img = boost.face_detection(original_img,original_img_color, 100, scale_step=10, saved_patches_dir=None)
    cv2.imwrite('Result_img1_%s.png' % boosting_type, result_img)

    original_img = cv2.imread('./Testing_Images/Face_2.jpg', cv2.IMREAD_GRAYSCALE)
    original_img_color = cv2.imread('./Testing_Images/Face_2.jpg')
    result_img = boost.face_detection(original_img,original_img_color, 100, scale_step=10, saved_patches_dir=None)
    cv2.imwrite('Result_img2_%s.png' % boosting_type, result_img)

    original_img = cv2.imread('./Testing_Images/Face_3.jpg', cv2.IMREAD_GRAYSCALE)
    original_img_color = cv2.imread('./Testing_Images/Face_3.jpg')
    result_img = boost.face_detection(original_img, original_img_color, 100, scale_step=10, saved_patches_dir=None)
    cv2.imwrite('Result_img3_%s.png' % boosting_type, result_img)

    #With hard negative mining
    original_img = cv2.imread('./Testing_Images/Non_face_1.jpg', cv2.IMREAD_GRAYSCALE)
    wrong_patches = boost.get_hard_negative_patches(original_img,200,scale_step=10,save_dir_wrong_patches = saved_wrong_patches_dir_image1, save_dir_pos_predcits = saved_wrong_patchespredict)
    #wrong_patches = wrong_patches[0]
    print("wrong_patches.shape", wrong_patches.shape)

    wrong_patches_to_add = []
    for i in range(0, wrong_patches.shape[0]):
        patch = wrong_patches[i]
        new_img = original_img[int(patch[1]):int(patch[3]), int(patch[0]):int(patch[2])]

        new_img = cv2.resize(new_img, (16, 16))
        wrong_patches_to_add.append(new_img)

    wrong_patches_to_add = np.array(wrong_patches_to_add)
    print(wrong_patches_to_add.shape)


    new_data = np.concatenate((data1, wrong_patches_to_add))
    data = integrate_images(normalize(new_data))
    print("data.shape", data.shape)


    print("labels.shape", labels.shape)
    labels_hnm  = np.full(wrong_patches_to_add.shape[0],-1 ,dtype=int )
    labels = np.concatenate((labels,labels_hnm))
    print("new labels.shape after hnm", labels.shape)

    boost = Boosting_Classifier(filters, data, labels, training_epochs, num_bins, drawer, num_cores, boosting_type)

    chosen_wc_cache_dir_hnm = 'chosen_wcs_hnm.pkl' if not flag_subset else 'chosen_wcs_subset_hnm.pkl'
    save_dir_sc_error_hnm = 'sc_error_hnm.pkl' if not flag_subset else 'sc_error_subset_hnm.pkl'
    save_dir_current_weights_hnm = 'current_weights_hnm.pkl' if not flag_subset else 'current_weights_subset_hnm.pk1'

    act_cache_dir = 'wc_activations_hnm.npy' if not flag_subset else 'wc_activations_subset_hnm.npy'

    # calculate filter values for all training images
    start = time.clock()
    wc_activations = boost.calculate_training_activations(act_cache_dir, act_cache_dir)
    end = time.clock()
    print('%f seconds for activation calculation' % (end - start))

    print("Performing Adaboost with hard negative mining..")

    boost.train(wc_activations, 100, chosen_wc_cache_dir_hnm, save_dir_sc_error_hnm, save_dir_current_weights_hnm,hnm=True)
    #sc_scores_100_hnm = boost.GetScoresAndLabels(100, wc_activations)

    original_img = cv2.imread('./Testing_Images/Face_1.jpg', cv2.IMREAD_GRAYSCALE)
    result_img = boost.face_detection(original_img, 100, scale_step=10, saved_patches_dir=None)
    cv2.imwrite('Result_img1_afterhnm_%s.png' % boosting_type, result_img)


    original_img = cv2.imread('./Testing_Images/Face_2.jpg', cv2.IMREAD_GRAYSCALE)
    result_img = boost.face_detection(original_img, 100, scale_step=20, saved_patches_dir=None)
    cv2.imwrite('Result_img2_afterhnm%s.png' % boosting_type, result_img)

    original_img = cv2.imread('./Testing_Images/Face_3.jpg', cv2.IMREAD_GRAYSCALE)
    result_img = boost.face_detection(original_img, 100, scale_step=10, saved_patches_dir=None)
    cv2.imwrite('Result_img3_afterhnm%s.png' % boosting_type, result_img)

if __name__ == '__main__':
    main()
