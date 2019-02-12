import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle
import matplotlib.pyplot as plt
import cv2
from weak_classifier import Ada_Weak_Classifier, Real_Weak_Classifier
from im_process import image2patches, nms, normalize
import time

class Boosting_Classifier:
    def __init__(self, haar_filters, data, labels, num_chosen_wc, num_bins, visualizer, num_cores, style):
        self.filters = haar_filters
        self.data = data

        self.labels = labels
        self.num_chosen_wc = num_chosen_wc
        self.num_bins = num_bins
        self.visualizer = visualizer
        self.num_cores = num_cores
        self.style = style
        self.chosen_wcs = []
        if style == 'Ada':
            self.weak_classifiers = [Ada_Weak_Classifier(i, filt[0], filt[1], self.num_bins) \
                                     for i, filt in enumerate(self.filters)]
        elif style == 'Real':
            self.weak_classifiers = [Real_Weak_Classifier(i, filt[0], filt[1], self.num_bins) \
                                     for i, filt in enumerate(self.filters)]

    def calculate_training_activations(self, save_dir=None, load_dir=None):
        print('Calcuate activations for %d weak classifiers, using %d imags.' % (
        len(self.weak_classifiers), self.data.shape[0]))
        print('shape of data is', self.data.shape)
        if load_dir is not None and os.path.exists(load_dir):
            print('[Find cached activations, %s loading...]' % load_dir)
            wc_activations = np.load(load_dir)
        else:
            if self.num_cores == 1:
                wc_activations = [wc.apply_filter(self.data) for wc in self.weak_classifiers]
            else:
                wc_activations = Parallel(n_jobs=self.num_cores)(
                    delayed(wc.apply_filter)(self.data) for wc in self.weak_classifiers)
            wc_activations = np.array(wc_activations)
            if save_dir is not None:
                print('Writing results to disk...')
                np.save(save_dir, wc_activations)
                print('[Saved calculated activations to %s]' % save_dir)
        print("wc.activations shape is", wc_activations.shape)
        for wc in self.weak_classifiers:
            wc.activations = wc_activations[wc.id, :]

        print("wc.activations shape is", wc.activations.shape)
        return wc_activations

##############################AdaBoost methods

    #Training for adaboost
    def train(self, wc_activations, T, save_dir_chosen_wcs=None, save_dir_sc_error=None, save_dir_current_weights=None,
              hnm=False ):

        current_weights = np.zeros(shape=self.data.shape[0])
        flag_to_be_run = True
        to_be_run_after_load_cache = False
        T_To_Run = T

        predicted_values_for_this_classfier = np.zeros(shape=self.data.shape[0])
        initial_wt = 1 / self.data.shape[0]
        self.chosen_wcs_ids = np.zeros(shape=T)
        self.chosen_wcs_errors = []
        self.training_errors_sc = []
        for i in range(0, self.data.shape[0]):
            current_weights[i] = initial_wt


        # assign initial wieght
        if save_dir_chosen_wcs is not None and os.path.exists(save_dir_chosen_wcs):
            print('[Find cached wc_errors, %s loading...]' % save_dir_chosen_wcs)
            self.chosen_wcs = np.load(save_dir_chosen_wcs)
            self.training_errors_sc = np.load(save_dir_sc_error)
            current_weights = np.load(save_dir_current_weights)

            if (len(self.chosen_wcs) != T and len(self.chosen_wcs) < T):
                flag_to_be_run = True
                to_be_run_after_load_cache = True
                T_To_Run = T - len(self.chosen_wcs)
                print(T_To_Run)
            else:
                flag_to_be_run = False

        if (flag_to_be_run):
            for i in range(0, T_To_Run):
                start = time.time()
                if self.num_cores == 1:
                    wc_polarities_wc_thresholds = [
                        wc.calculate_threshold_and_polarity(self.data, self.labels, current_weights, wc_activations,
                                                            self.num_cores, hnm) for wc in self.weak_classifiers]
                else:
                    wc_polarities_wc_thresholds = Parallel(n_jobs=self.num_cores)(
                        delayed(wc.calculate_threshold_and_polarity)(self.data, self.labels, current_weights,
                                                                     wc_activations, self.num_cores,hnm) for wc
                        in self.weak_classifiers)

                wc_polarities_wc_thresholds = np.array(wc_polarities_wc_thresholds)
                wc_polarities = wc_polarities_wc_thresholds[:, 0]
                wc_thresholds = wc_polarities_wc_thresholds[:, 1]

                if self.num_cores == 1:
                    wc_errors_polarities = [
                        wc.calc_error(current_weights, self.labels, self.data, wc_activations, wc_polarities[wc.id],
                                      wc_thresholds[wc.id]) for wc in self.weak_classifiers]
                else:
                    wc_errors_polarities = Parallel(n_jobs=self.num_cores)(
                        delayed(wc.calc_error)(current_weights, self.labels, self.data, wc_activations,
                                               wc_polarities[wc.id], wc_thresholds[wc.id]) for wc in
                        self.weak_classifiers)  # change

                # choose classifer with min error
                wc_errors_polarities = np.array(wc_errors_polarities)
                wc_errors = wc_errors_polarities[:, 0]
                wc_polarities = wc_errors_polarities[:, 1]
                wc_index = np.argmin(wc_errors)
                min_error_of_wc = np.min(wc_errors)
                print("min_error_of_wc", min_error_of_wc)
                chosen_classifier = self.weak_classifiers[wc_index]
                print("wc_index", wc_index)

                # assign wght for new classifer (assign to self.chosen_wcs)
                alpha_t = self.calculate_alpha(min_error_of_wc)
                self.chosen_wcs.append((alpha_t, chosen_classifier, wc_polarities[chosen_classifier.id],
                                        wc_thresholds[chosen_classifier.id]))
                self.chosen_wcs_errors.append(min_error_of_wc)
                for idx in range(self.data.shape[0]):
                    predicted_values_for_this_classfier[idx] = (
                        chosen_classifier.predict_image(wc_activations[chosen_classifier.id, idx],
                                                        wc_polarities[chosen_classifier.id],
                                                        wc_thresholds[chosen_classifier.id]))

                current_weights = self.Update_weights(current_weights, alpha_t, self.labels,
                                                      predicted_values_for_this_classfier)


                self.training_errors_sc.append(self.sc_training_error(T, wc_activations))

                if save_dir_chosen_wcs is not None:
                    pickle.dump(self.chosen_wcs, open(save_dir_chosen_wcs, 'wb'))

                if save_dir_sc_error is not None:
                    pickle.dump(self.training_errors_sc, open(save_dir_sc_error, 'wb'))

                if save_dir_current_weights is not None:
                    pickle.dump(current_weights, open(save_dir_current_weights, 'wb'))

                end = time.time()
                print('%f seconds for one epoch' % (end - start))

            if ((T == 10 or T == 50 or T == 100 or T == 1)):
                self.draw_wc_errors(wc_errors, T)
                pickle.dump(np.sort(wc_errors[0:1000]), open('wc_errors_100_' + str(T) + '.npy', 'wb'))

        if(T == 100):
            self.draw_top_haarfilters(self.chosen_wcs[0:20])


        self.draw_sc_training_error(T, self.training_errors_sc[0:T])
        if (flag_to_be_run):
            return np.sort(wc_errors[0:1000]), np.array(self.training_errors_sc[0:T])
        else:
            return None, np.array(self.training_errors_sc[0:T])

    def calculate_alpha(self, errorValue):
        a = (1 - errorValue) / errorValue
        b = np.log(a)
        return 0.5 * b

    def Update_weights(self, previous_weights, alpha, labels, predicted_values):
        new_weights = np.empty(shape=self.data.shape[0])

        z = 0
        for i in range(0, self.data.shape[0]):
            a = -1 * predicted_values[i] * alpha * labels[i]
            b = np.exp(a)

            c = previous_weights[i] * b
            z = z + c

        for i in range(0, self.data.shape[0]):
            a = -1 * predicted_values[i] * alpha * labels[i]
            b = np.exp(a)

            c = previous_weights[i] * b
            d = c / z
            new_weights[i] = d
        return new_weights

    def sc_training_error(self, T, wc_activations):
        train_predicts = []
        for idx in range(self.data.shape[0]):
            train_predicts.append(self.sc_function_train(T, wc_activations, idx))  # cyhange to train
        training_error_sc = 1 - np.mean(np.sign(train_predicts) == self.labels)
        print('training error is: ', training_error_sc)
        return training_error_sc

    def sc_function(self, image, T):
        if (len(self.chosen_wcs) < T):
            return np.sum([np.array(
                [alpha * wc.predict_image_test(image, polarity, threshold) for alpha, wc, polarity, threshold in
                 self.chosen_wcs])])
        else:
            return np.sum([np.array(
                [alpha * wc.predict_image_test(image, polarity, threshold) for alpha, wc, polarity, threshold in
                 self.chosen_wcs[0:T]])])

    def sc_function_train(self, T, wc_activations, image_idx):
        result = 0
        if (len(self.chosen_wcs) < T):
            x = len(self.chosen_wcs)
        else:
            x = T
        for i in range(0, x):
            alpha, wc, polarity, threshold = self.chosen_wcs[i]
            wc_activation = wc_activations[wc.id, image_idx]
            result = result + np.array(alpha * wc.predict_image(wc_activation, polarity, threshold))

        return result

    def load_trained_wcs(self, save_dir):
        self.chosen_wcs = pickle.load(open(save_dir, 'rb'))

    def GetScoresAndLabels(self, T, wc_activations):
        # this training accuracy should be the same as your training process,
        ##################################################################################
        train_predicts = []
        for idx in range(self.data.shape[0]):
            # train_predicts.append(self.sc_function(self.data[idx, ...],T))
            train_predicts.append(self.sc_function_train(T, wc_activations, idx))
        predicted_values_train = np.sign(train_predicts)
        print('Check training accuracy is: ', np.mean(np.sign(train_predicts) == self.labels))
        ##################################################################################
        return train_predicts

    def face_detection(self, img,img_color, T, scale_step=1, saved_patches_dir=None):

        scales = 1 / np.linspace(1, 8, scale_step)
        if saved_patches_dir is not None and os.path.exists(saved_patches_dir):
            print('[Find cached saved_patches, %s loading...]' % saved_patches_dir)
            patches_patch_xyxy = np.load(saved_patches_dir)
            print("shape of patches_patch_xyxy", len(patches_patch_xyxy))
            patches = patches_patch_xyxy[0]
            patch_xyxy = patches_patch_xyxy[1]
            print("shape of patches", len(patches))
            print("shape of patch_xyxy", len(patch_xyxy))

        else:
            patches, patch_xyxy = image2patches(scales, img)
            a = np.max(np.array(patch_xyxy)[:, 0])
            print("shape of patches", len(patches))
            print("shape of patch_xyxy", len(patch_xyxy))
            patches_patch_xyxy = (patches, patch_xyxy)
            print("shape of patches_patch_xyxy", len(patches_patch_xyxy))

            if saved_patches_dir is not None:
                pickle.dump(patches_patch_xyxy, open(saved_patches_dir, 'wb'))

        print('Face Detection in Progress ..., total %d patches' % patches.shape[0])
        predicts = [self.sc_function(patch, T) for patch in tqdm(patches)]
        print(np.mean(np.array(predicts) > 0), np.sum(np.array(predicts) > 0))
        threshold = 0.2 * np.max(predicts)
        pos_predicts_xyxy = np.array([patch_xyxy[idx] + [score] for idx, score in enumerate(predicts) if score > threshold])
        if pos_predicts_xyxy.shape[0] == 0:
            return
        xyxy_after_nms = nms(pos_predicts_xyxy, 0.01)
        # xyxy_after_nms = pos_predicts_xyxy
        print('after nms:', xyxy_after_nms.shape[0])
        for idx in range(xyxy_after_nms.shape[0]):
            pred = xyxy_after_nms[idx, :]
            cv2.rectangle(img_color, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), (0, 255, 0),
                          2)  # gree rectangular with line width 3

        return img_color

    def get_hard_negative_patches(self, img, T, scale_step=10, save_dir_wrong_patches=None,save_dir_pos_predcits=None):

        if save_dir_wrong_patches is not None and os.path.exists(save_dir_wrong_patches):
            print('[Find cached save_dir_wrong_patches, %s loading...]' % save_dir_wrong_patches)
            wrong_patches = np.load(save_dir_wrong_patches)

        else:
            print("didnt find cached wrong patches..")

            scales = 1 / np.linspace(1, 8, scale_step)
            patches, patch_xyxy = image2patches(scales, img, toNormalize = False)
            print('Get Hard Negative in Progress ..., total %d patches' % patches.shape[0])
            #a = np.max(np.array(patch_xyxy)[:,0])
            predicts = [self.sc_function(patch, T) for patch in tqdm(patches)]
            # predicts = Parallel(n_jobs=8)(delayed(self.sc_function)(patch,T) for patch in tqdm(patches))
            # error
            predicts = np.array(predicts)
            threshold = 0.2 * np.max(predicts)
            pos_predicts_xyxy = np.array([patch_xyxy[idx] + [score] for idx, score in enumerate(predicts) if score > threshold])
            if pos_predicts_xyxy.shape[0] == 0:
             return

            if save_dir_pos_predcits is not None:
                pickle.dump(pos_predicts_xyxy, open(save_dir_pos_predcits, 'wb'))

            wrong_patches = nms(pos_predicts_xyxy, 0.01)
            print('Number of wrong patches:', wrong_patches.shape)

            #wrong_patches = patches[np.where(predicts > 0), ...]
            if save_dir_wrong_patches is not None:
                pickle.dump(wrong_patches, open(save_dir_wrong_patches, 'wb'))

        return wrong_patches


    def visualize(self):
        self.visualizer.labels = self.labels
        self.visualizer.draw_histograms()
        self.visualizer.draw_rocs()
        self.visualizer.draw_wc_accuracies()

    def draw_sc_training_error(self, T, training_errors_sc):
        plt.figure()
        values = np.zeros(shape=T)
        for i in range(0, T):
            values[i] = i + 1
        # plt.plot(values[i], error, label='After %d Selection' % i)

        plt.plot(values, training_errors_sc, label='After %d Selection' % T)
        plt.ylabel('Training Error')
        plt.xlabel('Number of steps T')
        plt.title('Training error of strong classifier vs Number of steps')
        plt.legend(loc='upper right')
        plt.savefig('Training error strong classifier_ada' + str(T))

    def draw_wc_errors(self, wc_errors, T):
        values = np.zeros(shape=1000)
        plt.figure()

        for i in range(0, 1000):
            values[i] = i + 1

        print("np.sort(wc_errors[0:1000])",np.sort(wc_errors[0:1000]))
        top_errors= np.sort(wc_errors[0:1000])

        plt.plot(top_errors, label='For T = %d ' % T)
        plt.ylabel('Error')
        plt.xlabel('Weak Classifiers')

        plt.title('Top 1000 Weak Classifier Errors')
        plt.legend(loc='upper right')
        plt.savefig('Weak Classifier Errors alone' + str(T))

    def draw_top_haarfilters(self, chosen_wcs):
        print("Plotting top 20 Haar filters..")
        for i in range(0, len(chosen_wcs)):
            img = np.zeros([160, 160, 3], dtype=np.uint8)
            img.fill(0)
            wc = chosen_wcs[i]
            wc = (wc[1])
            plus_rects = np.array(wc.plus_rects[0]) * 10

            minus_rects = np.array(wc.minus_rects[0]) *10

            cv2.rectangle(img, (int(plus_rects[0]), int(plus_rects[1])), (int(plus_rects[2]), int(plus_rects[3])),
                          (255, 255, 0), 1)
            if (plus_rects[1] == minus_rects[1]):
                cv2.rectangle(img, (int(plus_rects[2]), int(minus_rects[1])),
                              (int(plus_rects[2] + minus_rects[2] - minus_rects[0]), int(minus_rects[3])),
                              (0, 0, 255), 1)
            elif (plus_rects[2] == minus_rects[2]):
                cv2.rectangle(img, (int(minus_rects[0]), int(plus_rects[3])),
                              (int(minus_rects[2]), int(plus_rects[3] + minus_rects[3] - minus_rects[1])),
                              (0, 0, 255), 1)
            cv2.imwrite('topHaarFilter_' + str(i) + '.png', img)

 ##############################RealBoost methods
    #Real boost training
    def train_real_boost(self, wc_activations, T, No_Of_bins, save_dir_chosen_wcs=None,
                         save_dir_sc_error=None, save_dir_current_weights=None):
        predicted_values_of_all_wcs = []
        sc_scores = []
        if self.num_cores == 1:
            wc_thresholds = [
                wc.calculate_thresholds(No_Of_bins, wc_activations[wc.id]) for wc in self.weak_classifiers[0:T]]
        else:
            wc_thresholds = Parallel(n_jobs=self.num_cores)(
                delayed(wc.calculate_thresholds)(No_Of_bins, wc_activations[wc.id]) for wc in
                self.weak_classifiers[0:T])

        if self.num_cores == 1:
            indices_bins = [
                wc.calculate_images_for_all_bins(No_Of_bins, wc_activations[wc.id], self.labels, wc_thresholds[i])
                for wc, i in zip(self.weak_classifiers[0:T], range(T))]
        else:
            indices_bins = Parallel(n_jobs=self.num_cores)(
                delayed(wc.calculate_images_for_all_bins)(No_Of_bins, wc_activations[wc.id], self.labels,
                                                          wc_thresholds[i])
                for wc, i in zip(self.weak_classifiers[0:T], range(T)))

        indices_bins = np.array(indices_bins)
        wc_pq_indices = indices_bins[:, 0]
        bins_of_all_images = indices_bins[:, 1]

        current_weights = np.zeros(shape=self.data.shape[0])
        initial_wt = 1 / self.data.shape[0]

        for i in range(0, self.data.shape[0]):
            current_weights[i] = initial_wt

        for i in range(0, T):
            # wc_chosen = chosen_wcs[i]
            print("T=", i)
            wc_pq = wc_pq_indices[i]
            bins_of_all_images_for_this_wc = bins_of_all_images[i]
            if self.num_cores == 1:
                ps_qs_of_all_bins = [
                    self.calculate_p_And_q_for_bin(bin[0], bin[1], current_weights) for bin in wc_pq]
            else:
                ps_qs_of_all_bins = Parallel(n_jobs=self.num_cores)(
                    delayed(self.calculate_p_And_q_for_bin)(bin[0], bin[1], current_weights) for bin in wc_pq)

            # ps_of_all_bins = ps_qs_of_all_bins[:,0]
            # qs_of_all_bins = ps_qs_of_all_bins[:,1]
            if self.num_cores == 1:
                h_of_bins = [
                    self.calculate_h_of_bin(pq[0], pq[1]) for pq in ps_qs_of_all_bins]
            else:
                h_of_bins = Parallel(n_jobs=self.num_cores)(
                    delayed(self.calculate_h_of_bin)(pq[0], pq[1]) for pq in ps_qs_of_all_bins)

            predicted_values = self.calculate_h_of_all_x(bins_of_all_images_for_this_wc, h_of_bins)
            predicted_values_of_all_wcs.append(predicted_values)
            # predicted values of all images for this classifier, pts of all bs , qts of all bs, ht of all bs
            current_weights = self.Update_weights_RealBoost(current_weights, self.labels, predicted_values,
                                                            ps_qs_of_all_bins, h_of_bins, No_Of_bins)

        predicted_values_of_all_wcs = np.array(predicted_values_of_all_wcs)
        for i in range(0, self.data.shape[0]):
            predicted_values_for_this_datapoint = predicted_values_of_all_wcs[:, i]
            sc_scores.append(np.sum(predicted_values_for_this_datapoint))

        return sc_scores

    def Update_weights_RealBoost(self, previous_weights, labels, predicted_values, pq_of_all_bins, h_of_bins,
                                 num_bins):
        new_weights = np.empty(shape=self.data.shape[0])

        # print("self.data.shape[0]",self.data.shape)
        # print("predicted_values.shape[0]", predicted_values.shape)
        # print("labels.shape[0]", labels.shape)
        z = 0
        for i in range(0, num_bins):
            # h_of_bin = 0.5 * np.log(pq_of_all_bins[i,0] * qs_of_this_qc[i])
            z = z + (pq_of_all_bins[i])[0] * np.exp(-1 * h_of_bins[i]) + (pq_of_all_bins[i])[1] * np.exp(
                1 * h_of_bins[i])

        for i in range(0, self.data.shape[0]):
            a = -1 * predicted_values[i] * labels[i]  # predicted_value keeps changing
            b = np.exp(a)

            c = previous_weights[i] * b
            d = c / z
            new_weights[i] = d
        return new_weights

    def calculate_p_And_q_for_bin(self, p_indices, q_indices, current_weights):
        p = 0
        q = 0
        for i in range(0, len(p_indices)):
            p = p + current_weights[p_indices[i]]
        for i in range(0, len(q_indices)):
            q = q + current_weights[q_indices[i]]

        p_q = []
        p_q.append(p)
        p_q.append(q)
        return p_q

    def calculate_h_of_all_x(self, bins_of_all_images, h_of_bins):
        h_of_all_x = []
        for i in range(0, len(bins_of_all_images)):
            # a = 0.5 * np.log(ps_of_all_bins[bins_of_all_images[i]]*qs_of_all_bins[bins_of_all_images[i]])
            a = h_of_bins[bins_of_all_images[i]]
            h_of_all_x.append(a)

        return h_of_all_x

    def calculate_h_of_bin(self, ps_of_bin, qs_of_bin):
        if ps_of_bin == 0:
            ps_of_bin = 0.000000000000000000001
        if qs_of_bin == 0:
            qs_of_bin = 0.000000000000000000001
        a = 0.5 * np.log(ps_of_bin / qs_of_bin)
        # if a< 0:
        # print("neg!")
        return 0.5 * np.log(ps_of_bin / qs_of_bin)

