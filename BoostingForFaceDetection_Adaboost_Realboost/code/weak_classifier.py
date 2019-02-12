from abc import ABC, abstractmethod
import numpy as np
from joblib import Parallel, delayed
import os


class Weak_Classifier(ABC):
    # initialize a harr filter with the positive and negative rects
    # rects are in the form of [x1, y1, x2, y2] 0-index
    def __init__(self, id, plus_rects, minus_rects, num_bins):
        self.id = id
        self.plus_rects = plus_rects
        self.minus_rects = minus_rects
        self.num_bins = num_bins
        self.activations = None

    # take in one integrated image and return the value after applying the image
    # integrated_image is a 2D np array
    # return value is the number BEFORE polarity is applied
    def apply_filter2image(self, integrated_image):
        pos = 0
        for rect in self.plus_rects:
            rect = [int(n) for n in rect]
            pos += integrated_image[rect[3], rect[2]] \
                   + (0 if rect[1] == 0 or rect[0] == 0 else integrated_image[rect[1] - 1, rect[0] - 1]) \
                   - (0 if rect[1] == 0 else integrated_image[rect[1] - 1, rect[2]]) \
                   - (0 if rect[0] == 0 else integrated_image[rect[3], rect[0] - 1])
        neg = 0
        for rect in self.minus_rects:
            rect = [int(n) for n in rect]
            neg += integrated_image[rect[3], rect[2]] \
                   + (0 if rect[1] == 0 or rect[0] == 0 else integrated_image[rect[1] - 1, rect[0] - 1]) \
                   - (0 if rect[1] == 0 else integrated_image[rect[1] - 1, rect[2]]) \
                   - (0 if rect[0] == 0 else integrated_image[rect[3], rect[0] - 1])
        return pos - neg

    # take in a list of integrated images and calculate values for each image
    # integrated images are passed in as a 3-D np-array
    # calculate activations for all images BEFORE polarity is applied
    # only need to be called once
    def apply_filter(self, integrated_images):
        values = []
        for idx in range(integrated_images.shape[0]):
            values.append(self.apply_filter2image(integrated_images[idx, ...]))
        if (self.id + 1) % 100 == 0:
            print('Weak Classifier No. %d has finished applying' % (self.id + 1))
        return values

    # using this function to compute the error of
    # applying this weak classifier to the dataset given current weights
    # return the error and potentially other identifier of this weak classifier
    # detailed implementation is up you and depends
    # your implementation of Boosting_Classifier.train()
    @abstractmethod
    def calc_error(self, weights, labels):
        pass

    @abstractmethod
    def predict_image(self, integrated_image):
        pass


class Ada_Weak_Classifier(Weak_Classifier):

    def __init__(self, id, plus_rects, minus_rects, num_bins):
        super().__init__(id, plus_rects, minus_rects, num_bins)
        self.polarity = None  # calc
        self.threshold = None  # calc

    def calculate_threshold_and_polarity(self, data, labels, weights_data_points, wc_activations_entire, num_cores,
                                         hnm=False):
        wc_activations = wc_activations_entire[self.id, :]

        # if (num_cores == 1):
        # save_dir_indices = './sorted_filters/wc_sortedindex' + str(self.id) + 'subset.npy'
        # save_dir_features = './sorted_filters/wc_sortedfeatures' + str(self.id) + 'subset.npy'
        # else:
        # save_dir_indices = './sorted_filters/wc_sortedindex' + str(self.id) + '.npy'
        # save_dir_features = './sorted_filters/wc_sortedfeatures' + str(self.id) + '.npy'

        if (num_cores == 1):
            if(hnm == False):
                save_dir_indices = './sorted_filters/wc_index' + str(self.id) + 'subset.npy'
                save_dir_features = './sorted_filters/wc_features' + str(self.id) + 'subset.npy'
            else:
                save_dir_indices = './sorted_filters/wc_index_hnm' + str(self.id) + 'subset.npy'
                save_dir_features = './sorted_filters/wc_features_hnm' + str(self.id) + 'subset.npy'
        else:
            if (hnm == False):
                save_dir_indices = './sorted_filters/wc_index' + str(self.id) + '.npy'
                save_dir_features = './sorted_filters/wc_features' + str(self.id) + '.npy'
            else:
                save_dir_indices = './sorted_filters/wc_index_hnm' + str(self.id) + '.npy'
                save_dir_features = './sorted_filters/wc_features_hnm' + str(self.id) + '.npy'


        # print("Entered calculate_threshold_and_polarity")
        if save_dir_indices is not None and os.path.exists(save_dir_indices):
            #print('[Find cached sorted_indicies, %s loading...]' % save_dir_indices)
            sorted_indicies = np.load(save_dir_indices)
        else:
            sorted_indicies = np.argsort(wc_activations)

            if save_dir_indices is not None:
                #print('Writing sorted indices of activations to disk...')
                np.save(save_dir_indices, sorted_indicies)
            # print('[Saved calculated sorted_indicies to %s]' % save_dir_indices)
        if save_dir_features is not None and os.path.exists(save_dir_features):
            #print('[Find cached sorted_indicies, %s loading...]' % save_dir_features)
            sorted_features = np.load(save_dir_features)
        else:
            sorted_features = np.sort(wc_activations)
            if save_dir_features is not None:
                #print('Writing sorted activations to disk...')
                np.save(save_dir_features, sorted_features)
            # print('[Saved calculated sorted_features to %s]' % save_dir_features)

        searchval = 1
        indicesOfFaces = np.where(labels == searchval)[0]
        searchval = -1
        indicesOfNonFaces = np.where(labels == searchval)[0]
        weightsOfFaces = np.array(list(weights_data_points[indicesOfFaces]))
        weightsOfNonFaces = np.array(list(weights_data_points[indicesOfNonFaces]))
        afs = np.sum(np.array(weightsOfFaces))
        abg = np.sum(np.array(weightsOfNonFaces))
        fs = 0
        bg = 0
        besterror = 99999999999999999999
        for i in range(0, data.shape[0]):
            idx = sorted_indicies[i]
            curent_label = labels[idx]
            if (curent_label == 1):
                #fs = fs + 1
                 fs = fs+ weights_data_points[idx]
            else:
                #bg = bg + 1
                bg = bg + weights_data_points[idx]
            left = bg + (afs - fs)
            right = fs + (abg - bg)
            error = min(left, right)
            # print("Debug:error ",error )
            if (left < right):
                polarity1 = -1
            else:
                polarity1 = 1

            if (error < besterror):
                besterror = error
                bestPolarity = polarity1
                bestThreshold = sorted_features[i]  # check i or idx

        self.polarity = bestPolarity
        self.threshold = bestThreshold
        polarity_threshold = np.empty(shape=2)
        polarity_threshold[0] = (bestPolarity)
        polarity_threshold[1] = bestThreshold
        return polarity_threshold

    # return np.array(bestThreshold), np.array(bestPolarity)

    def calc_error(self, weights, labels, data, wc_activations_entire, polarity, threshold):

        wc_activation = wc_activations_entire[self.id, :]

        # bestPolarity, bestThreshold =  self.calculate_threshold_and_polarity(data,labels,weights,wc_activation,save_dir_indices,save_dir_features )
        # self.calculate_threshold_and_polarity(data, labels, weights, wc_activation,
        # save_dir_indices, save_dir_features)

        # cal error value
        final_error = 0
        for i in range(0, data.shape[0]):
            # predicted_val = self.predict_image(data[i])
            # predicted_val = self.predict_image(wc_activation[i],bestPolarity,bestThreshold)
            predicted_val = self.predict_image(wc_activation[i], polarity, threshold)
            # result = predicted_val * labels[i]
            if (predicted_val != labels[i]):
                result = 1
            else:
                result = 0
            final_error = final_error + weights[i] * result

        if (final_error > 0.5):
            final_error = 1 - final_error
            polarity = polarity * -1

        error_polarity = np.empty(shape=2)
        error_polarity[0] = (final_error)
        error_polarity[1] = polarity
        return error_polarity

    # def predict_image(self, wc_activation):
    def predict_image(self, wc_activation, bestPolarity, bestThreshold):
        # value = self.apply_filter2image(integrated_image)
        value = wc_activation
        # print("Debug:value is ", value)
        polarity = bestPolarity
        threshold = bestThreshold

        return polarity * np.sign(value - threshold)

    def predict_image_test(self, integrated_image, bestPolarity, bestThreshold):
        value = self.apply_filter2image(integrated_image)
        polarity = bestPolarity
        threshold = bestThreshold
        return polarity * np.sign(value - threshold)


class Real_Weak_Classifier(Weak_Classifier):
    def __init__(self, id, plus_rects, minus_rects, num_bins):
        super().__init__(id, plus_rects, minus_rects, num_bins)
        # self.thresholds = [] #this is different from threshold in ada_weak_classifier, think about it
        self.bin_pqs = None
        self.train_assignment = []

    def calculate_thresholds(self, No_Of_bins, wc_activations):
        # print("shape of wc_activations_entire", wc_activations_entire.shape)
        wc_activations_sorted = np.sort(wc_activations)
        step_size = wc_activations.shape[0] / No_Of_bins
        thresholds = []
        j = 0
        for i in range(0, No_Of_bins):
            current_threshold = wc_activations_sorted[j]
            thresholds.append(current_threshold)
            j = int(j + step_size)

        if (j <= wc_activations.shape[0]):
            thresholds[i] = wc_activations_sorted[wc_activations.shape[0] - 1]
        print("shape of self.thresholds", len(thresholds))
        return np.array(thresholds)

    def calculate_images_for_all_bins(self, No_Of_bins, wc_activations, labels, thresholds):
        # self.train_assignment = np.ndarray(shape=(No_Of_bins, 2))
        self.train_assignment = np.empty(shape=((No_Of_bins, 2)), dtype=np.object)
        weights_assignment = np.empty(shape=((No_Of_bins, 2)), dtype=np.object)
        bins_of_all_images = list()

        for i in range(0, No_Of_bins):
            self.train_assignment[i] = [list(), list()]

        for i in range(0, wc_activations.shape[0]):

            bin_idx = np.sum(thresholds < wc_activations[i])
            bins_of_all_images.append(bin_idx)
            if (labels[i] == 1):
                self.train_assignment[bin_idx, 0].append(i)

            else:
                self.train_assignment[bin_idx, 1].append(i)

        indices_bins = []
        indices_bins.append(self.train_assignment)
        indices_bins.append(bins_of_all_images)
        return indices_bins

    def calc_error(self, weights, labels, No_Of_bins):
        #implemented this function in boosting_classifier
        return 0

    def predict_image(self, integrated_image):
        value = self.apply_filter2image(integrated_image)
        bin_idx = np.sum(self.thresholds < value)
        return 0.5 * np.log(self.bin_pqs[0, bin_idx] / self.bin_pqs[1, bin_idx])


def main():
    plus_rects = [(1, 2, 3, 4)]
    minus_rects = [(4, 5, 6, 7)]
    num_bins = 50
    ada_hf = Ada_Weak_Classifier(0, plus_rects, minus_rects, num_bins)
    real_hf = Real_Weak_Classifier(0, plus_rects, minus_rects, num_bins)


if __name__ == '__main__':
    main()


class mylist:

    def __init__(self, l):
        self.l = l

    def __repr__(self):
        return repr(self.l)

    def append(self, x):
        self.l.append(x)