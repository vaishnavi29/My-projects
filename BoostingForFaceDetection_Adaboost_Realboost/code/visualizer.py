import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

class Visualizer:
    def __init__(self, histogram_intervals, top_wc_intervals):
        self.histogram_intervals = histogram_intervals
        self.top_wc_intervals = top_wc_intervals
        self.weak_classifier_accuracies = {}
        self.strong_classifier_scores = []
        self.strong_classifier_errors = {}
        self.labels = None

    def draw_histograms(self):
        for t in (self.strong_classifier_scores):
            scores = self.strong_classifier_scores[t]
            # scores = self.strong_classifier_scores
            pos_scores = [scores[idx] for idx, label in enumerate(self.labels) if label == 1]
            neg_scores = [scores[idx] for idx, label in enumerate(self.labels) if label == -1]

            bins \
                = np.linspace(np.min(scores), np.max(scores), 100)

            plt.figure()
            plt.hist(pos_scores, bins, alpha=0.5, label='Faces')
            plt.hist(neg_scores, bins, alpha=0.5, label='Non-Faces')
            plt.legend(loc='upper right')
            plt.title('Using %d Weak Classifiers' % t)
            plt.savefig('histogram_%d_Ada.png' % t)

    def draw_rocs(self):
        plt.figure()
        for t in self.strong_classifier_scores:
            scores = self.strong_classifier_scores[t]
            fpr, tpr, _ = roc_curve(self.labels, scores)
            plt.plot(fpr, tpr, label='No. %d Weak Classifiers' % t)
        plt.legend(loc='lower right')
        plt.title('ROC Curve')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig('ROC Curve')

    def draw_wc_accuracies(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.6f'))
        for t in self.weak_classifier_accuracies:
            accuracies = self.weak_classifier_accuracies[t]
            print("t",t)
            ax.plot(accuracies, label='After %d Selection' % t)
        plt.ylabel('Error')
        plt.xlabel('Weak Classifiers')
        plt.title('Top 1000 Weak Classifier Errors')
        plt.legend(loc='center right')
        plt.savefig('Weak Classifier Errors')

    def draw_sc_training_errors(self):

        fig, ax = plt.subplots(nrows=10, ncols=2)
        colors = ['r', 'g', 'b', 'o']
        i = 1
        # for t in self.strong_classifier_errors:
        # errors = self.strong_classifier_errors[t]
        plt.subplot(3, 1, 1)
        plt.plot(self.strong_classifier_errors[10], 'r', label='T = 10')
        plt.legend()
        i = i + 1

        plt.subplot(3, 1, 2)
        plt.plot(self.strong_classifier_errors[50], 'b', label='T = 50')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(self.strong_classifier_errors[100], 'g', label='T = 100')
        plt.legend()

        plt.savefig('training error strong classifier over number of steps T')

if __name__ == '__main__':
    main()
