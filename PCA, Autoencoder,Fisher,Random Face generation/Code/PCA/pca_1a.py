import numpy as np
import os
import cv2
from skimage import io
import skimage
from skimage.color import rgb2grey
import PIL
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage import io, exposure, img_as_uint, img_as_float
import scipy
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

l = 1000
mn = 128 * 128

dirOfTrainingImages ="C:\\Users\\vaish\\OneDrive\\Desktop\\UCLA\\Prml\\Project_1\\Project_1\\images"
dirOfTestImages ="C:\\Users\\vaish\\OneDrive\\Desktop\\UCLA\\Prml\\Project_1\\Project_1\\Test_Images"
dirOfOutputsEigen  = "Outputs\Part1\Part a\EigenFaces"
dirOfOutputsRecons  = "Outputs\Part1\Part a\ReconstrucedImages"

def __FindMeanOfMatrix__(X, noOfImages):
    mean_img_col = np.mean(X, axis=0)
    return mean_img_col

def __TransformedMatrixWithMeanSubtractedAndFindEigen__( training_dir, noOfImages, mn, Eigen_Number, plot_eigen =False):

    L = np.empty(shape=(noOfImages,mn ))
    HValues = np.empty(shape=(noOfImages, mn))
    SValues = np.empty(shape=(noOfImages, mn))
    VValues = np.empty(shape=(noOfImages, mn))

    cur_img = 0
    print("reading input train images")
    for filename in os.listdir(training_dir):
        filename = os.path.join(training_dir, filename)
        img = io.imread(filename)
        inputImage =  skimage.color.rgb2hsv(img)
        v = inputImage[:, :, 2]
        h = inputImage[:, :, 0]
        s = inputImage[:, :, 1]

        img_col_h = np.array(h).flatten()  # flatten the 2d image into 1d
        HValues[cur_img, :] = img_col_h[:]

        img_col_s = np.array(s, dtype='float64').flatten()  # flatten the 2d image into 1d
        SValues[cur_img, :] = img_col_s[:]

        img_col_v = np.array(v, dtype='float64').flatten()  # flatten the 2d image into 1d

        L[cur_img, : ] = img_col_v[:]  # set the cur_img-th column to the current training image
        VValues[cur_img, :] = img_col_v[:]

        cur_img += 1

    print("Calculating the mean of the input images and subtracting from each image..")
    mean_img_col = __FindMeanOfMatrix__(L, noOfImages)
    for j in range(0, noOfImages):  # subtract from all training images
     L[j,: ] -= mean_img_col[:]

    print("Computing the eigen-vectors for eigen number: ", Eigen_Number)
    n_components = Eigen_Number
    pca = PCA(n_components=n_components, svd_solver='randomized',
              whiten=True).fit(L)

    eigenfaces = pca.components_.reshape((n_components, 128, 128))
    print("shape of eigen vectors extracted", eigenfaces.shape)

    eigenFaces_vectors = np.empty(shape=(n_components, 128*128))
    eigenFaces_vectors_display = np.empty(shape=(n_components, 128 * 128))
    for i in range(0, n_components):

        image_array = eigenfaces[i, ]
        image_array = image_array.flatten()
        eigenFaces_vectors_display[i, :] = eigenFaces_vectors[i,:] + mean_img_col
        scipy.misc.imsave(os.path.join(dirOfOutputsEigen,
                         str(i) + ".png"), eigenFaces_vectors_display[i, :].reshape(128,128))
        eigenFaces_vectors[i,:] = image_array

    def plot_filter_grid(units):
         filters = units.shape[0]
         plt.figure(0, figsize=(20, 12))
         n_columns = 8
         n_rows = math.ceil(filters / n_columns) + 1
         for i in range(0, n_components):
            plt.subplot(n_rows, n_columns, i + 1)
            plt.title('Eigen Face ' + str(i))
            plt.axis('off')
            plt.imshow(units[i, ].reshape(128,128), interpolation='nearest', cmap = 'gray')

    if(plot_eigen == True ):
      plot_filter_grid(eigenFaces_vectors_display[0:10,])
      plt.savefig('First 10 eigen-faces.png', bbox_inches='tight')
      plt.close()
      #plt.show()

    #projecting the train images into the eigen space
    X_train_pca = pca.transform(L)

    #recontructing the train images back to original..
    reconstructed_image = np.array(np.dot( X_train_pca,eigenFaces_vectors))
    for j in range(0, noOfImages):  # subtract from all training images
        reconstructed_image[j, :] = np.array(reconstructed_image[j, :])
        reconstructed_image[j, :] += mean_img_col[:]
        final = np.dstack((HValues[j, :].reshape(128, 128), SValues[j, :].reshape(128, 128),
                           reconstructed_image[j, :].reshape(128, 128)))

    return pca, eigenFaces_vectors, mean_img_col


def __TestProjectionAndReconstruction__(pca, eigenFaces_vectors, mean, test_dir, noOfImages, mn,Eigen_Number,plot_reconstruct = False):
    L = np.empty(shape=(noOfImages, mn))
    HValues = np.empty(shape=(noOfImages, mn))
    SValues = np.empty(shape=(noOfImages, mn))
    VValues = np.empty(shape=(noOfImages, mn))
    original_Test_images = np.empty(shape=(200, 128, 128, 3), dtype='float64')
    cur_img = 0

    print("reading the test images..")
    for filename in os.listdir(test_dir):
        filename = os.path.join(test_dir, filename)
        img = io.imread(filename)
        inputImage = skimage.color.rgb2hsv(img)
        original_Test_images[cur_img,] = skimage.color.hsv2rgb(inputImage)
        v = inputImage[:, :, 2]
        h = inputImage[:, :, 0]
        s = inputImage[:, :, 1]

        img_col_h = np.array(h).flatten()  # flatten the 2d image into 1d
        HValues[cur_img, :] = img_col_h[:]

        img_col_s = np.array(s, dtype='float64').flatten()  # flatten the 2d image into 1d
        SValues[cur_img, :] = img_col_s[:]

        img_col_v = np.array(v, dtype='float64').flatten()  # flatten the 2d image into 1d
        L[cur_img, :] = img_col_v[:]  # set the cur_img-th column to the current training image
        VValues[cur_img, :] = img_col_v[:]

        cur_img += 1

    for j in range(0, noOfImages):  # subtract from all training images
        L[j, :] -= mean[:]

    print("projecting the test images into the eigen-space")
    X_test_pca = pca.transform(L)
    final_reconstructed_Test_images = np.empty(shape=(200, 128, 128, 3), dtype='float64')
    reconstructed_image = np.array(np.dot(X_test_pca, eigenFaces_vectors))
    total_reconstruction_error = 0

    for j in range(0, noOfImages):  # subtract from all training images

        reconstructed_image[j, :] = np.array(reconstructed_image[j, :])
        reconstructed_image[j, :] += mean[:]
        total_reconstruction_error = total_reconstruction_error +  np.sum(np.square(reconstructed_image[j, :] - VValues[j, :]))
        final = np.dstack((HValues[j, :].reshape(128, 128), SValues[j, :].reshape(128, 128),
                           reconstructed_image[j, :].reshape(128, 128)))

        final_rgb_image = skimage.color.hsv2rgb(final)
        final_reconstructed_Test_images[j,] = final_rgb_image

    def plot_filter_grid(reconstructed, original, number_to_display):
         plt.figure(2, figsize=(20, 12))
         n_columns = 8
         n_rows = math.ceil(number_to_display / n_columns) + 1
         for i in range(0, number_to_display):
            plt.subplot(n_rows, n_columns, i + 1)
            plt.title('reconstructed image ' + str(i))
            plt.axis('off')
            plt.imshow(reconstructed[i, ].reshape(128,128,3), interpolation='nearest')

         j= 0
         for i in range(10, 20):
            plt.subplot(n_rows, n_columns, i + 1)
            plt.title('original image ' + str(i))
            plt.axis('off')
            plt.imshow(original[j,].reshape(128, 128,3), interpolation='nearest')
            j= j+1

    if(plot_reconstruct == True ):
      plot_filter_grid( final_reconstructed_Test_images[0:10,], original_Test_images[0:10,],10)
      plt.savefig('First_10_reconstructed.png', bbox_inches='tight')
      plt.close()
    total_reconstruction_error = total_reconstruction_error/(128*128*200)
    print("eigen-faces K =" + str(Eigen_Number) + ", total reconstruction error " + str(total_reconstruction_error))
    return total_reconstruction_error

pca, eigenFaces_vectors , mean= __TransformedMatrixWithMeanSubtractedAndFindEigen__( dirOfTrainingImages, 800, 128 * 128,10, True)
error = __TestProjectionAndReconstruction__(pca, eigenFaces_vectors, mean, dirOfTestImages, 200, 128 * 128, 10, True)

total_reconstruction_error = np.empty(shape=(11))

pca, eigenFaces_vectors, mean = __TransformedMatrixWithMeanSubtractedAndFindEigen__(
    dirOfTrainingImages, 800, 128 * 128, 1)
total_reconstruction_error[0] = __TestProjectionAndReconstruction__(pca, eigenFaces_vectors, mean, dirOfTestImages,
                                                                    200, 128 * 128, 1)

j= 1
k= 0
for i in range(1, 11):
    k = k + 5
    pca, eigenFaces_vectors , mean= __TransformedMatrixWithMeanSubtractedAndFindEigen__( dirOfTrainingImages, 800, 128 * 128,k, False)
    total_reconstruction_error[j] = __TestProjectionAndReconstruction__( pca, eigenFaces_vectors,mean, dirOfTestImages, 200, 128 * 128,k, False)
    j=j+1


print(total_reconstruction_error)
eigenNumber= [1,5,10,15,20,25,30,35,40,45,50]

plt.figure(1)
plt.plot(eigenNumber, total_reconstruction_error, label='linear')
plt.xlabel('eigenNumber')
plt.ylabel('total_reconstruction_error')
plt.title('total_reconstruction_error vs eigenNumber')

plt.legend()

plt.savefig('total_reconstruction_error_vs_eigenNumber.png', dpi = 'figure')
plt.close()

