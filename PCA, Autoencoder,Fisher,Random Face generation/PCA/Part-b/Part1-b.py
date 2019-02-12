import numpy as np
import os
import scipy.io
from skimage import io
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math
import skimage

def __FindMeanOfMatrix__(X, noOfImages):
    mean_img_col = np.mean(X, axis=0)
    return mean_img_col


l = 1000
mn = 128 * 128
dirOfTrainingLandmarks = "C:\\Users\\vaish\\OneDrive\\Desktop\\UCLA\\Prml\\Project_1\\Project_1\\Landmarks"
dirOfTrainingImages ="C:\\Users\\vaish\\OneDrive\\Desktop\\UCLA\\Prml\\Project_1\\Project_1\\images"
dirOfTestLandmarks = "C:\\Users\\vaish\\OneDrive\\Desktop\\UCLA\\Prml\\Project_1\\Project_1\\Test_Landmarks"
dirOfTestImages ="C:\\Users\\vaish\\OneDrive\\Desktop\\UCLA\\Prml\\Project_1\\Project_1\\Test_Images"
dirOfOutputsEigen  = "Outputs\Part1\Part a\EigenFaces"
dirOfOutputsRecons  = "Outputs\Part1\Part a\ReconstrucedImages"

training_dir_images="C:\\Users\\vaish\\OneDrive\\Desktop\\UCLA\\Prml\\Project_1\\Project_1\\images"
L = np.empty(shape=(800,mn ))
HValues = np.empty(shape=(800, mn))
SValues = np.empty(shape=(800, mn))
VValues = np.empty(shape=(800, mn))

cur_img = 0
print("reading input train landmarks")
for filename in os.listdir(training_dir_images):
    filename = os.path.join(training_dir_images, filename)
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

print("Computing the mean face for training landmarks..")
mean_image = np.empty(shape=(1, 128, 128, 3), dtype='float64')
mean_img_col = __FindMeanOfMatrix__(L, 800)

final = np.dstack((HValues[0, :].reshape(128, 128), SValues[0, :].reshape(128, 128),
                        mean_img_col[:].reshape(128, 128)))

mean_image = skimage.color.hsv2rgb(final)


def __TransformedLandmarksWithMeanSubtracted__( Landmarks_dir, noOfLandmarks, mn, mean_img_col ):
    L = np.empty(shape=(noOfLandmarks, mn), dtype='float64')

    cur_dataPoint = 0

    for filename in os.listdir(Landmarks_dir):
        filename = os.path.join(Landmarks_dir, filename)
        train_data = scipy.io.loadmat(filename)
        values = train_data.get('lms')

        img_col = np.array(values, dtype='float64').flatten()

        L[cur_dataPoint, :] = img_col[:]  # set the cur_img-th column to the current training image
        cur_dataPoint += 1


    if(mean_img_col is None):
     mean_img_col = __FindMeanOfMatrix__(L,noOfLandmarks)
    for j in range(0, noOfLandmarks):  # subtract from all training images
        L[j, :] -= mean_img_col[:]
    return L, mean_img_col


def plot_filter_grid(landmarks, mean_image):
    filters = landmarks.shape[0]
    fig = plt.figure(0, figsize=(20, 12))
    fig.suptitle('First 10 Eigen-Landmarks plotted on mean face')
    n_columns = 8
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(0, 10):
        landmarks_current = landmarks[i,:].reshape(68,2)
        plt.subplot(n_rows, n_columns, i + 4)
        plt.title('Eigen-Landmarks-' + str(i))
        plt.axis('off')
        plt.imshow(mean_image, interpolation='nearest')
        plt.scatter(x=landmarks_current[:, 0], y=landmarks_current[:, 1], c='r', s=10)


def __GetEigenVectorsAndValues__(L, mean_landmarks,EigenNumber,plot_eigen):

    print("finding the eigen vectors for the landmarks")
    n_components = EigenNumber
    pca = PCA(n_components=n_components, svd_solver='randomized',
              whiten=True).fit(L)

    eigenLandmarks = pca.components_.reshape((n_components, 68, 2))

    eigenLandmarks_matrix = np.empty(shape=(n_components, 68* 2))
    eigenLandmarks_matrix_display = np.empty(shape=(n_components, 68 * 2))
    for i in range(0, n_components):
        image_array = eigenLandmarks[i, :, :]
        image_array = image_array.flatten()
        eigenLandmarks_matrix[i, :] = image_array
        eigenLandmarks_matrix_display[i, :] = eigenLandmarks_matrix[i, :] + mean_landmarks


    if (plot_eigen == True):
            plot_filter_grid(eigenLandmarks_matrix_display[0:n_components, ],mean_image)
            plt.savefig('First 10 eigen-landmarks.png', bbox_inches='tight')
            plt.close()

    return eigenLandmarks_matrix, pca

def __GetProjectionAndRecontructionError__(pca, eigenLandmarks_matrix, original_landmarks,mean, Eigen_Number):

    print("Projecting the test landmarks into the eigen space")
    Landmarks_tranformed_pca = pca.transform(original_landmarks)

    print("Reconstructing the test landmarks from the eigen space")
    reconstructed_landmarks = np.array(np.dot(Landmarks_tranformed_pca, eigenLandmarks_matrix))
    print("reconstructed_image", reconstructed_landmarks.shape)
    j = 0
    total_reconstruction_error = 0
    for i in range (0,200):
        reconstructed_landmarks[j, :] = np.array(reconstructed_landmarks[j, :])
        reconstructed_landmarks[j, :] += mean[:]
        total_reconstruction_error = total_reconstruction_error + np.sum( np.square(reconstructed_landmarks[j, :] - original_landmarks[j, :]))

    total_reconstruction_error = total_reconstruction_error / (68 * 2 * 200)/10000000000
    print("eigen-Landmarks K =" + str(Eigen_Number) + ", total reconstruction error " + str(total_reconstruction_error))

    # reconstructed_image = np.array(reconstructed_image, dtype='float64').flatten()
    #for filename in os.listdir(Test_Images_dir):
        #filename = os.path.join(Test_Images_dir, filename)
        # img = io.imread(filename)
        # print("reconstructed_image", reconstructed_image[:,j].shape)
        # reconstructed_image[j, :,:] = reconstructed_image[j, :,:] + np.dot(X_train_pca[i,:],  eigenfaces[i,:,:])
        #reconstructed_landmarks_to_display = reconstructed_landmarks[j, :].reshape(68, 2)

        # img = io.imread(Imagefilename)
        #im = plt.imread(filename)
        #implot = plt.imshow(im)
        # plt.plot(im)
        #plt.scatter(x=reconstructed_landmarks_to_display[:, 0], y=reconstructed_landmarks_to_display[:, 1], c='r', s=10)
        # plt.show()
        # plt.savefig(os.path.join(reconstructed_landmarks_folder,filename))

    return total_reconstruction_error


#training
L, mean_img_col = __TransformedLandmarksWithMeanSubtracted__(dirOfTrainingLandmarks, 800, 68*2,None )
eigenLandmarks_matrix, pca = __GetEigenVectorsAndValues__(L, mean_img_col,10,True)

Transformed_Test_Landmarks, mean_img_col = __TransformedLandmarksWithMeanSubtracted__(dirOfTestLandmarks, 200, 68*2, mean_img_col)
error = __GetProjectionAndRecontructionError__(pca, eigenLandmarks_matrix,Transformed_Test_Landmarks, mean_img_col, 10)

#Test
total_reconstruction_error = np.empty(shape=(11))
Transformed_Test_Landmarks, mean_img_col = __TransformedLandmarksWithMeanSubtracted__(dirOfTestLandmarks, 200, 68*2, mean_img_col)
total_reconstruction_error[0] = __GetProjectionAndRecontructionError__(pca, eigenLandmarks_matrix,Transformed_Test_Landmarks, mean_img_col, 1)


j= 1
k= 0
for i in range(1, 11):
    k = k + 5
    eigenLandmarks_matrix, pca = __GetEigenVectorsAndValues__(L, mean_img_col,k,False)
    total_reconstruction_error[j] = __GetProjectionAndRecontructionError__( pca, eigenLandmarks_matrix,Transformed_Test_Landmarks, mean_img_col, k)
    j=j+1


eigenNumber= [1,5,10,15,20,25,30,35,40,45,50]

fig = plt.figure(1)
plt.plot(eigenNumber, total_reconstruction_error, label='linear')
plt.xlabel('eigenNumber')
plt.ylabel('total_reconstruction_error')
plt.title('total_reconstruction_error vs eigenNumber for landmarks')

plt.legend()

plt.savefig('Landmarks_reconstruction_error_vs_eigenNumber.png', dpi = 'figure')
plt.close()