import numpy as np
import os
import scipy.io
import warping
import matplotlib.pyplot as plt
import skimage
import PIL
import matplotlib
from sklearn.decomposition import PCA
from skimage import io
import math
def __FindMeanOfMatrix__( X, noOfDataPoints):
        mean_img_col = np.sum(X, axis=0) / noOfDataPoints
        print("mean is", mean_img_col.shape)
        return mean_img_col


dirOfTrainingLandmarks = "C:\\Users\\vaish\\OneDrive\\Desktop\\UCLA\\Prml\\Project_1\\Project_1\\Landmarks"
dirOfTrainingImages ="C:\\Users\\vaish\\OneDrive\\Desktop\\UCLA\\Prml\\Project_1\\Project_1\\images"
dirOfTestLandmarks = "C:\\Users\\vaish\\OneDrive\\Desktop\\UCLA\\Prml\\Project_1\\Project_1\\Test_Landmarks"
dirOfTestImages ="C:\\Users\\vaish\\OneDrive\\Desktop\\UCLA\\Prml\\Project_1\\Project_1\\Test_Images"
dirOfOutputs1  = "Outputs\Part1\Partc\ReconstructedImagesBeforeWarp"
dirOfOutput2  = "Outputs\Part1\Partc\ReconstructedImagesAfterWarp"

noOfTrainingLandmarks = 800
noOfTrainingImages = 800
noOfTestLandmarks = 200
noOfTestImages = 200
dimensionOfLandmarks = 68*2
dimensionOfImages = 128*128

#initializing some variables
Training_Landmarks_Matrix_BeforeMean = np.empty(shape=(noOfTrainingLandmarks, dimensionOfLandmarks ), dtype='float64')
Training_Landmarks_Matrix_AfterMean = np.empty(shape=(noOfTrainingLandmarks, dimensionOfLandmarks), dtype='float64')
cur_dataPoint = 0

Training_Images = np.empty(shape=(noOfTrainingImages,dimensionOfImages), dtype='float64')
Training_Images_After_Warping_To_Mean = np.empty(shape=(noOfTrainingImages,dimensionOfImages ), dtype='float64')


Training_Landmarks = np.empty(shape=(noOfTrainingLandmarks, dimensionOfLandmarks), dtype='float64')

#Find Mean landmark of all training landmarks
print("Reading training landmarks and finding their mean")
for filename in os.listdir(dirOfTrainingLandmarks):

    filename = os.path.join(dirOfTrainingLandmarks, filename)
    train_landmark = scipy.io.loadmat(filename)
    training_Landmark_values = train_landmark.get('lms')

    trainingLandmarks_col = np.array(training_Landmark_values, dtype='float64').flatten()  # flatten the 2d image into 1d

    Training_Landmarks_Matrix_BeforeMean[cur_dataPoint,: ] = trainingLandmarks_col[:]  # set the cur_img-th column to the current training image
    cur_dataPoint += 1


mean_Landmark_col = __FindMeanOfMatrix__(Training_Landmarks_Matrix_BeforeMean, noOfTrainingLandmarks)

#substract mean from all the training landmarks
for j in range(0, noOfTrainingLandmarks):  # subtract from all training images
    Training_Landmarks_Matrix_AfterMean[ j, :] = Training_Landmarks_Matrix_BeforeMean[j, :] - mean_Landmark_col[:]

i = 0
for filename in os.listdir(dirOfTrainingImages):
    filename = os.path.join(dirOfTrainingImages, filename)
    image =  matplotlib.pyplot.imread(filename)
    image = image[:, :, 2]
    a = PIL.Image.fromarray(image)
    img_col = np.array(image, dtype='float64').flatten()
    Training_Images[i, :] = img_col[:]
    i = i+1
i =0

HValues = np.empty(shape=(noOfTrainingImages, dimensionOfImages))
SValues = np.empty(shape=(noOfTrainingImages, dimensionOfImages))
VValues = np.empty(shape=(noOfTrainingImages, dimensionOfImages))

#warp the training images to mean landmark
training_dir = dirOfTrainingImages
noOfImages = 800
mn =128 * 128

L = np.empty(shape=(noOfImages, mn))
HValues = np.empty(shape=(noOfImages, mn))
SValues = np.empty(shape=(noOfImages, mn))
VValues = np.empty(shape=(noOfImages, mn))

cur_img = 0

print("warping the training images to mean landmark")
for filename in os.listdir(training_dir):
    filename = os.path.join(training_dir, filename)
    img = io.imread(filename)
    inputImage = skimage.color.rgb2hsv(img)
    v = inputImage[:, :, 2]
    h = inputImage[:, :, 0]
    s = inputImage[:, :, 1]

    final = np.dstack((h, s, v))

    img_col_h = np.array(h).flatten()
    HValues[cur_img, :] = img_col_h[:]

    img_col_s = np.array(s, dtype='float64').flatten()
    SValues[cur_img, :] = img_col_s[:]

    img_col_v = np.array(v, dtype='float64').flatten()


    L[cur_img, :] = img_col_v[:]  # set the cur_img-th column to the current training image
    VValues[cur_img, :] = img_col_v[:]

    warped_Image_To_Mean = warping.warp(img, Training_Landmarks_Matrix_BeforeMean[cur_img, :].reshape(68, 2),
                                        mean_Landmark_col[:].reshape(68, 2))
    # skimage.io.imshow(warped_Image_To_Mean)
    warped_Image_To_Mean = skimage.color.rgb2hsv(warped_Image_To_Mean)
    warped_Image_To_Mean = warped_Image_To_Mean[:, :, 2]
    warped_Image_To_Mean = PIL.Image.fromarray(warped_Image_To_Mean)

    img_col = np.array(warped_Image_To_Mean, dtype='float64').flatten()  # flatten the 2d image into 1d

    Training_Images_After_Warping_To_Mean[cur_img, :] = img_col[:]

    final = np.dstack((HValues[cur_img, :].reshape(128, 128), SValues[cur_img, :].reshape(128, 128),
                       VValues[cur_img, :].reshape(128, 128)))

    cur_img += 1


mean_img_col = __FindMeanOfMatrix__(L, noOfImages)


for i in range(0, noOfImages):
    image_array = L[i, :]
    image_array = image_array.reshape(128, 128)


def GetEigenFaces(eigen_number ):
    print("calculating eigen vectors for warped training images")
    n_components = eigen_number
    pca = PCA(n_components=n_components, svd_solver='randomized',
              whiten=True).fit(Training_Images_After_Warping_To_Mean)

    eigenfaces = pca.components_.reshape((n_components, 128, 128))
    eigenFaces_vectors = np.empty(shape=(n_components, 128 * 128))
    for i in range(0, n_components):
        image_array = eigenfaces[i, :, :]
        image_array = image_array.flatten()
        eigenFaces_vectors[i, :] = image_array

    return eigenFaces_vectors, pca

print("calculating eigen vectors for training landmarks")
n_components = 10
pca_Of_Geometry = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(Training_Landmarks_Matrix_AfterMean)

eigenLandmarks = pca_Of_Geometry.components_.reshape((n_components, 68, 2))

eigenLandmarks_matrix = np.empty(shape=(n_components, 68 * 2))
for i in range(0, n_components):
    image_array = eigenLandmarks[i, :, :]
    image_array = image_array.flatten()
    eigenLandmarks_matrix[i, :] = image_array

#Test Landmarks, project into first ten eigen lanmarks, calculate the reconsteruction landmarks
#Reading Test landmarks, subtracting the mean from them
print("Reading Test landmarks, subtracting the mean from them")
Test_Landmarks = np.empty(shape=(noOfTestLandmarks, dimensionOfLandmarks), dtype='float64')

cur_dataPoint = 0

for filename in os.listdir(dirOfTestLandmarks):
    filename = os.path.join(dirOfTestLandmarks, filename)
    train_data = scipy.io.loadmat(filename)
    values = train_data.get('lms')
    img_col = np.array(values, dtype='float64').flatten()

    Test_Landmarks[cur_dataPoint, :] = img_col[:]
    cur_dataPoint += 1


#project into first ten eigen lanmarks
print("projecting test landmarks into first ten eigen landmarks")
Projected_Test_Landmarks = pca_Of_Geometry.transform(Test_Landmarks)

print("shape of Projected_Test_Landmarks", Projected_Test_Landmarks.shape)

#calculating the reconsteruction landmarks
print("calculating the reconstructed landmarks")
reconstructed_Test_landmarks = np.array(np.dot(Projected_Test_Landmarks, eigenLandmarks_matrix))

j = 0
# reconstructed_image = np.array(reconstructed_image, dtype='float64').flatten()
for filename in os.listdir(dirOfTestImages):
    filename = os.path.join(dirOfTestImages, filename)
    # img = io.imread(filename)
    # print("reconstructed_image", reconstructed_image[:,j].shape)
    # reconstructed_image[j, :,:] = reconstructed_image[j, :,:] + np.dot(X_train_pca[i,:],  eigenfaces[i,:,:])
    reconstructed_Test_landmarks[j, :] = np.array(reconstructed_Test_landmarks[j, :])
    reconstructed_Test_landmarks[j, :] += mean_Landmark_col[:] #check
    reconstructed_landmarks_to_display = reconstructed_Test_landmarks[j, :].reshape(68, 2)
    #img = io.imread(filename)
    #im = plt.imread(filename)
    #implot = plt.imshow(im)
    # plt.plot(im)
    #plt.scatter(x=reconstructed_landmarks_to_display[:, 0], y=reconstructed_landmarks_to_display[:, 1], c='r', s=10)
    #plt.show()
    # plt.savefig(os.path.join(reconstructed_landmarks_folder,filename))


#Testimages, warp to mean position, project into first 50 eigen faces, calculate the reconsteruction
#warping the test images to mean landmark
print("warping the test images to mean landmark ")
Testing_Images_After_Warping_To_Mean = np.empty(shape=(noOfTestImages,dimensionOfImages ), dtype='float64')
Testing_Images_Before_Warping_To_Mean = np.empty(shape=(noOfTestImages,128,128,3 ), dtype='float64')

L = np.empty(shape=(200, mn))
HValues = np.empty(shape=(200, mn))
SValues = np.empty(shape=(200, mn))
VValues = np.empty(shape=(200, mn))

cur_img = 0

for filename in os.listdir(dirOfTestImages):

    filename = os.path.join(dirOfTestImages, filename)
    img = io.imread(filename)
    inputImage = skimage.color.rgb2hsv(img)
    Testing_Images_Before_Warping_To_Mean[cur_img,] = skimage.color.hsv2rgb(inputImage)
    v = inputImage[:, :, 2]
    h = inputImage[:, :, 0]
    s = inputImage[:, :, 1]

    final = np.dstack((h, s, v))
    img_col_h = np.array(h).flatten()
    HValues[cur_img, :] = img_col_h[:]

    img_col_s = np.array(s, dtype='float64').flatten()
    SValues[cur_img, :] = img_col_s[:]

    img_col_v = np.array(v, dtype='float64').flatten()

    L[cur_img, :] = img_col_v[:]
    VValues[cur_img, :] = img_col_v[:]

    final = np.dstack((HValues[cur_img, :].reshape(128, 128), SValues[cur_img, :].reshape(128, 128),
                       VValues[cur_img, :].reshape(128, 128)))

    warped_Image_To_Mean = warping.warp(img, Test_Landmarks[cur_img, :].reshape(68, 2),
                                        mean_Landmark_col[:].reshape(68, 2))
    warped_Image_To_Mean = skimage.color.rgb2hsv(warped_Image_To_Mean)
    warped_Image_To_Mean = warped_Image_To_Mean[:, :, 2]
    warped_Image_To_Mean = PIL.Image.fromarray(warped_Image_To_Mean)
    # a.show()
    img_col = np.array(warped_Image_To_Mean, dtype='float64').flatten()

    Testing_Images_After_Warping_To_Mean[cur_img, :] = img_col[:]

    cur_img += 1

def plot_filter_grid(reconstructed, original, number_to_display):
         fig = plt.figure(2, figsize=(30, 30))
         fig.suptitle('Plot of 20 reconstructed images')
         n_columns = 20
         k = 0
         n_rows = math.ceil(number_to_display / n_columns) + 1
         for i in range(0, number_to_display):
            plt.subplot(n_rows, n_columns, i + k + 1)
            plt.title('r- ' + str(i))
            plt.axis('off')
            plt.imshow(reconstructed[i, ].reshape(128,128,3), interpolation='nearest')
            plt.subplot(n_rows, n_columns, i + k+ 2)
            plt.title('o- ' + str(i))
            plt.axis('off')
            plt.imshow(original[i,].reshape(128,128,3), interpolation='nearest')
            k = k+1

def _ProjectAndReconstruct(X_train_pca, eigenFaces_vectors, eigen_number, Plot = False):
        total_reconstruction_error = 0
        print("Projecting the test images into eigen space and reconstructing")
        X_test_pca = X_train_pca.transform(Testing_Images_After_Warping_To_Mean)
        reconstructed_image = np.array(np.dot(X_test_pca, eigenFaces_vectors))

        final_reconstructed_Test_images = np.empty(shape=(noOfTestImages,128,128,3 ), dtype='float64')
        Reconstructed_Image_WarpedTo_Mean = np.empty(shape=(noOfTestImages,128,128,3 ), dtype='float64')

        for j in range(0, noOfTestImages):  # subtract from all training images
            reconstructed_image[j, :] = np.array(reconstructed_image[j, :])
            reconstructed_image[j, :] += mean_img_col[:]

            final = np.dstack((HValues[j, :].reshape(128, 128), SValues[j, :].reshape(128, 128),
                               reconstructed_image[j, :].reshape(128, 128)))
            final_rgb_image = skimage.color.hsv2rgb(final)
            final_reconstructed_Test_images[j,:,:,:] = final_rgb_image
            total_reconstruction_error = total_reconstruction_error + np.sum(np.square(reconstructed_image[j, :] - VValues[j, :]))
            if(Plot == True):
                scipy.misc.imsave(os.path.join(dirOfOutputs1,
                                               str(j) + ".jpg"), final_rgb_image)

        print("For eigen no: " + str(eigen_number) + "total_reconstruction_error = " , total_reconstruction_error/(128 * 128 * 200) )
        print("Warping to reconstructed landmarks")
        #Warp to reconstructed landmarks
        for j in range(0, noOfTestImages):
            Reconstructed_Image_WarpedTo_Mean[j,:,:,:] = warping.warp(final_reconstructed_Test_images[j,:,:,:], mean_Landmark_col[:].reshape(68, 2 ), Test_Landmarks[j,:].reshape(68, 2))
            if(Plot == True):
                scipy.misc.imsave(os.path.join(dirOfOutput2,
                                               str(j) + ".jpg"), Reconstructed_Image_WarpedTo_Mean[j,:,:,:])

        if(Plot == True):
            plot_filter_grid(Reconstructed_Image_WarpedTo_Mean, Testing_Images_Before_Warping_To_Mean,20)
            plt.savefig('First_20_reconstructedPart3.png', bbox_inches='tight')
            plt.close()
            return total_reconstruction_error/ (128 * 128 * 200)


total_reconstruction_error = np.empty(shape=(11))

eigenFaces_vectors, X_train_pca =  GetEigenFaces(50)
total_reconstruction_error[0] = _ProjectAndReconstruct(X_train_pca, eigenFaces_vectors,50, Plot = True)

eigenFaces_vectors, X_train_pca =  GetEigenFaces(1 )
total_reconstruction_error[0] = _ProjectAndReconstruct(X_train_pca, eigenFaces_vectors,1, Plot = False)



j= 1
k= 0
for i in range(1, 11):
    k = k + 5
    eigenFaces_vectors, X_train_pca =  GetEigenFaces(k )
    total_reconstruction_error[j] = _ProjectAndReconstruct(X_train_pca, eigenFaces_vectors,k ,Plot = False)
    j=j+1

print(total_reconstruction_error)
eigenNumber= [1,5,10,15,20,25,30,35,40,45,50]

plt.figure(1)
plt.plot(eigenNumber, total_reconstruction_error, label='linear')
plt.xlabel('eigenNumber')
plt.ylabel('total_reconstruction_error')
plt.title('total_reconstruction_error vs eigenNumber')

plt.legend()

plt.savefig('reconstruction_error_Graph_Partc.png', dpi = 'figure')
plt.close()
