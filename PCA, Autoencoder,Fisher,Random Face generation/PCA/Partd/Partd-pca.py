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
        return mean_img_col


dirOfTrainingLandmarks = "C:\\Users\\vaish\\OneDrive\\Desktop\\UCLA\\Prml\\Project_1\\Project_1\\Landmarks"
dirOfTrainingImages ="C:\\Users\\vaish\\OneDrive\\Desktop\\UCLA\\Prml\\Project_1\\Project_1\\images"
dirOfTestLandmarks = "C:\\Users\\vaish\\OneDrive\\Desktop\\UCLA\\Prml\\Project_1\\Project_1\\Test_Landmarks"
dirOfTestImages ="C:\\Users\\vaish\\OneDrive\\Desktop\\UCLA\\Prml\\Project_1\\Project_1\\Test_Images"
dirOfOutputs1  = "Outputs\Part1\Partd\SynthesizedImagesBeforeWarp"
dirOfOutput2  = "Outputs\Part1\Partd\SynthesizedImagesAfterWarp"

noOfTrainingLandmarks = 800
noOfTrainingImages = 800
noOfTestLandmarks = 200
noOfTestImages = 200
dimensionOfLandmarks = 68*2
dimensionOfImages = 128*128

#initializing  variables
Training_Landmarks_Matrix_BeforeMean = np.empty(shape=(noOfTrainingLandmarks, dimensionOfLandmarks ), dtype='float64')
Training_Landmarks_Matrix_AfterMean = np.empty(shape=(noOfTrainingLandmarks, dimensionOfLandmarks), dtype='float64')
cur_dataPoint = 0

Training_Images = np.empty(shape=(noOfTrainingImages,dimensionOfImages), dtype='float64')
Training_Images_After_Warping_To_Mean = np.empty(shape=(noOfTrainingImages,dimensionOfImages ), dtype='float64')

Training_Landmarks = np.empty(shape=(noOfTrainingLandmarks, dimensionOfLandmarks), dtype='float64')
print("Running...")
#Find Mean landmark of all training landmarks
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
    #a.show()
    img_col = np.array(image, dtype='float64').flatten()  # flatten the 2d image into 1d
    # print("shape of img_col", img_col.shape)
    # plt.imshow(warped_Image_To_Mean)
    Training_Images[i, :] = img_col[:]
    i = i+1
    # plt.imshow(warped_Image_To_Mean)
i =0

HValues = np.empty(shape=(noOfTrainingImages, dimensionOfImages))
SValues = np.empty(shape=(noOfTrainingImages, dimensionOfImages))
VValues = np.empty(shape=(noOfTrainingImages, dimensionOfImages))

#warp the training images to mean landmark
for filename in os.listdir(dirOfTrainingImages):
    filename = os.path.join(dirOfTrainingImages, filename)
    img = io.imread(filename)
    inputImage = skimage.color.rgb2hsv(img)
    v = inputImage[:, :, 2]
    h = inputImage[:, :, 0]
    s = inputImage[:, :, 1]

    final = np.dstack((h, s, v))
    # skimage.io.imshow(final)
    img_col_h = np.array(h).flatten()  # flatten the 2d image into 1d
    HValues[i, :] = img_col_h[:]

    img_col_s = np.array(s, dtype='float64').flatten()  # flatten the 2d image into 1d
    SValues[i, :] = img_col_s[:]

    img_col_v = np.array(v, dtype='float64').flatten()  # flatten the 2d image into 1d
    # print("image shape is", img_col.shape)
    VValues[i, :] = img_col_v[:]
    warped_Image_To_Mean = warping.warp(img, Training_Landmarks_Matrix_BeforeMean[i, :].reshape(68, 2),
                                        mean_Landmark_col[:].reshape(68, 2))
    #skimage.io.imshow(warped_Image_To_Mean)
    warped_Image_To_Mean = skimage.color.rgb2hsv(warped_Image_To_Mean)
    warped_Image_To_Mean = warped_Image_To_Mean[:, :, 2]
    warped_Image_To_Mean = PIL.Image.fromarray(warped_Image_To_Mean)
    # a.show()
    img_col = np.array(warped_Image_To_Mean, dtype='float64').flatten()  # flatten the 2d image into 1d
    # print("shape of img_col", img_col.shape)
    # plt.imshow(warped_Image_To_Mean)
    Training_Images_After_Warping_To_Mean[i, :] = img_col[:]

    i= i+1
    #plt.imshow(warped_Image_To_Mean)



training_dir = dirOfTrainingImages
noOfImages = 800
mn =128 * 128

#calculating eigen vectors for warped training images
L = np.empty(shape=(noOfImages, mn))
HValues = np.empty(shape=(noOfImages, mn))
SValues = np.empty(shape=(noOfImages, mn))
VValues = np.empty(shape=(noOfImages, mn))


cur_img = 0
# finalImageId + ".jpg")  # relative path

for filename in os.listdir(training_dir):
    # img = cv2.imread(path_to_img, 0)  # read a grayscale image
    filename = os.path.join(training_dir, filename)
    img = io.imread(filename)
    inputImage = skimage.color.rgb2hsv(img)

    v = inputImage[:, :, 2]
    h = inputImage[:, :, 0]
    s = inputImage[:, :, 1]

    final = np.dstack((h, s, v))
    # skimage.io.imshow(final)
    img_col_h = np.array(h).flatten()  # flatten the 2d image into 1d
    HValues[cur_img, :] = img_col_h[:]

    img_col_s = np.array(s, dtype='float64').flatten()  # flatten the 2d image into 1d
    SValues[cur_img, :] = img_col_s[:]

    img_col_v = np.array(v, dtype='float64').flatten()  # flatten the 2d image into 1d
    # print("image shape is", img_col.shape)

    L[cur_img, :] = img_col_v[:]  # set the cur_img-th column to the current training image
    VValues[cur_img, :] = img_col_v[:]

    warped_Image_To_Mean = warping.warp(img, Training_Landmarks_Matrix_BeforeMean[cur_img, :].reshape(68, 2),
                                        mean_Landmark_col[:].reshape(68, 2))
    # skimage.io.imshow(warped_Image_To_Mean)
    warped_Image_To_Mean = skimage.color.rgb2hsv(warped_Image_To_Mean)
    warped_Image_To_Mean = warped_Image_To_Mean[:, :, 2]
    warped_Image_To_Mean = PIL.Image.fromarray(warped_Image_To_Mean)
    # a.show()
    img_col = np.array(warped_Image_To_Mean, dtype='float64').flatten()  # flatten the 2d image into 1d
    Training_Images_After_Warping_To_Mean[cur_img, :] = img_col[:]

    final = np.dstack((HValues[cur_img, :].reshape(128, 128), SValues[cur_img, :].reshape(128, 128),
                       VValues[cur_img, :].reshape(128, 128)))

    cur_img += 1


mean_img_col = __FindMeanOfMatrix__(L, noOfImages)



for i in range(0, noOfImages):
    image_array = L[i, :]
    image_array = image_array.reshape(128, 128)

n_components = 50
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(Training_Images_After_Warping_To_Mean)

eigenfaces = pca.components_.reshape((n_components, 128, 128))
eigenValues = pca.explained_variance_

# evalues = evalues[0:10]  # reduce the number of eigenvectors/values to consider
# evectors = evectors[0:10]

# evectors = evectors.transpose()  # change eigenvectors from rows to columns
eigenFaces_vectors = np.empty(shape=(n_components, 128 * 128))
for i in range(0, n_components):
    image_array = eigenfaces[i, :, :]
    image_array = image_array.flatten()
    eigenFaces_vectors[i, :] = image_array


X_train_pca = pca.transform(Training_Images_After_Warping_To_Mean)

print("synthesizing faces..")
synthesized_faces = np.empty(shape=(50, 128 * 128))
for j in  range(0,50):
  for i in  range(0,50):
    random_coeffecient = np.random.normal(0, np.sqrt(eigenValues[i]))
    synthesized_faces[j,:] = synthesized_faces[j,:] + np.dot(random_coeffecient , eigenFaces_vectors[i,:])
    synthesized_faces[j,:]+= mean_img_col[:]



for i in range(0, 50):
    synthesized_face = synthesized_faces[i, :].reshape(128, 128)
    #matplotlib.pyplot.imshow(synthesized_face, cmap= 'gray')


#Get evectors of landmarks
n_components = 10
pca_Of_Geometry = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(Training_Landmarks_Matrix_AfterMean)

eigenLandmarks = pca_Of_Geometry.components_.reshape((n_components, 68, 2))

eigenValuesOfLandmarks = pca_Of_Geometry.explained_variance_
eigenLandmarks_matrix = np.empty(shape=(n_components, 68 * 2))
for i in range(0, n_components):
    image_array = eigenLandmarks[i, :, :]
    #print("size", image_array.shape)
    image_array = image_array.flatten()
    eigenLandmarks_matrix[i, :] = image_array

synthesized_landmarks = np.empty(shape=(50, 68 * 2))
for j in  range(0,50):
  for i in  range(0,10):
    random_coeffecient = np.random.normal(0, np.sqrt(eigenValuesOfLandmarks[i]))
    synthesized_landmarks[j,:] = synthesized_landmarks[j,:] + np.dot(random_coeffecient , eigenLandmarks_matrix[i,:])
    synthesized_landmarks[j,:]+= mean_Landmark_col[:]

print("shape of synthesized_faces", synthesized_landmarks.shape)


final_reconstructed_images = np.empty(shape=(50,128,128,3 ), dtype='float64')
reconstructed_image = np.empty(shape=(50,128,128,3 ), dtype='float64')
for j in range(0, 50):
    final = np.dstack((HValues[j, :].reshape(128, 128), SValues[j, :].reshape(128, 128),
                       synthesized_faces[j, :].reshape(128, 128)))
    final_rgb_image = skimage.color.hsv2rgb(final)
    final_reconstructed_images[j,:,:,:] = final_rgb_image
    scipy.misc.imsave(os.path.join(dirOfOutputs1,
                                   str(j) + ".jpg"), final_rgb_image)

def plot_filter_grid(images):
         filters = images.shape[0]
         fig = plt.figure(0, figsize=(20, 12))
         fig.suptitle('SynthesizeD faces by random sampling of landmarks and random sampling of appearance.')
         n_columns = 8
         n_rows = math.ceil(filters / n_columns) + 1
         for i in range(0, 50):
            plt.subplot(n_rows, n_columns, i + 1)
            plt.title('Eigen Face ' + str(i))
            plt.axis('off')
            plt.imshow(images[i, ].reshape(128,128,3)*255, interpolation='nearest')


#Warp to  landmarks
for j in range(0, 50):
    final_reconstructed_images[j,:,:,:] = warping.warp(final_reconstructed_images[j,:,:,:], mean_Landmark_col[:].reshape(68, 2 ), synthesized_landmarks[j,:].reshape(68, 2))
    scipy.misc.imsave(os.path.join(dirOfOutput2,
                                   str(j) + ".jpg"), final_reconstructed_images[j,:,:,:])

plot_filter_grid(final_reconstructed_images)
plt.savefig('synthesized-images.png', bbox_inches='tight')
plt.close()





