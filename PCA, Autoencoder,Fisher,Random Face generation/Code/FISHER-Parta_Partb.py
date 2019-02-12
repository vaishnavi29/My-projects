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

def __FindMeanOfMatrix__( X, noOfDataPoints):
        mean_img_col = np.sum(X, axis=0) / noOfDataPoints

        return mean_img_col

def __TransformedMatrixWithMeanSubtractedAndFindEigen__( training_dir, noOfImages, mn, Eigen_Number):
    L = np.empty(shape=(noOfImages,mn ))
    HValues = np.empty(shape=(noOfImages, mn))
    SValues = np.empty(shape=(noOfImages, mn))
    VValues = np.empty(shape=(noOfImages, mn))

    cur_img = 0

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

    mean_img_col = __FindMeanOfMatrix__(L, noOfImages)
    for j in range(0, noOfImages):  # subtract from all training images
     L[j,: ] -= mean_img_col[:]

    n_components = Eigen_Number
    pca = PCA(n_components=n_components, svd_solver='randomized',
              whiten=True).fit(L)

    eigenfaces = pca.components_.reshape((n_components, 128, 128))

    eigenFaces_vectors = np.empty(shape=(n_components, 128*128))
    eigenFaces_vectors_display = np.empty(shape=(n_components, 128 * 128))
    for i in range(0, n_components):

        image_array = eigenfaces[i, :,:]
        image_array = image_array.flatten()
        eigenFaces_vectors_display[i, :] = eigenFaces_vectors[i,:] + mean_img_col
        eigenFaces_vectors[i,:] = image_array

    X_train_pca = pca.transform(L)

    reconstructed_image = np.array(np.dot( X_train_pca,eigenFaces_vectors))

    for j in range(0, noOfImages):  # subtract from all training images
        reconstructed_image[j, :] = np.array(reconstructed_image[j, :])
        reconstructed_image[j, :] += mean_img_col[:]
        final = np.dstack((HValues[j, :].reshape(128, 128), SValues[j, :].reshape(128, 128),
                           reconstructed_image[j, :].reshape(128, 128)))

    return pca, eigenFaces_vectors, mean_img_col

def __TransformedLandmarkWithMeanSubtractedAndFindEigen__( Landmarks_dir, noOfLandmarks, mn ):
    Landmarks = np.empty(shape=(noOfLandmarks, mn), dtype='float64')

    cur_dataPoint = 0
    # finalImageId + ".jpg")  # relative path

    for filename in os.listdir(Landmarks_dir):
        filename = os.path.join(Landmarks_dir, filename)
        train_data = scipy.io.loadmat(filename)
        values = train_data.get('lms')

        img_col = np.array(values, dtype='float64').flatten()  # flatten the 2d image into 1d
        # print("image shape is", img_col.shape)

        Landmarks[cur_dataPoint, :] = img_col[:]  # set the cur_img-th column to the current training image
        cur_dataPoint += 1


    mean_img_col = __FindMeanOfMatrix__(Landmarks,noOfLandmarks)
    for j in range(0, noOfLandmarks):  # subtract from all training images
        Landmarks[j, :] -= mean_img_col[:]

    n_components = 10
    pca = PCA(n_components=n_components, svd_solver='randomized',
              whiten=True).fit(Landmarks)

    eigenLandmarks = pca.components_.reshape((n_components, 68, 2))

    eigenLandmarks_matrix = np.empty(shape=(n_components, 68* 2))
    for i in range(0, n_components):
        image_array = eigenLandmarks[i, :, :]
        image_array = image_array.flatten()
        eigenLandmarks_matrix[i, :] = image_array

    return pca, eigenLandmarks_matrix

def _findFisherFaces_(pcaModel, male_training_dir,female_training_dir, noOfMaleImages,noOfFemaleImages):

    #reading male images
    cur_img = 0
    mn = 128*128
    Male_training_images = np.empty(shape=(noOfMaleImages,128*128 ))
    Female_training_images = np.empty(shape=(noOfFemaleImages, 128 * 128))

    for filename in os.listdir(male_training_dir):
        filename = os.path.join(male_training_dir, filename)
        img = io.imread(filename)
        inputImage = skimage.color.rgb2hsv(img)
        v = inputImage[:, :, 2]
        img_col_v = np.array(v, dtype='float64').flatten()  # flatten the 2d image into 1d

        Male_training_images[cur_img, :] = img_col_v[:]  # set the cur_img-th column to the current training image

        cur_img += 1

    cur_img = 0
    for filename in os.listdir(female_training_dir):
        filename = os.path.join(female_training_dir, filename)
        img = io.imread(filename)
        inputImage = skimage.color.rgb2hsv(img)
        v = inputImage[:, :, 2]

        img_col_v = np.array(v, dtype='float64').flatten()  # flatten the 2d image into 1d

        Female_training_images[cur_img, :] = img_col_v[:]  # set the cur_img-th column to the current training image

        cur_img += 1

    Transformed_X_male = (pcaModel.transform(Male_training_images)).transpose()
    Transformed_X_female = pcaModel.transform(Female_training_images).transpose()
    Transformed_X_male_minusMean = np.empty(shape=( 50, noOfMaleImages))
    Transformed_X_female_minusMean = np.empty(shape=(50, noOfFemaleImages))


    meanMaleClass = Transformed_X_male.mean(axis=1)
    meanFemaleClass = Transformed_X_female.mean(axis=1)

    for i in range (0,400):
     Transformed_X_male_minusMean[:,i] = Transformed_X_male[:,i] - meanMaleClass
     Transformed_X_female_minusMean[:, i] = Transformed_X_female[:, i] - meanFemaleClass

    S1 = np.dot(Transformed_X_male_minusMean, (Transformed_X_male_minusMean).transpose())
    S2 = np.dot(Transformed_X_female_minusMean,Transformed_X_female_minusMean.transpose())
    Sw = S1 + S2
    SwInverse = np.linalg.inv(Sw)

    w = np.dot(SwInverse, meanMaleClass- meanFemaleClass)


    #projecting male and female into 1d fisher space
    projected_male_faces = np.dot(Transformed_X_male.transpose(), w)
    projected_female_faces = np.dot(Transformed_X_female.transpose(), w)
    threshold = np.dot(w, (meanFemaleClass + meanMaleClass) / 2)

    plt.figure(0)
    plt.title("Plot of male vs female fisher faces (appearance) projections of Training set")
    plt.plot(projected_male_faces,'*', color = 'blue')
    plt.plot(projected_female_faces, '*', color= 'pink')
    plt.savefig("Outputs\Part3-FisherFaces\Appearance-FisherProjections.png")
    plt.close()


    print("The fisher face for the key point ( appearance) is", threshold)
    return projected_male_faces, projected_female_faces,w, threshold

def _findFisherLandmarks_(pcaModel, male_training_dir,female_training_dir, noOfMaleLandmarks,noOfFemaleLandmarks):

    #reading male images
    print("reading input train images")
    cur_img = 0
    mn = 68*2
    Male_training_landmarks = np.empty(shape=(noOfMaleLandmarks,68*2 ))
    Female_training_landmarks = np.empty(shape=(noOfFemaleLandmarks, 68 * 2))

    cur_dataPoint = 0
    for filename in os.listdir(male_training_dir):
        filename = os.path.join(male_training_dir, filename)
        train_data = scipy.io.loadmat(filename)
        values = train_data.get('lms')
        img_col = np.array(values, dtype='float64').flatten()
        Male_training_landmarks[cur_dataPoint, :] = img_col[:]  # set the cur_img-th column to the current training image
        cur_dataPoint += 1

    cur_dataPoint = 0
    for filename in os.listdir(female_training_dir):
        filename = os.path.join(female_training_dir, filename)
        train_data = scipy.io.loadmat(filename)
        values = train_data.get('lms')
        img_col = np.array(values, dtype='float64').flatten()
        Female_training_landmarks[cur_dataPoint, :] = img_col[:]  # set the cur_img-th column to the current training image
        cur_dataPoint += 1

    Transformed_X_male = (pcaModel.transform(Male_training_landmarks)).transpose()
    Transformed_X_female = pcaModel.transform(Female_training_landmarks).transpose()
    Transformed_X_male_minusMean = np.empty(shape=( 10, noOfMaleLandmarks))
    Transformed_X_female_minusMean = np.empty(shape=(10, noOfFemaleLandmarks))

    meanMaleClass = Transformed_X_male.mean(axis=1)
    meanFemaleClass = Transformed_X_female.mean(axis=1)
    for i in range (0,400):
     Transformed_X_male_minusMean[:,i] = Transformed_X_male[:,i] - meanMaleClass
     Transformed_X_female_minusMean[:, i] = Transformed_X_female[:, i] - meanFemaleClass

    S1 = np.dot(Transformed_X_male_minusMean, (Transformed_X_male_minusMean).transpose())
    S2 = np.dot(Transformed_X_female_minusMean,Transformed_X_female_minusMean.transpose())
    Sw = S1 + S2
    SwInverse = np.linalg.inv(Sw)
    w = np.dot(SwInverse, meanMaleClass- meanFemaleClass)


    #projecting male and female into 1d fisher space
    projected_male_landmarks = np.dot(Transformed_X_male.transpose(), w)
    projected_female_landmarks = np.dot(Transformed_X_female.transpose(), w)
    threshold = np.dot(w, (meanFemaleClass + meanMaleClass) / 2)
    plt.figure(1)
    plt.title("Plot of male vs female fisher landmark projections of Training set" )
    plt.plot(projected_male_landmarks,'*', color = 'blue')
    plt.plot(projected_female_landmarks, '*', color= 'pink')
    plt.savefig("Outputs\Part3-FisherFaces\Landmarks-FisherProjections.png")
    plt.close()


    print("The fisher face for the key point (geometry) is", threshold)
    return projected_male_landmarks, projected_female_landmarks, w,threshold

#part 2
def _ProjectandClassifyTest(Images_dir_male,Images_dir_female, Landmarks_dir_male,Landmarks_dir_male_female, w_fisher_faces, w_fisher_landmarks,pcaAppearence, pcaLandmarks  ):

    testing_images= np.empty(shape=(200, 128 * 128))
    testing_landmarks = np.empty(shape=(200, 68 * 2))
    testing_images_male= np.empty(shape=(200, 128 * 128))
    testing_landmarks_male = np.empty(shape=(200, 68 * 2))
    testing_images_female= np.empty(shape=(200, 128 * 128))
    testing_landmarks_female = np.empty(shape=(200, 68 * 2))

    cur_img = 0
    for filename in os.listdir(Images_dir_male):
        filename = os.path.join(Images_dir_male, filename)
        img = io.imread(filename)
        inputImage = skimage.color.rgb2hsv(img)
        v = inputImage[:, :, 2]
        img_col_v = np.array(v, dtype='float64').flatten()  # flatten the 2d image into 1d
        testing_images_male[cur_img, :] = img_col_v[:]  # set the cur_img-th column to the current training image

        cur_img += 1

    for filename in os.listdir(Images_dir_female):
        filename = os.path.join(Images_dir_female, filename)
        img = io.imread(filename)
        inputImage = skimage.color.rgb2hsv(img)

        img_col_v = np.array(v, dtype='float64').flatten()  # flatten the 2d image into 1d
        testing_images_female[cur_img, :] = img_col_v[:]  # set the cur_img-th column to the current training image

        cur_img += 1

    cur_dataPoint = 0
    for filename in os.listdir(Landmarks_dir_male):
        filename = os.path.join(Landmarks_dir_male, filename)
        train_data = scipy.io.loadmat(filename)
        values = train_data.get('lms')
        img_col = np.array(values, dtype='float64').flatten()
        testing_landmarks_male[cur_dataPoint, :] = img_col[:]  # set the cur_img-th column to the current training image
        cur_dataPoint += 1

    for filename in os.listdir(Landmarks_dir_male_female):
        filename = os.path.join(Landmarks_dir_male_female, filename)
        train_data = scipy.io.loadmat(filename)
        values = train_data.get('lms')
        img_col = np.array(values, dtype='float64').flatten()
        testing_landmarks_female[cur_dataPoint, :] = img_col[:]  # set the cur_img-th column to the current training image
        cur_dataPoint += 1

    Transformed_test_faces_male = (pcaAppearence.transform(testing_images_male)).transpose()
    Transformed_test_landmarks_male = pcaLandmarks.transform(testing_landmarks_male).transpose()
    Transformed_test_faces_female = (pcaAppearence.transform(testing_images_female)).transpose()
    Transformed_test_landmarks_female = pcaLandmarks.transform(testing_landmarks_female).transpose()

    projected_appearence_male = np.dot(Transformed_test_faces_male.transpose(), w_fisher_faces)
    projected_landmarks_male = np.dot(Transformed_test_landmarks_male.transpose(), w_fisher_landmarks)
    projected_appearence_female = np.dot(Transformed_test_faces_female.transpose(), w_fisher_faces)
    projected_landmarks_female = np.dot(Transformed_test_landmarks_female.transpose(), w_fisher_landmarks)

    plt.figure(2)
    plt.scatter(projected_appearence_male, projected_landmarks_male, )
    plt.scatter(projected_appearence_female, projected_landmarks_female)
    plt.savefig("Outputs\Part3-FisherFaces\Part2-ProjectingAllTesting.png")
    plt.close()

#part 1
def _findFisherFacesCombined_(pca_appearance, pca_geometry, male_training_dir_images, female_training_dir_images , male_training_dir_landmarks,female_training_dir_landmarks ):

    #reading male images
    print("reading input train images")
    cur_img = 0
    Male_training_images = np.empty(shape=(400,128*128 ))
    Female_training_images = np.empty(shape=(400, 128 * 128))

    for filename in os.listdir(male_training_dir_images):
        filename = os.path.join(male_training_dir_images, filename)
        img = io.imread(filename)
        inputImage = skimage.color.rgb2hsv(img)
        v = inputImage[:, :, 2]
        img_col_v = np.array(v, dtype='float64').flatten()  # flatten the 2d image into 1d
        Male_training_images[cur_img, :] = img_col_v[:]  # set the cur_img-th column to the current training image

        cur_img += 1

    cur_img = 0
    for filename in os.listdir(female_training_dir_images):
        filename = os.path.join(female_training_dir_images, filename)
        img = io.imread(filename)
        inputImage = skimage.color.rgb2hsv(img)
        v = inputImage[:, :, 2]

        img_col_v = np.array(v, dtype='float64').flatten()  # flatten the 2d image into 1d

        Female_training_images[cur_img, :] = img_col_v[:]  # set the cur_img-th column to the current training image
        cur_img += 1

    Transformed_X_male_faces = (pca_appearance.transform(Male_training_images)).transpose()
    Transformed_X_female_faces = pca_appearance.transform(Female_training_images).transpose()

    Male_training_landmarks = np.empty(shape=(400,68*2 ))
    Female_training_landmarks = np.empty(shape=(400, 68 * 2))


    cur_dataPoint = 0
    for filename in os.listdir(male_training_dir_landmarks):
        filename = os.path.join(male_training_dir_landmarks, filename)
        train_data = scipy.io.loadmat(filename)
        values = train_data.get('lms')
        img_col = np.array(values, dtype='float64').flatten()
        Male_training_landmarks[cur_dataPoint, :] = img_col[:]  # set the cur_img-th column to the current training image
        cur_dataPoint += 1

    cur_dataPoint = 0
    for filename in os.listdir(female_training_dir_landmarks):
        filename = os.path.join(female_training_dir_landmarks, filename)
        train_data = scipy.io.loadmat(filename)
        values = train_data.get('lms')
        img_col = np.array(values, dtype='float64').flatten()
        Female_training_landmarks[cur_dataPoint, :] = img_col[:]  # set the cur_img-th column to the current training image
        cur_dataPoint += 1

    Transformed_X_male_landmarks = (pca_geometry.transform(Male_training_landmarks)).transpose()
    Transformed_X_female_landmarks = pca_geometry.transform(Female_training_landmarks).transpose()

    #combining appearence and geometry now
    Transformed_X_male_combined = np.vstack((Transformed_X_male_faces, Transformed_X_male_landmarks))

    Transformed_X_female_combined = np.vstack((Transformed_X_female_faces, Transformed_X_female_landmarks))

    Transformed_X_male_minusMean = np.empty(shape=( 60, 400))
    Transformed_X_female_minusMean = np.empty(shape=(60, 400))


    meanMaleClass = Transformed_X_male_combined.mean(axis=1)
    meanFemaleClass = Transformed_X_female_combined.mean(axis=1)
    for i in range (0,400):
     Transformed_X_male_minusMean[:,i] = Transformed_X_male_combined[:,i] - meanMaleClass
     Transformed_X_female_minusMean[:, i] = Transformed_X_female_combined[:, i] - meanFemaleClass

    S1 = np.dot(Transformed_X_male_minusMean, (Transformed_X_male_minusMean).transpose())
    S2 = np.dot(Transformed_X_female_minusMean,Transformed_X_female_minusMean.transpose())
    Sw = S1 + S2
    SwInverse = np.linalg.inv(Sw)
    complete_fisher_w = np.dot(SwInverse, meanMaleClass- meanFemaleClass)

    #projecting male and female into 1d fisher space
    projected_male_faces = np.dot(Transformed_X_male_combined.transpose(), complete_fisher_w)
    projected_female_faces = np.dot(Transformed_X_female_combined.transpose(), complete_fisher_w)

    #plt.plot(projected_male_faces,'*', color = 'blue')
    #plt.plot(projected_female_faces, '*', color= 'pink')

    threshold = np.dot(complete_fisher_w,(meanFemaleClass+meanMaleClass)/2)
    return projected_male_faces, projected_female_faces,complete_fisher_w, threshold

#Part1
def _ProjectandClassifyTestCombined(Images_dir_male,Images_dir_female, Landmarks_dir_male,Landmarks_dir_male_female, w_fisher_combined,pcaAppearence, pcaLandmarks,threshold  ):

    testing_images_male= np.empty(shape=(200, 128 * 128))
    testing_landmarks_male = np.empty(shape=(200, 68 * 2))
    testing_images_female= np.empty(shape=(200, 128 * 128))
    testing_landmarks_female = np.empty(shape=(200, 68 * 2))
    VValues = np.empty(shape=(200, 128*128))

    cur_img = 0
    for filename in os.listdir(Images_dir_male):
        filename = os.path.join(Images_dir_male, filename)
        img = io.imread(filename)
        inputImage = skimage.color.rgb2hsv(img)
        v = inputImage[:, :, 2]
        img_col_v = np.array(v, dtype='float64').flatten()  # flatten the 2d image into 1d
        testing_images_male[cur_img, :] = img_col_v[:]  # set the cur_img-th column to the current training image
        cur_img += 1

    cur_img = 0
    for filename in os.listdir(Images_dir_female):
        filename = os.path.join(Images_dir_female, filename)
        img = io.imread(filename)
        inputImage = skimage.color.rgb2hsv(img)
        v = inputImage[:, :, 2]
        img_col_v = np.array(v, dtype='float64').flatten()  # flatten the 2d image into 1d

        testing_images_female[cur_img, :] = img_col_v[:]  # set the cur_img-th column to the current training image
        VValues[cur_img, :] = img_col_v[:]
        cur_img += 1

    cur_dataPoint = 0
    for filename in os.listdir(Landmarks_dir_male):
        filename = os.path.join(Landmarks_dir_male, filename)
        train_data = scipy.io.loadmat(filename)
        values = train_data.get('lms')
        img_col = np.array(values, dtype='float64').flatten()
        testing_landmarks_male[cur_dataPoint, :] = img_col[:]  # set the cur_img-th column to the current training image
        cur_dataPoint += 1

    cur_dataPoint = 0
    for filename in os.listdir(Landmarks_dir_male_female):
        filename = os.path.join(Landmarks_dir_male_female, filename)
        train_data = scipy.io.loadmat(filename)
        values = train_data.get('lms')
        img_col = np.array(values, dtype='float64').flatten()
        testing_landmarks_female[cur_dataPoint, :] = img_col[:]  # set the cur_img-th column to the current training image
        cur_dataPoint += 1

    Transformed_test_faces_male = (pcaAppearence.transform(testing_images_male)).transpose()
    Transformed_test_landmarks_male = pcaLandmarks.transform(testing_landmarks_male).transpose()
    Transformed_test_faces_female = (pcaAppearence.transform(testing_images_female)).transpose()
    Transformed_test_landmarks_female = pcaLandmarks.transform(testing_landmarks_female).transpose()

    Transformed_X_male_combined = np.vstack((Transformed_test_faces_male, Transformed_test_landmarks_male))

    Transformed_X_female_combined = np.vstack((Transformed_test_faces_female, Transformed_test_landmarks_female))

    projected_combined_male = np.dot(Transformed_X_male_combined.transpose(), w_fisher_combined)
    projected_combined_female = np.dot(Transformed_X_female_combined.transpose(), w_fisher_combined)

    TruePredictions = 0
    FalsePredictions = 0
    print("threshold is", threshold)
    for i in range (0,200):
        if(projected_combined_male[i] > threshold):
            TruePredictions = TruePredictions+1
        else:
            FalsePredictions = FalsePredictions +1
        if (projected_combined_female[i] < threshold):
            TruePredictions = TruePredictions + 1
        else:
            FalsePredictions = FalsePredictions + 1

    total_error_rate  = FalsePredictions/(200)
    print("error rate(for geometry and appearence combined) is", total_error_rate)
    print("the fisher faces(for geometry and appearence combined) that distinguishes male from female is the point at", threshold)

pca_appearence, eigenFaces_vectors, mean = __TransformedMatrixWithMeanSubtractedAndFindEigen__(
    "C:\\Users\\vaish\\OneDrive\\Desktop\\UCLA\\Prml\\Project_1\\Project_1\\images", 800, 128 * 128, 50)

pca_geometry, eigenLandmarks_vectors = __TransformedLandmarkWithMeanSubtractedAndFindEigen__(
    "C:\\Users\\vaish\\OneDrive\\Desktop\\UCLA\\Prml\\Project_1\\Project_1\\landmarks", 800,68*2)

#-----------Part 1---------------------------
print("Running part 1: finding fihser for geometry and appearence combined")
projected_male_faces, projected_female_faces,complete_fisher_w, threshold = _findFisherFacesCombined_(pca_appearence, pca_geometry, "C:\\Users\\vaish\\OneDrive\\Desktop\\UCLA\\Prml\\Project_1\\Project_1\\male_training", "C:\\Users\\vaish\\OneDrive\\Desktop\\UCLA\\Prml\\Project_1\\Project_1\\female_training" , "C:\\Users\\vaish\\OneDrive\\Desktop\\UCLA\\Prml\\Project_1\\Project_1\\male_landmarks_train", "C:\\Users\\vaish\\OneDrive\\Desktop\\UCLA\\Prml\\Project_1\\Project_1\\female_landmarks_train" )

_ProjectandClassifyTestCombined("C:\\Users\\vaish\\OneDrive\\Desktop\\UCLA\\Prml\\Project_1\\Project_1\\Test_male_images_fisher",
                                "C:\\Users\\vaish\\OneDrive\\Desktop\\UCLA\\Prml\\Project_1\\Project_1\\Test_female_images_fisher",
                                "C:\\Users\\vaish\\OneDrive\\Desktop\\UCLA\\Prml\\Project_1\\Project_1\\Test_male_landmarks_fisher",
                                "C:\\Users\\vaish\\OneDrive\\Desktop\\UCLA\\Prml\\Project_1\\Project_1\\Test_female_landmarks_fisher",
                                complete_fisher_w, pca_appearence, pca_geometry ,threshold)

print("End of Part 1")
#-----------End of Part 1---------------------------

#-----------Part 2---------------------------
print("Running part 2: finding fihser for geometry and appearence individually")
projected_male_faces, projected_female_faces, w_fisher_faces,threshold = _findFisherFaces_(pca_appearence,  "C:\\Users\\vaish\\OneDrive\\Desktop\\UCLA\\Prml\\Project_1\\Project_1\\male_training", "C:\\Users\\vaish\\OneDrive\\Desktop\\UCLA\\Prml\\Project_1\\Project_1\\female_training", 400,400)

projected_male_landmarks, projected_female_landmarks, w_fisher_landmarks,threshold = _findFisherLandmarks_(pca_geometry, "C:\\Users\\vaish\\OneDrive\\Desktop\\UCLA\\Prml\\Project_1\\Project_1\\male_landmarks_train", "C:\\Users\\vaish\\OneDrive\\Desktop\\UCLA\\Prml\\Project_1\\Project_1\\female_landmarks_train", 400,400)




_ProjectandClassifyTest("C:\\Users\\vaish\\OneDrive\\Desktop\\UCLA\\Prml\\Project_1\\Project_1\\Test_male_images_fisher",
                                "C:\\Users\\vaish\\OneDrive\\Desktop\\UCLA\\Prml\\Project_1\\Project_1\\Test_female_images_fisher",
                                "C:\\Users\\vaish\\OneDrive\\Desktop\\UCLA\\Prml\\Project_1\\Project_1\\Test_male_landmarks_fisher",
                                "C:\\Users\\vaish\\OneDrive\\Desktop\\UCLA\\Prml\\Project_1\\Project_1\\Test_female_landmarks_fisher",w_fisher_faces, w_fisher_landmarks, pca_appearence, pca_geometry )

plt.figure(3)
plt.title("projecting all images in to the 2D-feature space learned by the fisher-faces")
plt.scatter(projected_male_faces, projected_male_landmarks )
plt.scatter(projected_female_faces, projected_female_landmarks )
plt.savefig("Outputs\Part3-FisherFaces\Part2-ProjectingAllImages.png")
plt.close
print("End of Part 2")
#-----------End of Part 2---------------------------