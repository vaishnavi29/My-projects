% xmode
Verbose
import tensorflow as tf
import numpy as np
from time import time
from datetime import datetime
import math
# import matplotlib.pyplot as plt
# from PIL import Image
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from skimage import io
import skimage
from PIL import Image
from google.colab import files
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
import scipy
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
# src = list(files.upload().values())[0]
# open('mywarper.py','wb').write(src)
import mywarper
import scipy.io

training_dir = "/content/gdrive/My Drive/prml-train-images"
testing_dir = "/content/gdrive/My Drive/prml-testing-images"
dirOfTrainingLandmarks = "/content/gdrive/My Drive/prml-training-landmarks"
dirOfTrainingImages = "/content/gdrive/My Drive/prml-train-images"
dirOfTestLandmarks = "/content/gdrive/My Drive/prml-test-landmarks"
dirOfTestImages = "/content/gdrive/My Drive/prml-testing-images"
# batch_size = 100

noOfTrainingLandmarks = 800
noOfTrainingImages = 800
noOfTestLandmarks = 200
noOfTestImages = 200
dimensionOfLandmarks = 68 * 2
dimensionOfImages = 128 * 128

Training_Landmarks_Matrix_BeforeMean = np.empty(shape=(noOfTrainingLandmarks, dimensionOfLandmarks), dtype='float64')
Training_Landmarks_Matrix_AfterMean = np.empty(shape=(noOfTrainingLandmarks, dimensionOfLandmarks), dtype='float64')
Test_Landmarks_Matrix_BeforeMean = np.empty(shape=(noOfTrainingLandmarks, dimensionOfLandmarks), dtype='float64')


def select_device(use_gpu=True):
    from tensorflow.python.client import device_lib
    print("list of  devices", device_lib.list_local_devices())
    device = '/device:GPU:0' if use_gpu else '/CPU:0'
    print('Using device: ', device)
    return device


device = select_device(use_gpu=True)


def __FindMeanOfMatrix__(X, noOfDataPoints):
    mean_img_col = np.sum(X, axis=0) / noOfDataPoints
    print("mean is", mean_img_col.shape)
    return mean_img_col


def kaiming_normal(shape):
    """
    He et al, *Delving Deep into Rectifiers: Surpassing Human-Level Performance on
    ImageNet Classification, ICCV 2015, https://arxiv.org/abs/1502.01852
    """
    if len(shape) == 2:
        fan_in, fan_out = shape[0], shape[1]
    elif len(shape) == 4:
        fan_in, fan_out = np.prod(shape[:3]), shape[3]
    return tf.random_normal(shape) * np.sqrt(2.0 / fan_in)


def load_data(dir, noOImages):
    width = 128
    height = 128
    channels = 3
    X_train_flattened = np.empty(shape=(noOImages, width * height * channels))
    X_train_Unflattened = np.empty(shape=(noOImages, width, height, channels))
    cur_image = 0
    for filename in os.listdir(dir):
        filename = os.path.join(dir, filename)
        img_input = io.imread(filename)

        # X_train_flattened[cur_image,:] = (img_input.flatten())
        # img_input = normalize
        X_train_Unflattened[cur_image,] = (img_input / 255)
        g = skimage.color.rgb2gray(X_train_Unflattened[cur_image,])

        # io.imshow(g, cmap = 'gray')
        cur_image += 1
    print("shape is", X_train_flattened.shape)
    print("shape X_train_Unflattened is", X_train_Unflattened.shape)
    return X_train_Unflattened  # , mean_pixel, std_pixel


def get_train_Landmark():
    cur_dataPoint = 0
    for filename in os.listdir(dirOfTrainingLandmarks):
        filename = os.path.join(dirOfTrainingLandmarks, filename)
        train_landmark = scipy.io.loadmat(filename)
        training_Landmark_values = train_landmark.get('lms')

        trainingLandmarks_col = np.array(training_Landmark_values,
                                         dtype='float64').flatten()  # flatten the 2d image into 1d

        Training_Landmarks_Matrix_BeforeMean[cur_dataPoint, :] = trainingLandmarks_col[
                                                                 :]  # set the cur_img-th column to the current training image
        cur_dataPoint += 1
        return Training_Landmarks_Matrix_BeforeMean


def get_test_Landmark():
    cur_dataPoint = 0
    for filename in os.listdir(dirOfTestLandmarks):
        filename = os.path.join(dirOfTestLandmarks, filename)
        test_landmark = scipy.io.loadmat(filename)
        test_Landmark_values = test_landmark.get('lms') / 128

        testLandmarks_col = np.array(test_Landmark_values,
                                     dtype='float64').flatten()  # flatten the 2d image into 1d

        Test_Landmarks_Matrix_BeforeMean[cur_dataPoint, :] = testLandmarks_col[
                                                             :]  # set the cur_img-th column to the current training image
        cur_dataPoint += 1
        return Test_Landmarks_Matrix_BeforeMean


print("reading the training landmarks")
Training_Landmarks_Matrix_BeforeMean = get_train_Landmark()
X_test_landmarks = get_test_Landmark()

print("computing the mean of training landmarks")
mean_Landmark_col = __FindMeanOfMatrix__(Training_Landmarks_Matrix_BeforeMean, noOfTrainingLandmarks)

for j in range(0, noOfTrainingLandmarks):  # subtract from all training images
    Training_Landmarks_Matrix_AfterMean[j, :] = Training_Landmarks_Matrix_BeforeMean[j, :] - mean_Landmark_col[:]
print("computed the training landmark after minusing mean")

X_train = load_data(training_dir, 800)

Training_Images_After_Warping_To_Mean = np.empty(shape=(noOfTrainingImages, 128, 128, 3), dtype='float64')
# warp the training images to mean landmark
print("warping the training images to mean landmark")
for i in range(0, noOfTrainingImages):
    img = X_train[i, :, :, :]
    warped_Image_To_Mean = mywarper.warp(img, Training_Landmarks_Matrix_BeforeMean[i, :].reshape(68, 2),
                                         mean_Landmark_col[:].reshape(68, 2))
    Training_Images_After_Warping_To_Mean[i, :, :, :] = warped_Image_To_Mean
    i = i + 1

conv_w1_encode = tf.Variable(kaiming_normal([5, 5, 3, 16]))
conv_b1_encode = tf.Variable(tf.zeros(16, ))

conv_w2_encode = tf.Variable(kaiming_normal([3, 3, 16, 32]))
conv_b2_encode = tf.Variable(tf.zeros(32, ))

conv_w3_encode = tf.Variable(kaiming_normal([3, 3, 32, 64]))
conv_b3_encode = tf.Variable(tf.zeros(64, ))

conv_w4_encode = tf.Variable(kaiming_normal([3, 3, 64, 128]))
conv_b4_encode = tf.Variable(tf.zeros(128, ))

conv_w5_encode = tf.Variable(kaiming_normal([8, 8, 128, 50]))
conv_b5_encode = tf.Variable(tf.zeros(50, ))


def encoder(x):
    # block 1
    x1_1_pad = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', constant_values=0)
    x1_1_conv = tf.nn.conv2d(x1_1_pad, conv_w1_encode, [1, 2, 2, 1], padding='VALID') + conv_b1_encode
    x1_3_relu = tf.nn.leaky_relu(x1_1_conv)

    # block 2
    x2_1_conv = tf.nn.conv2d(x1_3_relu, conv_w2_encode, [1, 2, 2, 1], padding='SAME') + conv_b2_encode
    x2_3_relu = tf.nn.leaky_relu(x2_1_conv)

    # block 3
    x3_1_conv = tf.nn.conv2d(x2_3_relu, conv_w3_encode, [1, 2, 2, 1], padding='SAME') + conv_b3_encode
    x3_3_relu = tf.nn.leaky_relu(x3_1_conv)

    # block 4
    x4_1_conv = tf.nn.conv2d(x3_3_relu, conv_w4_encode, [1, 2, 2, 1], padding='SAME') + conv_b4_encode
    x4_2_relu = tf.nn.leaky_relu(x4_1_conv)

    # block 5
    x5_2_conv = tf.nn.conv2d(x4_2_relu, conv_w5_encode, [1, 1, 1, 1], padding='SAME') + conv_b5_encode
    x5_3_relu = tf.nn.leaky_relu(x5_2_conv)

    print("output from encoder is", (x5_3_relu).get_shape().as_list())

    return x5_3_relu


conv_w1_decode = tf.Variable(kaiming_normal([8, 8, 128, 50]))
conv_b1_decode = tf.Variable(tf.zeros(128, ))

conv_w2_decode = tf.Variable(kaiming_normal([3, 3, 64, 128]))
conv_b2_decode = tf.Variable(tf.zeros(64, ))

conv_w3_decode = tf.Variable(kaiming_normal([3, 3, 32, 64]))
conv_b3_decode = tf.Variable(tf.zeros(32, ))

conv_w4_decode = tf.Variable(kaiming_normal([3, 3, 16, 32]))
conv_b4_decode = tf.Variable(tf.zeros(16, ))

conv_w5_decode = tf.Variable(kaiming_normal([5, 5, 3, 16]))
conv_b5_decode = tf.Variable(tf.zeros(3, ))


def decoder(x, batch_size):
    # block 1
    print("shape of x", tf.shape(x))

    print("shape of decoder x is", x.get_shape().as_list())
    x1_2_conv = tf.nn.conv2d_transpose(x, conv_w1_decode, output_shape=[batch_size, 8, 8, 128], strides=[1, 1, 1, 1],
                                       padding='SAME') + conv_b1_decode

    x1_3_relu = tf.nn.leaky_relu(x1_2_conv)
    print("shape of decoder x1_3_relu is", x1_3_relu.get_shape().as_list())

    # block 2
    x2_2_conv = tf.nn.conv2d_transpose(x1_3_relu, conv_w2_decode, output_shape=[batch_size, 16, 16, 64],
                                       strides=[1, 2, 2, 1], padding='SAME') + conv_b2_decode
    x2_3_relu = tf.nn.leaky_relu(x2_2_conv)
    print("shape of decoder x2_3_relu is", x2_3_relu.get_shape().as_list())

    # block 3
    x3_2_conv = tf.nn.conv2d_transpose(x2_3_relu, conv_w3_decode, output_shape=[batch_size, 32, 32, 32],
                                       strides=[1, 2, 2, 1], padding='SAME') + conv_b3_decode
    x3_3_relu = tf.nn.leaky_relu(x3_2_conv)
    print("shape of decoder x3_3_relu is", x3_3_relu.get_shape().as_list())

    # block 4

    x4_2_conv = tf.nn.conv2d_transpose(x3_3_relu, conv_w4_decode, output_shape=[batch_size, 64, 64, 16],
                                       strides=[1, 2, 2, 1], padding='SAME') + conv_b4_decode
    x4_3_relu = tf.nn.leaky_relu(x4_2_conv)
    print("shape of decoder x4_3_relu is", x4_3_relu.get_shape().as_list())

    # block 5
    x5_2_conv = tf.nn.conv2d_transpose(x4_3_relu, conv_w5_decode, output_shape=[batch_size, 128, 128, 3],
                                       strides=[1, 2, 2, 1], padding='SAME') + conv_b5_decode
    x5_3_relu = tf.sigmoid(x5_2_conv)

    print("output from decoder is", x5_3_relu.get_shape().as_list())
    return x5_3_relu


with tf.device(device):
    X = tf.placeholder("float", [None, None, None, None])  # check
    batch_size = tf.placeholder(tf.int32)
    encoder_op = encoder(X)
    decoder_op = decoder(encoder_op, batch_size)

y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X
# Prediction


learning_rate = 0.0007
num_steps = 7
# num_steps = 5

# Define loss and optimizer, minimize the squared error
print("shape of y_true is", np.shape(y_true))
print("shape of y_pred is", np.shape(y_pred))
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

display_step = 50

# Start Training
# Start a new TF session


with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    # Training

    for epoch in range(0, 100):
        j = 0
        for i in range(0, num_steps):
            # Prepare Data
            batch_x = tf.placeholder("float", [100, 128, 128, 3])
            batch_x = (Training_Images_After_Warping_To_Mean[j:j + 100, :, :, :])
            j = j + 100

            _, l = sess.run([optimizer, loss], feed_dict={X: batch_x, batch_size: 100})
            # Display logs per step
            # if i % display_step == 0 or i == 1:
            print('Step %i: epoch %i: Minibatch Loss: %f' % (i, epoch, l))

    # Encode and decode images from test set and visualize their reconstruction.
    X_Test = load_data(testing_dir, 200)
    n = 4
    j = 0

    Testing_Images_After_Warping_To_Mean = np.empty(shape=(200, 128, 128, 3), dtype='float64')
    # warp the training images to mean landmark
    print("warping the training images to mean landmark")
    for i in range(0, 200):
        img = X_train[i, :, :, :]
        warped_Image_To_Mean = mywarper.warp(img, X_test_landmarks[i,].reshape(68, 2),
                                             mean_Landmark_col[:].reshape(68, 2))
        Testing_Images_After_Warping_To_Mean[i, :, :, :] = warped_Image_To_Mean
        i = i + 1
        # plt.imshow(warped_Image_To_Mean)

    batch_x = (Testing_Images_After_Warping_To_Mean[0:0 + 200, :, :, :])
    output = (sess.run(decoder_op, feed_dict={X: batch_x, batch_size: 200}))
    reconstructed_Test_images = output


for i in range(0, 200):
    scipy.misc.imsave('/content/gdrive/My Drive/prml-outputs/reconstructed_images_before_warp/'
                      + str(i) + '.jpg', reconstructed_Test_images[i].reshape([128, 128, 3]))
    # plt.imshow(output[i].reshape([128,128,3]))
    # plt.show()

    # -------------------------------End of Appearance Training-------------------------------------------------------------------#

batch_size = 100


def load_data_landmarks(dir, noOImages):
    width = 68
    height = 2
    channels = 1
    cur_dataPoint = 0
    X_train_flattened = np.empty(shape=(noOImages, width * height * channels))
    X_train_Unflattened = np.empty(shape=(noOImages, width, height, channels))
    cur_image = 0
    for filename in os.listdir(dir):
        filename = os.path.join(dir, filename)
        train_landmarks = scipy.io.loadmat(filename)

        train_landmarks_values = (train_landmarks.get('lms') / 128)
        # train_landmarks_values =np.array((train_landmarks_values - np.min(train_landmarks_values)) / (np.max(train_landmarks_values) - np.min(train_landmarks_values)))


        img_col = np.array(train_landmarks_values, dtype='float64').flatten()  # flatten the 2d image into 1d
        # print("image shape is", img_col.shape)

        X_train_flattened[cur_dataPoint, :] = img_col[:]  # set the cur_img-th column to the current training image
        cur_dataPoint += 1

    print("shape of test is", X_train_flattened.shape)

    return X_train_flattened  # , mean_pixel, std_pixel


X_train_landmarks = load_data_landmarks(dirOfTrainingLandmarks, 800)


def encoder_landmarks(x):
    # block 1
    x1_1_fc = tf.contrib.layers.fully_connected(
        x, num_outputs=100,
        activation_fn=tf.nn.leaky_relu)  # weights_initializer=tf.random_normal_initializer(), biases_initializer=tf.random_normal_initializer())

    # block 2
    x2_1_fc = tf.contrib.layers.fully_connected(
        x1_1_fc, num_outputs=10,
        activation_fn=tf.nn.leaky_relu)  # weights_initializer=tf.random_normal_initializer(), biases_initializer=tf.random_normal_initializer())

    print("output from encoder is", (x2_1_fc).get_shape().as_list())

    return x2_1_fc


def decoder_landmarks(x):
    # block 1
    x1_1_fc = tf.contrib.layers.fully_connected(
        x, num_outputs=100,
        activation_fn=tf.nn.leaky_relu)  # weights_initializer=tf.random_normal_initializer(), biases_initializer=tf.random_normal_initializer())

    # block 2
    x2_1_fc = tf.contrib.layers.fully_connected(
        x1_1_fc, num_outputs=68 * 2,
        activation_fn=tf.nn.leaky_relu)  # weights_initializer=tf.random_normal_initializer(), biases_initializer=tf.random_normal_initializer())

    print("output from decoder is", (x2_1_fc).get_shape().as_list())

    return x2_1_fc


with tf.device(device):
    X = tf.placeholder("float", [None, 136])
    encoder_op_landmarks = encoder_landmarks(X)
    decoder_op_landmarks = decoder_landmarks(encoder_op_landmarks)

y_pred_landmarks = decoder_op_landmarks
# Targets (Labels) are the input data.
y_true_landmarks = X
# Prediction


learning_rate_landmarks = 0.0007
num_steps = 7
# num_steps = 5

# Define loss and optimizer, minimize the squared error
print("shape of y_true is", np.shape(y_true_landmarks))
print("shape of y_pred is", np.shape(y_pred_landmarks))
loss_landmarks = tf.reduce_mean(tf.pow(y_true_landmarks - y_pred_landmarks, 2))
optimizer_landmarks = tf.train.AdamOptimizer(learning_rate_landmarks).minimize(loss_landmarks)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

display_step = 50


print("training the geometry auto-encoder on X_train_landmarks")
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    for epoch in range(0, 1):
        j = 0
        for i in range(0, num_steps):
            # Prepare Data
            batch_x = tf.placeholder("float", [batch_size, 68 * 2])
            # print("shape of x_train is", np.shape(X_train))
            batch_x = (X_train_landmarks[j:j + batch_size, ])
            # print("shape of batch_x is", np.shape(batch_x))
            j = j + batch_size

            # Run optimization op (backprop) and cost op (to get loss value)
            _, l = sess.run([optimizer_landmarks, loss_landmarks], feed_dict={X: batch_x})
            # Display logs per step
            # if i % display_step == 0 or i == 1:
            print('Step %i, epoch %i: Minibatch Loss for landmarks: %f' % (i, epoch, l))


    n = 4
    j = 0

    batch_test_landmarks = tf.placeholder("float", [200, 68 * 2])
    batch_test_landmarks = (X_test_landmarks[0:200, ])

    reconstrcuted_test_landmarks = (sess.run(decoder_op_landmarks, feed_dict={X: batch_test_landmarks})) * 128
    test = reconstrcuted_test_landmarks[0,].reshape([68, 2])
    Imagefilename = os.path.join(dirOfTrainingImages, "000880.jpg")
    im = plt.imread(Imagefilename)
    implot = plt.imshow(im)
    plt.scatter(x=test[:, 0], y=test[:, 1], c='r', s=10)

    plt.savefig("/content/gdrive/My Drive/prml-outputs/Part2-ProjectingAllTrainingImages.png")
    plt.show()

# -------------------------------End of Landmarks Training-------------------------------------------------------------------#
# -------------------------------Begin Appearence recontrcution and warping to original landmarks-------------------------------------------------------------------#


Reconstructed_Image_WarpedTo_Mean = np.empty(shape=(noOfTestImages, 128, 128, 3), dtype='float64')

for j in range(0, noOfTestImages):
    Reconstructed_Image_WarpedTo_Mean[j,:,:,:] = mywarper.warp(reconstructed_Test_images[j], mean_Landmark_col[:].reshape(68, 2), (reconstrcuted_test_landmarks[j,:]).reshape(68, 2))

    plt.imshow(Reconstructed_Image_WarpedTo_Mean[0,:,:,:])
    plt.savefig(os.path.join("/content/gdrive/My Drive/prml-outputs/reconstructed_images_after_warp", str(j) + ".jpg"))
    # plt.show()
    scipy.misc.imsave(os.path.join("/content/gdrive/My Drive/prml-outputs/reconstructed_images_after_warp", str(j) + ".jpg"), Reconstructed_Image_WarpedTo_Mean[j,:,:,:])