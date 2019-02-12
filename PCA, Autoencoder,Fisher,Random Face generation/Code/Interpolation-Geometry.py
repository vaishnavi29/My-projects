import tensorflow as tf
import numpy as np
from time import time
from datetime import datetime
import math
import matplotlib.pyplot as plt
# from PIL import Image
import os
from skimage import io

from google.colab import files
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
import scipy
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler

src = list(files.upload().values())[0]
open('mywarper.py', 'wb').write(src)
import mywarper
import scipy.io

training_dir = "/content/gdrive/My Drive/prml-train-images"
testing_dir = "/content/gdrive/My Drive/prml-testing-images"
dirOfTrainingLandmarks = "/content/gdrive/My Drive/prml-training-landmarks"
dirOfTrainingImages = "/content/gdrive/My Drive/prml-train-images"
dirOfTestLandmarks = "/content/gdrive/My Drive/prml-test-landmarks"
dirOfTestImages = "/content/gdrive/My Drive/prml-testing-images"
# batch_size = 100

scaler = None


def select_device(use_gpu=True):
    from tensorflow.python.client import device_lib
    print("list of  devices", device_lib.list_local_devices())
    device = '/device:GPU:0' if use_gpu else '/CPU:0'
    print('Using device: ', device)
    return device


device = select_device(use_gpu=True)


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
        X_train_Unflattened[cur_image,] = (img_input)

        cur_image += 1
    print("shape is", X_train_flattened.shape)
    print("shape X_train_Unflattened is", X_train_Unflattened.shape)
    return X_train_Unflattened  # , mean_pixel, std_pixel


X_train_images = load_data(dirOfTrainingImages, 800)


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


def load_data_test(dir, noOImages):
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

        # checking landmark on one training image
        # Imagefilename = os.path.join(testing_dir_images,"000001.jpg")
        # im = plt.imread(Imagefilename)
        # implot = plt.imshow(im)
        # plt.scatter(x=train_landmarks_values[:,0], y=train_landmarks_values[:,1], c='r', s=10)
        # plt.show()
        img_col = np.array(train_landmarks_values, dtype='float64').flatten()  # flatten the 2d image into 1d
        # print("image shape is", img_col.shape)

        X_train_flattened[cur_dataPoint, :] = img_col[:]  # set the cur_img-th column to the current training image
        cur_dataPoint += 1

    print("shape of test is", X_train_flattened.shape)
    # scaler = MinMaxScaler()
    # scaler.fit(X_train_Unflattened)
    # MinMaxScaler(copy=True, feature_range=(0, 1))
    # scaler.transform(X_train_Unflattened)
    # print("orig", X_train_flattened[0,])
    return X_train_flattened  # , mean_pixel, std_pixel


X_train_landmarks = load_data_test(dirOfTrainingLandmarks, 800)

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
    x1_1_fc = tf.contrib.layers.fully_connected(
        x, num_outputs=100,
        activation_fn=tf.nn.leaky_relu)  # weights_initializer=tf.random_normal_initializer(), biases_initializer=tf.random_normal_initializer())

    # block 2
    x2_1_fc = tf.contrib.layers.fully_connected(
        x1_1_fc, num_outputs=10,
        activation_fn=tf.nn.leaky_relu)  # weights_initializer=tf.random_normal_initializer(), biases_initializer=tf.random_normal_initializer())

    print("output from encoder is", (x2_1_fc).get_shape().as_list())

    return x2_1_fc


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


def decoder(x):
    # block 1
    print("shape in decoder is", x.shape)
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
    X = tf.placeholder("float", [None, 136])  # check

    encoder_op = encoder(X)
    decoder_op = decoder(encoder_op)

y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X
# Prediction


learning_rate = 0.007
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

sess2 = tf.Session()
sess2.run(init)
for epoch in range(0, 300):
    j = 0
    for i in range(0, num_steps):
        # Prepare Data
        batch_x = tf.placeholder("float", [100, 68 * 2])
        # print("shape of x_train is", np.shape(X_train))
        batch_x = (X_train_landmarks[j:j + 100, ])
        # print("shape of batch_x is", np.shape(batch_x))
        j = j + 100

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess2.run([optimizer, loss], feed_dict={X: batch_x})
        output_from_encoder_landmarks = (sess2.run(encoder_op, feed_dict={X: batch_x}))
        # Display logs per step
        # if i % display_step == 0 or i == 1:
        print('Step %i, epoch %i: Minibatch Loss: %f' % (i, epoch, l))

with sess2.as_default() as sess:
    print("shape of output_from_encoder_landmarks is", output_from_encoder_landmarks.shape)

    # variance_of_latent_var = np.var(output_from_encoder, axis=1)
    # idx = np.argsort(variance_of_latent_var)
    # sorted_latent_var = output_from_encoder[idx]

    # picking the top 4 laten variables with max variance
    # five_latent = sorted_latent_var[0:4]
    # print("shape of output_from_encoder is", five_latent.shape)
    for i in range(0, 100):
        latent = output_from_encoder_landmarks[i]
        variance_of_latent_var = np.var(latent)
        idx = np.argsort(variance_of_latent_var)
        sorted_latent_var = latent[idx]
        output_from_encoder_landmarks[i] = latent
    ten_images_for_each_dimension = np.empty(shape=(10, 128, 128, 3), dtype='float64')
    print("shape of output_from_encoder_landmarks is", output_from_encoder_landmarks.shape)
    originaloutput_from_encoder = output_from_encoder_landmarks
    for i in range(0, 4):
        output_from_encoder_landmarks = originaloutput_from_encoder
        max_image_vl = np.amax(output_from_encoder_landmarks[:, i], axis=0)
        min_image_vl = np.amin(output_from_encoder_landmarks[:, i], axis=0)
        step_size = (max_image_vl - min_image_vl) / 10
        randValue = np.random.uniform(max_image_vl, min_image_vl)

        prevRandValue = randValue
        output_from_encoder_landmarks[0, i] = randValue
        value_to_feed = output_from_encoder_landmarks[0, :].reshape(1, 10)
        g1 = (decoder_op.eval(feed_dict={encoder_op: value_to_feed})) * 128

        # plt.imshow(g1.reshape(128, 128, 3))
        # plt.show()
        warped_image = mywarper.warp(X_train_images[0, :, :, :], X_train_landmarks[0, :].reshape(68, 2),
                                     g1.reshape(68, 2))
        ten_images_for_each_dimension[0,] = warped_image
        for j in range(1, 10):
            randValue = prevRandValue + step_size
            prevRandValue = randValue

            output_from_encoder_landmarks[0, i] = randValue
            value_to_feed = output_from_encoder_landmarks[0, :].reshape(1, 10)
            g1 = (decoder_op.eval(feed_dict={encoder_op: value_to_feed})) * 128

            warped_image = mywarper.warp(X_train_images[0, :, :, :], (X_train_landmarks[0, :] * 128).reshape(68, 2),
                                         g1.reshape(68, 2))
            ten_images_for_each_dimension[j,] = warped_image
            step_size = step_size * 2
            # plt.imshow(X_train_images[0, :, :, :].reshape([128, 128, 3])/255)
            # plt.savefig('/content/gdrive/My Drive/prml-outputs/interpolation-geo/interpolationgeometry_orig'+ str(i)+  str(j)+'.png')
            plt.imshow(ten_images_for_each_dimension[j].reshape([128, 128, 3]) / 255)
            plt.savefig('/content/gdrive/My Drive/prml-outputs/interpolation-geo/interpolationgeometry' + str(i) + str(
                j) + '.png')
            # plt.imshow(ten_images_for_each_dimension[i].reshape([128, 128, 3]))