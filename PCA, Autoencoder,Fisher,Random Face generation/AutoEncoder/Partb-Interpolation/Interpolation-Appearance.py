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

scaler = None


def select_device(use_gpu=True):
    from tensorflow.python.client import device_lib
    print("list of  devices", device_lib.list_local_devices())
    device = '/device:GPU:0' if use_gpu else '/CPU:0'
    print('Using device: ', device)
    return device


device = select_device(use_gpu=True)


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
        X_train_Unflattened[cur_image,] = (img_input / 255)

        cur_image += 1
    print("shape is", X_train_flattened.shape)
    print("shape X_train_Unflattened is", X_train_Unflattened.shape)
    return X_train_Unflattened  # , mean_pixel, std_pixel


X_train = load_data(dirOfTrainingImages, 800)

conv_w1_encode = tf.Variable(kaiming_normal([5, 5, 3, 16]))
conv_b1_encode = tf.Variable(tf.zeros(16, ))

conv_w2_encode = tf.Variable(kaiming_normal([3, 3, 16, 32]))
conv_b2_encode = tf.Variable(tf.zeros(32, ))

conv_w3_encode = tf.Variable(kaiming_normal([3, 3, 32, 64]))
conv_b3_encode = tf.Variable(tf.zeros(64, ))

conv_w4_encode = tf.Variable(kaiming_normal([3, 3, 64, 128]))
conv_b4_encode = tf.Variable(tf.zeros(128, ))

conv_w5_encode = tf.Variable(kaiming_normal([10, 10, 128, 50]))
conv_b5_encode = tf.Variable(tf.zeros(50, ))


def encoder(x):
    # block 1
    # x1_1_pad = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', constant_values=0)
    x1_1_conv = tf.nn.conv2d(x, conv_w1_encode, [1, 2, 2, 1], padding='SAME') + conv_b1_encode
    x1_3_relu = tf.nn.leaky_relu(x1_1_conv)

    # block 2
    # x2_1_pad = tf.pad(x1_3_relu, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', constant_values=0)
    x2_1_conv = tf.nn.conv2d(x1_3_relu, conv_w2_encode, [1, 2, 2, 1], padding='SAME') + conv_b2_encode
    x2_3_relu = tf.nn.leaky_relu(x2_1_conv)

    # block 3
    # x3_1_pad = tf.pad(x2_3_relu, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', constant_values=0)
    x3_1_conv = tf.nn.conv2d(x2_3_relu, conv_w3_encode, [1, 2, 2, 1], padding='SAME') + conv_b3_encode
    x3_3_relu = tf.nn.leaky_relu(x3_1_conv)

    # block 4
    # x4_1_pad = tf.pad(x3_3_relu, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', constant_values=0)
    x4_1_conv = tf.nn.conv2d(x3_3_relu, conv_w4_encode, [1, 2, 2, 1], padding='SAME') + conv_b4_encode
    x4_2_relu = tf.nn.leaky_relu(x4_1_conv)

    # block 5
    x5_1_pad = tf.pad(x4_2_relu, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', constant_values=0)
    x5_2_conv = tf.nn.conv2d(x5_1_pad, conv_w5_encode, [1, 2, 2, 1], padding='VALID') + conv_b5_encode
    x5_3_relu = tf.nn.leaky_relu(x5_2_conv)

    print("output from encoder is", (x5_3_relu).shape)

    return x5_3_relu


# conv_w1_decode = tf.Variable(kaiming_normal([1, 1, 50]))
# conv_b1_decode = tf.Variable(tf.zeros(128, ))

conv_w1_decode = tf.Variable(kaiming_normal([8, 8, 128, 50]))
conv_b1_decode = tf.Variable(tf.zeros(128, ))

conv_w2_decode = tf.Variable(kaiming_normal([3, 3, 64, 128]))
# conv_w2_decode = tf.Variable(kaiming_normal([3, 3, 64, 128]))
conv_b2_decode = tf.Variable(tf.zeros(64, ))

conv_w3_decode = tf.Variable(kaiming_normal([3, 3, 32, 64]))
conv_b3_decode = tf.Variable(tf.zeros(32, ))

conv_w4_decode = tf.Variable(kaiming_normal([3, 3, 16, 32]))
conv_b4_decode = tf.Variable(tf.zeros(16, ))

conv_w5_decode = tf.Variable(kaiming_normal([5, 5, 3, 16]))
conv_b5_decode = tf.Variable(tf.zeros(3, ))


def decoder(x):
    # block 1
    print("shape of x", x.shape)
    # x = tf.reshape(x, [8, 8, 50])

    x1_2_conv = tf.layers.conv2d_transpose(x, filters=128, kernel_size=(8, 8), strides=(1, 1), padding='VALID',
                                           activation=tf.nn.leaky_relu)

    print("shape of decoder x1_3_relu is", x1_2_conv.get_shape().as_list())

    x2_2_conv = tf.layers.conv2d_transpose(x1_2_conv, filters=64, kernel_size=(3, 3), strides=(2, 2), padding='SAME',
                                           activation=tf.nn.leaky_relu)

    print("shape of decoder x2_3_relu is", x2_2_conv.shape)

    x3_2_conv = tf.layers.conv2d_transpose(x2_2_conv, filters=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME',
                                           activation=tf.nn.leaky_relu)

    print("shape of decoder x3_3_relu is", x3_2_conv.get_shape().as_list())

    x4_2_conv = tf.layers.conv2d_transpose(x3_2_conv, 16, [3, 3], (2, 2), padding='SAME', activation=tf.nn.leaky_relu)

    print("shape of decoder x4_3_relu is", x4_2_conv.get_shape().as_list())

    x5_2_conv = tf.layers.conv2d_transpose(x4_2_conv, 3, [5, 5], (2, 2), padding='SAME', activation=tf.nn.sigmoid)

    print("output from decoder is", x5_2_conv.get_shape().as_list())
    return x5_2_conv


with tf.device(device):
    X = tf.placeholder("float", [None, 128, 128, 3])  # check
    encoder_op = encoder(X)
    decoder_op = decoder(encoder_op)

y_pred = decoder_op
y_true = X

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

# Training
sess1 = tf.Session()
sess1.run(init)
output_from_encoder = []
for epoch in range(0, 300):
    j = 0
    for i in range(0, num_steps):
        # Prepare Data
        batch_x = tf.placeholder("float", [100, 128, 128, 3])
        batch_x = (X_train[j:j + 100, :, :, :])
        j = j + 100
        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess1.run([optimizer, loss], feed_dict={X: batch_x})
        output_from_encoder = (sess1.run(encoder_op, feed_dict={X: batch_x}))

        print('Step %i epoch %i: Minibatch Loss: %f' % (i, epoch, l))

ten_images_for_each_dimension = np.empty(shape=(10, 128, 128, 3), dtype='float64')

with sess1.as_default() as sess:
    print("shape of output_from_encoder is", output_from_encoder.shape)
    for i in range(0, 100):
        latent = output_from_encoder[i, :, :, :]
        variance_of_latent_var = np.var(latent)
        idx = np.argsort(variance_of_latent_var)
        sorted_latent_var = latent[idx]
        output_from_encoder[i, :, :, :] = latent

    print("shape of output_from_encoder is", output_from_encoder.shape)
    originaloutput_from_encoder = output_from_encoder
    for i in range(0, 4):
        output_from_encoder = originaloutput_from_encoder
        max_image_vl = np.amax(output_from_encoder[:, :, :, i], axis=0)
        min_image_vl = np.amin(output_from_encoder[:, :, :, i], axis=0)
        step_size = (max_image_vl - min_image_vl) / 10
        randValue = np.random.uniform(max_image_vl, min_image_vl)
        prevRandValue = randValue
        output_from_encoder[0, :, :, i] = randValue

        value_to_feed = output_from_encoder[0, :, :, :].reshape(1, 1, 1, 50)
        g1 = (decoder_op.eval(feed_dict={encoder_op: value_to_feed}))

        plt.imshow(g1.reshape(128, 128, 3))
        plt.show()
        ten_images_for_each_dimension[0,] = g1
        for j in range(1, 10):
            randValue = prevRandValue + step_size
            prevRandValue = randValue
            # step_size= step_size *2
            output_from_encoder[0, :, :, i] = randValue
            # output_from_encoder[0,:, :,i] = np.random.uniform(max_image_vl, min_image_vl)

            value_to_feed = output_from_encoder[0, :, :, :].reshape(1, 1, 1, 50)
            g1 = (decoder_op.eval(feed_dict={encoder_op: value_to_feed}))

            ten_images_for_each_dimension[j,] = g1
            print("dispalying image " + str(j) + " for dimension ", str(i))
            plt.imshow(ten_images_for_each_dimension[j].reshape([128, 128, 3]))
            plt.savefig('/content/gdrive/My Drive/prml-outputs/interpolation/' + str(i) + str(j) + '.png')
            plt.show()