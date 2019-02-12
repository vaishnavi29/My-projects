from __future__ import division, print_function, absolute_import

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, TensorBoard

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# Hyper Parameter
batch_size = 86
epochs = 30

# Set up TensorBoard
tensorboard = TensorBoard(batch_size=batch_size)

with h5py.File("full_dataset_vectors.h5", 'r') as h5:
    X_train, y_train = h5["X_train"][:], h5["y_train"][:]
    X_test, y_test = h5["X_test"][:], h5["y_test"][:]



y_train = to_categorical(y_train, num_classes=10)
# y_test = to_categorical(y_test, num_classes=10)

X_train = X_train.reshape(-1, 16, 16, 16)
X_test  = X_test.reshape(-1, 16, 16, 16)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
# Conv2D layer
def Conv(filters=16, kernel_size=(3,3), activation='relu', input_shape=None):
    if input_shape:
        return Conv2D(filters=filters, kernel_size=kernel_size, padding='Same', activation=activation, input_shape=input_shape)
    else:
        return Conv2D(filters=filters, kernel_size=kernel_size, padding='Same', activation=activation)

# Define Model
def CNN(input_dim, num_classes):
    model = Sequential()

    model.add(Conv(8, (3,3), input_shape=input_dim))
    model.add(Conv(16, (3,3)))
    # model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv(32, (3,3)))
    model.add(Conv(64, (3,3)))
    model.add(BatchNormalization())
    model.add(MaxPool2D())
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    return model

# Train Model
def train(optimizer, scheduler, gen):
    global model

    print("Training...")
    model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])

    model.fit_generator(gen.flow(X_train, y_train, batch_size=batch_size),
                    epochs=epochs, validation_data=(X_val, y_val),
                    verbose=2, steps_per_epoch=X_train.shape[0]//batch_size,
                    callbacks=[scheduler, tensorboard])
def evaluate():
    global model

    pred = model.predict(X_test)
    pred = np.argmax(pred, axis=1)

    print(accuracy_score(pred,y_test))
    # Heat Map
    array = confusion_matrix(y_test, pred)
    cm = pd.DataFrame(array, index = range(10), columns = range(10))
    plt.figure(figsize=(20,20))
    sns.heatmap(cm, annot=True)
    plt.show()

def save_model():
    global model

    model_json = model.to_json()
    with open('model/model_2D.json', 'w') as f:
        f.write(model_json)

    model.save_weights('model/model_2D.h5')

    print('Model Saved.')

def load_model():
    f = open('model/model_2D.json', 'r')
    model_json = f.read()
    f.close()

    loaded_model = model_from_json(model_json)
    loaded_model.load_weights('model/model_2D.h5')

    print("Model Loaded.")
    return loaded_model

if __name__ == '__main__':

    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    scheduler = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=1e-5)

    model = CNN((16,16,16), 10)

    gen = ImageDataGenerator(rotation_range=10, zoom_range = 0.1, width_shift_range=0.1, height_shift_range=0.1)
    gen.fit(X_train)

    train(optimizer, scheduler, gen)
    evaluate()
    save_model()