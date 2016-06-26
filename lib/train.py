import os

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from keras.layers.core import Dropout, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adagrad
import h5py

from core import timeit


def string_labels_to_binary_vectors(y_labels, label_map):
    # Convert the string labels to numberical values.
    y = [label_map[label] for label in y_labels]
    # Now we need to convert labels from numerical value to 0/1 vectors
    # http://stackoverflow.com/questions/31997366/python-keras-shape-mismatch-error
    y = np_utils.to_categorical(y)
    return y


def expand_features_and_labels(x_feat, y_labels):
    """
    Take features and labels and expand them into labelled examples for the
    model.
    """
    x_expanded = []
    y_expanded = []
    for x, y in zip(x_feat, y_labels):
        for segment in x:
            x_expanded.append(segment)
            y_expanded.append(y)
    return x_expanded, y_expanded


def load_model_from_architecture_string(architecture_json_string):
    model = model_from_json(architecture_json_string)
    return model


def GetModel():
    # train a dnn
    # see: http://keras.io/
    model = Sequential()
    #model.add(Dropout(0.5, input_shape=(len(X[0]),)))

    # examples are 21 frames of 128-band mel features (21 * 128 = 2688)
    # examples are 21 frames of 40-band mel features (21 * 40 = 840)
    # input: 100x100 images with 3 channels -> (1, 128, 100) tensors.
    # this applies 8 convolution filters of size 8x3 each.
    #model.add(Reshape((1,128,21), input_shape=(2688,)))
    #model.add(Convolution2D(8, 8, 3, border_mode='valid', input_shape=(1, 128, 21)))
    #NUMBER_MELS = 40  # for later
    #model.add(Reshape((1,40,21), input_shape=(40*21,)))
    model.add(Reshape((1,21,40), input_shape=(40*21,)))
    NUM_CONVOLUTIONS = 8
    #model.add(Convolution2D(NUM_CONVOLUTIONS, 10, 7, border_mode='valid', input_shape=(1, 40, 21)))
    model.add(Convolution2D(NUM_CONVOLUTIONS, 7, 10, border_mode='valid', input_shape=(1, 21, 40)))
    convout1 = Activation('relu')  # named variable used for visualization
    model.add(convout1)
    #model.add(Convolution2D(32, 3, 3))
    #model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(output_dim=32, init="uniform"))  # MAYBE glorot_uniform IS BAD?, input_dim=len(X[0]),
    model.add(Activation("relu"))
    model.add(Dropout(0.25))

    #model.add(Dense(output_dim=16, init="uniform"))  # MAYBE glorot_uniform IS BAD?, input_dim=len(X[0]),
    #model.add(Activation("relu"))
    #model.add(Dropout(0.25))

    model.add(Dense(output_dim=5, init="uniform"))
    model.add(Activation("softmax"))

    # Good configuartions (?), for when using 23 ms training examples:
    # 128, .5dr, 32, .5dr
    # 256, .5dr, 32, .5dr
    # .5dr, 256, .5dr, 64, .5dr
    # .25dr, 256, .5dr, 64, .5dr
    # 128, .5dr, 64, .5dr, 32, .5dr
    # 128, .5dr, 64, .5dr, 32, .5dr, 32, .5dr

    # when we "flatten" the data (over half second interval), we may have trouble making the correlations
    # we need to make without doing a convolution. Convolution works though!
    return model


# TODO: implement this beyond the stub function that it is. Look for "loss" and
# "optimizer"
@timeit
def compile_model(model, training_parameters=None):
    if training_parameters == None:
        model.compile(loss='categorical_crossentropy', optimizer=Adagrad())
    else:
        # We haven't implemented anything else yet.
        model.compile(loss='categorical_crossentropy', optimizer=Adagrad())


# TODO: parse validation split.
@timeit
def train_model(model, X, y, training_parameters):

    kwargs = {}

    if ('early_stopping' in training_parameters and
        training_parameters['early_stopping'] == True):
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        kwargs['callbacks'] = [early_stopping]

    if 'batch_size' in training_parameters:
        kwargs['batch_size'] = training_parameters['batch_size']

    if 'validation_split' in training_parameters:
        kwargs['validation_split'] = training_parameters['validation_split']

    kwargs['show_accuracy'] = True

    return model.fit(X, y, **kwargs)
    # nb_epoch=2, show_accuracy=True)


# http://keras.io/faq/#how-can-i-save-a-keras-model
def save_model_to_path_stub(model, path):
    json_string = model.to_json()
    architecture_filepath = path + '_architecture.json'
    weights_filepath = path + '_weights.h5'
    open(architecture_filepath, 'w').write(json_string)
    model.save_weights(weights_filepath)


# Companion function to save_model_to_path_stub()
def load_model_from_path_stub(path):
    architecture_filepath = path + '_architecture.json'
    weights_filepath = path + '_weights.h5'
    model = model_from_json(open(architecture_filepath).read())
    model.load_weights(weights_filepath)
    return model
