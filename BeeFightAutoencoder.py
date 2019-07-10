import random
import time
import numpy as np
import cv2
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.optimizers import Adam, SGD, Adadelta
from keras.backend import manual_variable_initialization
from keras.utils import to_categorical
from matplotlib import patches
from skimage import io
from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
import os
from keras.applications.resnet50 import ResNet50
from keras import Sequential, Input, Model, backend
from keras.models import load_model
from keras.layers import Flatten, Dense, Dropout, regularizers, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from skimage.transform import resize, rotate
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
from sklearn.utils import shuffle, class_weight
from tensorflow.python.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from video_selector import VideoSelector

# setting seeds for consistent results
import random
from numpy.random import seed
from tensorflow import set_random_seed
set_random_seed(2)
seed(1)
random.seed(1)

im_size = 40
tot_mean = None
tot_std = None
min_max = None


def load_data(rotate_append=False):
    global tot_mean, tot_std, min_max
    positive = None
    negative = None
    p_test = None
    n_test = None

    root_path = "C:/Users/beekmanpc/Documents/BeeCounter/all_segments_fight_training/positive_sm/"
    for im in os.listdir(root_path):
        if positive is None:
            positive = np.array([io.imread(root_path + im)])
        else:
            positive = np.concatenate((positive, np.array([io.imread(root_path + im)])))
    #positive = positive.reshape((positive.shape[0], positive.shape[1], positive.shape[2], 1))

    root_path = "C:/Users/beekmanpc/Documents/BeeCounter/all_segments_fight_testing/positive_sm/"
    for im in os.listdir(root_path):
        if p_test is None:
            p_test = np.array([io.imread(root_path + im)])
        else:
            p_test = np.concatenate((p_test, np.array([io.imread(root_path + im)])))
    #p_test = p_test.reshape((p_test.shape[0], p_test.shape[1], p_test.shape[2], 1))

    root_path = "C:/Users/beekmanpc/Documents/BeeCounter/all_segments_fight_training/negative_sm/"
    # neg_images = os.listdir(root_path)
    # neg_sample = np.array(neg_images)[sample_without_replacement(len(neg_images), 1500, random_state=0)]
    for im in os.listdir(root_path):
        if negative is None:
            negative = np.array([io.imread(root_path + im)])
        else:
            negative = np.concatenate((negative, np.array([io.imread(root_path + im)])))
    # negative = negative.reshape((negative.shape[0], negative.shape[1], negative.shape[2], 1))

    root_path = "C:/Users/beekmanpc/Documents/BeeCounter/all_segments_fight_testing/negative_sm/"
    for im in os.listdir(root_path):
        if n_test is None:
            n_test = np.array([io.imread(root_path + im)])
        else:
            n_test = np.concatenate((n_test, np.array([io.imread(root_path + im)])))
    #n_test = n_test.reshape((n_test.shape[0], n_test.shape[1], n_test.shape[2], 1))

    if rotate_append:
        rot = [90, 180, 270]
        pos2 = positive.copy()
        for i, im in enumerate(pos2):
            pos2[i] = rotate(im, angle=rot[i%3])
        positive = np.concatenate((positive, pos2))
        neg2 = positive.copy()
        for i, im in enumerate(neg2):
            neg2[i] = rotate(im, angle=rot[i%3])
        negative = np.concatenate((negative, neg2))


    # pos_sm = np.array([cv2.resize(p, (im_size, im_size)) for p in positive])
    # neg_sm = np.array([cv2.resize(n, (im_size, im_size)) for n in negative])
    # # calculate the means
    # tot_mean = np.concatenate((pos_sm, neg_sm)).mean(axis=0)
    # tot_std = np.concatenate((pos_sm, neg_sm)).mean(axis=0)
    #
    # # compute min max scaling
    # # rescaled_pos = ((pos_sm - tot_mean) / tot_std)
    # # rescaled_neg = ((neg_sm - tot_mean) / tot_std)
    # rescaled_pos = pos_sm
    # rescaled_neg = neg_sm
    # r_min = np.concatenate((rescaled_pos[:,:,:,0], rescaled_neg[:,:,:,0])).min()
    # r_max = np.concatenate((rescaled_pos[:,:,:,0], rescaled_neg[:,:,:,0])).max()
    # g_min = np.concatenate((rescaled_pos[:,:,:,1], rescaled_neg[:,:,:,1])).min()
    # g_max = np.concatenate((rescaled_pos[:,:,:,1], rescaled_neg[:,:,:,1])).max()
    # b_min = np.concatenate((rescaled_pos[:,:,:,2], rescaled_neg[:,:,:,2])).min()
    # b_max = np.concatenate((rescaled_pos[:,:,:,2], rescaled_neg[:,:,:,2])).max()
    # min_max = [r_min, r_max, g_min, g_max, b_min, b_max]
    #
    # # apply standardization and min-max scaling
    positive = pre_process_data(positive)
    negative = pre_process_data(negative)
    p_test = pre_process_data(p_test)
    n_test = pre_process_data(n_test)

    positive = shuffle(positive)
    negative = shuffle(negative)
    p_test = shuffle(p_test)
    n_test = shuffle(n_test)

    # positive = positive / 255
    # p_test = p_test / 255
    # negative = negative / 255
    # n_test = n_test / 255

    # positive[np.where(positive > .6)] = np.random.rand(np.where(positive > .6)[0].shape[0])
    # p_test[np.where(p_test > .6)] = np.random.rand(np.where(p_test > .6)[0].shape[0])
    # negative[np.where(negative > .6)] = np.random.rand(np.where(negative > .6)[0].shape[0])
    # n_test[np.where(n_test > .6)] = np.random.rand(np.where(n_test > .6)[0].shape[0])

    # positive = np.expand_dims(positive, axis=3)
    # p_test = np.expand_dims(p_test, axis=3)
    # negative = np.expand_dims(negative, axis=3)
    # n_test = np.expand_dims(n_test, axis=3)

    return positive, p_test, negative, n_test


def pre_process_data(data):
    global tot_mean, tot_std, min_max
    # if tot_mean is None or tot_std is None or min_max is None:
    #     raise ValueError("mean and/or standard deviation have not been calculated on the training data")

    data = np.array([cv2.resize(im, (im_size, im_size)) / 255 for im in data])
    # data = ((data - tot_mean) / tot_std)
    # data[:, :, :, 0] = np.clip(((data[:, :, :, 0] - min_max[0]) / (min_max[1] - min_max[0])), a_min=0, a_max=1)
    # data[:, :, :, 1] = np.clip(((data[:, :, :, 1] - min_max[2]) / (min_max[3] - min_max[2])), a_min=0, a_max=1)
    # data[:, :, :, 2] = np.clip(((data[:, :, :, 2] - min_max[4]) / (min_max[5] - min_max[4])), a_min=0, a_max=1)
    return data


def tune():
    pos_train, neg_train, pos_test, neg_test = load_data()
    hyper_param_tuning(pos_train, pos_test, np.concatenate((neg_test, neg_train)))

'''
Takes as input a dictionary of parameters and then train the autoencoder.
The weights are then saved and the training/validation loss are plotted to ensure learning.
'''
def autoencode_params(params=None, data=None):
    global im_size
    if params is None:
        params = {'batch_size': 20, 'conv_1_filter': 5, 'conv_1_layers': 32, 'learning_rate': 0.01, 'num_epochs': 100, 'optimizer': Adam}
    if data is None:
        pos_train, pos_test, neg_train, neg_test = load_data(rotate_append=False)
    else:
        pos_train = data[0]
        pos_test = data[1]
        neg_train = data[2]
        neg_test = data[3]

    input_img = Input(shape=(im_size, im_size, 3))  # adapt this if using `channels_first` image data format

    # ENCODER
    x = Conv2D(params['conv_1_layers'], (params['conv_1_filter'], params['conv_1_filter']), activation=params['activation'], padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(params['conv_2_layers'], (params['conv_2_filter'], params['conv_2_filter']), activation=params['activation'], padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    # x = Conv2D(32, (3, 3), activation='tanh', padding='same')(x)
    # encoded = MaxPooling2D((2, 2), padding='same')(x)

    # DECODER
    # x = Conv2D(32, (3, 3), activation='tanh', padding='same')(encoded)
    # x = UpSampling2D((2, 2))(x)
    x = Conv2D(params['conv_2_layers'], (params['conv_2_filter'], params['conv_2_filter']), activation=params['activation'], padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(params['conv_1_layers'], (params['conv_1_filter'], params['conv_1_filter']), activation=params['activation'], padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)

    # load the model with the good weights I want to use
    other_params = {
        'batch_size': 15, 'conv_1_filter': 3, 'conv_1_layers': 2,
        'conv_2_filter': 3, 'conv_2_layers': 8, 'learning_rate': 0.0001,
        'activation': 'tanh', 'dense_layers': 32, 'dropout': .7, 'regularization': .01,
        'num_epochs': 100, 'optimizer': Adam
    }
    good_weights_model = get_fully_connected_model(other_params)
    good_weights_model.load_weights('bee_curvature_seperation_weights.h5')
    # get the weights from the pretrained model
    for l1, l2 in zip(autoencoder.layers[1:3], good_weights_model.layers[1:3]):
        l1.set_weights(l2.get_weights())
    # hold the specified layers fixed
    for layer in autoencoder.layers[1:3]:
        layer.trainable = False

    autoencoder.compile(optimizer=params['optimizer'](lr=params['learning_rate']), loss='mean_squared_error')

    curr_t = time.gmtime()
    train_history = autoencoder.fit(pos_train, pos_train,
                                    epochs=40, # params['num_epochs']
                                    batch_size=params['batch_size'],
                                    shuffle=True,
                                    validation_data=(pos_test, pos_test),
                                    verbose=0)#,
                                    # callbacks=[TensorBoard(log_dir='tmp/autoencoder_%d-%d-%d' % (curr_t.tm_hour, curr_t.tm_min, curr_t.tm_sec))])
    # autoencoder.save_weights('autoencoder.h5')
    #
    # loss = train_history.history['loss']
    # val_loss = train_history.history['val_loss']
    # epochs = range(40)
    # plt.figure()
    # plt.plot(epochs, loss, 'g--', label='Training loss')
    # plt.plot(epochs, val_loss, 'm', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.legend()
    # plt.show()

    # print("accuracy:", calculate_RSS(autoencoder, pos_train, pos_test, np.concatenate((neg_test, neg_train))))
    return autoencoder


saved_weights = None
def autoencode_fully_connected(params, data=None, visualize=False):
    global saved_weights, im_size
    autoencoder = autoencode_params(params, data=data)

    if data is None:
        pos_train, pos_test, neg_train, neg_test = load_data(rotate_append=False)
    else:
        pos_train = data[0]
        pos_test = data[1]
        neg_train = data[2]
        neg_test = data[3]

    X_train = np.concatenate((pos_train, neg_train))
    X_test = np.concatenate((pos_test, neg_test))
    y_train = np.array([1]*len(pos_train) + [0]*len(neg_train))
    y_test = np.array([1]*len(pos_test) + [0]*len(neg_test))
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)

    input_img = Input(shape=(im_size, im_size, 3))

    # ENCODER
    x = Conv2D(params['conv_1_layers'], (params['conv_1_filter'], params['conv_1_filter']), activation=params['activation'], padding='same')(input_img) # , kernel_regularizer=regularizers.l2(0.1)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(params['conv_2_layers'], (params['conv_2_filter'], params['conv_2_filter']), activation=params['activation'], padding='same')(x) # , kernel_regularizer=regularizers.l2(0.1)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = BatchNormalization()(x)

    flat = Flatten()(x)
    den = Dropout(rate=params['dropout'])(flat)
    den = Dense(params['dense_layers'], activation='relu', kernel_regularizer=regularizers.l2(params['regularization']))(den) #
    out = Dense(1, activation='sigmoid')(den)

    full_model = Model(input_img, out)
    # get the weights from the pretrained model
    for l1, l2 in zip(full_model.layers[:5], autoencoder.layers[0:5]):
        l1.set_weights(l2.get_weights())
    # hold them steady
    for layer in full_model.layers[0:5]:
        layer.trainable = False
    # compile and train the model
    full_model.compile(loss=binary_crossentropy, optimizer=Adam(lr=0.001), metrics=['accuracy'])
    curr_t = time.gmtime()
    train_history = full_model.fit(X_train, y_train,
                                   epochs=params['num_epochs'],
                                   batch_size=params['batch_size'],
                                   validation_data=(X_test, y_test),
                                   verbose=0,
                                   callbacks=[
                                       # EarlyStopping(monitor='val_loss', min_delta=0.01, patience=20, mode='min', restore_best_weights=True)
                                       # TensorBoard(log_dir='tmp/[0]autoencoder_fully_connected(layer#=%d)(reg=%.04f)_%d-%d-%d' % (dense_layer_nodes, reg, curr_t.tm_hour, curr_t.tm_min, curr_t.tm_sec))
                                   ])

    # # plot the train and validation loss
    # loss = train_history.history['loss']
    # val_loss = train_history.history['val_loss']
    # epochs = range(params['num_epochs'])
    # plt.figure()
    # plt.plot(epochs, loss, 'g--', label='Training loss')
    # plt.plot(epochs, val_loss, 'm', label='Validation loss')
    # plt.title('[1]Training and validation loss')
    # plt.legend()
    # plt.show()
    #
    # # plot the train and validation acc
    # acc = train_history.history['acc']
    # val_acc = train_history.history['val_acc']
    # epochs = range(params['num_epochs'])
    # plt.figure()
    # plt.plot(epochs, acc, 'g--', label='Training acc')
    # plt.plot(epochs, val_acc, 'm', label='Validation acc')
    # plt.title('[1]Training and validation acc')
    # plt.legend()
    # plt.show()

    # make all layers trainable
    for layer in full_model.layers[0:5]:
        layer.trainable = True
    # fine tune all of the weights
    full_model.compile(loss=binary_crossentropy, optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    train_history = full_model.fit(X_train, y_train,
                                   epochs=params['num_epochs'],
                                   batch_size=params['batch_size'],
                                   validation_data=(X_test, y_test),
                                   verbose=0,
                                   callbacks=[
                                       EarlyStopping(monitor='val_loss', min_delta=0.05, patience=20, mode='min', restore_best_weights=True)
                                       # TensorBoard(log_dir='tmp/[1]autoencoder_fully_connected(layer#=%d)(reg=%.04f)_%d-%d-%d' % (dense_layer_nodes, reg, curr_t.tm_hour, curr_t.tm_min, curr_t.tm_sec))
                                   ])

    # # plot the train and validation loss
    # loss = train_history.history['loss']
    # val_loss = train_history.history['val_loss']
    # epochs = range(params['num_epochs'])
    # plt.figure()
    # plt.plot(epochs, loss, 'g--', label='Training loss')
    # plt.plot(epochs, val_loss, 'm', label='Validation loss')
    # plt.title('[2]Training and validation loss')
    # plt.legend()
    # plt.show()
    #
    # # plot the train and validation acc
    # acc = train_history.history['acc']
    # val_acc = train_history.history['val_acc']
    # epochs = range(params['num_epochs'])
    # plt.figure()
    # plt.plot(epochs, acc, 'g--', label='Training acc')
    # plt.plot(epochs, val_acc, 'm', label='Validation acc')
    # plt.title('[2]Training and validation acc')
    # plt.legend()
    # plt.show()

    pos_train_acc = accuracy_score(np.round(full_model.predict(pos_train)).astype(int), np.array([1] * len(pos_train)))
    neg_train_acc = accuracy_score(np.round(full_model.predict(neg_train)).astype(int), np.array([0] * len(neg_train)))
    pos_test_acc = accuracy_score(np.round(full_model.predict(pos_test)).astype(int), np.array([1] * len(pos_test)))
    neg_test_acc = accuracy_score(np.round(full_model.predict(neg_test)).astype(int), np.array([0] * len(neg_test)))
    print("pos_train:%.03f\nneg_train:%.03f\npos_test:%.03f\nneg_test:%.03f" % (pos_train_acc, neg_train_acc, pos_test_acc, neg_test_acc))

    saved_weights = full_model.get_weights()
    full_model.save_weights('autoencoder_classification.h5')

    if visualize:
        visualize_conv_layers(params)
    return full_model, np.array([pos_train_acc, neg_train_acc, pos_test_acc, neg_test_acc])


def get_fully_connected_model(params):
    global im_size
    input_img = Input(shape=(im_size, im_size, 3))

    # ENCODER
    x = Conv2D(params['conv_1_layers'], (params['conv_1_filter'], params['conv_1_filter']), activation=params['activation'], padding='same')(input_img) # , kernel_regularizer=regularizers.l2(0.1)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(params['conv_2_layers'], (params['conv_2_filter'], params['conv_2_filter']), activation=params['activation'], padding='same')(x) # , kernel_regularizer=regularizers.l2(0.1)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = BatchNormalization()(x)

    flat = Flatten()(x)
    den = Dropout(rate=params['dropout'])(flat)
    den = Dense(params['dense_layers'], activation='relu', kernel_regularizer=regularizers.l2(params['regularization']))(den) #
    out = Dense(1, activation='sigmoid')(den)

    full_model = Model(input_img, out)
    full_model.compile(loss=binary_crossentropy, optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return full_model


# heavily based on: https://github.com/gabrielpierobon/cnnshapes/blob/master/README.md
def visualize_conv_layers(params):
    global tot_mean, tot_std, min_max

    manual_variable_initialization(True)
    model = get_fully_connected_model(params)
    model.load_weights('autoencoder_classification.h5')

    print("num model layers %d" % len(model.layers))
    layer_outputs = [layer.output for layer in model.layers[1:5]]#[1:]  # Extracts the outputs of the top 12 layers
    activation_model = Model(inputs=model.input, outputs=layer_outputs)  # Creates a model that will return these outputs, given the model input

    pos_test_img = np.array([io.imread("C:/Users/beekmanpc/Documents/BeeCounter/all_segments_fight_training/positive_sm/10-35-13_fight(145,33).png")])
    neg_test_img = np.array([io.imread("C:/Users/beekmanpc/Documents/BeeCounter/all_segments_fight_training/negative_sm/14-31-18_fight(447,96).png")])

    pos_test_img = pre_process_data(pos_test_img)
    neg_test_img = pre_process_data(neg_test_img)

    # pos_test_img = ((pos_test_img - tot_mean) / tot_std)
    # neg_test_img = ((neg_test_img - tot_mean) / tot_std)
    #
    # pos_test_img[:, :, :, 0] = ((pos_test_img[:, :, :, 0] - min_max[0]) / (min_max[1] - min_max[0])).clip(0)
    # pos_test_img[:, :, :, 1] = ((pos_test_img[:, :, :, 1] - min_max[2]) / (min_max[3] - min_max[2])).clip(0)
    # pos_test_img[:, :, :, 2] = ((pos_test_img[:, :, :, 2] - min_max[4]) / (min_max[5] - min_max[4])).clip(0)
    #
    # neg_test_img[:, :, :, 0] = ((neg_test_img[:, :, :, 0] - min_max[0]) / (min_max[1] - min_max[0])).clip(0)
    # neg_test_img[:, :, :, 1] = ((neg_test_img[:, :, :, 1] - min_max[2]) / (min_max[3] - min_max[2])).clip(0)
    # neg_test_img[:, :, :, 2] = ((neg_test_img[:, :, :, 2] - min_max[4]) / (min_max[5] - min_max[4])).clip(0)

    pos_activations = activation_model.predict(pos_test_img)
    neg_activations = activation_model.predict(neg_test_img)

    layer_names = []
    for layer in model.layers[1:5]:
        layer_names.append(layer.name)

    # model_weights = [(name, layer.get_weights()) for name, layer in zip(layer_names, model.layers[1:5]) if "conv2d" in name]
    #
    # for layer in model_weights:
    #     # num_filters = layer[1][0].shape[3]
    #     colors = ['Red', "Green", "Blue"]
    #     for i, c in enumerate(colors):
    #         fig, ax = plt.subplots(2, 2)
    #         plt.suptitle(c + "channel")
    #         ax[0, 0].imshow(layer[1][0][:,:,:,0].squeeze()[:,:,i], cmap='gray') #ax[0, 0]
    #         ax[0, 0].set_title(layer[0]+"_filter=0")
    #         ax[0, 1].imshow(layer[1][0][:,:,:,1].squeeze()[:,:,i], cmap='gray') # ax[0, 1]
    #         ax[0, 1].set_title(layer[0]+"_filter=1")
    #         ax[1, 0].imshow(layer[1][0][:,:,:,2].squeeze()[:,:,i], cmap='gray') # ax[0, 2]
    #         ax[1, 0].set_title(layer[0]+"_filter=2")
    #         ax[1, 1].imshow(layer[1][0][:,:,:,3].squeeze()[:,:,i], cmap='gray') # ax[0, 3]
    #         ax[1, 1].set_title(layer[0]+"_filter=3")
    #         # ax[1, 0].imshow(layer[1][0][:,:,:,4].squeeze()[:,:,i], cmap='gray')
    #         # ax[1, 0].set_title(layer[0]+"_filter=4")
    #         # ax[1, 1].imshow(layer[1][0][:,:,:,5].squeeze()[:,:,i], cmap='gray')
    #         # ax[1, 1].set_title(layer[0]+"_filter=5")
    #         # ax[1, 2].imshow(layer[1][0][:,:,:,6].squeeze()[:,:,i], cmap='gray')
    #         # ax[1, 2].set_title(layer[0]+"_filter=6")
    #         # ax[1, 3].imshow(layer[1][0][:,:,:,7].squeeze()[:,:,i], cmap='gray')
    #         # ax[1, 3].set_title(layer[0]+"_filter=7")
    #         plt.show()
    #         plt.clf()

    print(layer_names)
    images_per_row = 2

    io.imshow(pos_test_img[0])#.reshape((40,40)), cmap='gray')
    io.show()

    for layer_name, layer_activation in zip(layer_names, pos_activations):
        n_features = layer_activation.shape[-1]  # Number of features in the feature map
        size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):  # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                :, :,
                                col * images_per_row + row]
                channel_image = (channel_image-channel_image.mean())/channel_image.std()  # Post-processes the feature to make it visually palatable
                # channel_image /=
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,  # Displays the grid
                row * size: (row + 1) * size] = channel_image

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title("pos_" + layer_name)
        plt.grid(True)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.show()

    io.imshow(neg_test_img[0])#.reshape((40,40)), cmap='gray')
    io.show()

    for layer_name, layer_activation in zip(layer_names, neg_activations):
        n_features = layer_activation.shape[-1]  # Number of features in the feature map
        size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):  # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                :, :,
                                col * images_per_row + row]
                channel_image = (channel_image-channel_image.mean())/channel_image.std()  # Post-processes the feature to make it visually palatable
                # channel_image /=
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,  # Displays the grid
                row * size: (row + 1) * size] = channel_image

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title("neg_" + layer_name)
        plt.grid(True)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.show()


def test_with_frame(params):
    global tot_mean, tot_std, min_max, saved_weights, im_size
    # vid = VideoSelector()
    # detail_name, filename, hive = vid.download_video(type='fight')
    frame = None
    frame_num = -1
    play_video = True
    ret = None

    manual_variable_initialization(True)
    full_model = get_fully_connected_model(params)
    full_model.load_weights('autoencoder_classification.h5')

    # cap = cv2.VideoCapture("C:/Users/beekmanpc/Documents/BeeCounter/bee_videos/14-06-15.h264")# + filename) 17-26-55.h264
    cap = cv2.VideoCapture("C:/Users/beekmanpc/Documents/BeeCounter/bee_videos/11-00-18.h264") # 15-56-31.h264
    while True:
        locations = []
        sub_images = None
        frame_num += 1
        if frame_num % 100 == 0:
            print("at frame %d" % frame_num)
        if play_video:
            ret, frame = cap.read()
        if ret:
            cv2.namedWindow("fightz")
            key = cv2.waitKey(1)
            # if frame_num < 1100:
            #     continue
            if key == 112: # 'p'
                play_video = not play_video
            #elif key == 115: # 's'
            # split the image into small 40x40 windows
            h, w, colors = frame.shape
            stride = 10

            sub_images = np.array([frame[i:i + im_size, j:j + im_size, :] for i in range(0, h - im_size, stride) for j in range(0, w - im_size, stride)])
            locations = np.array([(j, i) for i in range(0, h - im_size, stride) for j in range(0, w - im_size, stride)])

            sub_images = pre_process_data(sub_images)

            predictions = full_model.predict(sub_images).reshape(-1)
            if frame_num % 5 == 0:
                print("max pred is %.03f" % predictions.max())
            # max_pred_loc = locations[np.argmax(predictions)]
            # if predictions.max() >= .95:
            #     cv2.rectangle(frame, tuple(np.flip(max_pred_loc)), (max_pred_loc[1]+im_size, max_pred_loc[0]+im_size), (0,255,0))
            # else:
            #     cv2.rectangle(frame, tuple(np.flip(max_pred_loc)), (max_pred_loc[1]+im_size, max_pred_loc[0]+im_size), (255,0,0))
            # cv2.imshow("fightz", frame)
            fight_predictions = np.where(predictions >= .75)
            if len(fight_predictions[0]) >= 1:
                print("%d fights found at frame %d" % (len(fight_predictions[0]), frame_num))
            # save all predicted fights and the surrounding context
            for idx, loc in enumerate(np.array(locations)[fight_predictions[0]]):
                if predictions[fight_predictions[0][idx]] < .85:
                    color = (255,0,0)
                elif predictions[fight_predictions[0][idx]] >= .85 and predictions[fight_predictions[0][idx]] < .91:
                    color = (0,0,255)
                else:
                    color = (0,255,0)
                cv2.rectangle(frame, tuple(loc), (loc[0]+im_size, loc[1]+im_size), color)

                # curr_sub = frame[loc[0]:loc[0]+im_size, loc[1]:loc[1]+im_size, :]
                # # curr_sub = frame[max(0,loc[0]-40):min(loc[0]+80, h), max(0,loc[1]-40):min(loc[1]+80,w), :]
                # # cv2.rectangle(curr_sub, (40,40), (80,80), (0,255,0), 3)
                # detail_name = "14-06-15_"
                # cv2.imwrite("C:/Users/beekmanpc/Documents/stigma/found_fights/"
                #             +detail_name+"fight[%d](frame=%d)pred=%.03f.png" % (idx, frame_num, predictions[fight_predictions[0][idx]]),
                #             curr_sub)
                # curr_sub = frame[max(0,loc[0]-im_size):min(loc[0]+80, h), max(0,loc[1]-im_size):min(loc[1]+80,w), :]
                # # cv2.rectangle(curr_sub, (40,40), (80,80), (0,255,0), 3)
                # cv2.imwrite("C:/Users/beekmanpc/Documents/stigma/found_fights/"
                #             +detail_name+"fight[%d](frame=%d)pred=%.03fCONTEXT.png" % (idx, frame_num, predictions[fight_predictions[0][idx]]),
                #             curr_sub)
            cv2.imshow("fightz", frame)
        else:
            cap.release()
            cv2.destroyAllWindows()
            break

    print("yo")
    pass


def calculate_RSS(autoencoder, pos_train, pos_test, neg_test, output=True):
    # Residual Sum of Squares
    pred_pos = autoencoder.predict(pos_train)
    RSS_pos = ((pos_train - pred_pos) ** 2).sum(axis=(1,2,3))
    pred_p_test = autoencoder.predict(pos_test)
    RSS_p_test = ((pos_test - pred_p_test) ** 2).sum(axis=(1,2,3))
    pred_neg = autoencoder.predict(neg_test)
    RSS_neg = ((neg_test - pred_neg) ** 2).sum(axis=(1,2,3))

    # find the best threshold for the RSS to get the highest accuracy
    # overall accuracy is defined as the average of the positive train and test multiplied by the negative test
    best_acc = 0
    best_thresh = 0
    seperate_acc = [0,0,0]
    thresholds = np.arange(int(RSS_neg.min()), int(RSS_p_test.max()), .5)
    p = []
    pt = []
    nt = []
    for t in thresholds:
        p.append((RSS_pos <= t).sum() / len(RSS_pos))
        pt.append((RSS_p_test <= t).sum() / len(RSS_p_test))
        nt.append((RSS_neg > t).sum() / len(RSS_neg))

        positive_acc = (RSS_pos <= t).sum() / len(RSS_pos)
        p_test_acc = (RSS_p_test <= t).sum() / len(RSS_p_test)
        n_test_acc = (RSS_neg > t).sum() / len(RSS_neg)

        acc = ((positive_acc + p_test_acc) / 2) * n_test_acc
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
            seperate_acc = [positive_acc, p_test_acc, n_test_acc]

    if output:
        print("Best calculated ACC:", best_acc, "Best thresh:", best_thresh)
        plt.figure()
        plt.title("Reconstruction threshold picker\n(train=%.02f, pos_test=%.02f, neg_test=%.02f)" % tuple(seperate_acc))
        plt.plot(thresholds, p, label="pos_train_acc")
        plt.plot(thresholds, pt, label='pos_test_acc')
        plt.plot(thresholds, nt, label='neg_test_acc')
        plt.plot([best_thresh, best_thresh], [0, 1], label='best found thresh')
        plt.legend()
        plt.xlabel("threshold")
        plt.ylabel("accuracy")
        plt.show()

    return best_acc, seperate_acc


def full_param_tuning():
    positive, p_test, negative, n_test = load_data()
    cross_val = 1
    best_score = 0
    best_params = []
    best_accs = []

    params = {
        'batch_size': [15, 30, 90],
        'dense_layers': [4, 8, 16, 32],
        'dropout': [.6, .7, .8],
        'conv_1_layers': [2, 4, 8],
        'conv_1_filter': [3],
        'conv_2_layers': [2, 4, 8, 16],
        'conv_2_filter': [3, 5],
        'learning_rate': [0.0001],
        'num_epochs': [100],
        'optimizer': [Adam],
        'regularization': [0.1, 0.01],
        'activation': ['tanh']
    }

    count = 0
    param_len = len(ParameterGrid(params))
    print("-------" * 2 + "Grid Search Started" + "-------" * 2)
    param_list = list(ParameterGrid(params))
    random.shuffle(param_list)
    for p in param_list:
        cv_acc = 0
        for _ in range(cross_val):
            model, accuracies = autoencode_fully_connected(p, data=[positive, p_test, negative, n_test])
            cum_acc = np.mean(accuracies)
            cv_acc += cum_acc
        cv_acc /= cross_val
        count += 1
        if cv_acc > best_score:
            best_score = cv_acc
            best_params = p
            best_accs = accuracies
            model.save_weights("best_weights.h5")
            print("[*](%d/%d) %.03f Params:" % (count, param_len, cv_acc), p)
        else:
            print("[.](%d/%d) %.03f Params:" % (count, param_len, cv_acc), p)

    print("Best Cumulative Accuracy: %.03f\nBest params:" % best_score, best_params)
    print("Best ACC's:", best_accs)


def hyper_param_tuning(pos_train, pos_test, neg_test):
    im_size = 20
    input_img = Input(shape=(im_size, im_size, 3))
    best_params = {}
    best_score = 0
    best_model = ""
    cross_val = 2

    params_1 = {
              "conv_1_layers": [8, 16, 32, 64, 128],
              "conv_1_filter": [3, 5],
              "num_epochs": [100],
              "batch_size": [15, 20, 50],
              "optimizer": [Adam],
              "learning_rate": [.01, .001, .0001]
              }

    params_2 = {
              "conv_1_layers": [8, 16, 24, 32],
              "conv_1_filter": [3],
              "conv_2_layers": [8, 16, 24, 32],
              "conv_2_filter": [3],
              "num_epochs": [100],
              "batch_size": [15, 50, 75, 100],
              "optimizer": [Adam, SGD],
              "learning_rate": [.001, 0.0001, .00001]
              }

    params_3 = {
              "conv_1_layers": [16, 32, 64, 128, 256],
              "conv_1_filter": [3, 5],
              "conv_2_layers": [8, 16, 32, 64, 128],
              "conv_2_filter": [3, 5],
              "conv_3_layers": [8, 16, 32, 64, 128, 256],
              "conv_3_filter": [3, 5],
              "num_epochs": [100],
              "batch_size": [15, 30],
              "optimizer": [Adam],
              "learning_rate": [.1, .01, .001]
              }

    # count = 0
    # param_1_len = len(ParameterGrid(params_1))
    # print("-------" * 2 + "Model 1" + "-------" * 2)
    # for params in ParameterGrid(params_1):
    #     decoded = model_1(input_img, params)
    #     autoencoder = Model(input_img, decoded)
    #     autoencoder.compile(optimizer=params['optimizer'](lr=params['learning_rate']), loss='mean_squared_error')
    #     autoencoder.fit(pos_train, pos_train,
    #                     epochs=params['num_epochs'],
    #                     batch_size=params['batch_size'],
    #                     shuffle=True,
    #                     verbose=0,
    #                     validation_data=(pos_test, pos_test))
    #     curr_acc, seperate_acc = calculate_RSS(autoencoder, pos_train, pos_test, neg_test, output=False)
    #     print("(%d/%d) Model_1: %.03f Params:" % (count, param_1_len, curr_acc), params)
    #     count += 1
    #     if curr_acc > best_score:
    #         best_score = curr_acc
    #         best_params = params
    #         best_model = "model_1"

    count = 0
    param_2_len = len(ParameterGrid(params_2))
    print("-------" * 2 + "Model 2" + "-------" * 2)
    for params in ParameterGrid(params_2):
        cv_acc = 0
        for _ in range(cross_val):
            decoded = model_2(input_img, params)
            autoencoder = Model(input_img, decoded)
            autoencoder.compile(optimizer=params['optimizer'](lr=params['learning_rate']), loss='mean_squared_error')
            autoencoder.fit(pos_train, pos_train,
                            epochs=params['num_epochs'],
                            batch_size=params['batch_size'],
                            shuffle=True,
                            verbose=0,
                            validation_data=(pos_test, pos_test))
            curr_acc, seperate_acc = calculate_RSS(autoencoder, pos_train, pos_test, neg_test, output=False)
            cv_acc += curr_acc
        cv_acc /= cross_val
        count += 1
        if cv_acc > best_score:
            best_score = cv_acc
            best_params = params
            best_model = "model_2"
            print("(%d/%d) *** Model_2: %.03f Params:" % (count, param_2_len, cv_acc), params)
        else:
            print("(%d/%d) Model_2: %.03f Params:" % (count, param_2_len, cv_acc), params)

    count = 0
    param_3_len = len(ParameterGrid(params_3))
    print("-------" * 2 + "Model 3" + "-------" * 2)
    for params in ParameterGrid(params_3):
        cv_acc = 0
        for _ in range(cross_val):
            decoded = model_3(input_img, params)
            autoencoder = Model(input_img, decoded)
            autoencoder.compile(optimizer=params['optimizer'](lr=params['learning_rate']), loss='mean_squared_error')
            autoencoder.fit(pos_train, pos_train,
                            epochs=params['num_epochs'],
                            batch_size=params['batch_size'],
                            shuffle=True,
                            verbose=0,
                            validation_data=(pos_test, pos_test))
            curr_acc, seperate_acc = calculate_RSS(autoencoder, pos_train, pos_test, neg_test, output=False)
            cv_acc += curr_acc
        cv_acc /= cross_val
        count += 1
        if cv_acc > best_score:
            best_score = cv_acc
            best_params = params
            best_model = "model_3"
            print("(%d/%d) *** Model_3: %.03f Params:" % (count, param_3_len, cv_acc), params)
        else:
            print("(%d/%d) Model_3: %.03f Params:" % (count, param_3_len, cv_acc), params)

    print("Best Cumulative Accuracy: %.03f\nBest model: %s\nBest params:" % (best_score, best_model), best_params)
    print("yooooo")
    pass


def model_1(input_img, params):
    # ENCODER
    x = Conv2D(params['conv_1_layers'], (params['conv_1_filter'], params['conv_1_filter']), activation='tanh', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # DECODER
    x = Conv2D(params['conv_1_layers'], (params['conv_1_filter'], params['conv_1_filter']), activation='tanh', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    return decoded


def model_2(input_img, params):
    # ENCODER
    x = Conv2D(params['conv_1_layers'], (params['conv_1_filter'], params['conv_1_filter']), activation='tanh', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(params['conv_2_layers'], (params['conv_2_filter'], params['conv_2_filter']), activation='tanh', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # DECODER
    x = Conv2D(params['conv_2_layers'], (params['conv_2_filter'], params['conv_2_filter']), activation='tanh', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(params['conv_1_layers'], (params['conv_1_filter'], params['conv_1_filter']), activation='tanh', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    return decoded


def model_3(input_img, params):
    # ENCODER
    x = Conv2D(params['conv_1_layers'], (params['conv_1_filter'], params['conv_1_filter']), activation='tanh', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(params['conv_2_layers'], (params['conv_2_filter'], params['conv_2_filter']), activation='tanh', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(params['conv_3_layers'], (params['conv_3_filter'], params['conv_3_filter']), activation='tanh', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # DECODER
    x = Conv2D(params['conv_3_layers'], (params['conv_3_filter'], params['conv_3_filter']), activation='tanh', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(params['conv_2_layers'], (params['conv_2_filter'], params['conv_2_filter']), activation='tanh', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(params['conv_1_layers'], (params['conv_1_filter'], params['conv_1_filter']), activation='tanh', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    return decoded

def main():
    # build_autoencoder()
    # autoencode_params(params={'batch_size': 15, 'conv_1_filter': 5, 'conv_1_layers': 32, 'conv_2_filter': 5, 'conv_2_layers': 64, 'learning_rate': 0.001, 'num_epochs': 100, 'optimizer': Adam})
    # tune()

    params = {
        'batch_size': 15, 'conv_1_filter': 3, 'conv_1_layers': 2,
        'conv_2_filter': 3, 'conv_2_layers': 2, 'learning_rate': 0.0001,
        'activation': 'tanh', 'dense_layers': 8, 'dropout': .8, 'regularization': .01,
        'num_epochs': 100, 'optimizer': Adam
    }
    # visualize_conv_layers(params)

    full_model, accuracies = autoencode_fully_connected(params=params, visualize=False)
    # test_with_frame(params)

    # full_param_tuning()

    # for _ in [0,1,2]:
    #     varying_data_train()

    # [*](074/1728) 0.911 Params: {'activation': 'tanh', 'batch_size': 15, 'conv_1_filter': 3, 'conv_1_layers': 2, 'conv_2_filter': 3, 'conv_2_layers': 8, 'dense_layers': 32, 'dropout': 0.6, 'learning_rate': 0.0001, 'num_epochs': 100, 'optimizer': <class 'keras.optimizers.Adam'>, 'regularization': 0.01}
    # These are the parameters that I used to get the good weights where the edge of the bee is highlighted: /stigma/bee_curvature_seperation_weights.h5


def varying_data_train():
    pos_train, pos_test, neg_train, neg_test = load_data()

    params = {
        'batch_size': 15, 'conv_1_filter': 3, 'conv_1_layers': 2,
        'conv_2_filter': 3, 'conv_2_layers': 4, 'learning_rate': 0.0001,
        'activation': 'tanh', 'dense_layers': 8, 'dropout': .8, 'regularization': .01,
        'num_epochs': 100, 'optimizer': Adam
    }

    accuracies = []
    labels = ['pos_train', 'pos_test', 'neg_train', 'neg_test']
    percentages = np.arange(0.1, 1.1, .1)

    for perc in percentages:
        _, acc = autoencode_fully_connected(params=params, data=[
            pos_train[0:int(len(pos_train) * perc)],
            pos_test[0:int(len(pos_test) * perc)],
            neg_train[0:int(len(neg_train) * perc)],
            neg_test[0:int(len(neg_test) * perc)]])
        accuracies.append(acc)

    accuracies = np.array(accuracies)
    avg_acc = np.mean(accuracies, axis=1)
    for i, acc in enumerate(accuracies.T):
        plt.plot(percentages, acc, '.-', label=labels[i])
    plt.plot(percentages, avg_acc, '.-', label='average')
    plt.title("accuracies as amount of data increases")
    plt.xlabel("percent of available data used")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

    variance = np.var(accuracies, axis=1)
    plt.plot(percentages, variance, '.-')
    plt.title("Accuracy variance of pos/neg train/test sets")
    plt.xlabel("percent of available data")
    plt.ylabel("variance")
    plt.show()


if __name__ == "__main__":
    main()

