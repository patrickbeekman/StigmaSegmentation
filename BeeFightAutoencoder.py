import time

import numpy as np
import cv2
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.optimizers import Adam, SGD, Adadelta
from keras.utils import to_categorical
from matplotlib import patches
from skimage import io
from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
# from PIL import Image
import os
from keras.applications.resnet50 import ResNet50
from keras import Sequential, Input, Model
from keras.models import load_model
from keras.layers import Flatten, Dense, Dropout, regularizers, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, \
    AveragePooling2D
from skimage.transform import resize, rotate
from sklearn.metrics import mean_squared_error, roc_auc_score, auc, accuracy_score
from sklearn.model_selection import ParameterGrid
from sklearn.utils._random import sample_without_replacement
from tensorflow.python.keras import backend
from sklearn.utils import shuffle, class_weight
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

from video_selector import VideoSelector


def load_data(rotate_append=False):
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
    print("positive_shape:", positive.shape)

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

    positive = positive / 255
    p_test = p_test / 255
    negative = negative / 255
    n_test = n_test / 255
    return positive, p_test, negative, n_test


def build_autoencoder():
    global stigma_center
    stop = 25
    count = 0
    im_size = 40

    # root_path = "C:/Users/beekmanpc/Documents/BeeCounter/all_segments_fight_training/positive_sm/"
    # for im in os.listdir(root_path):
    #     if positive is None:
    #         positive = np.array([rgb2hsv(io.imread(root_path + im))])
    #     else:
    #         positive = np.concatenate((positive, np.array([rgb2hsv(io.imread(root_path + im))])))
    # #positive = positive.reshape((positive.shape[0], positive.shape[1], positive.shape[2], 1))
    #
    # root_path = "C:/Users/beekmanpc/Documents/BeeCounter/all_segments_fight_testing/positive_sm/"
    # for im in os.listdir(root_path):
    #     if p_test is None:
    #         p_test = np.array([rgb2hsv(io.imread(root_path + im))])
    #     else:
    #         p_test = np.concatenate((p_test, np.array([rgb2hsv(io.imread(root_path + im))])))
    # #p_test = p_test.reshape((p_test.shape[0], p_test.shape[1], p_test.shape[2], 1))
    #
    # root_path = "C:/Users/beekmanpc/Documents/BeeCounter/all_segments_fight_training/negative_sm/"
    # # neg_images = os.listdir(root_path)
    # # neg_sample = np.array(neg_images)[sample_without_replacement(len(neg_images), 1500, random_state=0)]
    # for im in os.listdir(root_path):
    #     if negative is None:
    #         negative = np.array([rgb2hsv(io.imread(root_path + im))])
    #     else:
    #         negative = np.concatenate((negative, np.array([rgb2hsv(io.imread(root_path + im))])))
    # # negative = negative.reshape((negative.shape[0], negative.shape[1], negative.shape[2], 1))
    # print("positive_shape:", positive.shape)

    #in_out_shape = positive.shape[1] * positive.shape[2] * positive.shape[3]

    # positive = positive.astype('float32') / 255
    # negative = negative.astype('float32') / 255
    # p_test = p_test.astype('float32') / 255
    # n_test = n_test.astype('float32') / 255
    #positive = positive.reshape((len(positive), -1))
    # p_test = p_test.reshape((len(p_test), -1))

    pos_train, pos_test, neg_test = load_data()

    # hidden_1 = Dense(4096, activation='tanh')(input_img)
    # hidden_2 = Dense(2048, activation='tanh')(hidden_1)
    # code = Dense(1024, activation='tanh')(hidden_2)
    # hidden_3 = Dense(2048, activation='tanh')(code)
    # hidden_4 = Dense(4096, activation='tanh')(hidden_3)
    # decoded = Dense(in_out_shape, activation='sigmoid')(hidden_4)

    input_img = Input(shape=(im_size,im_size, 3))  # adapt this if using `channels_first` image data format

    # ENCODER
    x = Conv2D(32, (5, 5), activation='tanh', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='tanh', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    # x = Conv2D(32, (3, 3), activation='tanh', padding='same')(x)
    # encoded = MaxPooling2D((2, 2), padding='same')(x)

    # DECODER
    # x = Conv2D(32, (3, 3), activation='tanh', padding='same')(encoded)
    # x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='tanh', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (5, 5), activation='tanh', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    autoencoder.fit(pos_train, pos_train,
                    epochs=50,
                    batch_size=20,
                    shuffle=True,
                    validation_data=(pos_test, pos_test),
                    verbose=2,
                    callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

    autoencoder.save("data/autoencoder.h5")
    # autoencoder.load_weights("data/autoencoder.h5")

    pos_pred = autoencoder.predict(pos_train)
    neg_pred = autoencoder.predict(neg_test)

    io.imshow(pos_train[15])#.reshape((80,80)), cmap='gray')
    io.show()
    io.imshow(pos_pred[15])#.reshape((80,80)), cmap='gray')
    io.show()
    io.imshow(neg_test[15])#.reshape((80,80)), cmap='gray')
    io.show()
    io.imshow(neg_pred[15])#.reshape((80,80)), cmap='gray')
    io.show()

    # Residual Sum of Squares
    pred_pos = autoencoder.predict(pos_train)
    RSS_pos = ((pos_train - pred_pos) ** 2).sum(axis=(1,2,3))
    pred_p_test = autoencoder.predict(pos_test)
    RSS_p_test = ((pos_test - pred_p_test) ** 2).sum(axis=(1,2,3))
    pred_neg = autoencoder.predict(neg_test)
    RSS_neg = ((neg_test - pred_neg) ** 2).sum(axis=(1,2,3))

    print("positive", RSS_pos.mean())
    # print("negative", negative_mse.mean())
    print("p_test", RSS_p_test.mean())
    print("n_test", RSS_neg.mean())

    best_acc = 0
    threshold = RSS_neg.min()
    for t in np.arange(int(RSS_neg.min()), int(RSS_p_test.max()), .5):
        positive_acc = (RSS_pos <= t).sum() / len(RSS_pos)
        p_test_acc = (RSS_p_test <= t).sum() / len(RSS_p_test)
        n_test_acc = (RSS_neg > t).sum() / len(RSS_neg)
        acc = (positive_acc + p_test_acc) / 2 * n_test_acc
        if acc > best_acc:
            best_acc = acc
            threshold = t
    print("Done finding best split: Best ACC=%.02f   with threshold=%.01f" % (best_acc, threshold))
    positive_acc = (RSS_pos <= threshold).sum() / len(RSS_pos)
    p_test_acc = (RSS_p_test <= threshold).sum() / len(RSS_p_test)
    negative_acc = (RSS_neg > threshold).sum() / len(RSS_neg)
    print("pos_train_acc=%.03f\npos_test_acc=%.03f\nneg_test_acc=%.03f" % (positive_acc, p_test_acc, negative_acc))

    fig, ax = plt.subplots()
    ax.boxplot([RSS_pos, RSS_p_test, RSS_neg], positions=np.array(range(3))+1, labels=['pos_train', 'pos_test', 'neg_test'], meanline=True)
    plt.title("RSS reconstruction errors: total acc=%.03f, pos_train=%.02f,\npos_test=%.02f, neg_test=%.02f, thresh=%.01f" % (best_acc, positive_acc, p_test_acc, negative_acc, threshold))
    plt.show()
    plt.clf()

    print("yo")


def tune():
    pos_train, neg_train, pos_test, neg_test = load_data()
    hyper_param_tuning(pos_train, pos_test, np.concatenate((neg_test, neg_train)))

'''
Takes as input a dictionary of parameters and then train the autoencoder.
The weights are then saved and the training/validation loss are plotted to ensure learning.
'''
def autoencode_params(params=None):
    if params is None:
        params = {'batch_size': 20, 'conv_1_filter': 5, 'conv_1_layers': 32, 'learning_rate': 0.01, 'num_epochs': 100, 'optimizer': Adam}
    pos_train, pos_test, neg_train, neg_test = load_data(rotate_append=False)

    input_img = Input(shape=(40, 40, 3))  # adapt this if using `channels_first` image data format

    # ENCODER
    x = Conv2D(params['conv_1_layers'], (params['conv_1_filter'], params['conv_1_filter']), activation='tanh', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(params['conv_2_layers'], (params['conv_2_filter'], params['conv_2_filter']), activation='tanh', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    # x = Conv2D(32, (3, 3), activation='tanh', padding='same')(x)
    # encoded = MaxPooling2D((2, 2), padding='same')(x)

    # DECODER
    # x = Conv2D(32, (3, 3), activation='tanh', padding='same')(encoded)
    # x = UpSampling2D((2, 2))(x)
    x = Conv2D(params['conv_2_layers'], (params['conv_2_filter'], params['conv_2_filter']), activation='tanh', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(params['conv_1_layers'], (params['conv_1_filter'], params['conv_1_filter']), activation='tanh', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=params['optimizer'](lr=params['learning_rate']), loss='mean_squared_error')

    curr_t = time.gmtime()
    train_history = autoencoder.fit(pos_train, pos_train,
                                    epochs=75, # params['num_epochs']
                                    batch_size=params['batch_size'],
                                    shuffle=True,
                                    validation_data=(pos_test, pos_test),
                                    verbose=2)#,
                                    # callbacks=[TensorBoard(log_dir='tmp/autoencoder_%d-%d-%d' % (curr_t.tm_hour, curr_t.tm_min, curr_t.tm_sec))])
    autoencoder.save_weights('autoencoder.h5')

    loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']
    epochs = range(75)
    plt.figure()
    plt.plot(epochs, loss, 'g--', label='Training loss')
    plt.plot(epochs, val_loss, 'm', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

    print("accuracy:", calculate_RSS(autoencoder, pos_train, pos_test, np.concatenate((neg_test, neg_train))))
    return autoencoder


def autoencode_fully_connected(params, visualize=False):
    dense_layer_nodes = 256
    reg = 0.01
    autoencoder = autoencode_params(params)

    pos_train, pos_test, neg_train, neg_test = load_data(rotate_append=False)
    X_train = np.concatenate((pos_train, neg_train))
    X_test = np.concatenate((pos_test, neg_test))
    y_train = np.array([1]*len(pos_train) + [0]*len(neg_train))
    y_test = np.array([1]*len(pos_test) + [0]*len(neg_test))
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)
    input_img = Input(shape=(40, 40, 3))

    # ENCODER
    x = Conv2D(params['conv_1_layers'], (params['conv_1_filter'], params['conv_1_filter']), activation='tanh', padding='same')(input_img) # , kernel_regularizer=regularizers.l2(0.1)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(params['conv_2_layers'], (params['conv_2_filter'], params['conv_2_filter']), activation='tanh', padding='same')(x) # , kernel_regularizer=regularizers.l2(0.1)
    x = MaxPooling2D((2, 2), padding='same')(x)

    flat = Flatten()(x)
    den = Dense(dense_layer_nodes, activation='relu')(flat) #, kernel_regularizer=regularizers.l2(reg)
    den = Dropout(rate=.6)(den)
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
                                   verbose=2,
                                   callbacks=[
                                       EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, mode='min', restore_best_weights=True),
                                       TensorBoard(log_dir='tmp/[0]autoencoder_fully_connected(layer#=%d)(reg=%.04f)_%d-%d-%d' % (dense_layer_nodes, reg, curr_t.tm_hour, curr_t.tm_min, curr_t.tm_sec))
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
                                   verbose=2,
                                   callbacks=[
                                       EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, mode='min', restore_best_weights=True),
                                       TensorBoard(log_dir='tmp/[1]autoencoder_fully_connected(layer#=%d)(reg=%.04f)_%d-%d-%d' % (dense_layer_nodes, reg, curr_t.tm_hour, curr_t.tm_min, curr_t.tm_sec))
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

    full_model.save_weights('autoencoder_classification.h5')
    if visualize:
        visualize_conv_layers(full_model)
    return full_model


# heavily based on: https://github.com/gabrielpierobon/cnnshapes/blob/master/README.md
def visualize_conv_layers(model):
    print("num model layers %d" % len(model.layers))
    layer_outputs = [layer.output for layer in model.layers[1:5]]#[1:]  # Extracts the outputs of the top 12 layers
    activation_model = Model(inputs=model.input, outputs=layer_outputs)  # Creates a model that will return these outputs, given the model input

    pos_test_img = np.array([io.imread("C:/Users/beekmanpc/Documents/BeeCounter/all_segments_fight_training/positive_sm/10-50-40_fight(494,56).png")])
    neg_test_img = np.array([io.imread("C:/Users/beekmanpc/Documents/BeeCounter/all_segments_fight_training/negative_sm/10-30-28_fight(128,261).png")])

    pos_activations = activation_model.predict(pos_test_img)
    neg_activations = activation_model.predict(neg_test_img)

    layer_names = []
    for layer in model.layers[1:5]:
        layer_names.append(layer.name)

    model_weights = [(name, layer.get_weights()) for name, layer in zip(layer_names, model.layers[1:5]) if "conv2d" in name]

    for layer in model_weights:
        # num_filters = layer[1][0].shape[3]
        colors = ['Red', "Green", "Blue"]
        for i, c in enumerate(colors):
            fig, ax = plt.subplots(2, 2)
            plt.suptitle(c + "channel")
            ax[0, 0].imshow(layer[1][0][:,:,:,0].squeeze()[:,:,i], cmap='gray') #ax[0, 0]
            ax[0, 0].set_title(layer[0]+"_filter=0")
            ax[0, 1].imshow(layer[1][0][:,:,:,1].squeeze()[:,:,i], cmap='gray') # ax[0, 1]
            ax[0, 1].set_title(layer[0]+"_filter=1")
            ax[1, 0].imshow(layer[1][0][:,:,:,2].squeeze()[:,:,i], cmap='gray') # ax[0, 2]
            ax[1, 0].set_title(layer[0]+"_filter=2")
            ax[1, 1].imshow(layer[1][0][:,:,:,3].squeeze()[:,:,i], cmap='gray') # ax[0, 3]
            ax[1, 1].set_title(layer[0]+"_filter=3")
            # ax[1, 0].imshow(layer[1][0][:,:,:,4].squeeze()[:,:,i], cmap='gray')
            # ax[1, 0].set_title(layer[0]+"_filter=4")
            # ax[1, 1].imshow(layer[1][0][:,:,:,5].squeeze()[:,:,i], cmap='gray')
            # ax[1, 1].set_title(layer[0]+"_filter=5")
            # ax[1, 2].imshow(layer[1][0][:,:,:,6].squeeze()[:,:,i], cmap='gray')
            # ax[1, 2].set_title(layer[0]+"_filter=6")
            # ax[1, 3].imshow(layer[1][0][:,:,:,7].squeeze()[:,:,i], cmap='gray')
            # ax[1, 3].set_title(layer[0]+"_filter=7")
            plt.show()
            plt.clf()

    print(layer_names)
    images_per_row = 4

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

    print("h")
    pass


def test_with_frame(full_model):

    # vid = VideoSelector()
    # detail_name, filename, hive = vid.download_video(type='fight')
    frame = None
    frame_num = -1
    play_video = True
    ret = None

    cap = cv2.VideoCapture("C:/Users/beekmanpc/Documents/BeeCounter/bee_videos/rpi12b@2018-10-07@16-21-36.mp4")# + filename) 17-26-55.h264
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
            cv2.imshow("fightz", frame)
            key = cv2.waitKey(1)
            # if frame_num < 1100:
            #     continue
            if key == 112: # 'p'
                play_video = not play_video
            #elif key == 115: # 's'
            # split the image into small 40x40 windows
            h, w, colors = frame.shape
            im_size = 40
            stride = 20
            # cycle through the frame finding all (im_size X im_size) images with a stride and locations
            for i in range(0, h - im_size, stride):
                for j in range(0, w - im_size, stride):
                    if sub_images is None:
                        sub_images = np.array([frame[i:i + im_size, j:j + im_size, :]])
                    else:
                        sub_images = np.concatenate((sub_images, [frame[i:i + im_size, j:j + im_size, :]]))
                    locations.append((i, j))
            predictions = full_model.predict(sub_images).reshape(-1)
            fight_predictions = np.where(predictions >= .5)
            if len(fight_predictions[0]) >= 1:
                print("jksdjlaf")
            # save all predicted fights and the surrounding context
            for idx, loc in enumerate(np.array(locations)[fight_predictions[0]]):
                curr_sub = frame[loc[0]:loc[0]+40, loc[1]:loc[1]+40, :]
                # curr_sub = frame[max(0,loc[0]-40):min(loc[0]+80, h), max(0,loc[1]-40):min(loc[1]+80,w), :]
                # cv2.rectangle(curr_sub, (40,40), (80,80), (0,255,0), 3)
                detail_name = "12b_16-21-36_"
                cv2.imwrite("C:/Users/beekmanpc/Documents/stigma/found_fights/"
                            +detail_name+"fight[%d](frame=%d).png" % (idx, frame_num),
                            curr_sub)
                curr_sub = frame[max(0,loc[0]-40):min(loc[0]+80, h), max(0,loc[1]-40):min(loc[1]+80,w), :]
                # cv2.rectangle(curr_sub, (40,40), (80,80), (0,255,0), 3)
                cv2.imwrite("C:/Users/beekmanpc/Documents/stigma/found_fights/"
                            +detail_name+"fight[%d](frame=%d)CONTEXT.png" % (idx, frame_num),
                            curr_sub)
                #break # breakout because we have collected all sub images of a frame with fights in it
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
    full_model = autoencode_fully_connected(params={
        'batch_size': 50, 'conv_1_filter': 3, 'conv_1_layers': 4,
        'conv_2_filter': 5, 'conv_2_layers': 8, 'learning_rate': 0.0001,
        'num_epochs': 100, 'optimizer': Adam}, visualize=True)
    # test_with_frame(full_model)



if __name__ == "__main__":
    main()

