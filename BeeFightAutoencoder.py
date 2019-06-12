import numpy as np
import pandas as pd
from keras.optimizers import Adam, SGD, Adadelta
from matplotlib import patches
from skimage import io
from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
# from PIL import Image
import os
from keras.applications.resnet50 import ResNet50
from keras import Sequential, Input, Model
from keras.models import load_model
from keras.layers import Flatten, Dense, Dropout, regularizers, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from skimage.transform import resize, rotate
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
from sklearn.utils._random import sample_without_replacement
from tensorflow.python.keras import backend
from sklearn.utils import shuffle, class_weight
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt


def build_autoencoder():
    global stigma_center
    stop = 25
    positive = None
    negative = None
    p_test = None
    n_test = None
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
    print("positive_shape:", positive.shape)

    rot = [90, 180, 270]
    pos2 = positive.copy()
    for i, im in enumerate(pos2):
        pos2[i] = rotate(im, angle=rot[i%3])
    positive = np.concatenate((positive, pos2))

    positive = positive / 255
    p_test = p_test / 255
    negative = negative / 255

    input_img = Input(shape=(im_size,im_size, 3))  # adapt this if using `channels_first` image data format

    # hidden_1 = Dense(4096, activation='tanh')(input_img)
    # hidden_2 = Dense(2048, activation='tanh')(hidden_1)
    # code = Dense(1024, activation='tanh')(hidden_2)
    # hidden_3 = Dense(2048, activation='tanh')(code)
    # hidden_4 = Dense(4096, activation='tanh')(hidden_3)
    # decoded = Dense(in_out_shape, activation='sigmoid')(hidden_4)

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

    autoencoder.fit(positive, positive,
                    epochs=50,
                    batch_size=20,
                    shuffle=True,
                    validation_data=(p_test, p_test),
                    verbose=2,
                    callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

    autoencoder.save("data/autoencoder.h5")
    # autoencoder.load_weights("data/autoencoder.h5")

    pos_pred = autoencoder.predict(positive)
    neg_pred = autoencoder.predict(negative)

    io.imshow(positive[15])#.reshape((80,80)), cmap='gray')
    io.show()
    io.imshow(pos_pred[15])#.reshape((80,80)), cmap='gray')
    io.show()
    io.imshow(negative[15])#.reshape((80,80)), cmap='gray')
    io.show()
    io.imshow(neg_pred[15])#.reshape((80,80)), cmap='gray')
    io.show()

    # Residual Sum of Squares
    pred_pos = autoencoder.predict(positive)
    RSS_pos = ((positive - pred_pos) ** 2).sum(axis=(1,2,3))
    pred_p_test = autoencoder.predict(p_test)
    RSS_p_test = ((p_test - pred_p_test) ** 2).sum(axis=(1,2,3))
    pred_neg = autoencoder.predict(negative)
    RSS_neg = ((negative - pred_neg) ** 2).sum(axis=(1,2,3))

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
    print("positive_acc=%.03f\np_test_acc=%.03f\nn_test_acc=%.03f" % (positive_acc, p_test_acc, negative_acc))

    fig, ax = plt.subplots()
    ax.boxplot([RSS_pos, RSS_p_test, RSS_neg], positions=np.array(range(3))+1, labels=['positive', 'pos_test', 'neg_test'], meanline=True)
    plt.title("RSS reconstruction errors: total acc=%.03f, pos=%.02f,\np_test=%.02f, n_test=%.02f, thresh=%.01f" % (best_acc, positive_acc, p_test_acc, negative_acc, threshold))
    plt.show()
    plt.clf()

    print("yo")

    # plt.hist(positive_mse, bins=20)
    # plt.title("positive_mse %.02f" % positive_mse.mean())
    # plt.show()
    # plt.clf()

    # plt.hist(negative_mse, bins=20)
    # plt.title("negative_mse %.02f" % negative_mse.mean())
    # plt.show()
    # plt.clf()
    #
    # plt.hist(p_test_mse, bins=20)
    # plt.title("p_test_mse %.02f" % p_test_mse.mean())
    # plt.show()
    # plt.clf()
    #
    # plt.hist(n_test_mse, bins=20)
    # plt.title("n_test_mse %.02f" % n_test_mse.mean())
    # plt.show()
    # plt.clf()



# NEXT STEP: Change this model to be like the datacamp example with encoder and fully connected for classification
# I should decide on a scoring metric? Mae
'''
hyperparameter tuning for autoencoder
- different models (number of convolutions, num of max pools)
- size of conv filters
- learning rate
- num epochs
- batch_size
- optimizier and its parameters
'''


def calculate_RSS(autoencoder, pos_train, pos_test, neg_test):
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
    threshold = RSS_neg.min()
    for t in np.arange(int(RSS_neg.min()), int(RSS_p_test.max()), .5):
        positive_acc = (RSS_pos <= t).sum() / len(RSS_pos)
        p_test_acc = (RSS_p_test <= t).sum() / len(RSS_p_test)
        n_test_acc = (RSS_neg > t).sum() / len(RSS_neg)
        acc = (positive_acc + p_test_acc) / 2 * n_test_acc
        if acc > best_acc:
            best_acc = acc
            threshold = t
    return best_acc
    pass


def hyper_param_tuning(pos_train, pos_test, neg_test):
    im_size = 40
    input_img = Input(shape=(im_size, im_size, 3))
    best_params = {}
    best_score = 0
    best_model = ""

    params_1 = {
              "conv_1_layers": [8, 16, 32, 64, 128, 256, 512],
              "conv_1_filter": [3, 5, 7],
              "num_epochs": [25, 50, 75, 100],
              "batch_size": [15, 20, 30, 50, 60],
              "optimizer": ['adam', 'adadelta', 'sgd'],
              "learning_rate": [.1, .01, .001, .0001, .00001]
              }

    params_2 = {
              "conv_1_layers": [8, 16, 32, 64, 128, 256, 512],
              "conv_1_filter": [3, 5, 7],
              "conv_2_layers": [8, 16, 32, 64, 128, 256, 512],
              "conv_2_filter": [3, 5, 7],
              "num_epochs": [25, 50, 75, 100],
              "batch_size": [15, 20, 30, 50, 60],
              "optimizer": ['adam', 'adadelta', 'sgd'],
              "learning_rate": [.1, .01, .001, .0001, .00001]
              }

    params_3 = {
              "conv_1_layers": [8, 16, 32, 64, 128, 256, 512],
              "conv_1_filter": [3, 5, 7],
              "conv_2_layers": [8, 16, 32, 64, 128, 256, 512],
              "conv_2_filter": [3, 5, 7],
              "conv_3_layers": [8, 16, 32, 64, 128, 256, 512],
              "conv_3_filter": [3, 5, 7],
              "num_epochs": [25, 50, 75, 100],
              "batch_size": [15, 20, 30, 50, 60],
              "optimizer": [Adam, Adadelta, SGD],
              "learning_rate": [.1, .01, .001, .0001, .00001]
              }

    count = 0
    param_1_len = len(ParameterGrid(params_1))
    for params in ParameterGrid(params_1):
        decoded = model_1(input_img, params)
        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer=params['optimizer'](lr=params['learning_rate']), loss='mean_squared_error')
        autoencoder.fit(pos_train, pos_train,
                        epochs=params['num_epochs'],
                        batch_size=params['batch_size'],
                        shuffle=True,
                        validation_data=(pos_test, pos_test))
        curr_acc = calculate_RSS(autoencoder, pos_train, pos_test, neg_test)
        print("(%d/%d) Model_1: %.03f Params:" % (count, param_1_len, curr_acc), params)
        count += 1
        if curr_acc > best_score:
            best_score = curr_acc
            best_params = params
            best_model = "model_1"

    count = 0
    param_2_len = len(ParameterGrid(params_2))
    for params in ParameterGrid(params_2):
        decoded = model_2(input_img, params)
        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer=params['optimizer'](lr=params['learning_rate']), loss='mean_squared_error')
        autoencoder.fit(pos_train, pos_train,
                        epochs=params['num_epochs'],
                        batch_size=params['batch_size'],
                        shuffle=True,
                        validation_data=(pos_test, pos_test))
        curr_acc = calculate_RSS(autoencoder, pos_train, pos_test, neg_test)
        print("(%d/%d) Model_2: %.03f Params:" % (count, param_2_len, curr_acc), params)
        count += 1
        if curr_acc > best_score:
            best_score = curr_acc
            best_params = params
            best_model = "model_2"

    count = 0
    param_3_len = len(ParameterGrid(params_3))
    for params in ParameterGrid(params_3):
        decoded = model_3(input_img, params)
        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer=params['optimizer'](lr=params['learning_rate']), loss='mean_squared_error')
        autoencoder.fit(pos_train, pos_train,
                        epochs=params['num_epochs'],
                        batch_size=params['batch_size'],
                        shuffle=True,
                        validation_data=(pos_test, pos_test))
        curr_acc = calculate_RSS(autoencoder, pos_train, pos_test, neg_test)
        print("(%d/%d) Model_3: %.03f Params:" % (count, param_3_len, curr_acc), params)
        count += 1
        if curr_acc > best_score:
            best_score = curr_acc
            best_params = params
            best_model = "model_3"

    print("Best Cumulative Accuracy: %.03f\nBest model: %s\nBest params:" % (best_score, best_model), best_params)
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
    build_autoencoder()


if __name__ == "__main__":
    main()

