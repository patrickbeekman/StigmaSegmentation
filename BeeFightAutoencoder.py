import numpy as np
import pandas as pd
from keras.optimizers import Adam
from matplotlib import patches
from skimage import io
from skimage.color import rgb2gray, rgb2hsv
# from PIL import Image
import os
from keras.applications.resnet50 import ResNet50
from keras import Sequential, Input, Model
from keras.models import load_model
from keras.layers import Flatten, Dense, Dropout, regularizers, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from skimage.transform import resize
from sklearn.metrics import mean_squared_error
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
    im_size = 80

    root_path = "C:/Users/beekmanpc/Documents/BeeCounter/all_segments_fight_training/positive/"
    for im in os.listdir(root_path):
        if positive is None:
            positive = np.array([rgb2hsv(io.imread(root_path + im))])
        else:
            positive = np.concatenate((positive, np.array([rgb2hsv(io.imread(root_path + im))])))
    #positive = positive.reshape((positive.shape[0], positive.shape[1], positive.shape[2], 1))

    root_path = "C:/Users/beekmanpc/Documents/BeeCounter/all_segments_fight_testing/positive/"
    for im in os.listdir(root_path):
        if p_test is None:
            p_test = np.array([rgb2hsv(io.imread(root_path + im))])
        else:
            p_test = np.concatenate((p_test, np.array([rgb2hsv(io.imread(root_path + im))])))
    #p_test = p_test.reshape((p_test.shape[0], p_test.shape[1], p_test.shape[2], 1))

    root_path = "C:/Users/beekmanpc/Documents/BeeCounter/all_segments_fight_training/negative/"
    neg_images = os.listdir(root_path)
    neg_sample = np.array(neg_images)[sample_without_replacement(len(neg_images), 1500, random_state=0)]
    for im in neg_sample:
        if negative is None:
            negative = np.array([rgb2hsv(io.imread(root_path + im))])
        else:
            negative = np.concatenate((negative, np.array([rgb2hsv(io.imread(root_path + im))])))
    # negative = negative.reshape((negative.shape[0], negative.shape[1], negative.shape[2], 1))
    print("positive_shape:", positive.shape)

    #in_out_shape = positive.shape[1] * positive.shape[2] * positive.shape[3]

    # positive = positive.astype('float32') / 255
    # negative = negative.astype('float32') / 255
    # p_test = p_test.astype('float32') / 255
    # n_test = n_test.astype('float32') / 255
    #positive = positive.reshape((len(positive), -1))
    # p_test = p_test.reshape((len(p_test), -1))

    input_img = Input(shape=(im_size,im_size, 3))  # adapt this if using `channels_first` image data format

    # hidden_1 = Dense(4096, activation='tanh')(input_img)
    # hidden_2 = Dense(2048, activation='tanh')(hidden_1)
    # code = Dense(1024, activation='tanh')(hidden_2)
    # hidden_3 = Dense(2048, activation='tanh')(code)
    # hidden_4 = Dense(4096, activation='tanh')(hidden_3)
    # decoded = Dense(in_out_shape, activation='sigmoid')(hidden_4)

    # ENCODER
    x = Conv2D(32, (7, 7), activation='tanh', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (5, 5), activation='tanh', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='tanh', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # DECODER
    x = Conv2D(8, (3, 3), activation='tanh', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (5, 5), activation='tanh', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (7, 7), activation='tanh', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    autoencoder.fit(positive, positive,
                    epochs=50,
                    batch_size=40,
                    shuffle=True,
                    validation_data=(p_test, p_test),
                    verbose=2,
                    callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

    autoencoder.save("data/autoencoder.h5")
    # autoencoder.load_weights("data/autoencoder.h5")

    # positive_mse = mean_squared_error(positive.reshape(len(positive), -1), autoencoder.predict(positive).reshape(len(positive), -1),
    #                                   multioutput='raw_values')
    # negative_mse = mean_squared_error(negative.reshape(len(negative), -1), autoencoder.predict(negative).reshape(len(negative), -1),
    #                                   multioutput='raw_values')
    # p_test_mse = mean_squared_error(p_test.reshape(len(p_test), -1), autoencoder.predict(p_test).reshape(len(p_test), -1),
    #                                   multioutput='raw_values')
    # n_test_mse = mean_squared_error(n_test.reshape(len(n_test), -1), autoencoder.predict(n_test).reshape(len(n_test), -1),
    #                                   multioutput='raw_values')

    # print("positive", positive_mse.mean())
    # # print("negative", negative_mse.mean())
    # print("p_test", p_test_mse.mean())
    # print("negative", negative_mse.mean())
    #
    # fig, ax = plt.subplots()
    # ax.boxplot([positive_mse, p_test_mse, negative_mse], positions=np.array(range(3))+1, labels=['positive', 'p_test', 'n_test'], meanline=True)
    # plt.show()
    # plt.clf()

    pos_pred = autoencoder.predict(positive)
    neg_pred = autoencoder.predict(negative)

    io.imshow(positive[10])#.reshape((80,80)), cmap='gray')
    io.show()
    io.imshow(pos_pred[10])#.reshape((80,80)), cmap='gray')
    io.show()
    io.imshow(negative[10])#.reshape((80,80)), cmap='gray')
    io.show()
    io.imshow(neg_pred[10])#.reshape((80,80)), cmap='gray')
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
    ax.boxplot([RSS_pos, RSS_p_test, RSS_neg], positions=np.array(range(3))+1, labels=['positive', 'p_test', 'negative'], meanline=True)
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


def main():
    build_autoencoder()


if __name__ == "__main__":
    main()

