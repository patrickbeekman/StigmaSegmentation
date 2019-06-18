import cv2
import time

import numpy as np
import pandas as pd
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import to_categorical
from matplotlib import patches
from skimage import io
# from PIL import Image
import os
from keras.applications.resnet50 import ResNet50
from keras import Sequential, Input, Model
from keras.models import load_model
from keras.layers import Flatten, Dense, Dropout, regularizers, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from skimage.color import rgb2hsv
from skimage.draw import circle, rectangle, rectangle_perimeter, circle_perimeter
from skimage.transform import resize
from sklearn.metrics import mean_squared_error, accuracy_score
from tensorflow.python.keras import backend
from sklearn.utils import shuffle, class_weight
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
#from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, UpSampling2D

from image_extract import InceptionFeatureExtractor
import keras_metrics as km
import glob
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

stigma_center = None
im_size = 64
positive, negative, p_test, n_test = None, None, None, None


def main():
    global stigma_center, positive, negative, p_test, n_test
    stigma_center = pd.read_json("stigma_locations/stigma_locations.json").T
    if not os.path.isdir("data"):
        os.mkdir("data")
    # segment_images()
    # transform_data()
    # build_model()
    positive, negative, p_test, n_test = load_data()
    # build_autoencoder()
    full_model = autoencode_fully_connected()
    test_model(full_model)
    # test_model()
    pass


# open each image listed 2560x1920
# move a 200x200 window around
# if pixel relative position (100,100) (center) is within the start and end bounds then save to positive class
def segment_images(count_num=0, stig=1):
    global stigma_center
    # stigma_center = pd.read_json("stigma_locations/stigma_locations%d.json" % stig).T
    stride = 50
    win_size = 200
    half_size = win_size/2
    count = count_num

    for stigma in stigma_center.iterrows():
        if count > 47:
            break
        # if count == 9: # idk broken folder?
        #     count+=1
        #     continue
        count += 1
        stigma = stigma[1]
        if not os.path.exists("my_stigma_locations/" + stigma.name[:-stigma.name[::-1].find("/")] + "positive/"):
            os.mkdir("my_stigma_locations/" + stigma.name[:-stigma.name[::-1].find("/")] + "positive/")
        if not os.path.exists("my_stigma_locations/" + stigma.name[:-stigma.name[::-1].find("/")] + "negative/"):
            os.mkdir("my_stigma_locations/" + stigma.name[:-stigma.name[::-1].find("/")] + "negative/")
        else: # someone else is working on it
            continue

        im = io.imread("stigma_locations/" + stigma.name)
        # cycle through all 200x200 images
        for i in np.arange(0, im.shape[0]-win_size, stride): # x direction
            for j in np.arange(0, im.shape[1]-win_size, stride): # y direction
                # if the center pixel is considered a stigma then save positive example
                curr_im = im[i:i+win_size, j:j+win_size]
                if j + half_size >= stigma.start[0]+50 and j + half_size <= stigma.end[0]+50 and i + half_size >= stigma.start[1]-50 and i + half_size <= stigma.end[1]-50:
                    io.imsave("my_stigma_locations/" + stigma.name[:stigma.name.rfind("/")] + "/positive" + stigma.name[stigma.name.rfind("/"):][:-4] + "(i=%dj=%d).png" % (i,j), curr_im)
                else:
                    io.imsave("my_stigma_locations/" + stigma.name[:stigma.name.rfind("/")] + "/negative" + stigma.name[stigma.name.rfind("/"):][:-4] + "(i=%dj=%d).png" % (i,j), curr_im)
        print("[%d]p=%d %s segmented and saved" % (count-1, stig, stigma.name))
    return

def transform_data():
    global stigma_center
    ife = InceptionFeatureExtractor()
    # filepaths = ["my_stigma_locations/" + fn for fn in list(stigma_center.index)]
    count = 0
    positive = []
    negative = []
    size = 100

    for stigma in stigma_center.iterrows():
        if count > 47:
            break
        # if count == 9:
        #     count +=1
        #     continue
        count+=1
        stigma = stigma[1]
        name = stigma.name[:-4].replace('/','@')
        if os.path.exists("data/%s_negative[0].npy" % name):
            print("[%d] skipping: %s" % (count-1,name))
            continue
        pos_path = "my_stigma_locations/" + stigma.name[:stigma.name.rfind("/")] + "/positive/"
        neg_path = "my_stigma_locations/" + stigma.name[:stigma.name.rfind("/")] + "/negative/"
        p = [pos_path+name for name in os.listdir(pos_path)]
        pos = ife.transform(p)
        print("[%d] about to save:" % (count-1), name, pos.shape)
        np.save(open("data/%s_positive.npy" % name, "wb"), pos)
        del p
        n = [neg_path+name for name in os.listdir(neg_path)]
        for i in range(int(np.ceil(len(n)/size))):
            np.save(open("data/%s_negative[%d].npy" % (name, i), "wb"), ife.transform(n[i*size:i*size+size]))
        del n


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = backend.sum(backend.round(backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + backend.epsilon())
    return recall


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = backend.sum(backend.round(backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + backend.epsilon())
    return precision


def build_model():
    global stigma_center
    positive = None
    negative = None
    p_test = None
    n_test = None
    count = 0
    for stigma in stigma_center.iterrows():
        if count > 47:
            break
        # if count == 9:
        #     count += 1
        #     continue
        count+=1
        stigma = stigma[1]
        name = stigma.name[:-4].replace('/','@')
        if count <= 40: # use samples 0-42 for training
            if positive is None:
                positive = np.load(open("data/%s_positive.npy" % name, "rb"))
            else:
                positive = np.concatenate((positive, np.load(open("data/%s_positive.npy" % name, "rb"))))
            if negative is None:
                negative = np.load(open("data/%s_negative[0].npy" % name, "rb"))
            else:
                for f in glob.glob("data/%s_negative*" % name):
                    negative = np.concatenate((negative, np.load(open(f, "rb")))) # "data/%s_negative.npy" % name
        elif count <= 47: # use samples 43-50 for testing
            if p_test is None:
                p_test = np.load(open("data/%s_positive.npy" % name, "rb"))
            else:
                p_test = np.concatenate((p_test, np.load(open("data/%s_positive.npy" % name, "rb"))))
            if n_test is None:
                n_test = np.load(open("data/%s_negative[0].npy" % name, "rb"))
            else:
                for f in glob.glob("data/%s_negative*" % name):
                    n_test = np.concatenate((n_test, np.load(open(f, "rb")))) # "data/%s_negative.npy" % name
        else: # validation
            pass
        print("[%d] appended %s, curr size is %d" % (count-1, name, (len(positive)+len(negative))))

    # oversample (with replacement) the positive class to be twice as large
    oversample_index = np.random.choice(list(range(len(positive))), len(positive)*4)
    positive = np.concatenate((positive, positive[oversample_index]))

    print("creating Train and Test sets")
    X_train = np.concatenate((positive, negative))
    y_train = np.array([1]*len(positive) + [0]*len(negative))
    X_test = np.concatenate((p_test, n_test))
    y_test = np.array([1]*len(p_test) + [0]*len(n_test))
    # shuffle the data
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)

    print("starting to build and train model")
    model = Sequential()
    model.add(Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Dropout(rate=0.6))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(lr=0.0001, decay=10**-6),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])#, precision, recall])

    history = model.fit(X_train, y_train,
              epochs=60,
              batch_size=60,
              validation_data=(X_test, y_test),
              sample_weight=class_weight.compute_sample_weight('balanced', y_train))

    # Creates a HDF5 file 'my_model.h5'
    model.save('data/my_model.h5')

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model acc score')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.clf()

    # plt.plot(history.history['precision'])
    # plt.plot(history.history['val_precision'])
    # plt.title('Model precision score')
    # plt.ylabel('Precision')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    # plt.clf()
    #
    # plt.plot(history.history['recall'])
    # plt.plot(history.history['val_recall'])
    # plt.title('Model recall score')
    # plt.ylabel('Recall')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    # plt.clf()


def load_data():
    global stigma_center, im_size, positive, negative, p_test, n_test
    count = 0
    stop = 50

    # load the data
    for stigma in stigma_center.iterrows():
        if count > stop:
            break
        # if count == 9:
        #     count += 1
        #     continue
        count+=1
        stigma = stigma[1]
        pos_path = 'my_stigma_locations/' + stigma.name[:stigma.name.rfind("/")] + "/positive/"
        neg_path = 'my_stigma_locations/' + stigma.name[:stigma.name.rfind("/")] + "/negative/"
        if count <= int(stop*.8): # use samples 0-42 for training
            for im in os.listdir(pos_path):
                if positive is None:
                    positive = np.array([resize(rgb2hsv(io.imread(pos_path + im)), output_shape=(im_size,im_size,3))])
                else:
                    positive = np.concatenate((positive, np.array([resize(rgb2hsv(io.imread(pos_path + im)), output_shape=(im_size,im_size,3))])))
            for im in np.random.choice(os.listdir(neg_path), len(os.listdir(pos_path))):
                if negative is None:
                    negative = np.array([resize(rgb2hsv(io.imread(neg_path + im)), output_shape=(im_size,im_size,3))])
                else:
                    negative = np.concatenate((negative, np.array([resize(rgb2hsv(io.imread(neg_path + im)), output_shape=(im_size,im_size,3))])))
            if count == int(stop*.8):
                print("-------------" * 4 + "Testing below" + "-------------" * 4)
        elif count <= stop: # use samples 43-50 for testing
            for im in os.listdir(pos_path):
                if p_test is None:
                    p_test = np.array([resize(rgb2hsv(io.imread(pos_path + im)), output_shape=(im_size,im_size,3))])
                else:
                    p_test = np.concatenate((p_test, np.array([resize(rgb2hsv(io.imread(pos_path + im)), output_shape=(im_size,im_size,3))])))
            for im in np.random.choice(os.listdir(neg_path), len(os.listdir(pos_path))):
                try:
                    if n_test is None:
                        n_test = np.array([resize(rgb2hsv(io.imread(neg_path + im)), output_shape=(im_size,im_size,3))])
                    else:
                        n_test = np.concatenate((n_test, np.array([resize(rgb2hsv(io.imread(neg_path + im)), output_shape=(im_size,im_size,3))])))
                except FileNotFoundError:
                    continue
        else: # validation
            pass
        try:
            print("[%d] appended %s | pos_train size:%d | neg_train:%d | pos_test:%d | neg_test:%d |" % (count-1, stigma.name, len(positive), len(negative), len(p_test), len(n_test)))
        except TypeError:
            print("[%d] appended %s | pos_train size:%d | neg_train:%d | pos_test:%d | neg_test:%d |" % (count-1, stigma.name, len(positive), len(negative), 0, 0))
    return positive, negative, p_test, n_test


def build_autoencoder():
    global im_size, positive, negative, p_test, n_test
    num_epochs = 60

    # root_path = "C:/Users/beekmanpc/Documents/BeeCounter/all_segments_fight_training/positive/"
    # for im in os.listdir(root_path):
    #     if positive is None:
    #         positive = np.array([io.imread(root_path + im)])
    #     else:
    #         positive = np.concatenate((positive, np.array([io.imread(root_path + im)])))
    # root_path = "C:/Users/beekmanpc/Documents/BeeCounter/all_segments_fight_training/negative/"
    # count = 0
    # for im in os.listdir(root_path):
    #     if count > 1500:
    #         break
    #     if negative is None:
    #         negative = np.array([io.imread(root_path + im)])
    #     else:
    #         negative = np.concatenate((negative, np.array([io.imread(root_path + im)])))
    #     count+=1

    print("positive_shape:", positive.shape)
    # print("negative_shape:", negative.shape)
    # print(p_test.shape)

    # positive = np.apply_along_axis(my_resize, 0, positive)
    # p_test = np.apply_along_axis(my_resize, 0, p_test)
    #
    # print(positive.shape)
    # print(p_test.shape)

    #in_out_shape = positive.shape[1] * positive.shape[2] * positive.shape[3]

    # positive = positive.astype('float32') / 255
    # # negative = negative.astype('float32') / 255
    # p_test = p_test.astype('float32') / 255
    # n_test = n_test.astype('float32') / 255
    #positive = positive.reshape((len(positive), -1))
    # p_test = p_test.reshape((len(p_test), -1))

    input_img = Input(shape=(im_size,im_size,3))  # adapt this if using `channels_first` image data format

    # ENCODER
    x = Conv2D(64, (7, 7), activation='tanh', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (5, 5), activation='tanh', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='tanh', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # DECODER
    x = Conv2D(16, (3, 3), activation='tanh', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (5, 5), activation='tanh', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (7, 7), activation='tanh', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    curr_t = time.gmtime()
    train_history = autoencoder.fit(positive, positive,
                                    epochs=num_epochs,
                                    batch_size=20,
                                    shuffle=True,
                                    validation_data=(p_test, p_test),
                                    verbose=2,
                                    callbacks=[TensorBoard(log_dir='tmp/autoencoder_%d-%d-%d' % (curr_t.tm_hour, curr_t.tm_min, curr_t.tm_sec))])
    autoencoder.save("data/autoencoder.h5")
    # autoencoder.load_weights("data/autoencoder.h5")

    loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']
    epochs = range(num_epochs)
    plt.figure()
    plt.plot(epochs, loss, 'g--', label='Training loss')
    plt.plot(epochs, val_loss, 'm', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

    print("accuracy:", calculate_RSS(autoencoder, positive, p_test, np.concatenate((n_test, negative))))
    return autoencoder


def autoencode_fully_connected():
    global im_size, positive, negative, p_test, n_test
    dense_layer_nodes = 512
    reg = 0.0001
    num_epochs = 100
    autoencoder = build_autoencoder()

    # pos_train, pos_test, neg_train, neg_test = load_data(rotate_append=True)
    input_img = Input(shape=(im_size, im_size, 3))

    # ENCODER
    x = Conv2D(64, (7, 7), activation='tanh', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (5, 5), activation='tanh', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='tanh', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    flat = Flatten()(encoded)
    den = Dense(dense_layer_nodes, activation='relu', kernel_regularizer=regularizers.l2(reg))(flat)#
    den = Dropout(rate=.6)(den)
    # den = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(den)
    out = Dense(2, activation='softmax')(den)

    full_model = Model(input_img, out)
    # get the weights from the pretrained model
    for l1, l2 in zip(full_model.layers[:6], autoencoder.layers[0:6]):
        l1.set_weights(l2.get_weights())
    # hold them steady
    for layer in full_model.layers[0:6]:
        layer.trainable = False
    # compile and train the model
    full_model.compile(loss=categorical_crossentropy, optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    curr_t = time.gmtime()
    train_history = full_model.fit(np.concatenate((positive, negative)), to_categorical(np.array([1]*len(positive) + [0]*len(negative)), 2),
                                   epochs=num_epochs,
                                   batch_size=20,
                                   shuffle=True,
                                   validation_data=(np.concatenate((p_test, n_test)), to_categorical(np.array([1]*len(p_test) + [0]*len(n_test)), 2)),
                                   verbose=2,
                                   sample_weight=None,
                                   callbacks=[TensorBoard(log_dir='tmp/autoencoder_fully_connected[0](layer#=%d)(reg=%.04f)_%d-%d-%d' % (dense_layer_nodes, reg, curr_t.tm_hour, curr_t.tm_min, curr_t.tm_sec))])

    # set weights to trainable
    for layer in full_model.layers[0:6]:
        layer.trainable = True
    full_model.compile(loss=categorical_crossentropy, optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    curr_t = time.gmtime()
    train_history = full_model.fit(np.concatenate((positive, negative)), to_categorical(np.array([1]*len(positive) + [0]*len(negative)), 2),
                                   epochs=num_epochs,
                                   batch_size=20,
                                   shuffle=True,
                                   validation_data=(np.concatenate((p_test, n_test)), to_categorical(np.array([1]*len(p_test) + [0]*len(n_test)), 2)),
                                   verbose=2,
                                   sample_weight=None,
                                   callbacks=[TensorBoard(log_dir='tmp/autoencoder_fully_connected[1](layer#=%d)(reg=%.04f)_%d-%d-%d' % (dense_layer_nodes, reg, curr_t.tm_hour, curr_t.tm_min, curr_t.tm_sec))])

    # plot the train and validation loss
    loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']
    epochs = range(num_epochs)
    plt.figure()
    plt.plot(epochs, loss, 'g--', label='Training loss')
    plt.plot(epochs, val_loss, 'm', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

    # plot the train and validation acc
    acc = train_history.history['acc']
    val_acc = train_history.history['val_acc']
    epochs = range(num_epochs)
    plt.figure()
    plt.plot(epochs, acc, 'g--', label='Training acc')
    plt.plot(epochs, val_acc, 'm', label='Validation acc')
    plt.title('Training and validation acc')
    plt.legend()
    plt.show()

    pos_train_acc = accuracy_score(np.round(full_model.predict(positive)).astype(int), to_categorical(np.array([1] * len(positive))))
    neg_train_acc = accuracy_score(np.round(full_model.predict(negative)).astype(int), np.flip(to_categorical(np.array([1] * len(negative))), axis=1))
    pos_test_acc = accuracy_score(np.round(full_model.predict(p_test)).astype(int), to_categorical(np.array([1] * len(p_test))))
    neg_test_acc = accuracy_score(np.round(full_model.predict(n_test)).astype(int), np.flip(to_categorical(np.array([1] * len(n_test))), axis=1))
    print("pos_train:%.03f\nneg_train:%.03f\npos_test:%.03f\nneg_test:%.03f" % (pos_train_acc, neg_train_acc, pos_test_acc, neg_test_acc))

    full_model.save_weights('autoencoder_classification.h5')
    return full_model


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


def test_model(full_model):
    test_paths = [
        'my_stigma_locations/22.06.18_0870751_pos1_kurz+lang/113MEDIA/Y0060538.jpg',
        'my_stigma_locations/02.07.18_4982033_pos3_kurz/168MEDIA/Y0150018.jpg',
        # 'my_stigma_locations/02.07.18_4982033_pos3_kurz/168MEDIA/Y0150474.jpg', # seen similar
        'my_stigma_locations/19.06.18_0870751_pos1_kurz/101MEDIA/Y0030052.jpg',
        # 'my_stigma_locations/19.06.18_0870751_pos1_kurz/101MEDIA/Y0030616.jpg', # seen similar
        # 'my_stigma_locations/19.06.18_4982033_pos3_kurz/101MEDIA/Y0010346.jpg', # seen similar
        # 'my_stigma_locations/19.06.18_4982033_pos3_kurz/102MEDIA/Y0010483.jpg', # seen similar
        'my_stigma_locations/20.06.18_0870751_pos1_kurz/106MEDIA/Y0040159.jpg',
        # 'my_stigma_locations/20.06.18_3403289_pos2_kurz/106MEDIA/Y0020183.jpg', # seen similar
        # 'my_stigma_locations/21.06.18_0870751_pos1_kurz/109MEDIA/Y0050316.jpg', # seen similar
        'my_stigma_locations/21.06.18_3403289_pos2_kurz/106MEDIA/Y0030737.jpg',
        'my_stigma_locations/22.06.18_0870751_pos1_kurz+lang/115MEDIA/Y0060185.jpg',
        'my_stigma_locations/27.06.18_4111145_pos4_kurz/159MEDIA/Y0131140.jpg',
        'my_stigma_locations/27.06.18_4111145_pos4_kurz/159MEDIA/Y0131537.jpg',
        'my_stigma_locations/28.06.18_1654305_pos6_kurz/170MEDIA/Y0170254.jpg',
        # 'my_stigma_locations/28.06.18_4237688_pos5_kurz/221MEDIA/Y0210041.jpg',
        'my_stigma_locations/29.06.18_3403289_pos2_kurz/172MEDIA/Y0170746.jpg',
        'my_stigma_locations/30.06.18_1654305_pos6_kurz/172MEDIA/Y0180506.jpg',
        'my_stigma_locations/30.06.18_3403289_pos2_kurz/174MEDIA/Y0181141.jpg',
        'my_stigma_locations/01.07.18_4237688_pos5_kurz/216MEDIA/Y0190290.jpg',
        # 'my_stigma_location/01.07.18_4237688_pos5_kurz/216MEDIA/Y0200983.jpg',

    ]

    print("Testing on different unseen images.")
    for path in test_paths:
        predict_stigma_center(full_model, path)


def predict_stigma_center(full_model, file_path):
    global im_size
    stride = 50
    win_size = 200
    X = None
    locations = []
    file_name = file_path[file_path.rfind("/")+1:file_path.find(".jpg")]

    im = io.imread(file_path)
    # cycle through all 200x200 images
    for i in np.arange(0, im.shape[0] - win_size, stride):  # x direction
        for j in np.arange(0, im.shape[1] - win_size, stride):  # y direction
            locations.append((j,i))
            if X is None:
                X = np.array([resize(rgb2hsv(im[i:i + win_size, j:j + win_size]), output_shape=(im_size,im_size,3))])
            else:
                X = np.concatenate((X, np.array([resize(rgb2hsv(im[i:i + win_size, j:j + win_size]), output_shape=(im_size,im_size,3))])))

    results = np.round(full_model.predict(X))
    locations = np.array(locations)
    stigma_locs = locations[np.where(results[:,1] == 1)]
    center_of_stigma = np.median(stigma_locs, axis=0).astype(int)#stigma_locs.mean(axis=0).astype(int)

    image = cv2.imread(file_path)
    calculated_center = tuple(center_of_stigma + 100)
    print("Predicted center is (%d, %d)" % calculated_center)
    cv2.circle(image, calculated_center, 100, thickness=5, color=(0, 255, 0))
    cv2.imwrite("stigma_predictions/%s_(%d, %d).png" % (file_name, calculated_center[0], calculated_center[1]), image)


if __name__ == "__main__":
    main()
