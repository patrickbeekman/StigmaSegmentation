import numpy as np
import pandas as pd
from keras.optimizers import Adam
from matplotlib import patches
from skimage import io
# from PIL import Image
import os
from keras.applications.resnet50 import ResNet50
from keras import Sequential
from keras.models import load_model
from keras.layers import Flatten, Dense, Dropout, regularizers
from tensorflow.python.keras import backend
from sklearn.utils import shuffle, class_weight
import tensorflow as tf
from image_extract import InceptionFeatureExtractor
import keras_metrics as km
import glob
import matplotlib.pyplot as plt

stigma_center = None


def main():
    global stigma_center
    stigma_center = pd.read_json("stigma_locations/stigma_locations.json").T
    # segment_images()
    # transform_data()
    # build_model()
    test_model()
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
        if count > 50:
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
                if j + half_size >= stigma.start[0] and j + half_size <= stigma.end[0] and i + half_size >= stigma.start[1] and i + half_size <= stigma.end[1]:
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
        if count > 50:
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
        if count > 50:
            break
        # if count == 9:
        #     count += 1
        #     continue
        count+=1
        stigma = stigma[1]
        name = stigma.name[:-4].replace('/','@')
        if count <= 41: # use samples 0-42 for training
            if positive is None:
                positive = np.load(open("data/%s_positive.npy" % name, "rb"))
            else:
                positive = np.concatenate((positive, np.load(open("data/%s_positive.npy" % name, "rb")))) 
            if negative is None:
                negative = np.load(open("data/%s_negative[0].npy" % name, "rb"))
            else:
                for f in glob.glob("data/%s_negative*" % name):
                    negative = np.concatenate((negative, np.load(open(f, "rb")))) # "data/%s_negative.npy" % name
        elif count <= 50: # use samples 43-50 for testing
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

    print("creating Train and Test sets")
    X_train = np.concatenate((positive, negative))
    y_train = np.array([1]*len(positive) + [0]*len(negative))
    X_test = np.concatenate((p_test, n_test))
    y_test = np.array([1]*len(p_test) + [0]*len(n_test))
    # shuffle the data
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)
    print(y_train[:30])

    print("starting to build and train model")
    # create
    model = Sequential()
    # model.add(Flatten(input_shape=X_train.shape[1:]))
    model.add(Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Dropout(rate=0.6))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='sgd',#Adam(lr=0.0001, decay=10**-6),
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


def test_model():
    stride = 50
    win_size = 200
    half_size = win_size/2
    X = None
    locations = []
    ife = InceptionFeatureExtractor()

    im = io.imread("my_stigma_locations/20.06.18_3403289_pos2_kurz/104MEDIA/Y0020859.jpg")
    # cycle through all 200x200 images
    for i in np.arange(0, im.shape[0] - win_size, stride):  # x direction
        for j in np.arange(0, im.shape[1] - win_size, stride):  # y direction
            locations.append((i,j))
            if X is None:
                X = np.array([im[i:i + win_size, j:j + win_size]])
            else:
                X = np.concatenate((X, [im[i:i + win_size, j:j + win_size]]))

    # Returns a compiled model identical to the previous one
    model = load_model('data/my_model.h5')

    # model.compile(optimizer=Adam(lr=0.0001),
    #               loss='binary_crossentropy',
    #               metrics=['accuracy', precision(), recall()])
    y_pred = None
    size = 200
    for i in range(int(np.ceil(len(X)/size))):
        if y_pred is None:
            y_pred = model.predict(ife.transform(my_X=X[i*size:i*size+size]))
        else:
            y_pred = np.concatenate((y_pred, model.predict(ife.transform(my_X=X[i*size:i*size+size]))))

    try:
        stigma_location = locations[int(np.median(np.where(y_pred > .5)[0]))]
    except ValueError:
        stigma_location = (0, 0)
    # plot the location as a square
    print("stigma top left loc:", stigma_location)
    fig, ax = plt.subplots(1)
    ax.imshow(im)
    for r in np.where(y_pred > .5)[0]:
        rect = patches.Rectangle(locations[int(r)], size, size, linewidth=3, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()
    print("hi")





if __name__ == "__main__":
    main()
