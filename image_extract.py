import sys
import os
import time
import tarfile
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from six.moves import urllib
from skimage.transform import resize
from tensorflow.contrib.slim.nets import inception
from skimage import io

class InceptionFeatureExtractor:
    width = 299
    height = 299
    channels = 3

    TF_MODELS_URL = "http://download.tensorflow.org/models"
    INCEPTION_V3_URL = TF_MODELS_URL + "/inception_v3_2016_08_28.tar.gz"
    INCEPTION_PATH = os.path.join("datasets", "inception")
    INCEPTION_V3_CHECKPOINT_PATH = os.path.join(INCEPTION_PATH, "inception_v3.ckpt")

    def __init__(self):
        self.fetch_pretrained_inception_v3()
        self.init_graph()

    @staticmethod
    def download_progress(count, block_size, total_size):
        percent = count * block_size * 100 // total_size
        sys.stdout.write("\rDownloading: {}%".format(percent))
        sys.stdout.flush()

    @staticmethod
    def fetch_pretrained_inception_v3(url=None, path=None):
        if url is None: 
            url = InceptionFeatureExtractor.INCEPTION_V3_URL
        if path is None:
            path = InceptionFeatureExtractor.INCEPTION_PATH

        if os.path.exists(InceptionFeatureExtractor.INCEPTION_V3_CHECKPOINT_PATH):
            return
        os.makedirs(path, exist_ok=True)
        tgz_path = os.path.join(path, "inception_v3.tgz")
        urllib.request.urlretrieve(url, tgz_path, 
                                   reporthook=InceptionFeatureExtractor.download_progress)
        inception_tgz = tarfile.open(tgz_path)
        inception_tgz.extractall(path=path)
        inception_tgz.close()
        os.remove(tgz_path)

    def init_graph(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.height, self.width, self.channels], name="X")
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            logits, end_points = inception.inception_v3(self.X, num_classes=1001)
        self.prelogits = tf.squeeze(end_points["PreLogits"], axis=[1, 2])
        self.init = tf.global_variables_initializer()

    @staticmethod
    def prepare_image(image_path=None, im=None):
        if im is None:
            image = io.imread(image_path)
        else:
            image = im
        target_width = InceptionFeatureExtractor.width
        target_height = InceptionFeatureExtractor.height

        # crop image
        height = image.shape[0]
        width = image.shape[1]
        image_ratio = width / height
        target_image_ratio = target_width / target_height
        crop_vertically = image_ratio < target_image_ratio
        crop_width = width if crop_vertically else int(height * target_image_ratio)
        crop_height = int(width / target_image_ratio) if crop_vertically else height

        # select the middle/center bounding box
        x0 = int((width - crop_width) / 2)
        y0 = int((height - crop_height) / 2)
        x1 = x0 + crop_width
        y1 = y0 + crop_height

        # crop the image using the bounding box
        image = image[y0:y1, x0:x1]

        # resize the image
        image = resize(image, (target_width, target_height), mode='reflect', anti_aliasing=True)
        image = image.astype(np.float32)
        return image

    def transform(self, image_paths=None, my_X=None):
        X = []
        if my_X is None and image_paths is not None:
            for image_path in image_paths:
                X.append(InceptionFeatureExtractor.prepare_image(image_path))
            X = np.array(X)
        else:
            for k in range(len(my_X)):
                X.append(InceptionFeatureExtractor.prepare_image(im=my_X[k]))
            X = np.array(X)

        with tf.Session() as sess:
            self.init.run()
            X_prime = sess.run(self.prelogits, feed_dict={self.X: X})

        return X_prime


if __name__ == '__main__':
    """ This example only runs when run at the command line:
    $ python image_extract.py
    """
    t0 = time.time()

    image_paths = ['Y%07d.jpg' % i for i in range(190001, 190011)]
    image_paths.extend(image_paths)
    image_paths.extend(image_paths)
    ife = InceptionFeatureExtractor()
    X = ife.transform(image_paths)
    t1 = time.time()
    print(X)
    print(X.shape)
    print(t1-t0, 'seconds')


