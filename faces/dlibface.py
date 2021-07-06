#ref: https://github.com/serengil/deepface.git
from ast import dump
from tensorflow.keras import backend as K
from tensorflow.keras.layers import add
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model
import os
import zipfile
import bz2

import cv2
import gdown
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt

from sklearn.svm import SVC
import pickle

import tensorflow as tf
tf_version = int(tf.__version__.split(".")[0])

if tf_version == 1:
    import keras
    from keras.preprocessing.image import load_img, save_img, img_to_array
    from keras.applications.imagenet_utils import preprocess_input
    from keras.preprocessing import image
elif tf_version == 2:
    from tensorflow import keras
    from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
    from tensorflow.keras.applications.imagenet_utils import preprocess_input
    from tensorflow.keras.preprocessing import image

class DlibMetaData:
    def __init__(self):
        self.input_shape = [[1, 150, 150, 3]]

class DlibResNet:

    def __init__(self):

        self.__FileFolder = os.path.dirname(os.path.realpath(__file__))
        self.file_weights = self.__FileFolder + \
            '/weights/dlib_face_recognition_resnet_model_v1.dat'
        # this is not a must dependency
        import dlib  # 19.20.0

        self.layers = [DlibMetaData()]

        # download pre-trained model if it does not exist
        if os.path.isfile(self.file_weights) != True:
            print("dlib_face_recognition_resnet_model_v1.dat is going to be downloaded")

            url = "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
            output = self.file_weights+url.split("/")[-1]
            gdown.download(url, output, quiet=False)

            zipfile = bz2.BZ2File(output)
            data = zipfile.read()
            #newfilepath = output[:-4]  # discard .bz2 extension
            open(self.file_weights, 'wb').write(data)

        # ---------------------

        model = dlib.face_recognition_model_v1(self.file_weights)
        self.__model = model

        # ---------------------

        return None  # classes must return None

    def predict(self, img_aligned):

        # functions.detectFace returns 4 dimensional images
        if len(img_aligned.shape) == 4:
            img_aligned = img_aligned[0]

        # functions.detectFace returns bgr images
        img_aligned = img_aligned[:, :, ::-1]  # bgr to rgb

        # deepface.detectFace returns an array in scale of [0, 1] but dlib expects in scale of [0, 255]
        if img_aligned.max() <= 1:
            img_aligned = img_aligned * 255

        img_aligned = img_aligned.astype(np.uint8)

        model = self.__model

        img_representation = model.compute_face_descriptor(img_aligned)

        img_representation = np.array(img_representation)
        img_representation = np.expand_dims(img_representation, axis=0)

        return img_representation

class FaceNet:
    def __init__(self):
        self.__FileFolder = os.path.dirname(os.path.realpath(__file__))
        self.__model = self.loadModel()
        pass

    def scaling(self, x, scale):
        return x * scale

    def InceptionResNetV2(self):

        inputs = Input(shape=(160, 160, 3))
        x = Conv2D(32, 3, strides=2, padding='valid',
                   use_bias=False, name='Conv2d_1a_3x3')(inputs)
        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                               scale=False, name='Conv2d_1a_3x3_BatchNorm')(x)
        x = Activation('relu', name='Conv2d_1a_3x3_Activation')(x)
        x = Conv2D(32, 3, strides=1, padding='valid',
                   use_bias=False, name='Conv2d_2a_3x3')(x)
        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                               scale=False, name='Conv2d_2a_3x3_BatchNorm')(x)
        x = Activation('relu', name='Conv2d_2a_3x3_Activation')(x)
        x = Conv2D(64, 3, strides=1, padding='same',
                   use_bias=False, name='Conv2d_2b_3x3')(x)
        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                               scale=False, name='Conv2d_2b_3x3_BatchNorm')(x)
        x = Activation('relu', name='Conv2d_2b_3x3_Activation')(x)
        x = MaxPooling2D(3, strides=2, name='MaxPool_3a_3x3')(x)
        x = Conv2D(80, 1, strides=1, padding='valid',
                   use_bias=False, name='Conv2d_3b_1x1')(x)
        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                               scale=False, name='Conv2d_3b_1x1_BatchNorm')(x)
        x = Activation('relu', name='Conv2d_3b_1x1_Activation')(x)
        x = Conv2D(192, 3, strides=1, padding='valid',
                   use_bias=False, name='Conv2d_4a_3x3')(x)
        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                               scale=False, name='Conv2d_4a_3x3_BatchNorm')(x)
        x = Activation('relu', name='Conv2d_4a_3x3_Activation')(x)
        x = Conv2D(256, 3, strides=2, padding='valid',
                   use_bias=False, name='Conv2d_4b_3x3')(x)
        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                               scale=False, name='Conv2d_4b_3x3_BatchNorm')(x)
        x = Activation('relu', name='Conv2d_4b_3x3_Activation')(x)

        # 5x Block35 (Inception-ResNet-A block):
        branch_0 = Conv2D(32, 1, strides=1, padding='same',
                          use_bias=False, name='Block35_1_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block35_1_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation(
            'relu', name='Block35_1_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(32, 1, strides=1, padding='same',
                          use_bias=False, name='Block35_1_Branch_1_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block35_1_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block35_1_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(32, 3, strides=1, padding='same', use_bias=False,
                          name='Block35_1_Branch_1_Conv2d_0b_3x3')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block35_1_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block35_1_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
        branch_2 = Conv2D(32, 1, strides=1, padding='same',
                          use_bias=False, name='Block35_1_Branch_2_Conv2d_0a_1x1')(x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block35_1_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
        branch_2 = Activation(
            'relu', name='Block35_1_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False,
                          name='Block35_1_Branch_2_Conv2d_0b_3x3')(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block35_1_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
        branch_2 = Activation(
            'relu', name='Block35_1_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False,
                          name='Block35_1_Branch_2_Conv2d_0c_3x3')(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block35_1_Branch_2_Conv2d_0c_3x3_BatchNorm')(branch_2)
        branch_2 = Activation(
            'relu', name='Block35_1_Branch_2_Conv2d_0c_3x3_Activation')(branch_2)
        branches = [branch_0, branch_1, branch_2]
        mixed = Concatenate(axis=3, name='Block35_1_Concatenate')(branches)
        up = Conv2D(256, 1, strides=1, padding='same',
                    use_bias=True, name='Block35_1_Conv2d_1x1')(mixed)
        up = Lambda(self.scaling, output_shape=K.int_shape(up)
                    [1:], arguments={'scale': 0.17})(up)
        x = add([x, up])
        x = Activation('relu', name='Block35_1_Activation')(x)

        branch_0 = Conv2D(32, 1, strides=1, padding='same',
                          use_bias=False, name='Block35_2_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block35_2_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation(
            'relu', name='Block35_2_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(32, 1, strides=1, padding='same',
                          use_bias=False, name='Block35_2_Branch_1_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block35_2_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block35_2_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(32, 3, strides=1, padding='same', use_bias=False,
                          name='Block35_2_Branch_1_Conv2d_0b_3x3')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block35_2_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block35_2_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
        branch_2 = Conv2D(32, 1, strides=1, padding='same',
                          use_bias=False, name='Block35_2_Branch_2_Conv2d_0a_1x1')(x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block35_2_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
        branch_2 = Activation(
            'relu', name='Block35_2_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False,
                          name='Block35_2_Branch_2_Conv2d_0b_3x3')(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block35_2_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
        branch_2 = Activation(
            'relu', name='Block35_2_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False,
                          name='Block35_2_Branch_2_Conv2d_0c_3x3')(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block35_2_Branch_2_Conv2d_0c_3x3_BatchNorm')(branch_2)
        branch_2 = Activation(
            'relu', name='Block35_2_Branch_2_Conv2d_0c_3x3_Activation')(branch_2)
        branches = [branch_0, branch_1, branch_2]
        mixed = Concatenate(axis=3, name='Block35_2_Concatenate')(branches)
        up = Conv2D(256, 1, strides=1, padding='same',
                    use_bias=True, name='Block35_2_Conv2d_1x1')(mixed)
        up = Lambda(self.scaling, output_shape=K.int_shape(up)
                    [1:], arguments={'scale': 0.17})(up)
        x = add([x, up])
        x = Activation('relu', name='Block35_2_Activation')(x)

        branch_0 = Conv2D(32, 1, strides=1, padding='same',
                          use_bias=False, name='Block35_3_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block35_3_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation(
            'relu', name='Block35_3_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(32, 1, strides=1, padding='same',
                          use_bias=False, name='Block35_3_Branch_1_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block35_3_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block35_3_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(32, 3, strides=1, padding='same', use_bias=False,
                          name='Block35_3_Branch_1_Conv2d_0b_3x3')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block35_3_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block35_3_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
        branch_2 = Conv2D(32, 1, strides=1, padding='same',
                          use_bias=False, name='Block35_3_Branch_2_Conv2d_0a_1x1')(x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block35_3_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
        branch_2 = Activation(
            'relu', name='Block35_3_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False,
                          name='Block35_3_Branch_2_Conv2d_0b_3x3')(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block35_3_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
        branch_2 = Activation(
            'relu', name='Block35_3_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False,
                          name='Block35_3_Branch_2_Conv2d_0c_3x3')(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block35_3_Branch_2_Conv2d_0c_3x3_BatchNorm')(branch_2)
        branch_2 = Activation(
            'relu', name='Block35_3_Branch_2_Conv2d_0c_3x3_Activation')(branch_2)
        branches = [branch_0, branch_1, branch_2]
        mixed = Concatenate(axis=3, name='Block35_3_Concatenate')(branches)
        up = Conv2D(256, 1, strides=1, padding='same',
                    use_bias=True, name='Block35_3_Conv2d_1x1')(mixed)
        up = Lambda(self.scaling, output_shape=K.int_shape(up)
                    [1:], arguments={'scale': 0.17})(up)
        x = add([x, up])
        x = Activation('relu', name='Block35_3_Activation')(x)

        branch_0 = Conv2D(32, 1, strides=1, padding='same',
                          use_bias=False, name='Block35_4_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block35_4_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation(
            'relu', name='Block35_4_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(32, 1, strides=1, padding='same',
                          use_bias=False, name='Block35_4_Branch_1_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block35_4_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block35_4_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(32, 3, strides=1, padding='same', use_bias=False,
                          name='Block35_4_Branch_1_Conv2d_0b_3x3')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block35_4_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block35_4_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
        branch_2 = Conv2D(32, 1, strides=1, padding='same',
                          use_bias=False, name='Block35_4_Branch_2_Conv2d_0a_1x1')(x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block35_4_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
        branch_2 = Activation(
            'relu', name='Block35_4_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False,
                          name='Block35_4_Branch_2_Conv2d_0b_3x3')(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block35_4_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
        branch_2 = Activation(
            'relu', name='Block35_4_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False,
                          name='Block35_4_Branch_2_Conv2d_0c_3x3')(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block35_4_Branch_2_Conv2d_0c_3x3_BatchNorm')(branch_2)
        branch_2 = Activation(
            'relu', name='Block35_4_Branch_2_Conv2d_0c_3x3_Activation')(branch_2)
        branches = [branch_0, branch_1, branch_2]
        mixed = Concatenate(axis=3, name='Block35_4_Concatenate')(branches)
        up = Conv2D(256, 1, strides=1, padding='same',
                    use_bias=True, name='Block35_4_Conv2d_1x1')(mixed)
        up = Lambda(self.scaling, output_shape=K.int_shape(up)
                    [1:], arguments={'scale': 0.17})(up)
        x = add([x, up])
        x = Activation('relu', name='Block35_4_Activation')(x)

        branch_0 = Conv2D(32, 1, strides=1, padding='same',
                          use_bias=False, name='Block35_5_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block35_5_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation(
            'relu', name='Block35_5_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(32, 1, strides=1, padding='same',
                          use_bias=False, name='Block35_5_Branch_1_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block35_5_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block35_5_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(32, 3, strides=1, padding='same', use_bias=False,
                          name='Block35_5_Branch_1_Conv2d_0b_3x3')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block35_5_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block35_5_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
        branch_2 = Conv2D(32, 1, strides=1, padding='same',
                          use_bias=False, name='Block35_5_Branch_2_Conv2d_0a_1x1')(x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block35_5_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
        branch_2 = Activation(
            'relu', name='Block35_5_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False,
                          name='Block35_5_Branch_2_Conv2d_0b_3x3')(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block35_5_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
        branch_2 = Activation(
            'relu', name='Block35_5_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False,
                          name='Block35_5_Branch_2_Conv2d_0c_3x3')(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block35_5_Branch_2_Conv2d_0c_3x3_BatchNorm')(branch_2)
        branch_2 = Activation(
            'relu', name='Block35_5_Branch_2_Conv2d_0c_3x3_Activation')(branch_2)
        branches = [branch_0, branch_1, branch_2]
        mixed = Concatenate(axis=3, name='Block35_5_Concatenate')(branches)
        up = Conv2D(256, 1, strides=1, padding='same',
                    use_bias=True, name='Block35_5_Conv2d_1x1')(mixed)
        up = Lambda(self.scaling, output_shape=K.int_shape(up)
                    [1:], arguments={'scale': 0.17})(up)
        x = add([x, up])
        x = Activation('relu', name='Block35_5_Activation')(x)

        # Mixed 6a (Reduction-A block):
        branch_0 = Conv2D(384, 3, strides=2, padding='valid',
                          use_bias=False, name='Mixed_6a_Branch_0_Conv2d_1a_3x3')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Mixed_6a_Branch_0_Conv2d_1a_3x3_BatchNorm')(branch_0)
        branch_0 = Activation(
            'relu', name='Mixed_6a_Branch_0_Conv2d_1a_3x3_Activation')(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding='same',
                          use_bias=False, name='Mixed_6a_Branch_1_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Mixed_6a_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Mixed_6a_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(192, 3, strides=1, padding='same', use_bias=False,
                          name='Mixed_6a_Branch_1_Conv2d_0b_3x3')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Mixed_6a_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Mixed_6a_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
        branch_1 = Conv2D(256, 3, strides=2, padding='valid', use_bias=False,
                          name='Mixed_6a_Branch_1_Conv2d_1a_3x3')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Mixed_6a_Branch_1_Conv2d_1a_3x3_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Mixed_6a_Branch_1_Conv2d_1a_3x3_Activation')(branch_1)
        branch_pool = MaxPooling2D(
            3, strides=2, padding='valid', name='Mixed_6a_Branch_2_MaxPool_1a_3x3')(x)
        branches = [branch_0, branch_1, branch_pool]
        x = Concatenate(axis=3, name='Mixed_6a')(branches)

        # 10x Block17 (Inception-ResNet-B block):
        branch_0 = Conv2D(128, 1, strides=1, padding='same',
                          use_bias=False, name='Block17_1_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_1_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation(
            'relu', name='Block17_1_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same',
                          use_bias=False, name='Block17_1_Branch_1_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_1_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block17_1_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same',
                          use_bias=False, name='Block17_1_Branch_1_Conv2d_0b_1x7')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_1_Branch_1_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block17_1_Branch_1_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same',
                          use_bias=False, name='Block17_1_Branch_1_Conv2d_0c_7x1')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_1_Branch_1_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block17_1_Branch_1_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_1_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same',
                    use_bias=True, name='Block17_1_Conv2d_1x1')(mixed)
        up = Lambda(self.scaling, output_shape=K.int_shape(up)
                    [1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_1_Activation')(x)

        branch_0 = Conv2D(128, 1, strides=1, padding='same',
                          use_bias=False, name='Block17_2_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_2_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation(
            'relu', name='Block17_2_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same',
                          use_bias=False, name='Block17_2_Branch_2_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_2_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block17_2_Branch_2_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same',
                          use_bias=False, name='Block17_2_Branch_2_Conv2d_0b_1x7')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_2_Branch_2_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block17_2_Branch_2_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same',
                          use_bias=False, name='Block17_2_Branch_2_Conv2d_0c_7x1')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_2_Branch_2_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block17_2_Branch_2_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_2_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same',
                    use_bias=True, name='Block17_2_Conv2d_1x1')(mixed)
        up = Lambda(self.scaling, output_shape=K.int_shape(up)
                    [1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_2_Activation')(x)

        branch_0 = Conv2D(128, 1, strides=1, padding='same',
                          use_bias=False, name='Block17_3_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_3_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation(
            'relu', name='Block17_3_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same',
                          use_bias=False, name='Block17_3_Branch_3_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_3_Branch_3_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block17_3_Branch_3_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same',
                          use_bias=False, name='Block17_3_Branch_3_Conv2d_0b_1x7')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_3_Branch_3_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block17_3_Branch_3_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same',
                          use_bias=False, name='Block17_3_Branch_3_Conv2d_0c_7x1')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_3_Branch_3_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block17_3_Branch_3_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_3_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same',
                    use_bias=True, name='Block17_3_Conv2d_1x1')(mixed)
        up = Lambda(self.scaling, output_shape=K.int_shape(up)
                    [1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_3_Activation')(x)

        branch_0 = Conv2D(128, 1, strides=1, padding='same',
                          use_bias=False, name='Block17_4_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_4_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation(
            'relu', name='Block17_4_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same',
                          use_bias=False, name='Block17_4_Branch_4_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_4_Branch_4_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block17_4_Branch_4_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same',
                          use_bias=False, name='Block17_4_Branch_4_Conv2d_0b_1x7')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_4_Branch_4_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block17_4_Branch_4_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same',
                          use_bias=False, name='Block17_4_Branch_4_Conv2d_0c_7x1')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_4_Branch_4_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block17_4_Branch_4_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_4_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same',
                    use_bias=True, name='Block17_4_Conv2d_1x1')(mixed)
        up = Lambda(self.scaling, output_shape=K.int_shape(up)
                    [1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_4_Activation')(x)

        branch_0 = Conv2D(128, 1, strides=1, padding='same',
                          use_bias=False, name='Block17_5_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_5_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation(
            'relu', name='Block17_5_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same',
                          use_bias=False, name='Block17_5_Branch_5_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_5_Branch_5_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block17_5_Branch_5_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same',
                          use_bias=False, name='Block17_5_Branch_5_Conv2d_0b_1x7')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_5_Branch_5_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block17_5_Branch_5_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same',
                          use_bias=False, name='Block17_5_Branch_5_Conv2d_0c_7x1')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_5_Branch_5_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block17_5_Branch_5_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_5_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same',
                    use_bias=True, name='Block17_5_Conv2d_1x1')(mixed)
        up = Lambda(self.scaling, output_shape=K.int_shape(up)
                    [1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_5_Activation')(x)

        branch_0 = Conv2D(128, 1, strides=1, padding='same',
                          use_bias=False, name='Block17_6_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_6_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation(
            'relu', name='Block17_6_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same',
                          use_bias=False, name='Block17_6_Branch_6_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_6_Branch_6_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block17_6_Branch_6_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same',
                          use_bias=False, name='Block17_6_Branch_6_Conv2d_0b_1x7')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_6_Branch_6_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block17_6_Branch_6_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same',
                          use_bias=False, name='Block17_6_Branch_6_Conv2d_0c_7x1')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_6_Branch_6_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block17_6_Branch_6_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_6_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same',
                    use_bias=True, name='Block17_6_Conv2d_1x1')(mixed)
        up = Lambda(self.scaling, output_shape=K.int_shape(up)
                    [1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_6_Activation')(x)

        branch_0 = Conv2D(128, 1, strides=1, padding='same',
                          use_bias=False, name='Block17_7_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_7_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation(
            'relu', name='Block17_7_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same',
                          use_bias=False, name='Block17_7_Branch_7_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_7_Branch_7_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block17_7_Branch_7_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same',
                          use_bias=False, name='Block17_7_Branch_7_Conv2d_0b_1x7')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_7_Branch_7_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block17_7_Branch_7_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same',
                          use_bias=False, name='Block17_7_Branch_7_Conv2d_0c_7x1')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_7_Branch_7_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block17_7_Branch_7_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_7_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same',
                    use_bias=True, name='Block17_7_Conv2d_1x1')(mixed)
        up = Lambda(self.scaling, output_shape=K.int_shape(up)
                    [1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_7_Activation')(x)

        branch_0 = Conv2D(128, 1, strides=1, padding='same',
                          use_bias=False, name='Block17_8_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_8_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation(
            'relu', name='Block17_8_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same',
                          use_bias=False, name='Block17_8_Branch_8_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_8_Branch_8_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block17_8_Branch_8_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same',
                          use_bias=False, name='Block17_8_Branch_8_Conv2d_0b_1x7')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_8_Branch_8_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block17_8_Branch_8_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same',
                          use_bias=False, name='Block17_8_Branch_8_Conv2d_0c_7x1')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_8_Branch_8_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block17_8_Branch_8_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_8_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same',
                    use_bias=True, name='Block17_8_Conv2d_1x1')(mixed)
        up = Lambda(self.scaling, output_shape=K.int_shape(up)
                    [1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_8_Activation')(x)

        branch_0 = Conv2D(128, 1, strides=1, padding='same',
                          use_bias=False, name='Block17_9_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_9_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation(
            'relu', name='Block17_9_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same',
                          use_bias=False, name='Block17_9_Branch_9_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_9_Branch_9_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block17_9_Branch_9_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same',
                          use_bias=False, name='Block17_9_Branch_9_Conv2d_0b_1x7')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_9_Branch_9_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block17_9_Branch_9_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same',
                          use_bias=False, name='Block17_9_Branch_9_Conv2d_0c_7x1')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_9_Branch_9_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block17_9_Branch_9_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_9_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same',
                    use_bias=True, name='Block17_9_Conv2d_1x1')(mixed)
        up = Lambda(self.scaling, output_shape=K.int_shape(up)
                    [1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_9_Activation')(x)

        branch_0 = Conv2D(128, 1, strides=1, padding='same',
                          use_bias=False, name='Block17_10_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block17_10_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation(
            'relu', name='Block17_10_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same',
                          use_bias=False, name='Block17_10_Branch_10_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_10_Branch_10_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block17_10_Branch_10_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False,
                          name='Block17_10_Branch_10_Conv2d_0b_1x7')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_10_Branch_10_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block17_10_Branch_10_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False,
                          name='Block17_10_Branch_10_Conv2d_0c_7x1')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_10_Branch_10_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block17_10_Branch_10_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_10_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same',
                    use_bias=True, name='Block17_10_Conv2d_1x1')(mixed)
        up = Lambda(self.scaling, output_shape=K.int_shape(up)
                    [1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_10_Activation')(x)

        # Mixed 7a (Reduction-B block): 8 x 8 x 2080
        branch_0 = Conv2D(256, 1, strides=1, padding='same',
                          use_bias=False, name='Mixed_7a_Branch_0_Conv2d_0a_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Mixed_7a_Branch_0_Conv2d_0a_1x1_BatchNorm')(branch_0)
        branch_0 = Activation(
            'relu', name='Mixed_7a_Branch_0_Conv2d_0a_1x1_Activation')(branch_0)
        branch_0 = Conv2D(384, 3, strides=2, padding='valid', use_bias=False,
                          name='Mixed_7a_Branch_0_Conv2d_1a_3x3')(branch_0)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Mixed_7a_Branch_0_Conv2d_1a_3x3_BatchNorm')(branch_0)
        branch_0 = Activation(
            'relu', name='Mixed_7a_Branch_0_Conv2d_1a_3x3_Activation')(branch_0)
        branch_1 = Conv2D(256, 1, strides=1, padding='same',
                          use_bias=False, name='Mixed_7a_Branch_1_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Mixed_7a_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Mixed_7a_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(256, 3, strides=2, padding='valid', use_bias=False,
                          name='Mixed_7a_Branch_1_Conv2d_1a_3x3')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Mixed_7a_Branch_1_Conv2d_1a_3x3_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Mixed_7a_Branch_1_Conv2d_1a_3x3_Activation')(branch_1)
        branch_2 = Conv2D(256, 1, strides=1, padding='same',
                          use_bias=False, name='Mixed_7a_Branch_2_Conv2d_0a_1x1')(x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Mixed_7a_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
        branch_2 = Activation(
            'relu', name='Mixed_7a_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
        branch_2 = Conv2D(256, 3, strides=1, padding='same', use_bias=False,
                          name='Mixed_7a_Branch_2_Conv2d_0b_3x3')(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Mixed_7a_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
        branch_2 = Activation(
            'relu', name='Mixed_7a_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
        branch_2 = Conv2D(256, 3, strides=2, padding='valid', use_bias=False,
                          name='Mixed_7a_Branch_2_Conv2d_1a_3x3')(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Mixed_7a_Branch_2_Conv2d_1a_3x3_BatchNorm')(branch_2)
        branch_2 = Activation(
            'relu', name='Mixed_7a_Branch_2_Conv2d_1a_3x3_Activation')(branch_2)
        branch_pool = MaxPooling2D(
            3, strides=2, padding='valid', name='Mixed_7a_Branch_3_MaxPool_1a_3x3')(x)
        branches = [branch_0, branch_1, branch_2, branch_pool]
        x = Concatenate(axis=3, name='Mixed_7a')(branches)

        # 5x Block8 (Inception-ResNet-C block):

        branch_0 = Conv2D(192, 1, strides=1, padding='same',
                          use_bias=False, name='Block8_1_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block8_1_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation(
            'relu', name='Block8_1_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding='same',
                          use_bias=False, name='Block8_1_Branch_1_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block8_1_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block8_1_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding='same',
                          use_bias=False, name='Block8_1_Branch_1_Conv2d_0b_1x3')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block8_1_Branch_1_Conv2d_0b_1x3_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block8_1_Branch_1_Conv2d_0b_1x3_Activation')(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding='same',
                          use_bias=False, name='Block8_1_Branch_1_Conv2d_0c_3x1')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block8_1_Branch_1_Conv2d_0c_3x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block8_1_Branch_1_Conv2d_0c_3x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block8_1_Concatenate')(branches)
        up = Conv2D(1792, 1, strides=1, padding='same',
                    use_bias=True, name='Block8_1_Conv2d_1x1')(mixed)
        up = Lambda(self.scaling, output_shape=K.int_shape(up)
                    [1:], arguments={'scale': 0.2})(up)
        x = add([x, up])
        x = Activation('relu', name='Block8_1_Activation')(x)

        branch_0 = Conv2D(192, 1, strides=1, padding='same',
                          use_bias=False, name='Block8_2_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block8_2_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation(
            'relu', name='Block8_2_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding='same',
                          use_bias=False, name='Block8_2_Branch_2_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block8_2_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block8_2_Branch_2_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding='same',
                          use_bias=False, name='Block8_2_Branch_2_Conv2d_0b_1x3')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block8_2_Branch_2_Conv2d_0b_1x3_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block8_2_Branch_2_Conv2d_0b_1x3_Activation')(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding='same',
                          use_bias=False, name='Block8_2_Branch_2_Conv2d_0c_3x1')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block8_2_Branch_2_Conv2d_0c_3x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block8_2_Branch_2_Conv2d_0c_3x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block8_2_Concatenate')(branches)
        up = Conv2D(1792, 1, strides=1, padding='same',
                    use_bias=True, name='Block8_2_Conv2d_1x1')(mixed)
        up = Lambda(self.scaling, output_shape=K.int_shape(up)
                    [1:], arguments={'scale': 0.2})(up)
        x = add([x, up])
        x = Activation('relu', name='Block8_2_Activation')(x)

        branch_0 = Conv2D(192, 1, strides=1, padding='same',
                          use_bias=False, name='Block8_3_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block8_3_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation(
            'relu', name='Block8_3_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding='same',
                          use_bias=False, name='Block8_3_Branch_3_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block8_3_Branch_3_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block8_3_Branch_3_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding='same',
                          use_bias=False, name='Block8_3_Branch_3_Conv2d_0b_1x3')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block8_3_Branch_3_Conv2d_0b_1x3_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block8_3_Branch_3_Conv2d_0b_1x3_Activation')(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding='same',
                          use_bias=False, name='Block8_3_Branch_3_Conv2d_0c_3x1')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block8_3_Branch_3_Conv2d_0c_3x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block8_3_Branch_3_Conv2d_0c_3x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block8_3_Concatenate')(branches)
        up = Conv2D(1792, 1, strides=1, padding='same',
                    use_bias=True, name='Block8_3_Conv2d_1x1')(mixed)
        up = Lambda(self.scaling, output_shape=K.int_shape(up)
                    [1:], arguments={'scale': 0.2})(up)
        x = add([x, up])
        x = Activation('relu', name='Block8_3_Activation')(x)

        branch_0 = Conv2D(192, 1, strides=1, padding='same',
                          use_bias=False, name='Block8_4_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block8_4_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation(
            'relu', name='Block8_4_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding='same',
                          use_bias=False, name='Block8_4_Branch_4_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block8_4_Branch_4_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block8_4_Branch_4_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding='same',
                          use_bias=False, name='Block8_4_Branch_4_Conv2d_0b_1x3')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block8_4_Branch_4_Conv2d_0b_1x3_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block8_4_Branch_4_Conv2d_0b_1x3_Activation')(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding='same',
                          use_bias=False, name='Block8_4_Branch_4_Conv2d_0c_3x1')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block8_4_Branch_4_Conv2d_0c_3x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block8_4_Branch_4_Conv2d_0c_3x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block8_4_Concatenate')(branches)
        up = Conv2D(1792, 1, strides=1, padding='same',
                    use_bias=True, name='Block8_4_Conv2d_1x1')(mixed)
        up = Lambda(self.scaling, output_shape=K.int_shape(up)
                    [1:], arguments={'scale': 0.2})(up)
        x = add([x, up])
        x = Activation('relu', name='Block8_4_Activation')(x)

        branch_0 = Conv2D(192, 1, strides=1, padding='same',
                          use_bias=False, name='Block8_5_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block8_5_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation(
            'relu', name='Block8_5_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding='same',
                          use_bias=False, name='Block8_5_Branch_5_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block8_5_Branch_5_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block8_5_Branch_5_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding='same',
                          use_bias=False, name='Block8_5_Branch_5_Conv2d_0b_1x3')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block8_5_Branch_5_Conv2d_0b_1x3_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block8_5_Branch_5_Conv2d_0b_1x3_Activation')(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding='same',
                          use_bias=False, name='Block8_5_Branch_5_Conv2d_0c_3x1')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block8_5_Branch_5_Conv2d_0c_3x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block8_5_Branch_5_Conv2d_0c_3x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block8_5_Concatenate')(branches)
        up = Conv2D(1792, 1, strides=1, padding='same',
                    use_bias=True, name='Block8_5_Conv2d_1x1')(mixed)
        up = Lambda(self.scaling, output_shape=K.int_shape(up)
                    [1:], arguments={'scale': 0.2})(up)
        x = add([x, up])
        x = Activation('relu', name='Block8_5_Activation')(x)

        branch_0 = Conv2D(192, 1, strides=1, padding='same',
                          use_bias=False, name='Block8_6_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block8_6_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation(
            'relu', name='Block8_6_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding='same',
                          use_bias=False, name='Block8_6_Branch_1_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block8_6_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block8_6_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding='same',
                          use_bias=False, name='Block8_6_Branch_1_Conv2d_0b_1x3')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block8_6_Branch_1_Conv2d_0b_1x3_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block8_6_Branch_1_Conv2d_0b_1x3_Activation')(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding='same',
                          use_bias=False, name='Block8_6_Branch_1_Conv2d_0c_3x1')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name='Block8_6_Branch_1_Conv2d_0c_3x1_BatchNorm')(branch_1)
        branch_1 = Activation(
            'relu', name='Block8_6_Branch_1_Conv2d_0c_3x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block8_6_Concatenate')(branches)
        up = Conv2D(1792, 1, strides=1, padding='same',
                    use_bias=True, name='Block8_6_Conv2d_1x1')(mixed)
        up = Lambda(self.scaling, output_shape=K.int_shape(up)
                    [1:], arguments={'scale': 1})(up)
        x = add([x, up])

        # Classification block
        x = GlobalAveragePooling2D(name='AvgPool')(x)
        x = Dropout(1.0 - 0.8, name='Dropout')(x)
        # Bottleneck
        x = Dense(128, use_bias=False, name='Bottleneck')(x)
        x = BatchNormalization(momentum=0.995, epsilon=0.001,
                               scale=False, name='Bottleneck_BatchNorm')(x)

        # Create model
        model = Model(inputs, x, name='inception_resnet_v1')

        return model

    def loadModel(self, url='https://drive.google.com/uc?id=1971Xk5RwedbudGgTIrGAL4F7Aifu7id1'):
        
        model = self.InceptionResNetV2()

        self.file_weight = self.__FileFolder+'/weights/facenet_weights.h5'
        #self.file_weight = self.__FileFolder+'/weights/facenet_keras_weights.h5'

        if os.path.isfile(self.file_weight) != True:
            print("facenet_weights.h5 will be downloaded..." + url)

            gdown.download(url, self.file_weight, quiet=False)

        model.load_weights(self.file_weight)

        return model

    def predict(self, img_aligned):
        return self.__model.predict(img_aligned)
        pass

class DlibDetector:
    def __init__(self):

        self.__FileFolder = os.path.dirname(os.path.realpath(__file__))

        import dlib  # this requirement is not a must that's why imported here
        self.file_weights = self.__FileFolder + \
            '/weights/shape_predictor_5_face_landmarks.dat'
        # print(os.path.isfile(self.file_weights))
        # exit(0)
        # check required file exists in the home/.deepface/weights folder
        if os.path.isfile(self.file_weights) != True:

            print("shape_predictor_5_face_landmarks.dat.bz2 is going to be downloaded")

            url = "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2"
            output = self.file_weights+url.split("/")[-1]

            gdown.download(url, output, quiet=False)

            zipfile = bz2.BZ2File(output)
            data = zipfile.read()
            # newfilepath = output[:-4]  # discard .bz2 extension
            open(self.file_weights, 'wb').write(data)

        face_detector = dlib.get_frontal_face_detector()

        sp = dlib.shape_predictor(self.file_weights)

        detector = {}
        detector["face_detector"] = face_detector
        detector["sp"] = sp
        self.detector = detector
        pass

    def detect_face(self, imgInput, align=True):
        """[summary]

        Args:
            imgInput ([type]): [cv2.imread]
            align (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [List of tuple(face,rect)]
        """
        import dlib  # this requirement is not a must that's why imported here
        detector = self.detector
        sp = detector["sp"]
        #print(sp)
        #exit(0)
        #img_region = [0, 0, imgInput.shape[0], imgInput.shape[1]]

        face_detector = detector["face_detector"]
        detections = face_detector(imgInput, 1)
        listDetected=[]
        
        if len(detections) > 0:
            for idx, d in enumerate(detections):                
                left = d.left()
                right = d.right()
                top = d.top()
                bottom = d.bottom()
                detected_face = imgInput[top:bottom, left:right]
                detected_face_region = [left, top, right - left, bottom - top]  
                #print(detected_face_region)          
                if align:
                    img_shape = sp(imgInput, d)
                    #print("img_shape")
                    #print(img_shape)

                    detected_face = dlib.get_face_chip(
                        imgInput, img_shape, size=detected_face.shape[0])

                listDetected.append((detected_face, detected_face_region))

        return listDetected

    def normalize_face(self,img, w, h):
        img = cv2.resize(img, (w, h))
        img_pixels = image.img_to_array(img)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        #tf.squeeze(img_pixels)
        img_pixels /= 255  # normalize input in [0, 1]
        #print(img_pixels.shape)
        #print(img_pixels.ndim)
        #print(img_pixels)
        return img_pixels
        pass

    def normalize(self,image, fixed=False):
        if fixed:
            return (np.float32(image) - 127.5) / 127.5
        else:
            mean = np.mean(image)
            std = np.std(image)
            std_adj = np.maximum(std, 1.0 / np.sqrt(image.size))
            y = np.multiply(np.subtract(image, mean), 1 / std_adj)
            return y

class VectorCompare:
    def __init__(self):
        pass

    def findCosineDistance(self, source_representation, test_representation):
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    def findEuclideanDistance(self, source_representation, test_representation):
        if type(source_representation) == list:
            source_representation = np.array(source_representation)

        if type(test_representation) == list:
            test_representation = np.array(test_representation)

        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(
            euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance

    def l2_normalize(self, x):
        return x / np.sqrt(np.sum(np.multiply(x, x)))

class SvmFaceClassifier:
    def __init__(self, vectors=[], labels=[]):
        self.model = SVC(kernel='linear', probability=True)

    def Train(self,vectors=[], labels=[]):
        
        self.faceVectors = vectors
        self.faceLabels = labels        
        self.model.fit(self.faceVectors, self.faceLabels)

        pass

    def SaveModel(self,modelPath=""):

        self.__FileFolder = os.path.dirname(os.path.realpath(__file__))
        if(modelPath==""):
            modelPath= self.__FileFolder+"/svm.pkl"

        pickle.dump(self.model,open(modelPath, 'wb'))
    
    def LoadModel(self,modelPath="" ):
        self.__FileFolder = os.path.dirname(os.path.realpath(__file__))
        if(modelPath==""):
            modelPath= self.__FileFolder+"/svm.pkl"

        self.model = pickle.load(open(modelPath, 'rb'))
        #result = loaded_model.score(X_test, Y_test)
    
    def Predict(self, vector):
        return self.model.predict(vector)
        pass

class UnitTest:

    def Run(self):
        currentDir = os.path.dirname(os.path.realpath(__file__))


        # multiface= cv2.imread(currentDir+"/imgtest/multiface.png")
        # lstFace=detector.detect_face(multiface)
        # for f in lstFace:
        #     cv2.imshow("face found", f[0])
        #     cv2.waitKey(0)
        #     pass
        # exit(0)

        kimlien = cv2.imread(currentDir+"/imgtest/kimlien.jpg")
        kimlien1 = cv2.imread(currentDir+"/imgtest/kimlien1.jpg")
        kimlien2 = cv2.imread(currentDir+"/imgtest/kimlien2.png")
        kimlien3 = cv2.imread(currentDir+"/imgtest/kimlien3.jpg")
        du = cv2.imread(currentDir+"/imgtest/du.png")
        multiface = cv2.imread(currentDir+"/imgtest/multiface.png")

        detector = DlibDetector()
        encoderDlib = DlibResNet()
        faceNetEncoder = FaceNet()
        comparer = VectorCompare()

        listImgTest=[kimlien1,kimlien2,kimlien3,du,multiface]
        listImgTestLbl=["kimlien1","kimlien2","kimlien3","du","multiface"]

        du1 = "C:/Users/Admin/Desktop/bak/du1.aligned.png"
        du2 = "C:/Users/Admin/Desktop/bak/du2.aligned.png"

        kimlien = cv2.imread(du1)
        listImgTest=[cv2.imread(du2)]

        (face_croped, region_face) = detector.detect_face(kimlien)[0]

        vectorDlib = encoderDlib.predict(detector.normalize_face(face_croped, 150, 150))[0].tolist()

        vectorFacenet = faceNetEncoder.predict(detector.normalize_face(face_croped, 160, 160))[0].tolist()

        for idx,img in enumerate( listImgTest):
            foundFaces = detector.detect_face(img)
            for i,f in enumerate(foundFaces):
                vdlib=  encoderDlib.predict(detector.normalize_face(f[0], 150, 150))[0].tolist()        
                distanceDlib = round(np.float64(comparer.findCosineDistance(vectorDlib, vdlib)), 10)
                     
                distanceDlibEcl = round(np.float64(comparer.findEuclideanDistance(vectorDlib, vdlib)), 10)

                vfnet=  faceNetEncoder.predict(detector.normalize_face(f[0], 160, 160))[0].tolist()
                distanceFnet = round(np.float64(comparer.findCosineDistance(vectorFacenet, vfnet)), 10)
                distanceFnetEcl = round(np.float64(comparer.findEuclideanDistance(vectorFacenet, vfnet)), 10)

                print("{} {} {}".format(listImgTestLbl[idx], idx,i))
                print("{} {} distanceDlib {}".format(idx,i,distanceDlib))                
                print("{} {} distanceDlibEcl {}".format(idx,i,distanceDlibEcl))

                print("{} {} distanceFnet {}".format(idx,i,distanceFnet))                
                print("{} {} distanceFnetEcl {}".format(idx,i,distanceFnetEcl))

                dx0=f[1][0]
                dy0=f[1][1]
                dx1=f[1][0]+f[1][2]
                dy1=f[1][1]+f[1][3]
                cv2.rectangle(img,(dx0,dy0),(dx1,dy1),(255,255,0,255),2)

                resizeToSeeDetail=img
                if img.shape[0]<600:
                    resizeToSeeDetail = cv2.resize(img,(600,int( 600*img.shape[0]/img.shape[1])))

                imgw=f[0].shape[0]
                imgh=f[0].shape[1]
                resizeToSeeDetail[0:imgw,0:imgh,:] = f[0][0:imgw,0:imgh,:]

                cv2.putText(resizeToSeeDetail, "{}".format(listImgTestLbl[idx])
                ,(0,30), cv2.FONT_HERSHEY_SIMPLEX, 1,  (255, 0, 0, 255),  2) 
                
                cv2.putText(resizeToSeeDetail, "dlib: {} fnet:{}".format(distanceDlib,distanceFnet)
                ,(0,70), cv2.FONT_HERSHEY_SIMPLEX, 1,  (255, 0, 0, 255),  2) 
                
                cv2.putText(resizeToSeeDetail, "size: {}".format(f[0].shape)
                ,(0,110), cv2.FONT_HERSHEY_SIMPLEX, 1,  (255, 0, 0, 255),  2) 
                
                cv2.putText(resizeToSeeDetail, "region: {}".format(f[1])
                ,(0,140), cv2.FONT_HERSHEY_SIMPLEX, 1,  (255, 0, 0, 255),  2) 

                cv2.imshow("",resizeToSeeDetail)
                cv2.waitKey(0)


        cv2.destroyAllWindows()       

#UnitTest().Run()

# dlibDetector=DlibDetector()
# imgXxx= dlibDetector.normalize_face(cv2.imread("./imgtest/du.png"),150,150)
# imgXxx = tf.squeeze(imgXxx)
# plt.imshow(imgXxx)
# plt.waitforbuttonpress()

# imgXxx=  dlibDetector.normalize(cv2.imread("./imgtest/du.png"))
# plt.imshow(imgXxx)
# plt.waitforbuttonpress()

# cv2.imshow("normalize",dlibDetector.normalize(cv2.imread("./imgtest/du.png")))
# cv2.waitKey()
# exit()

class CameraCapturer:

    def Run(self, actionCallback=None):
        #UnitTest().Run()
        currentDir = os.path.dirname(os.path.realpath(__file__))
        du = cv2.imread(currentDir+"/imgtest/du.png")
        lien = cv2.imread(currentDir+"/imgtest/kimlien3.jpg")

        listFaceImg=[du,lien]
        arrVector=[]
        arrLabel=["du","lien"]

        detector = DlibDetector()
        encoderDlib = DlibResNet()
        faceNetEncoder = FaceNet()
        comparer = VectorCompare()

        # init data
        for f in listFaceImg:
            
            # dupython=faceNetEncoder.predict(detector.normalize_face(f, 160, 160))[0].tolist() 
            # ducsharp=[-0.08461350202560425,0.10870248079299927,0.06432205438613892,-0.0835186094045639,-0.07422533631324768,0.00490811001509428,-0.10617043823003769,-0.11593830585479736,0.16171839833259583,-0.08627026528120041,0.19098907709121704,0.027474718168377876,-0.1588139832019806,-0.11673740297555923,-0.03184450417757034,0.17447572946548462,-0.21998348832130432,-0.11274232715368271,-0.04906761646270752,-0.03833162412047386,0.015536666847765446,-0.013968057930469513,0.06154670938849449,-0.015136182308197021,-0.004737555980682373,-0.3857226073741913,-0.1531461626291275,-0.033451147377491,0.1293250322341919,-0.014566851779818535,-0.06157102435827255,0.04404592141509056,-0.13947558403015137,-0.036244578659534454,0.04797746613621712,0.12615805864334106,-0.04094607010483742,-0.09578743577003479,0.20858000218868256,0.009477553889155388,-0.19284509122371674,0.009535165503621101,0.03893325477838516,0.2014131247997284,0.20033597946166992,0.05419013649225235,0.09267432987689972,-0.10418299585580826,0.1710759699344635,-0.13257479667663574,0.07788347452878952,0.20047542452812195,0.11017601937055588,0.040150273591279984,0.06579692661762238,-0.14281445741653442,0.014650404453277588,0.11880778521299362,-0.0845942348241806,-0.0032528629526495934,0.06777388602495193,-0.07942613959312439,-0.03352980688214302,-0.05450119078159332,0.1645974963903427,0.13432051241397858,-0.0822133868932724,-0.28002259135246277,0.1726679801940918,-0.1224498599767685,-0.09288617968559265,0.03702991083264351,-0.1787620484828949,-0.12023984640836716,-0.3203735053539276,0.016981882974505424,0.3917646110057831,0.13254716992378235,-0.13457857072353363,0.05706636235117912,-0.05355007201433182,-0.04941961169242859,0.08157823979854584,0.19810020923614502,-0.11364033818244934,0.06015434116125107,-0.11575307697057724,-0.03986159712076187,0.19171485304832458,-0.04510653018951416,-0.06438742578029633,0.1563309133052826,0.012417611666023731,0.15993359684944153,-0.013175349682569504,0.00687784468755126,-0.0715840607881546,0.03327440842986107,-0.16201171278953552,-0.0642997995018959,0.056524571031332016,0.0012944573536515236,0.0010995147749781609,0.14070287346839905,-0.12210346758365631,0.10675622522830963,-0.022018637508153915,0.053566139191389084,-0.01362593099474907,0.0036583601031452417,-0.08977406471967697,-0.061004411429166794,0.13083568215370178,-0.18438611924648285,0.15581904351711273,0.22185635566711426,0.056422159075737,0.10172639042139053,0.21145308017730713,0.10877689719200134,0.04128382354974747,-0.03956277295947075,-0.17385803163051605,-0.03877662122249603,0.022608324885368347,0.02234923653304577,0.018250638619065285,0.041948456317186356]
            # ducsharp=[-0.10398014634847641,0.10794883966445923,0.057899948209524155,-0.08929821103811264,-0.07987412065267563,0.005356641951948404,-0.10203596204519272,-0.1123555600643158,0.17756831645965576,-0.06860089302062988,0.1777464896440506,0.03762524574995041,-0.14587555825710297,-0.10983993858098984,-0.04997478052973747,0.15778887271881104,-0.24016454815864563,-0.10648773610591888,-0.037661362439394,-0.04075287654995918,0.012031548656523228,-0.007087382487952709,0.065708227455616,-0.019251834601163864,-0.011060167104005814,-0.3992408514022827,-0.14881759881973267,-0.05585815757513046,0.12496912479400635,-0.0065727876499295235,-0.044018879532814026,0.04138009622693062,-0.13536880910396576,-0.03814224153757095,0.05569831281900406,0.12151418626308441,-0.043977513909339905,-0.08311054110527039,0.2262917459011078,0.014616166241466999,-0.17973816394805908,0.012652965262532234,0.0415906123816967,0.21818502247333527,0.19471222162246704,0.060581497848033905,0.09066444635391235,-0.09763170033693314,0.18194372951984406,-0.13420668244361877,0.08961959183216095,0.20242926478385925,0.1085427775979042,0.03259436413645744,0.07740429043769836,-0.14349225163459778,0.014706777408719063,0.1084599569439888,-0.09015931934118271,0.009052915498614311,0.07297670841217041,-0.08070959150791168,-0.04748072102665901,-0.04667025804519653,0.15473026037216187,0.13682478666305542,-0.07261842489242554,-0.27443331480026245,0.17670060694217682,-0.12373635172843933,-0.09139660745859146,0.02678590640425682,-0.17544400691986084,-0.12786467373371124,-0.3131335973739624,0.017419148236513138,0.3820036053657532,0.1474863588809967,-0.13180862367153168,0.052374646067619324,-0.04782287776470184,-0.03899341821670532,0.08858342468738556,0.1909077763557434,-0.12646952271461487,0.06799647957086563,-0.10953369736671448,-0.03736007958650589,0.18051879107952118,-0.03375735878944397,-0.07824654132127762,0.15552803874015808,0.019757352769374847,0.15766434371471405,-0.008526738733053207,-0.010268501937389374,-0.06539805978536606,0.028726182878017426,-0.1784919649362564,-0.05876629054546356,0.07027806341648102,0.012774527072906494,0.006041618995368481,0.13448187708854675,-0.1346362829208374,0.1005270928144455,-0.021102702245116234,0.03701802343130112,-0.028610320761799812,-0.006408474408090115,-0.09627418965101242,-0.043287940323352814,0.14199160039424896,-0.18153926730155945,0.17177186906337738,0.22252316772937775,0.06252174079418182,0.11996880918741226,0.21088352799415588,0.07453229278326035,0.04539269208908081,-0.0383022204041481,-0.1682717651128769,-0.0440840981900692,0.018686611205339432,0.027798915281891823,0.025131896138191223,0.04252734035253525]
            # distanceDlib = round(np.float64(comparer.findCosineDistance(dupython, ducsharp)), 10)
            # print(distanceDlib)
       
            # distanceDlib = round(np.float64(comparer.findEuclideanDistance(dupython, ducsharp)), 10)
            # print(distanceDlib)

            ffound=detector.detect_face(f)
            print(len(ffound))
            if(len(ffound)>0):
                fcrop,rrect = ffound[0]
                xxxVector =encoderDlib.predict(detector.normalize_face(fcrop, 150, 150)) #[[]]
                # for x in xxxVector:

                #     f = open("demofile21.txt", "a")
                #     f.write(json.dumps(x.tolist()))
                #     f.write("\r\n\r\n")
                #     f.write(json.dumps(xxxVector.tolist()))
                #     f.close()
                # exit()
                vector=xxxVector[0].tolist()#[[]]
                arrVector.append(vector)                
        
        svmFaceClassifier = SvmFaceClassifier()
        svmFaceClassifier.Train(arrVector, arrLabel)
        svmFaceClassifier.SaveModel()

        
        # define a video capture object
      
        cameraUrl = "rtsp://admin:LSCOJS@192.168.3.80:554/H.265"
        cameraUrl = "rtsp://admin:Omt123123@192.168.3.110:554"
        cameraUrl="rtsp://admin:Omt123123@192.168.3.109:554/Streamming/Channels/101"
        cameraUrl=0

        vid = cv2.VideoCapture(cameraUrl)
        
        while(True):
            
            # Capture the video frame
            # by frame
            ret, frame = vid.read()
            
            foundFace = detector.detect_face(frame)
            #foundFace=[]
            print(len(foundFace))
            cv2.putText(frame, "Press 'q' to quit"
                        ,(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,  (255, 255, 0, 255),  2) 

            for ffound in foundFace:
                (face_croped, region_face)=ffound
                dx0=region_face[0]
                dy0=region_face[1]
                dx1=region_face[0]+region_face[2]
                dy1=region_face[1]+region_face[3]
                cv2.rectangle(frame,(dx0,dy0),(dx1,dy1),(255,255,0,255),2)

                vector=encoderDlib.predict(detector.normalize_face(face_croped, 150, 150))[0].tolist()
                              
                resCompare=[]
                for idx, fDec in enumerate( arrVector):
                    distanceDlib = round(np.float64(comparer.findCosineDistance(fDec, vector)), 10)
                    resCompare.append(distanceDlib)

                
                svmResult= svmFaceClassifier.Predict([vector])
                    
                if(len(resCompare)>0):
                    resCompare=np.array(resCompare)
                    minDistanceIdx = np.argmin( resCompare)
                    minDistanceVal = resCompare[minDistanceIdx]
                    cv2.putText(frame, "{} {} svm:{}".format(arrLabel[minDistanceIdx],minDistanceVal, svmResult)
                        ,(dx0 - dx0,dy0), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (255, 0, 0, 255), 2) 

            #if(actionCallback!= None):
            #    actionCallback(frame)
            #else
            # Display the resulting frame// fake video stream =))

            #frame= cv2.Canny(frame,100,100)

            cv2.imshow('frame', frame)            
            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # After the loop release the cap object
        vid.release()
        # Destroy all the windows
        cv2.destroyAllWindows()

CameraCapturer().Run()