#ref: https://github.com/serengil/deepface.git
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

import matplotlib.pyplot as plt

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
            [type]: [List of tuple]
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
        img_pixels /= 255  # normalize input in [0, 1]
        return img_pixels
        pass


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

(face_croped, region_face) = detector.detect_face(kimlien)[0]

vectorDlib = encoderDlib.predict(detector.normalize_face(face_croped, 150, 150))[0].tolist()

vectorFacenet = faceNetEncoder.predict(detector.normalize_face(face_croped, 160, 160))[0].tolist()

for idx,img in enumerate( listImgTest):
    foundFaces = detector.detect_face(img)
    for i,f in enumerate(foundFaces):
        vdlib=  encoderDlib.predict(detector.normalize_face(f[0], 150, 150))[0].tolist()        
        distanceDlib = round(np.float64(comparer.findCosineDistance(vectorDlib, vdlib)), 5)

        vfnet=  faceNetEncoder.predict(detector.normalize_face(f[0], 160, 160))[0].tolist()
        distanceFnet = round(np.float64(comparer.findCosineDistance(vectorFacenet, vfnet)), 5)
        print("{} {} {}".format(listImgTestLbl[idx], idx,i))
        print("{} {} distanceDlib {}".format(idx,i,distanceDlib))
        print("{} {} distanceFnet {}".format(idx,i,distanceFnet))
        cv2.imshow("",f[0])
        cv2.waitKey(0)


cv2.destroyAllWindows()

exit(0)

(f, r) = detector.detect_face(kimlien)[0]
(f1, r1) = detector.detect_face(kimlien1)[0]
(f2, r2) = detector.detect_face(kimlien2)[0]
(f3, r3) = detector.detect_face(kimlien3)[0]
(fdu, rdu) = detector.detect_face(du)[0]

# print(f)
# print(r)
# faceCrop = kimlien[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]
# cv2.imshow("", f)
# cv2.waitKey(0)

encoder = DlibResNet()

vector = encoder.predict(detector.normalize_face(f, 150, 150))[0].tolist()
vector1 = encoder.predict(detector.normalize_face(f1, 150, 150))[0].tolist()
vector2 = encoder.predict(detector.normalize_face(f2, 150, 150))[0].tolist()
vector3 = encoder.predict(detector.normalize_face(f3, 150, 150))[0].tolist()
vectordu1 = encoder.predict(detector.normalize_face(fdu, 150, 150))[0].tolist()

faceNetEncoder = FaceNet().loadModel()

vector4 = faceNetEncoder.predict(
    detector.normalize_face(f, 160, 160))[0].tolist()
vector5 = faceNetEncoder.predict(
    detector.normalize_face(f1, 160, 160))[0].tolist()
vector6 = faceNetEncoder.predict(
    detector.normalize_face(f2, 160, 160))[0].tolist()
vector7 = faceNetEncoder.predict(
    detector.normalize_face(f3, 160, 160))[0].tolist()
vectordu2 = faceNetEncoder.predict(
    detector.normalize_face(fdu, 160, 160))[0].tolist()

comparer = VectorCompare()

distance1 = round(np.float64(
    comparer.findCosineDistance(vector, vector1)), 5)
distance2 = round(np.float64(
    comparer.findCosineDistance(vector, vector2)), 5)
distance3 = round(np.float64(
    comparer.findCosineDistance(vector, vector3)), 5)
distancedu1 = round(np.float64(
    comparer.findCosineDistance(vector, vectordu1)), 5)

distance4 = round(np.float64(
    comparer.findCosineDistance(vector4, vector5)), 5)
distance5 = round(np.float64(
    comparer.findCosineDistance(vector4, vector6)), 5)
distance6 = round(np.float64(
    comparer.findCosineDistance(vector4, vector7)), 5)
distancedu2 = round(np.float64(
    comparer.findCosineDistance(vector4, vectordu2)), 5)

print("Distance1")
print(distance1)
print("Distance2")
print(distance2)
print("Distance3")
print(distance3)
print("distancedu1")
print(distancedu1)

print("Distance4")
print(distance4)
print("Distance5")
print(distance5)
print("Distance6")
print(distance6)
print("distancedu2")
print(distancedu2)

cv2.imshow("f0 - Ogininal - ", f)
cv2.waitKey(0)
cv2.imshow("Dlib - Distance1 - "+str(distance1) , f1)
cv2.waitKey(0)
cv2.imshow("Dlib - Distance2 - "+str(distance2), f2)
cv2.waitKey(0)
cv2.imshow("Dlib - Distance3 - "+str(distance3), f3)
cv2.waitKey(0)
cv2.imshow("Dlib - DistanceDu1 - "+str(distancedu1), fdu)
cv2.waitKey(0)

cv2.waitKey(0)
cv2.imshow("FaceNet - Distance4 - "+str(distance4) , f1)
cv2.waitKey(0)
cv2.imshow("FaceNet - Distance5 - "+str(distance5), f2)
cv2.waitKey(0)
cv2.imshow("FaceNet - Distance6 - "+str(distance6), f3)
cv2.waitKey(0)
cv2.imshow("FaceNet - DistanceDu2 - "+str(distancedu2),  fdu)
cv2.waitKey(0)

cv2.destroyAllWindows()
