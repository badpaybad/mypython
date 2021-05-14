from numpy.lib import type_check
import tensorflow as tf
import os
import io
import cv2
import uuid
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

# https://keras.io/api/applications/
# https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/
# https://keras.io/examples/vision/retinanet/
(train_images, train_labels), (test_images,
                               test_labels) = tf.keras.datasets.fashion_mnist.load_data()


train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
train_images = train_images/255
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
test_images = test_images/255

# for img in test_images:
#     print(type(img))
#     print(img.shape)
#     plt.figure()
#     plt.imshow(img)
#     plt.colorbar()
#     plt.grid(False)
#     plt.show()
#     exit(0)

# pass

folderModel = os.path.join(os.getcwd(), "trained_model")

if os.path.exists(folderModel) == False:

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=128, kernel_size=(
            3, 3), activation=tf.nn.relu, input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(filters=64, kernel_size=(
            3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(input_shape=(28, 28)),

        tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
        tf.keras.layers.Dropout(rate=0.1),
        tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy)

    model.fit(train_images, train_labels, batch_size=600, epochs=100, use_multiprocessing=True,
              workers=12, validation_data=(test_images, test_labels))

    print("Evaluate")
    #model.evaluate(test_images, test_labels)
    """
    for img in test_images:
        r = model.predict(img.shape[0])
        print(r)
        pass
    """
    model.save(folderModel)
else:
    model = keras.models.load_model(folderModel)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


imgTest = plt.imread("2.1.jpg")

imgTest = rgb2gray(imgTest)

print(test_images[0].shape)
print(imgTest.shape)

imgTest = imgTest.reshape(test_images[0].shape)

# #img from mnist by idx
# idxImgToTest=4
# imgTest=test_images[idxImgToTest]
#print("test_label idx: {}".format(idxImgToTest))
#print("test_label: {}".format(test_labels[idxImgToTest]))
#print("index must be:2: {}".format(np.argmax([1,2,9,3,4,5])))

plt.figure()
plt.imshow(imgTest)
plt.colorbar()
plt.grid(False)
plt.show()

imgTestNdims = np.expand_dims(imgTest, 0)

print(imgTestNdims.shape)

r = model.predict(imgTestNdims)

print("\r\nAll: ")
print(r[0])

lblPredicted = np.argmax(r[0])

idxLblPredicted = -1
idxCounter = 0
for x in test_labels:
    if x == lblPredicted:
        idxLblPredicted = idxCounter
        break
    idxCounter = idxCounter+1

print("lable prediected idx: {}".format(idxLblPredicted))
print("lable predicted: {}".format(lblPredicted))

plt.figure()
plt.imshow(test_images[idxLblPredicted])
plt.colorbar()
plt.grid(False)
plt.show()
