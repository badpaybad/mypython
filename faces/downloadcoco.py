import os
import re
import zipfile

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt

import matplotlib.patches as patches

import tensorflow_datasets as tfds

from tensorflow.keras import layers
import base64
import cv2


#Downloading the COCO2017 dataset
url = "https://github.com/srihari-humbarwadi/datasets/releases/download/v0.1.0/data.zip"
filename = os.path.join(os.getcwd(), "data.zip")
keras.utils.get_file(filename, url)

coco_lables="person\nbicycle\ncar\nmotorcycle\nairplane\nbus\ntrain\ntruck\nboat\ntraffic light\nfire hydrant\nstop sign\nparking meter\nbench\nbird\ncat\ndog\nhorse\nsheep\ncow\nelephant\nbear\nzebra\ngiraffe\nbackpack\numbrella\nhandbag\ntie\nsuitcase\nfrisbee\nskis\nsnowboard\nsports ball\nkite\nbaseball bat\nbaseball glove\nskateboard\nsurfboard\ntennis racket\nbottle\nwine glass\ncup\nfork\nknife\nspoon\nbowl\nbanana\napple\nsandwich\norange\nbroccoli\ncarrot\nhot dog\npizza\ndonut\ncake\nchair\ncouch\npotted plant\nbed\ndining table\ntoilet\ntv\nlaptop\nmouse\nremote\nkeyboard\ncell phone\nmicrowave\noven\ntoaster\nsink\nrefrigerator\nbook\nclock\nvase\nscissors\nteddy bear\nhair drier\ntoothbrush".split("\n")
with zipfile.ZipFile("data.zip", "r") as z_fp:
    z_fp.extractall("./")

(train_dataset, val_dataset), dataset_info = tfds.load(
    "coco/2017", split=["train", "validation"], with_info=True, data_dir="data"
)

def swap_xy(boxes):
    """Swaps order the of x and y coordinates of the boxes.

    Arguments:
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes.

    Returns:
      swapped boxes with shape same as that of boxes.
    """
    return tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)

for sample in train_dataset.take(3):

    image =np.array( sample["image"], dtype=np.uint8)
    boxs = np.array(sample["objects"]["bbox"], dtype=np.float32)
    imgName =sample["image/filename"].numpy().decode("utf-8") 
    class_id =np.array(sample["objects"]["label"],dtype=np.uint8)

    plt.clf()
    plt.imshow(image)
    ax = plt.gca()
    linewidth=1
    color=[0, 0, 1]
    for idx, box in enumerate( boxs):        
        y1, x1, y2, x2 = box

        x1=x1 * image.shape[1]
        y1=y1 * image.shape[0]
        x2=x2 * image.shape[1]
        y2=y2 * image.shape[0]
        
        w, h =  x2 - x1,y2 - y1

        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
        )
        ax.add_patch(patch)       
        text="clsid:{} lbl: {}".format(class_id[idx], coco_lables[class_id[idx]])
        ax.text(
            x1,
            y1,
            text,
            bbox={"facecolor": [1,1,1], "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )
        
    
    plt.waitforbuttonpress()    

exit()