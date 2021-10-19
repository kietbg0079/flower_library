import math
import random
import os
import numpy as np
import shutil

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models,layers,optimizers

from tensorflow.keras.layers import Dropout, Flatten, Dense, BatchNormalization

#from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping,  ModelCheckpoint, LearningRateScheduler
from keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet import ResNet101, preprocess_input



img_width = 224
img_height = 224

def create_model():
    adam = Adam(learning_rate=3e-4)

    resnet101_base = ResNet101(include_top=True, weights='imagenet',
                               input_shape=(img_width, img_height, 3))

    output = resnet101_base.get_layer(index=-1).output
    output = Flatten()(output)

    output = Dense(512, activation="relu")(output)
    output = BatchNormalization()(output)
    output = Dropout(0.2)(output)
    output = Dense(512, activation="relu")(output)
    output = BatchNormalization()(output)
    output = Dropout(0.2)(output)
    output = Dense(102, activation='softmax')(output)

    resnet101_model = Model(resnet101_base.input, output)
    for layer in resnet101_model.layers[:-7]:
        layer.trainable = False
    resnet101_model.summary()

    resnet101_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    resnet101_model.load_weights('weights.h5')

    return resnet101_model

def pre_image(link):
    img = image.load_img(link, target_size=(224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, 0)
    arr = preprocess_input(arr)

    return arr


def dictionary(id):
    dic = {"21": "fire lily",
         "3": "canterbury bells",
         "45": "bolero deep blue",
         "1": "pink primrose",
         "34": "mexican aster",
         "27": "prince of wales feathers", "7": "moon orchid", "16": "globe-flower", "25": "grape hyacinth", "26": "corn poppy", "79": "toad lily", "39": "siam tulip", "24": "red ginger", "67": "spring crocus", "35": "alpine sea holly", "32": "garden phlox", "10": "globe thistle", "6": "tiger lily", "93": "ball moss", "33": "love in the mist", "9": "monkshood", "102": "blackberry lily", "14": "spear thistle", "19": "balloon flower", "100": "blanket flower", "13": "king protea", "49": "oxeye daisy", "15": "yellow iris", "61": "cautleya spicata", "31": "carnation", "64": "silverbush", "68": "bearded iris", "63": "black-eyed susan", "69": "windflower", "62": "japanese anemone", "20": "giant white arum lily", "38": "great masterwort", "4": "sweet pea", "86": "tree mallow", "101": "trumpet creeper", "42": "daffodil", "22": "pincushion flower", "2": "hard-leaved pocket orchid", "54": "sunflower", "66": "osteospermum", "70": "tree poppy", "85": "desert-rose", "99": "bromelia", "87": "magnolia", "5": "english marigold", "92": "bee balm", "28": "stemless gentian", "97": "mallow", "57": "gaura", "40": "lenten rose", "47": "marigold", "59": "orange dahlia", "48": "buttercup", "55": "pelargonium", "36": "ruby-lipped cattleya", "91": "hippeastrum", "29": "artichoke", "71": "gazania", "90": "canna lily", "18": "peruvian lily", "98": "mexican petunia", "8": "bird of paradise", "30": "sweet william", "17": "purple coneflower", "52": "wild pansy", "84": "columbine", "12": "colt's foot", "11": "snapdragon", "96": "camellia", "23": "fritillary", "50": "common dandelion", "44": "poinsettia", "53": "primula", "72": "azalea", "65": "californian poppy", "80": "anthurium", "76": "morning glory", "37": "cape flower", "56": "bishop of llandaff", "60": "pink-yellow dahlia", "82": "clematis", "58": "geranium", "75": "thorn apple", "41": "barbeton daisy", "95": "bougainvillea", "43": "sword lily", "83": "hibiscus", "78": "lotus", "88": "cyclamen", "94": "foxglove", "81": "frangipani", "74": "rose", "89": "watercress", "73": "water lily", "46": "wallflower", "77": "passion flower", "51": "petunia"}

    return dic[str(id)]



