import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display
import tensorflow as tf

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

from mrcnn.model import log


# Import Food256
from food256 import Food256config
from food256 import Food256dataset

import foodutils as fu

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


# Config
config = Food256config()
config.display()

# Load Data
filename = 'data_food256.pkl'  # pkl dump in dataset root
train_ids, val_ids = fu.getSplit(filename, data_dir='data/UECFOOD256')


train_data = Food256dataset()
#train_data.prepareCocoFormat(filename)
train_data.load_food256(filename, train_ids)
train_data.prepare()

val_data = Food256dataset()
val_data.load_food256(filename, val_ids)
val_data.prepare()


# Create model in training mode
with tf.device('/cpu:0'):
    model = modellib.MaskRCNN(mode="training", config=config,
                            model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last


if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)


# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.


if len (config.GPU_LIST) > 0:
    with tf.device('/gpu:{}'.format(config.GPU_LIST[0])):
        model.train(train_data, val_data,
                    learning_rate=config.LEARNING_RATE,
                    epochs=config.NUM_EPOCHS,
                    layers='heads')

# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
weight_file="food256_t1"
model_path = os.path.join(MODEL_DIR, weight_file+".h5")
model.keras_model.save_weights(model_path)

class InferenceConfig(Food256config):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
model_path = os.path.join(MODEL_DIR, weight_file+".h5")
#model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)
