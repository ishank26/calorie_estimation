import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import skimage
from tqdm import tqdm
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


# Directory to save logs and trained model
MODEL_DIR = 'logs'
DATA_DIR ='data/UECFOOD256'
DEFAULT_LOGS_DIR = os.path.join(MODEL_DIR,'food25620180910T1118')

parser = argparse.ArgumentParser(
        description='Food segmentation')
parser.add_argument("mode",
                    metavar="<mode>",
                    help="'train' or 'evaluate'")
parser.add_argument("weights",
                    metavar="</path/to/weights>",
                    help="coco or imagenet or last or /path/to/weights")
parser.add_argument("--stage",
                    metavar="<stage>",
                    help="Training stage: heads, 4+, all", required = False)
parser.add_argument("--logs", metavar="</path/to/logs>", default='logs', required = False, help="/path/to/logs")
args = parser.parse_args()

print("Mode: ", args.mode)
print("Weight file: ", args.weights)
print("Training Stage: ", args.stage)
print("Log dir: ", args.logs)


'''#parser.add_argument('--weights', required=True,
#                    metavar="/path/to/weights.h5",
#                    help="Path to weights .h5 file or 'coco'")
parser.add_argument('--logs', required=False,
                    default=DEFAULT_LOGS_DIR,
                    metavar="/path/to/logs/",
                    help='Logs and checkpoints directory (default=logs/)')
parser.add_argument('--subset', required=False,
                    metavar="Dataset sub-directory",
                    help="Subset of dataset to run prediction on")'''
args = parser.parse_args()


def evaluate(model, class_names, filelist, data_dir = None):
    count = 0
    result_dir = 'results'
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    with open(os.path.join(data_dir, filelist), 'r') as file:
        lines = file.readlines()
        for line in tqdm(lines):
            meta = line.rstrip().split('/')
            imgLabel = int(meta[1])  # image_label
            imgName = meta[2]
            imgID = count # image id
            image_path = os.path.join(data_dir, str(imgLabel),
                                      imgName)
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            r = model.detect([image], verbose = 1)[0]

            visualize.display_instances(
                image, r['rois'], r['masks'], r['class_ids'],
                class_names, r['scores'],
                title="Predictions")
            plt.savefig("{}/{}".format(result_dir, imgName))

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Local path to trained weights file for coco
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


# Configurations
if args.mode == "train":
    config = Food256config()
else:
    class InferenceConfig(Food256config):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0
    config = InferenceConfig()
config.display()


# Load Data
filename = 'data_food256.pkl'  # pkl dump and load data
train_ids, val_ids = fu.getSplit(filename, DATA_DIR)
train_data = Food256dataset()
# uncomment below to generate pickle file in coco format
#train_data.prepareCocoFormat(filename)
train_data.load_food256(filename, train_ids)
train_data.prepare()

val_data = Food256dataset()
val_data.load_food256(filename, val_ids)
val_data.prepare()


# Create model in training mode
if args.mode == "train":
    with tf.device('/cpu:0'):
        model = modellib.MaskRCNN(mode="training", config=config,
                                model_dir=args.logs)
else:
    with tf.device('/cpu:0'):
        model = modellib.MaskRCNN(mode="inference", config=config,
                                model_dir=args.logs)


# Select weights file to load
if args.weights.lower() == "coco":
    # Start from COCO trained weigths
    weight_path = COCO_MODEL_PATH
elif args.weigths.lower() == "last":
    # Find last trained weights
    weight_path = model.find_last()
elif args.weights.lower() == "imagenet":
    # Start from ImageNet trained weights
    weight_path = model.get_imagenet_weights()
else:
    weight_path = args.weights

model.load_weights(weight_path, by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                            "mrcnn_bbox", "mrcnn_mask"])


# Training schedule
if args.mode == "train" and len(config.GPU_LIST) > 0 :
    assert args.stage
    with tf.device('/gpu:{}'.format(config.GPU_LIST[0])):
            # stage 1
            if args.stage == "heads":
                print("Training network heads")
                model.train(train_data, val_data,
                            learning_rate=config.LEARNING_RATE,
                            epochs=config.NUM_EPOCHS,
                            layers='heads')
            # stage = 2
            if args.stage == "4+":
                print("Fine tune Resnet stage 4 and up")
                model.train(train_data, val_data,
                            learning_rate=config.LEARNING_RATE,
                            epochs=config.NUM_EPOCHS,
                            layers='4+')
            # satge 3
            if args.stage == "all":
                print("Fine tune all layers")
                model.train(train_data, val_data,
                            learning_rate=config.LEARNING_RATE,
                            epochs=config.NUM_EPOCHS,
                            layers='all')
elif args.mode == "evaluate":
    class_names = train_data.class_names
    filelist = 'testfile.txt'
    evaluate(model, class_names, filelist, DATA_DIR)

#######################3
# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
weight_file="food256_1109"
model_path = os.path.join(MODEL_DIR, weight_file+".h5")
model.keras_model.save_weights(model_path)
