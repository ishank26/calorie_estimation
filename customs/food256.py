import os
import sys
import time
from collections import defaultdict
from os.path import join

import numpy as np
import skimage.draw
from tqdm import tqdm

import foodutils
from foodutils import foodutils as utilClass

from mrcnn import utils
from mrcnn.config import Config

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library


class Food256config(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "food256"

    BACKBONE = "resnet50" 

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4  # batch size
    GPU_LIST = [6]

    # Number of classes (including background)
    NUM_CLASSES = 1 + 256  # background + 256 food labels

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 120

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 50

    DETECTION_MAX_INSTANCES = 100
    USE_MINI_MASK = True

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 256
    IMAGE_MIN_SCALE = 0.0

    NUM_EPOCHS = 20





'''
 # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads":
            r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+":
            r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+":
            r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+":
            r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all":
            ".*",
        }
'''


class Food256dataset(utils.Dataset, utilClass):
    def __init__(self):
        super(Food256dataset, self).__init__()
        self.data_dir = 'data/UECFOOD256'
        self.category_file = 'category.txt'
        self.filelist = 'filelist.txt'
        self.bbfilelist = 'bbfilelist.txt'
        self.id2bb = None
        self.clsid2name = self.clsid2Name()
        self.class_info = [{"source": "food256", "id": 0, "name": "BG"}]
        self.train_ids = []
        self.val_ids = []

    def prepareCocoFormat(self, filename=None, dataset='food256'):
        '''Return data in COCO format'''
        assert filename != None, "Enter a pkl filename to save!"
        global data
        data = defaultdict(list)
        self.id2bb = self.read_bbox_file(self.bbfilelist)

        print("Preparing images metadata...\n")
        count = 0
        with open(os.path.join(self.data_dir, self.filelist), 'r') as file:
            lines = file.readlines()
            for line in tqdm(lines):
                meta = line.rstrip().split('/')
                imgLabel = int(meta[1])  # image_label
                imgName = meta[2]
                imgID = count # image id
                image_path = os.path.join(self.data_dir, str(imgLabel),
                                          imgName)
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]

                bbinfo = self.id2bb[imgLabel][np.int32(meta[2].split('.')[0])]

                # check if bb coords exceeds image dimensions
                if dataset =='food256':
                    for i in range(len(bbinfo)):
                        arr = bbinfo[i]
                        for j in range(len(arr['y'])):
                            y = arr['y'][j] 
                            if y >= height:
                                y-=2
                                bbinfo[i]['y'][j] = y
                            x = arr['x'][j]
                            if x >= width:
                                x-=2
                                bbinfo[i]['x'][j] = x


                data['images'].append({
                    'height': height,
                    'width': width,
                    'path': image_path,
                    'image_id': imgID
                })
                data['categories'].append({
                    'category_id':
                    imgLabel,
                    'category_name':
                    self.clsid2name[imgLabel],
                    'supercategory':
                    'food256',
                    'image_id':
                    imgID
                })
                data['annotations'].append({'bb': bbinfo, 'image_id': imgID})
                count += 1

        foodutils.pklSave(data, os.path.join(self.data_dir, filename))
        return data


    def load_food256(self, filename, idarr):
        '''
        @subset: train or test
        # TODO:
        # accomodate reading test split
        '''
        assert(len(idarr) > 0), "Give train or validation split ids !"
        self.id2bb = self.read_bbox_file(self.bbfilelist)
        if filename:
            print("Reading image files...\n")
            filepath = os.path.join(self.data_dir, filename)
            pkldata = foodutils.pklLoad(filepath)
            for id in tqdm(idarr):
                assert id == pkldata['annotations'][id]['image_id'], "Image IDs doesn't match!"
                bbdata = pkldata['annotations'][id]['bb']
                self.add_image(source="food256",
                image_id=id,
                path=pkldata['images'][id]['path'],
                height=pkldata['images'][id]['height'],
                width=pkldata['images'][id]['width'],
                category_id=pkldata['categories'][id]['category_id'],
                bb=bbdata)

            for k, v in self.clsid2name.items():
                self.add_class("food256", k, v)
            print "\nDone!\n"

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_id = int(image_id)
        image_info = self.image_info[image_id]
        if image_info["source"] != "food256":
            return super(self.__class__, self).load_mask(image_id)
        # Convert bb to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        class_ids = []
        mask = np.zeros(
            [info["height"], info["width"],
             len(info["bb"])],
            dtype=np.uint8)
        for i, p in enumerate(info["bb"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['y'], p['x'])
            mask[rr, cc, i] = 1
            class_ids.append(info['category_id'])
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask.astype(np.bool), class_ids


    def prepare(self, class_map=None):
        """Prepares the Dataset class for use."""

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        self.source_class_ids = {}
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)
        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {
            "{}.{}".format(info['source'], info['id']): id
            for info, id in zip(self.class_info, self.class_ids)
        }
        self.image_from_source_map = {
            "{}.{}".format(info['source'], info['id']): id
            for info, id in zip(self.image_info, self.image_ids)
        }

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "food256":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


if __name__ == '__main__':
    filename = 'data_coco.pkl'  # pkl dump in dataset root
    train_data = Food256dataset()
    train_data.prepareCocoFormat(filename)
    #train_data.getSplit(filename)
    #train_data.load_food256(filename)
    #train_data.load_mask(134)
    #train_data.prepare()
