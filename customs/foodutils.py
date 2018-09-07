import json
import os
import sys
import time
from collections import defaultdict
from os.path import join
import numpy as np
import numpy as np
from tqdm import tqdm
import cPickle as pickle
# Import Mask RCNN
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library


class foodutils(object):
    def __init__(self):
        self.data_dir = None  # 'data/UECFOOD256/'
        self.category_file = None  # 'category.txt'
        self.split_dir = None  # uecfood_split/uecfood256_split

    def read_bbox_file(self, bbfilelist):
        #id2bb = defaultdict(lambda: defaultdict(list))
        id2bb = defaultdict(lambda: defaultdict(list))
        print('\n Reading bounding box info...\n')
        with open(os.path.join(self.data_dir, bbfilelist), 'r') as f:
            lines = f.readlines()
            for line in lines:
                name = line.rstrip()
                classid = int(name.split('/')[1])
                with open(os.path.join(self.data_dir, name)) as bbf:
                    sentences = bbf.readlines()[1:]
                    for sent in sentences:
                        (idx, x1, y1, x2, y2) = sent.rstrip().split(' ')
                        # Food256 bb cordinates are not matrix idxs
                        ids = int(idx)
                        x1 = int(x1)
                        x2 = int(x2)
                        y1 = int(y1)
                        y2 = int(y2)
                        # assert x1, y1, y
                        id2bb[classid][ids].append({'x': [x1, x2, x2, x1],'y': [y1, y1, y2, y2]})
            print('Done.\n')
        return id2bb

    def clsid2Name(self):
        clsid2name = {}
        with open(os.path.join(self.data_dir, self.category_file), 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                clsids, name = line.rstrip().split('\t')
                clsid2name[int(clsids)] = name
        return clsid2name

    def rdict(self):
        return defaultdict(self.rdict)


    def gen_filelist(self, data_split='train', cropped=False):
        ''' crop: False otherwise crop to bbox '''
        assert data_split in ('train, val'), 'data_split should be train|val'
        output = ''
        split_file_dir = join(self.data_dir, self.split_dir)
        split_files = sorted(os.listdir(split_file_dir))
        split_files = [
            join(split_file_dir, split_file) for split_file in split_files
        ]

        if data_split == 'train':
            split_files = split_files[:-1]
        else:
            split_files = split_files[-1:]

        for file in split_files:
            with open(file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                root, category, image = line.rstrip().split('/')
                if cropped:
                    root += '_cropped'
                new_line = root + '/' + category + '/' + \
                    image + ' ' + str(int(category) - 1) + '\n'
                output += new_line
        if cropped:
            filename = join(self.data_dir, data_split + '_cropped.txt')
        else:
            filename = join(self.data_dir, data_split + '.txt')
        with open(filename, 'w') as f1:
            f1.write(output)

    def getDataSplit(self, data, val_ratio=0.3):
        train_num = int(len(data) * (1.0 - val_ratio))
        val_num = -1 * int(len(data) * val_ratio)
        return train_num, val_num



def getSplit(filename, data_dir, val_ratio = 0.3):
    filepath = os.path.join(data_dir, filename)
    pkldata = pklLoad(filepath)
    size = int(len(pkldata['images']))
    train_num = int(size * (1.0 - val_ratio))
    val_num = -1 * int(size * val_ratio)
    ids = np.arange(size)
    np.random.shuffle(ids)
    train_ids = ids[:train_num]
    val_ids = ids[val_num:]
    return train_ids, val_ids


def pklSave(obj, filename):
    pickle.dump(obj, open(filename, 'wb'))

def pklLoad(filename):
    return pickle.load(open(filename, 'rb'))
