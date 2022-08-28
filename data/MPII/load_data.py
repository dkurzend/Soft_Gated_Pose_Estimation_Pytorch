import numpy as np
import h5py
import os
from imageio import imread
from PIL import Image


annot_dir = 'data/MPII/annot'
img_dir = 'data/MPII/images'

class MPII():
    def __init__(self):
        #  dictionary train_f keys: ['center', 'imgname', 'index', 'name', 'name_', 'normalize', 'part', 'person', 'scale', 'torsoangle', 'visible']
        train_f = h5py.File(os.path.join(annot_dir, 'train.h5'), 'r')
        val_f = h5py.File(os.path.join(annot_dir, 'valid.h5'), 'r')

        self.t_center = train_f['center'][()] # shape (22246, 2)
        t_scale = train_f['scale'][()] # shape (22246,)
        t_part = train_f['part'][()]  # shape (22246, 16, 2)
        t_visible = train_f['visible'][()]  # shape (22246, 16)
        t_normalize = train_f['normalize'][()] # shape (22246,)
        t_imgname = [None] * len(self.t_center)
        for i in range(len(self.t_center)):
            t_imgname[i] = train_f['imgname'][i].decode('UTF-8')

        self.v_center = val_f['center'][()]
        v_scale = val_f['scale'][()]
        v_part = val_f['part'][()]
        v_visible = val_f['visible'][()]
        v_normalize = val_f['normalize'][()]
        v_imgname = [None] * len(self.v_center)
        for i in range(len(self.v_center)):
            v_imgname[i] = val_f['imgname'][i].decode('UTF-8')

        self.center = np.append(self.t_center, self.v_center, axis=0)
        self.scale = np.append(t_scale, v_scale)
        self.part = np.append(t_part, v_part, axis=0)
        self.visible = np.append(t_visible, v_visible, axis=0)
        self.normalize = np.append(t_normalize, v_normalize)
        self.imgname = t_imgname + v_imgname

        self.num_train = len(self.t_center)
        self.num_valid = len(self.v_center)

        # Part reference
        self.parts = {'mpii':['rank', 'rkne', 'rhip',
                        'lhip', 'lkne', 'lank',
                        'pelv', 'thrx', 'neck', 'head',
                        'rwri', 'relb', 'rsho',
                        'lsho', 'lelb', 'lwri']}

        # eg rank is flipped with lank, or rwri is flipped with lwri
        self.flipped_parts = {'mpii':[5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10]}

        self.part_pairs = {'mpii':[[0, 5], [1, 4], [2, 3], [6], [7], [8], [9], [10, 15], [11, 14], [12, 13]]}

        self.pair_names = {'mpii':['ankle', 'knee', 'hip', 'pelvis', 'thorax', 'neck', 'head', 'wrist', 'elbow', 'shoulder']}

    def getAnnots(self, idx):
        '''
        returns h5 file for train or val set
        '''
        return self.imgname[idx], self.part[idx], self.visible[idx], self.center[idx], self.scale[idx], self.normalize[idx]


    # returns length trainingset, length validation set
    def getLength(self):
        return self.num_train, self.num_valid



    def setup_val_split(self):
        '''
        returns index for train and validation imgs
        index for validation images starts after that of train images
        so that loadImage can tell them apart
        '''
        valid = [i+self.num_train for i in range(self.num_valid)] # = [22246, 22247, ...]
        train = [i for i in range(self.num_train)] # = [0, 1, 2, ... ,22245]
        return np.array(train), np.array(valid)

    # return image at index idx as numpy array
    def get_img(self, idx):
        imgname, __, __, __, __, __ = self.getAnnots(idx)
        path = os.path.join(img_dir, imgname)

        # imageio imread: Reads an image from the specified file.
        # Returns a numpy array, which comes with a dict of meta data at its ‘meta’ attribute.
        img = imread(path)

        return img

    # return image at index idx as numpy array
    def get_PIL_img(self, idx):
        imgname, __, __, __, __, __ = self.getAnnots(idx)
        path = os.path.join(img_dir, imgname)

        # PIL image
        img = Image.open(path)
        return img

    # return path of image at index idx
    def get_path(self, idx):
        imgname, __, __, __, __, __ = self.getAnnots(idx)
        path = os.path.join(img_dir, imgname)
        return path

    # returns numpy array of shape (1, 16, 3),
    def get_kps(self, idx):
        __, part, visible, __, __, __ = self.getAnnots(idx)
        kp2 = np.insert(part, 2, visible, axis=1)
        kps = np.zeros((1, 16, 3))
        kps[0] = kp2
        return kps # kp coordinates come in (x, y) = (width, height) format

    def get_normalized(self, idx):
        __, __, __, __, __, n = self.getAnnots(idx)
        return n

    def get_center(self, idx):
        __, __, __, c, __, __ = self.getAnnots(idx)
        return c

    # person scale w.r.t. 200 px height
    def get_scale(self, idx):
        __, __, __, __, s, __ = self.getAnnots(idx)
        return s
