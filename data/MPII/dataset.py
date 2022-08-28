import numpy as np
from data.MPII.load_data import MPII
from data.MPII.heatmap import GenerateHeatmap
import cv2
import sys
import os
import torch
import torch.utils.data
import utils.img
import PIL



class MPII_Dataset(torch.utils.data.Dataset):
    def __init__(self, config, mode='train', transform=None):
        self.input_res = config['train']['input_res']
        self.output_res = config['train']['output_res']
        self.generateHeatmap = GenerateHeatmap(self.output_res, config['inference']['num_parts'])
        self.ds = MPII()
        self.transform = transform
        self.mode = mode

        train, valid = self.ds.setup_val_split()
        self.index = train
        if mode == 'valid':
            self.index = valid


    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # loadImage returns image, label_heatmap
        return self.loadImage(self.index[idx % len(self.index)])

    def preprocess(self, data):
        # random hue and saturation
        data = cv2.cvtColor(data, cv2.COLOR_RGB2HSV);
        delta = (np.random.random() * 2 - 1) * 0.2
        data[:, :, 0] = np.mod(data[:,:,0] + (delta * 360 + 360.), 360.)

        delta_sature = np.random.random() + 0.5
        data[:, :, 1] *= delta_sature
        data[:,:, 1] = np.maximum( np.minimum(data[:,:,1], 1), 0 )
        data = cv2.cvtColor(data, cv2.COLOR_HSV2RGB)

        # adjust brightness
        delta = (np.random.random() * 2 - 1) * 0.3
        data += delta

        # adjust contrast
        mean = data.mean(axis=2, keepdims=True)
        data = (data - mean) * (np.random.random() + 0.5) + mean
        data = np.minimum(np.maximum(data, 0), 1)
        return data

    def loadImage(self, idx):
        ## load + crop
        orig_img = self.ds.get_img(idx)
        path =  self.ds.get_path(idx)
        orig_keypoints =  self.ds.get_kps(idx)
        kptmp = orig_keypoints.copy()
        c =  self.ds.get_center(idx)
        s =  self.ds.get_scale(idx)
        normalize =  self.ds.get_normalized(idx)

        cropped = utils.img.crop(orig_img, c, s, (self.input_res, self.input_res))
        for i in range(np.shape(orig_keypoints)[1]):
            if orig_keypoints[0,i,0] > 0:
                orig_keypoints[0,i,:2] = utils.img.transform(orig_keypoints[0,i,:2], c, s, (self.input_res, self.input_res))
        keypoints = np.copy(orig_keypoints)

        ## augmentation -- to be done to cropped image
        height, width = cropped.shape[0:2]
        center = np.array((width/2, height/2))
        scale = max(height, width)/200

        aug_rot=0

        aug_rot = (np.random.random() * 2 - 1) * 30.
        aug_scale = np.random.random() * (1.25 - 0.75) + 0.75
        scale *= aug_scale

        mat_mask = utils.img.get_transform(center, scale, (self.output_res, self.output_res), aug_rot)[:2]

        mat = utils.img.get_transform(center, scale, (self.input_res, self.input_res), aug_rot)[:2]
        inp = cv2.warpAffine(cropped, mat, (self.input_res, self.input_res)).astype(np.float32)/255
        keypoints[:,:,0:2] = utils.img.kpt_affine(keypoints[:,:,0:2], mat_mask)
        if np.random.randint(2) == 0:
            inp = self.preprocess(inp)
            inp = inp[:, ::-1] # flip input image
            keypoints = keypoints[:,  self.ds.flipped_parts['mpii']]
            keypoints[:, :, 0] = self.output_res - keypoints[:, :, 0] # only the x axis
            orig_keypoints = orig_keypoints[:,  self.ds.flipped_parts['mpii']]
            orig_keypoints[:, :, 0] = self.input_res - orig_keypoints[:, :, 0]

        ## set keypoints to 0 when were not visible initially
        for i in range(np.shape(orig_keypoints)[1]):
            if kptmp[0,i,0] == 0 and kptmp[0,i,1] == 0:
                keypoints[0,i,0] = 0
                keypoints[0,i,1] = 0
                orig_keypoints[0,i,0] = 0
                orig_keypoints[0,i,1] = 0

        ## generate heatmaps on outres
        heatmaps = self.generateHeatmap(keypoints)
        heatmaps = torch.from_numpy(heatmaps)
        inp = torch.from_numpy(inp.copy()) # inp shape = [256, 256, 3]

        inp = inp.permute(2, 0, 1) # change shape (h, w, c) to (c, h, w )

        # inp shape: [3, 256, 256] (normalized input)
        # heatmaps shape: [16, 64, 64]
        return inp.type(torch.FloatTensor), heatmaps.type(torch.FloatTensor)





class MPII_ValidationDataset(torch.utils.data.Dataset):
    def __init__(self, config, mode='train', transform=None):
        self.input_res = config['train']['input_res']
        self.output_res = config['train']['output_res']
        self.generateHeatmap = GenerateHeatmap(self.output_res, config['inference']['num_parts'])
        self.ds = MPII()
        self.transform = transform
        self.mode = mode

        train, valid = self.ds.setup_val_split()
        self.index = train
        if mode == 'valid':
            self.index = valid


    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # loadImage returns image, label_heatmap
        return self.loadImage(self.index[idx % len(self.index)])


    def loadImage(self, idx):
        ## load + crop
        orig_img = self.ds.get_img(idx)
        path =  self.ds.get_path(idx)
        orig_keypoints =  self.ds.get_kps(idx)
        kptmp = orig_keypoints.copy()
        c =  self.ds.get_center(idx)
        s =  self.ds.get_scale(idx)
        normalize =  self.ds.get_normalized(idx)

        cropped = utils.img.crop(orig_img, c, s, (self.input_res, self.input_res))
        cropped_normed = cropped / 255

        inp = torch.from_numpy(cropped_normed.copy()) # returns shape  [256, 256, 3]

        inp = inp.permute(2, 0, 1) # change shape (h, w, c) to (c, h, w )

        # process keypoints to generate label heatmap
        for i in range(np.shape(orig_keypoints)[1]):
            if orig_keypoints[0,i,0] > 0:
                orig_keypoints[0,i,:2] = utils.img.transform(orig_keypoints[0,i,:2], c, s, (self.input_res, self.input_res))
        keypoints = np.copy(orig_keypoints)

        height, width = cropped.shape[0:2]
        center = np.array((width/2, height/2))
        scale = max(height, width)/200

        mat_mask = utils.img.get_transform(center, scale, (self.output_res, self.output_res))[:2]
        keypoints[:,:,0:2] = utils.img.kpt_affine(keypoints[:,:,0:2], mat_mask)

        for i in range(np.shape(orig_keypoints)[1]):
            if kptmp[0,i,0] == 0 and kptmp[0,i,1] == 0:
                keypoints[0,i,0] = 0
                keypoints[0,i,1] = 0
                orig_keypoints[0,i,0] = 0
                orig_keypoints[0,i,1] = 0

        ## generate heatmaps on outres
        heatmaps = self.generateHeatmap(keypoints)
        heatmaps = torch.from_numpy(heatmaps)

        # inp shape: [3, 256, 256] (normalized input)
        # hm shape: [16, 64, 64]
        orig_img = torch.from_numpy(orig_img)
        orig_img = orig_img.permute(2, 0, 1) # change shape (h, w, c) to (c, h, w )

        result = {
            'image': inp.type(torch.FloatTensor),
            'heatmaps': heatmaps.type(torch.FloatTensor),
            'orig_keypoints': kptmp[0],
            'center': c,
            'scale': s,
            'normalize': normalize,
            'input_res': self.input_res,
            'orig_img': orig_img
            }
        return result
