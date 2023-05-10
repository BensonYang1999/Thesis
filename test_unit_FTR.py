import os
import random
import sys
from glob import glob
import logging

import cv2
import numpy as np
import torchvision.transforms.functional as F
from skimage.color import rgb2gray
from skimage.feature import canny
from torch.utils.data import Dataset
import pickle
import skimage.draw
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader

import torch.nn as nn
from src.models.TSR_model import EdgeLineGPT256RelBCE, EdgeLineGPTConfig
import torchvision.transforms as T

import torchvision.transforms as transforms
from src.Fuseformer.utils import create_random_shape_with_random_motion
from src.Fuseformer.utils import Stack, ToTorchFormatTensor, GroupRandomHorizontalFlip
from PIL import Image

import gc
gc.collect()
torch.cuda.empty_cache()

"""
FuseFormer imports
"""
import numpy as np
import time
import math
from functools import reduce
import torchvision.models as models
from torch.nn import functional as nnF

from src.models.fuseformer import InpaintGenerator

to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor(), ])

sys.path.append('..')

logger = logging.getLogger(__name__)

def to_int(x):
    return tuple(map(int, x))

class ImgDataset_video(torch.utils.data.Dataset):

    # write a video version of __init__ function
    def __init__(self, opts, sample=5, size=(432, 240), spit='train', name='YoutubeVOS', root='./datasets'):
        self.split = split
        self.sample_length = sample
        self.input_size = self.w, self.h = size
        self.opts = opts
        if name == 'YouTubeVOS':
            vid_lst_prefix = os.path.join(root, name, split+'_all_frames/JPEGImages')
        vid_lst = os.listdir(vid_lst_prefix)
        self.video_names = [os.path.join(vid_lst_prefix, name) for name in vid_lst]

        # don't read all the frames at once, save the video names and read them later
        # TODO: reading specific masking input

    def __len__(self): # return the number of videos
        return len(self.video_names)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('Loading error in video {}'.format(self.video_names[index]))
            item = self.load_item(0)
        return item

    def load_name(self, index): # load the base name of the video with given index
        return os.path.basename(self.video_names[index])

    def load_item(self, index):
        video_name = self.video_names[index]
        all_frames = [os.path.join(video_name, name) for name in sorted(os.listdir(video_name))]
        all_masks = create_random_shape_with_random_motion(
            len(all_frames), imageHeight=self.h, imageWidth=self.w)
    
        ref_index = self.get_ref_index(len(all_frames), self.sample_length)
        frames = []
        masks = []

        for idx in ref_index:
            img = Image.open(all_frames[idx]).convert('RGB')
            img = img.resize(self.input_size)

            frames.append(img)
            masks.append(all_masks[idx])

        if self.split=='train':
            prob = random.random()
            frames = GroupRandomHorizontalFlip()(frames, prob)

        batch = dict()
        batch['frames'] = to_tensors(frames)*2.0 - 1.0 # [0, 1] => [-1, 1] normalize
        batch['masks'] = to_tensors(masks)
        batch['name'] = self.load_name(index)
        batch['ref_index'] = ref_index
        return batch


    def get_ref_index(self, length, sample_length):
        if random.uniform(0, 1) > 0.5:
            ref_index = random.sample(range(length), sample_length)
            ref_index.sort()
        else:
            pivot = random.randint(0, length-sample_length)
            ref_index = [pivot+i for i in range(sample_length)]
        return ref_index

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item

# main
if __name__ == '__main__':
    split = 'train'

    dataset = ImgDataset_video(opts=None, sample=5, size=(432, 240), spit='train', name='YouTubeVOS', root='./datasets')
    # print the first 10 video names
    # print(dataset.video_names[:10])

    # # test the load_name function
    # print(dataset.load_name(0))

    # test the load_item function
    print(dataset.load_item(0))

    # test the create_iterator function
    iterator = dataset.create_iterator(batch_size=2)
    for i in range(5):
        item = next(iterator)
        print(item['frames'].shape, item['masks'].shape, item['name'], item['ref_index'])
        # save the loaded item by frames and masks with jpg files
        for j in range(5):
            frames = item['frames'][0][j].numpy().transpose(1, 2, 0)
            masks = item['masks'][0][j].numpy().transpose(1, 2, 0)
            frames = (frames+1.0)*127.5
            masks = masks*255.0
            frames = cv2.cvtColor(frames, cv2.COLOR_RGB2BGR)
            cv2.imwrite('frame'+str(i)+str(j)+'.jpg', frames)
            cv2.imwrite('mask'+str(i)+str(j)+'.jpg', masks)
            # print reference index
            print(item['ref_index'][j])
